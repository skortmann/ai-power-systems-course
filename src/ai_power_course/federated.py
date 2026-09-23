"""Federated learning: aggregation, local training, and communication accounting.

Tutorial 11 uses this. Everything here is deliberately small enough to read:
the point of the tutorial is that students see what a federation *does*, not
that they call a framework that does it for them. Flower appears only after
this loop has been built by hand.

Three cautions are wired into the naming and the docstrings, because they are
the errors this topic invites.

**Raw training samples stay local; updates do not.** Under the standard
federated architecture each client keeps its own dataset and sends model
parameters. "No data leaves the client" is false — parameters are a function
of the data, and §13 of the tutorial shows what can be recovered from them.

**Nothing here provides a formal privacy guarantee.** :func:`clip_update` and
:func:`gaussian_noise` are the two arithmetic steps a DP-SGD mechanism is built
from, and they are useful for showing the privacy-utility trade-off. They are
not a differentially private mechanism: there is no accountant, no calibrated
sigma, and no epsilon. The tutorial says so wherever it uses them.

**Simulated secure aggregation is not secure aggregation.** :func:`
federated_average` computes a sum the way a server with plaintext access would.
A real deployment would run a cryptographic protocol so the server learns only
the sum; that is a different thing and this module does not implement it.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor, nn

__all__ = [
    "ClientData",
    "FederationHistory",
    "federated_average",
    "state_dict_bytes",
    "communication_table",
    "trainable_state_dict",
    "local_train",
    "run_federation",
    "clip_update",
    "gaussian_noise",
    "parameter_delta",
]


# --------------------------------------------------------------------------
# Clients
# --------------------------------------------------------------------------


@dataclass
class ClientData:
    """One participant in the federation, with its own private dataset.

    ``name`` is the DSO label used in every table and plot. ``description``
    carries the one-line characterisation ("rural, wind-dominated") so that a
    per-client result can be read against what that client actually operates,
    which is how heterogeneity becomes visible rather than abstract.
    """

    name: str
    x_train: Tensor
    y_train: Tensor
    x_valid: Tensor
    y_valid: Tensor
    description: str = ""

    @property
    def n_train(self) -> int:
        return int(self.x_train.shape[0])

    def summary(self) -> dict[str, float | str]:
        return {
            "client": self.name,
            "description": self.description,
            "n_train": self.n_train,
            "n_valid": int(self.x_valid.shape[0]),
            "y_mean": float(self.y_train.mean()),
            "y_std": float(self.y_train.std()),
        }


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def federated_average(
    client_states: Sequence[Mapping[str, Tensor]],
    client_sizes: Sequence[int],
) -> dict[str, Tensor]:
    r"""Weighted parameter average — the aggregation step of FedAvg.

    .. math::

        w_{t+1} = \sum_k \frac{n_k}{\sum_j n_j}\, w_{t+1}^{(k)}

    The weighting by dataset size is not cosmetic. It makes the federated
    update equal to one large-batch step over the union of the clients' data
    when each client takes a single full-batch gradient step, which is the
    sense in which FedAvg approximates centralised training. It also means a
    client with ten times the data has ten times the influence -- worth
    remembering when reading per-client results (§30 of the tutorial).

    Returns a NEW dict and never mutates its inputs: a federation that
    accidentally aliases the global tensors will appear to converge while every
    client is silently editing the same object.
    """
    if len(client_states) == 0:
        raise ValueError("cannot aggregate an empty list of client states")
    if len(client_states) != len(client_sizes):
        raise ValueError(
            f"{len(client_states)} client states but {len(client_sizes)} sizes"
        )
    total = float(sum(client_sizes))
    if total <= 0:
        raise ValueError("total dataset size across clients must be positive")

    keys = set(client_states[0].keys())
    for index, state in enumerate(client_states[1:], start=1):
        if set(state.keys()) != keys:
            missing = keys.symmetric_difference(state.keys())
            raise ValueError(
                f"client {index} has different parameter keys; symmetric difference: "
                f"{sorted(missing)[:5]}"
            )

    averaged: dict[str, Tensor] = {}
    for key in client_states[0]:
        stacked = torch.stack(
            [state[key].detach().to(torch.float64) * (size / total)
             for state, size in zip(client_states, client_sizes, strict=True)]
        )
        averaged[key] = stacked.sum(dim=0).to(client_states[0][key].dtype)
    return averaged


def parameter_delta(
    new_state: Mapping[str, Tensor], old_state: Mapping[str, Tensor]
) -> dict[str, Tensor]:
    """``new - old`` per tensor: what a client would transmit as an update."""
    return {k: new_state[k].detach() - old_state[k].detach() for k in new_state}


# --------------------------------------------------------------------------
# Communication accounting
# --------------------------------------------------------------------------


def state_dict_bytes(state: Mapping[str, Tensor]) -> int:
    """Total bytes of a parameter payload, at its real dtype.

    ``element_size()`` rather than an assumed 4 bytes: a float16 or int8
    payload is half or a quarter the size, and the whole argument for
    parameter-efficient federation is about this number.
    """
    return int(sum(t.numel() * t.element_size() for t in state.values()))


def communication_table(
    payloads: Mapping[str, Mapping[str, Tensor]],
    n_rounds: int,
    n_clients: int,
) -> list[dict[str, float]]:
    """Per-method communication cost over a whole federation.

    Counts BOTH directions: each round the server ships parameters down to
    every client and every client ships its update back up. Reporting only the
    upload halves the number, which is the usual way this comparison is made to
    look better than it is.
    """
    rows = []
    for method, payload in payloads.items():
        per_message = state_dict_bytes(payload)
        per_round = per_message * n_clients * 2
        rows.append(
            {
                "method": method,
                "tensors": len(payload),
                "parameters": int(sum(t.numel() for t in payload.values())),
                "bytes_per_message": per_message,
                "MB_per_round": per_round / 1e6,
                "MB_total": per_round * n_rounds / 1e6,
            }
        )
    return rows


def trainable_state_dict(model: nn.Module) -> dict[str, Tensor]:
    """Only the parameters with ``requires_grad``.

    This is what federated PEFT transmits. Calling it on a fully trainable
    model returns everything, which is the point of the comparison in §24.
    """
    return {
        name: param.detach()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


# --------------------------------------------------------------------------
# Local training
# --------------------------------------------------------------------------


def local_train(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    *,
    epochs: int = 1,
    lr: float = 0.01,
    batch_size: int = 32,
    mu: float = 0.0,
    global_state: Mapping[str, Tensor] | None = None,
    loss_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
    generator: torch.Generator | None = None,
) -> float:
    r"""Train ``model`` on one client's data in place; return the mean loss.

    ``mu > 0`` adds FedProx's proximal term

    .. math::

        \frac{\mu}{2}\,\lVert w - w^{t} \rVert^2

    which penalises drift away from the global parameters the round started
    from. ``mu = 0`` is exactly FedAvg, so the two algorithms differ here by
    one term and nothing else -- which is the cleanest way to see what FedProx
    actually changes.
    """
    if mu > 0 and global_state is None:
        raise ValueError("FedProx needs the global state it should stay close to")
    criterion = loss_fn or nn.functional.mse_loss
    optimiser = torch.optim.SGD(model.parameters(), lr=lr)
    anchor = (
        {k: v.detach().clone() for k, v in global_state.items()}
        if global_state is not None
        else None
    )

    n = int(x.shape[0])
    losses: list[float] = []
    for _ in range(epochs):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, batch_size):
            index = order[start : start + batch_size]
            optimiser.zero_grad()
            loss = criterion(model(x[index]), y[index])
            if mu > 0:
                proximal = torch.zeros((), dtype=loss.dtype)
                for name, param in model.named_parameters():
                    if name in anchor:
                        proximal = proximal + ((param - anchor[name]) ** 2).sum()
                loss = loss + 0.5 * mu * proximal
            loss.backward()
            optimiser.step()
            losses.append(float(loss.detach()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def evaluate(model: nn.Module, x: Tensor, y: Tensor) -> float:
    """Mean absolute error, in the units of the target."""
    model.eval()
    value = float((model(x) - y).abs().mean())
    model.train()
    return value


# --------------------------------------------------------------------------
# The federation loop
# --------------------------------------------------------------------------


@dataclass
class FederationHistory:
    """Everything a federation run produced, for plotting and for tables."""

    rounds: list[int] = field(default_factory=list)
    global_mae: list[float] = field(default_factory=list)
    per_client_mae: list[dict[str, float]] = field(default_factory=list)
    local_losses: list[dict[str, float]] = field(default_factory=list)
    drift: list[float] = field(default_factory=list)

    def worst_client(self) -> float:
        """The worst per-client MAE in the final round.

        Reported alongside the mean throughout the tutorial, because a mean
        over clients is exactly the statistic that hides a client the
        federation is failing.
        """
        return max(self.per_client_mae[-1].values()) if self.per_client_mae else float("nan")

    def client_spread(self) -> float:
        """Standard deviation of the final per-client MAE across clients."""
        if not self.per_client_mae:
            return float("nan")
        return float(np.std(list(self.per_client_mae[-1].values())))


def _client_drift(
    client_states: Sequence[Mapping[str, Tensor]], global_state: Mapping[str, Tensor]
) -> float:
    """Mean L2 distance from each client's post-training parameters to the global ones.

    This is the quantity "client drift" names. It rises with local epochs and
    with heterogeneity, and §10 of the tutorial plots it against both.
    """
    distances = []
    for state in client_states:
        total = 0.0
        for key, value in state.items():
            total += float(((value - global_state[key]) ** 2).sum())
        distances.append(total**0.5)
    return float(np.mean(distances))


def run_federation(
    make_model: Callable[[], nn.Module],
    clients: Sequence[ClientData],
    *,
    n_rounds: int = 10,
    local_epochs: int = 1,
    lr: float = 0.01,
    batch_size: int = 32,
    mu: float = 0.0,
    participation: float = 1.0,
    clip_norm: float | None = None,
    noise_std: float = 0.0,
    seed: int = 0,
    global_eval: tuple[Tensor, Tensor] | None = None,
) -> tuple[nn.Module, FederationHistory]:
    """Run FedAvg (or FedProx when ``mu > 0``) and record what happened.

    ``participation < 1`` samples a subset of clients each round, which is the
    realistic case even in cross-silo federations where a partner may be
    offline for maintenance.

    ``clip_norm`` and ``noise_std`` apply the two arithmetic steps of DP-SGD to
    each client's update. They let the tutorial measure the utility cost of
    adding noise. **They do not make this differentially private** -- there is
    no accountant and no epsilon, and the tutorial says so at the point of use.
    """
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)

    global_model = make_model()
    global_state = {k: v.detach().clone() for k, v in global_model.state_dict().items()}
    history = FederationHistory()

    for round_index in range(1, n_rounds + 1):
        n_selected = max(1, int(round(participation * len(clients))))
        selected = (
            list(clients)
            if n_selected >= len(clients)
            else [clients[i] for i in rng.choice(len(clients), n_selected, replace=False)]
        )

        client_states: list[dict[str, Tensor]] = []
        sizes: list[int] = []
        losses: dict[str, float] = {}

        for client in selected:
            model = make_model()
            model.load_state_dict(copy.deepcopy(global_state))
            losses[client.name] = local_train(
                model,
                client.x_train,
                client.y_train,
                epochs=local_epochs,
                lr=lr,
                batch_size=batch_size,
                mu=mu,
                global_state=global_state if mu > 0 else None,
                generator=generator,
            )
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}

            if clip_norm is not None or noise_std > 0:
                update = parameter_delta(state, global_state)
                if clip_norm is not None:
                    update = clip_update(update, clip_norm)
                if noise_std > 0:
                    update = gaussian_noise(update, noise_std, generator=generator)
                state = {k: global_state[k] + update[k] for k in state}

            client_states.append(state)
            sizes.append(client.n_train)

        history.drift.append(_client_drift(client_states, global_state))
        global_state = federated_average(client_states, sizes)
        global_model.load_state_dict(global_state)

        per_client = {
            c.name: evaluate(global_model, c.x_valid, c.y_valid) for c in clients
        }
        history.rounds.append(round_index)
        history.per_client_mae.append(per_client)
        history.local_losses.append(losses)
        history.global_mae.append(
            evaluate(global_model, *global_eval)
            if global_eval is not None
            else float(np.mean(list(per_client.values())))
        )

    return global_model, history


# --------------------------------------------------------------------------
# The two arithmetic steps a DP mechanism is built from
# --------------------------------------------------------------------------


def clip_update(update: Mapping[str, Tensor], max_norm: float) -> dict[str, Tensor]:
    """Scale an update so its global L2 norm is at most ``max_norm``.

    Clipping bounds one client's influence on the aggregate, which is what
    makes a noise level meaningful: without a bound on the contribution there
    is no sensitivity to calibrate noise against. On its own it is not a
    privacy mechanism.
    """
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    total = float(sum(float((t**2).sum()) for t in update.values())) ** 0.5
    if total <= max_norm or total == 0.0:
        return {k: v.clone() for k, v in update.items()}
    scale = max_norm / total
    return {k: v * scale for k, v in update.items()}


def gaussian_noise(
    update: Mapping[str, Tensor],
    std: float,
    generator: torch.Generator | None = None,
) -> dict[str, Tensor]:
    """Add zero-mean Gaussian noise of standard deviation ``std`` to an update.

    **This is not differential privacy.** A DP mechanism calibrates ``std`` to
    the clipping norm and a target ``(epsilon, delta)``, and tracks the budget
    across rounds with an accountant. This function does the arithmetic and
    reports nothing about a guarantee, which is exactly why the tutorial uses
    it to show the utility cost and stops short of quoting an epsilon.
    """
    if std < 0:
        raise ValueError("noise std must be non-negative")
    return {
        k: v + torch.normal(0.0, std, size=v.shape, generator=generator)
        for k, v in update.items()
    }


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """``(trainable, total)`` parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def describe_clients(clients: Iterable[ClientData]):
    """A DataFrame of the per-client summaries, for the heterogeneity section."""
    import pandas as pd

    return pd.DataFrame([c.summary() for c in clients]).set_index("client")


# --------------------------------------------------------------------------
# Four DSOs, built from genuinely different grids
# --------------------------------------------------------------------------

#: Each entry is one simulated Distribution System Operator: a base network
#: (topology heterogeneity) plus sampling knobs (distribution heterogeneity).
#:
#: Randomly splitting one homogeneous table would produce clients that differ
#: only by sampling noise, and every federated method would look equally good
#: on it. These four differ in the way real operators differ -- a rural feeder
#: with high DER variance is a different learning problem from an urban one
#: with heavy demand and little generation -- which is what makes the non-IID
#: experiments in sections 8 to 10 show anything.
DSO_PROFILES: dict[str, dict] = {
    "DSO_A": {
        "network": "case33bw",
        "description": "residential radial feeder, high PV penetration",
        "load_scale": (0.7, 1.1),
        "sgen_scale": (0.8, 2.0),
        "load_spread": 0.18,
    },
    "DSO_B": {
        "network": "case14",
        "description": "rural, strong and variable wind generation",
        "load_scale": (0.5, 0.9),
        "sgen_scale": (0.0, 2.2),
        "load_spread": 0.25,
    },
    "DSO_C": {
        "network": "case30",
        "description": "urban, heavy demand with EV charging, little local DER",
        "load_scale": (1.0, 1.4),
        "sgen_scale": (0.0, 0.4),
        "load_spread": 0.12,
    },
    "DSO_D": {
        "network": "case57",
        "description": "mixed load and generation (held out for transfer)",
        "load_scale": (0.7, 1.2),
        "sgen_scale": (0.3, 1.2),
        "load_spread": 0.15,
    },
}


def operating_point_features(point) -> tuple[list[float], float]:
    """Turn one solved operating point into ``(features, target)``.

    The task is a **state-estimation surrogate**: predict the minimum bus
    voltage of the network from quantities an operator reads off SCADA without
    solving a power flow. Features are aggregate injections and a little
    topology; the target is the quantity that actually decides whether the
    operating point is acceptable.

    Every feature is an aggregate, so it is the same length whatever the
    network size -- which is what lets four DSOs with different topologies
    train one shared model at all.
    """
    import numpy as _np

    nodes = point.graph["node_features"].numpy()
    # Node feature layout is fixed by grid.graphs.NODE_FEATURE_NAMES; the
    # leading block is state (p, q, vm, va) and the rest is equipment.
    p, q = nodes[:, 0], nodes[:, 1]
    features = [
        float(p.sum()),
        float(q.sum()),
        float(_np.abs(p).max()),
        float(_np.abs(q).max()),
        float(p.std()),
        float(point.load_scale),
        float(nodes.shape[0]),
        float(point.graph["edge_index"].shape[1]),
    ]
    return features, float(point.vm_pu.min())


def build_dso_clients(
    names: Sequence[str] | None = None,
    *,
    n_samples: int = 120,
    valid_fraction: float = 0.25,
    seed: int = 20260101,
    verbose: bool = False,
) -> list[ClientData]:
    """Sample operating points per DSO and package them as federation clients.

    Each DSO gets its own derived seed, so adding or removing a participant
    does not change anybody else's data -- otherwise every ablation would
    silently re-draw the whole federation.
    """
    from .grid.sampling import SamplingConfig, sample_operating_points

    chosen = list(names) if names is not None else list(DSO_PROFILES)
    clients: list[ClientData] = []

    for offset, name in enumerate(chosen):
        profile = DSO_PROFILES[name]
        config = SamplingConfig(
            n_samples=n_samples,
            load_scale=profile["load_scale"],
            sgen_scale=profile["sgen_scale"],
            load_spread=profile["load_spread"],
            seed=seed + 1000 * (offset + 1),
        )
        points, _ = sample_operating_points(
            profile["network"], config, verbose=verbose
        )
        rows = [operating_point_features(p) for p in points]
        x = torch.tensor([r[0] for r in rows], dtype=torch.float32)
        y = torch.tensor([[r[1]] for r in rows], dtype=torch.float32)

        cut = int(len(x) * (1.0 - valid_fraction))
        clients.append(
            ClientData(
                name=name,
                x_train=x[:cut],
                y_train=y[:cut],
                x_valid=x[cut:],
                y_valid=y[cut:],
                description=f"{profile['network']} — {profile['description']}",
            )
        )
    return clients


def standardise_clients(clients: Sequence[ClientData]) -> list[ClientData]:
    """Standardise features using each client's OWN training statistics.

    Deliberately per-client, and it is worth being clear why. Computing one
    global mean and standard deviation over the union of the clients' data
    would be a small federated analytics computation that every participant
    contributed to -- which is exactly the point section 34 makes, and exactly
    the kind of step that is easy to perform accidentally by pooling the data
    first. Local statistics need no coordination and leak nothing extra.

    It is not free: clients end up in slightly different feature spaces, which
    is one more source of the heterogeneity the federation has to absorb.
    """
    out = []
    for c in clients:
        mean = c.x_train.mean(dim=0, keepdim=True)
        std = c.x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
        out.append(
            ClientData(
                name=c.name,
                x_train=(c.x_train - mean) / std,
                y_train=c.y_train,
                x_valid=(c.x_valid - mean) / std,
                y_valid=c.y_valid,
                description=c.description,
            )
        )
    return out
