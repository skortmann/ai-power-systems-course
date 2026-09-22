"""Generating many operating points across many grids.

A foundation model needs a *broad distribution* to pretrain on. For grids, that
breadth has three axes, and this module varies all three:

1. **Loading** — global demand level plus per-bus variation.
2. **Generation** — renewable in-feed, which can push flows in reverse.
3. **Topology** — single branch outages (N-1), so the model sees more than one
   graph per network.

The result is a dataset of solved AC states. Operating points the power-flow
solver cannot converge on are recorded as failures rather than silently
dropped: their rate is itself a useful diagnostic, and tutorial 10 reports it.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import pandapower as pp
import torch

from .graphs import net_to_graph
from .networks import load_network, run_power_flow

__all__ = ["SamplingConfig", "OperatingPoint", "sample_operating_points", "build_grid_dataset"]


@dataclass
class SamplingConfig:
    """How to perturb a base network into a population of operating points."""

    n_samples: int = 200
    #: Global demand multiplier, sampled uniformly.
    load_scale: tuple[float, float] = (0.6, 1.3)
    #: Log-normal sigma of the per-load deviation around the global scale.
    load_spread: float = 0.15
    #: Multiplier on static generators (PV/wind), sampled uniformly.
    sgen_scale: tuple[float, float] = (0.0, 1.6)
    #: Multiplier on dispatchable generator set-points.
    gen_scale: tuple[float, float] = (0.85, 1.15)
    #: Probability of taking one random in-service line out (an N-1 case).
    outage_probability: float = 0.15
    #: Reject a sample if an outage de-energises more than this share of buses.
    #: Losing a couple of buses behind a radial branch is a normal N-1 outcome;
    #: losing half the network is a system split, which is a different problem
    #: and would dominate the pretraining distribution if left in.
    min_energised_fraction: float = 0.8
    seed: int = 20260101
    max_attempts_factor: int = 3


@dataclass
class OperatingPoint:
    """One solved AC state of one grid, as tensors plus provenance."""

    network: str
    graph: dict[str, torch.Tensor]
    vm_pu: np.ndarray
    va_degree: np.ndarray
    max_line_loading: float
    n_overloaded: int
    has_outage: bool
    load_scale: float
    meta: dict[str, float] = field(default_factory=dict)

    @property
    def n_buses(self) -> int:
        return int(self.graph["node_features"].shape[0])


def _perturb(net: pp.pandapowerNet, cfg: SamplingConfig, rng: np.random.Generator) -> dict:
    """Apply one random perturbation in place. Returns what was done."""
    info: dict[str, float] = {}

    scale = float(rng.uniform(*cfg.load_scale))
    info["load_scale"] = scale
    if len(net.load):
        jitter = rng.lognormal(mean=0.0, sigma=cfg.load_spread, size=len(net.load))
        factors = scale * jitter
        net.load["p_mw"] = net.load["p_mw"].to_numpy() * factors
        net.load["q_mvar"] = net.load["q_mvar"].to_numpy() * factors

    if len(net.sgen):
        sgen_factor = float(rng.uniform(*cfg.sgen_scale))
        info["sgen_scale"] = sgen_factor
        net.sgen["p_mw"] = net.sgen["p_mw"].to_numpy() * sgen_factor

    if len(net.gen):
        gen_factor = float(rng.uniform(*cfg.gen_scale))
        info["gen_scale"] = gen_factor
        net.gen["p_mw"] = net.gen["p_mw"].to_numpy() * gen_factor

    info["has_outage"] = 0.0
    if rng.random() < cfg.outage_probability and len(net.line) > 1:
        candidates = net.line.index[net.line.in_service].to_numpy()
        if candidates.size > 1:
            net.line.loc[rng.choice(candidates), "in_service"] = False
            info["has_outage"] = 1.0
    return info


def sample_operating_points(
    network: str, config: SamplingConfig | None = None, verbose: bool = False
) -> tuple[list[OperatingPoint], dict[str, int]]:
    """Sample solved operating points for one catalogued grid.

    Returns ``(operating_points, statistics)`` where ``statistics`` counts
    attempts, convergence failures and islanded cases. The base network is
    reloaded for every sample so perturbations never accumulate.
    """
    cfg = config or SamplingConfig()
    rng = np.random.default_rng(cfg.seed)
    points: list[OperatingPoint] = []
    stats = {"attempts": 0, "converged": 0, "failed": 0, "islanded": 0}

    # Building a pandapower network from its stored definition costs ~400 ms,
    # twenty times a power flow. Build it once and deep-copy the pristine,
    # unsolved base for each sample so perturbations never accumulate.
    base = load_network(network)

    max_attempts = cfg.n_samples * cfg.max_attempts_factor
    while len(points) < cfg.n_samples and stats["attempts"] < max_attempts:
        stats["attempts"] += 1
        net = copy.deepcopy(base)
        info = _perturb(net, cfg, rng)
        if not run_power_flow(net):
            stats["failed"] += 1
            continue
        energised = int(net.res_bus.vm_pu.notna().sum())
        if energised < cfg.min_energised_fraction * len(net.bus):
            stats["islanded"] += 1
            continue
        stats["converged"] += 1

        loadings = net.res_line.loading_percent.to_numpy() if len(net.res_line) else np.array([0.0])
        loadings = loadings[np.isfinite(loadings)]
        if loadings.size == 0:
            loadings = np.array([0.0])
        try:
            graph = net_to_graph(net)
        except ValueError:
            # A "converged" solution can still be degenerate or contain
            # non-finite entries. Treat it as a failed sample rather than
            # poisoning the dataset with NaNs.
            stats["failed"] += 1
            stats["converged"] -= 1
            continue
        # The state is taken from the graph, not from res_bus: after an outage
        # the graph covers only the energised buses, and the two must align.
        points.append(
            OperatingPoint(
                network=network,
                graph=graph,
                vm_pu=graph["vm_pu"].numpy(),
                va_degree=graph["va_degree"].numpy(),
                max_line_loading=float(loadings.max()),
                n_overloaded=int((loadings > 100.0).sum()),
                has_outage=bool(info["has_outage"]),
                load_scale=float(info["load_scale"]),
                meta=info,
            )
        )
    if verbose:
        attempts = max(stats["attempts"], 1)
        print(
            f"{network:16s} {len(points):5d} states from {stats['attempts']:5d} attempts  "
            f"({stats['failed'] / attempts:5.1%} no convergence, "
            f"{stats['islanded'] / attempts:5.1%} islanded)"
        )
    return points, stats


def build_grid_dataset(
    networks: tuple[str, ...],
    config: SamplingConfig | None = None,
    verbose: bool = True,
) -> list[OperatingPoint]:
    """Sample operating points for several grids and concatenate them.

    Each grid gets its own derived seed so that adding a network to the list
    does not change the samples drawn for the others — otherwise every ablation
    would silently change the whole dataset.
    """
    cfg = config or SamplingConfig()
    dataset: list[OperatingPoint] = []
    for offset, name in enumerate(networks):
        per_grid = SamplingConfig(**{**cfg.__dict__, "seed": cfg.seed + 1000 * (offset + 1)})
        points, _ = sample_operating_points(name, per_grid, verbose=verbose)
        dataset.extend(points)
    return dataset
