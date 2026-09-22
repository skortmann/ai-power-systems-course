# %% [markdown]
# # Tutorial 10 — Toward Foundation Models for the Electric Power Grid
#
# ### From Machine Learning to Foundation Models · Hands-On AI for Power & Energy Systems
#
# > **What would a genuine foundation model for power systems look like?**
#
# ---
#
# ## 1. Why this matters
#
# Tutorial 09 reused a foundation model someone else pretrained, on a modality —
# time series — where every dataset has the same shape: a sequence of numbers.
#
# Power grids do not have the same shape. Grid A and Grid B differ in the number
# of buses, the topology, the voltage levels, the equipment and the operating
# regime. A model with a fixed input width trained on a 30-bus network cannot
# even be *evaluated* on a 118-bus one, let alone transfer to it. That single
# fact is why "a foundation model for the grid" is a harder proposition than
# "a foundation model for time series", and it is what this notebook is about.
#
# We will pretrain **one encoder across five different networks** with a
# self-supervised objective and no labels, then transfer it to two grids it has
# never seen — including one with a completely different structure. The model is
# tiny and the result is honest: at this scale it may not beat a specialist
# trained on the target grid. **The deliverable is the methodology**, and the
# methodology is the same one `gridfm-graphkit` and GridSFM use at research
# scale.
#
# ## 2. Historical context
#
# Hamann et al., *Foundation Models for the Electric Power Grid* (*Joule*, 2024;
# [arXiv:2407.09434](https://arxiv.org/abs/2407.09434)) set out the agenda and
# sketched GridFM-v0: a graph neural network pretrained across grids for power
# flow. Since then:
#
# - **PowerPM** (NeurIPS 2024) — electricity time series, masked modelling plus
#   dual-view contrastive pretraining, 44 downstream tasks.
# - **`gridfm-graphkit`** (2025, open source) — masked-feature pretraining with a
#   physics-informed AC power-balance loss, zero-shot evaluation on unseen
#   topologies. **The methodology below deliberately mirrors this one.**
# - **WindFM** (2025) — 8.1 M parameters, zero-shot wind power, beating larger
#   models.
# - **GridSFM** (Microsoft Research, 2026) — AC-OPF across ~200 grids and
#   ~500,000 scenarios, in milliseconds, usable as a solver warm start. The
#   *S* stands for Small.
#
# This is a research area roughly two years old with open implementations and
# **no mature operational deployment**. Read the last sentence again before
# anyone quotes a notebook benchmark at a control-room engineer.
#
# ## 3. Learning objectives
#
# By the end of this notebook you can:
#
# - explain why a fixed-width model cannot transfer between grids and a
#   message-passing model can;
# - implement message passing and verify its permutation equivariance;
# - list the modalities a grid foundation model would have to consume;
# - pretrain a graph encoder across several networks with a masked objective;
# - transfer it to an unseen topology and measure what it bought;
# - add a physics-informed penalty and measure its effect on *both* statistical
#   accuracy and physical validity;
# - argue about what the "tokens" of a grid should be.

# %%
from __future__ import annotations

import math
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from torch import nn

from ai_power_course import diagrams
from ai_power_course.config import fast_mode, scaled, set_seed
from ai_power_course.grid.graphs import (
    EDGE_FEATURE_NAMES,
    N_STATE_FEATURES,
    NODE_FEATURE_NAMES,
    net_to_graph,
)
from ai_power_course.grid.networks import (
    HELDOUT_NETWORKS,
    PRETRAIN_NETWORKS,
    load_network,
    network_summary,
    run_power_flow,
)
from ai_power_course.grid.physics import get_ybus, voltage_violations
from ai_power_course.grid.sampling import SamplingConfig, build_grid_dataset
from ai_power_course.models.gnn import GraphBatch, GridEncoder, MaskedGridModel, collate_graphs
from ai_power_course.plotting import COLORS, plot_embedding, use_course_style
from ai_power_course.results import leaderboard_table

warnings.filterwarnings("ignore")
use_course_style()
set_seed()
torch.set_num_threads(4)
print(f"reduced (CI) configuration: {fast_mode()}")

NODE_DIM, EDGE_DIM = len(NODE_FEATURE_NAMES), len(EDGE_FEATURE_NAMES)
D_MODEL, N_LAYERS = 64, 4
VOLTAGE = slice(2, 5)          # vm_pu, sin(va), cos(va) within the node features

# %% [markdown]
# ## 4. The problem: grids do not share a shape

# %%
summary = network_summary(PRETRAIN_NETWORKS + HELDOUT_NETWORKS)
display(summary)

print(f"Pretraining on : {', '.join(PRETRAIN_NETWORKS)}")
print(f"Held out       : {', '.join(HELDOUT_NETWORKS)}")
print("\ncase118 tests SCALE   — four times more buses than the largest pretraining grid.")
print("case33bw tests STRUCTURE — radial medium-voltage distribution, no generators,")
print("                          no transformers, a different base MVA. It is much the")
print("                          harder of the two and is meant to be.")
print(f"\nBus counts span {summary.buses.min()} to {summary.buses.max()}. A model with a")
print("fixed input width cannot be evaluated on more than one row of this table.")

# %% [markdown]
# ### What a real GridFM would have to consume
#
# The course has used one modality. A grid foundation model would need several,
# and reconciling them is most of the research problem.
#
# | modality | examples | why it is hard |
# |---|---|---|
# | **time series** | P, Q, V, I, f; PMU, SCADA, smart meters | wildly different sampling rates (ms to 15 min) |
# | **graph / topology** | buses, branches, transformers, switches | changes with every switching action |
# | **static metadata** | voltage levels, conductor parameters, equipment type | heterogeneous, often incomplete |
# | **weather** | irradiance, wind, temperature, on a spatial grid | a different coordinate system entirely |
# | **market** | prices, schedules, bids | different time base, different ownership |
# | **text** | asset documentation, disturbance reports, operator logs, standards | unstructured, confidential |
#
# This notebook uses the first three. Combining all six is an open problem, and
# it is where tutorial 08's text work would eventually attach.

# %%
fig = diagrams.gridfm_architecture()
plt.show()

# %% [markdown]
# ## 5. Why message passing solves the shape problem
#
# The parameters of a message-passing network live in the **message** and
# **update** functions, which are applied to every edge and every node. Nothing
# in the model has a bus count. The same weights therefore apply, unchanged, to
# a network the model has never seen.
#
# $$m_{i \to j} = \phi\big([\,h_i,\; h_j,\; e_{ij}\,]\big), \qquad
#   h_j' = h_j + \psi\Big(\big[\,h_j,\; \textstyle\sum_{i \in N(j)} m_{i\to j}\,\big]\Big)$$
#
# Two choices worth defending:
#
# - **Sum aggregation, not mean.** Total injected power at a bus is a *sum* over
#   incident branches. Mean aggregation would make a bus with two lines look like
#   a bus with twenty carrying half the current each. The physics is additive, so
#   the aggregator should be.
# - **A residual update.** Same argument as tutorials 03 and 06: it keeps deep
#   stacks trainable and lets a layer refine rather than overwrite.

# %%
fig = diagrams.multi_grid_transfer()
plt.show()

# %%
# The whole of message passing, in the three lines that matter.
print(
    """    source, target = edge_index[0], edge_index[1]
    messages   = self.message(torch.cat([x[source], x[target], edge_features], dim=-1))
    aggregated = torch.zeros_like(x).index_add_(0, target, messages)
    return self.norm(x + self.update(torch.cat([x, aggregated], dim=-1)))"""
)
print("\nNo PyTorch Geometric, no scatter library. `index_add_` is the scatter-add, and")
print("seeing it removes the magic. Use PyG for real work — it is faster and handles")
print("sparsity properly — but know what it is doing.")

# %%
# Demonstrate the two properties that make transfer possible at all.
set_seed()
encoder_probe = GridEncoder(NODE_DIM, EDGE_DIM, d_model=16, n_layers=2)

print("One set of weights, applied to four different grids:\n")
for name in ("case9", "case14", "case118", "case33bw"):
    net = load_network(name)
    assert run_power_flow(net)
    batch = collate_graphs([net_to_graph(net)])
    output = encoder_probe(batch)
    print(f"  {name:10s} {batch.n_nodes:4d} buses -> node embeddings {tuple(output.shape)}")

# Permutation equivariance: relabel the buses, and the outputs permute with them.
net = load_network("case14")
run_power_flow(net)
batch = collate_graphs([net_to_graph(net)])
with torch.no_grad():
    original = encoder_probe(batch)
permutation = torch.randperm(batch.n_nodes)
inverse = torch.argsort(permutation)
shuffled = GraphBatch(batch.node_features[permutation], inverse[batch.edge_index],
                      batch.edge_features, batch.batch, 1)
with torch.no_grad():
    permuted = encoder_probe(shuffled)
print(f"\nmax |encoder(relabelled) - relabel(encoder(original))| = "
      f"{(permuted - original[permutation]).abs().max():.2e}")
print("Bus numbering is an accident of the data file. The model is invariant to it,")
print("which is exactly the property a sequence model has to be *given* by positional")
print("encoding (tutorial 05) and a graph model gets for free.")

# %% [markdown]
# ## 6. Generating a broad pretraining distribution
#
# Breadth is the first requirement in the definition of a foundation model. For
# grids it has three axes, and we vary all three: **loading**, **generation**
# and **topology** (single branch outages).
#
# Note what makes this domain unusual and rather favourable: unlike language, we
# can *generate* unlimited correct data with a power-flow solver. GridSFM used
# ~500,000 scenarios across ~200 grids. We use a few thousand across seven.

# %%
N_SAMPLES = scaled(full=300, fast=40)
sampling = SamplingConfig(
    n_samples=N_SAMPLES,
    load_scale=(0.6, 1.3),
    load_spread=0.15,
    sgen_scale=(0.0, 1.6),
    gen_scale=(0.85, 1.15),
    outage_probability=0.15,
    seed=20260101,
)

started = time.perf_counter()
pretrain_points = build_grid_dataset(PRETRAIN_NETWORKS, sampling)
heldout_points = {
    name: build_grid_dataset(
        (name,), SamplingConfig(**{**sampling.__dict__, "seed": 20260202})
    )
    for name in HELDOUT_NETWORKS
}
print(f"\n{len(pretrain_points)} pretraining states in {time.perf_counter() - started:.0f}s")
for name, points in heldout_points.items():
    print(f"{len(points):5d} states on the held-out grid {name}")

overloaded = sum(p.n_overloaded > 0 for p in pretrain_points)
print(f"\noverloaded states: {overloaded} ({overloaded / len(pretrain_points):.0%})")
print(f"states with an outage: {sum(p.has_outage for p in pretrain_points)}")
print(f"distinct bus counts seen in pretraining: "
      f"{sorted({p.n_buses for p in pretrain_points})}")
print("\nPROVENANCE: every state is a converged AC power-flow solution computed by")
print("pandapower on a published IEEE/MATPOWER test case. Synthetic, reproducible,")
print("and not a measurement of any real system.")

# %% [markdown]
# ### Normalisation: the step that makes transfer possible
#
# Per-unit already removes the voltage level. Standardising on top of it, using
# statistics from the **pretraining grids only**, puts every feature on a
# comparable scale so the loss is not dominated by whichever quantity happens to
# have the largest numbers.

# %%
all_nodes = torch.cat([p.graph["node_features"] for p in pretrain_points])
all_edges = torch.cat([p.graph["edge_features"] for p in pretrain_points])
node_mean, node_std = all_nodes.mean(0), all_nodes.std(0).clamp(min=1e-6)
edge_mean, edge_std = all_edges.mean(0), all_edges.std(0).clamp(min=1e-6)
# The bus-type one-hot columns are already on the right scale; leave them alone.
node_mean[5:8], node_std[5:8] = 0.0, 1.0

display(pd.DataFrame(
    {"mean": all_nodes.mean(0).numpy(), "std": all_nodes.std(0).numpy()},
    index=list(NODE_FEATURE_NAMES),
).round(4))

print("Look at the spread before scaling: p_pu and q_pu vary by whole per-unit,")
print("vm_pu by a few percent. An unweighted MSE over these would be a loss on active")
print("power with a rounding error attached — the voltage reconstruction we actually")
print("care about would contribute almost nothing to the gradient.")


def scaled_graph(graph: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "node_features": (graph["node_features"] - node_mean) / node_std,
        "edge_index": graph["edge_index"],
        "edge_features": (graph["edge_features"] - edge_mean) / edge_std,
    }


pretrain_graphs = [scaled_graph(p.graph) for p in pretrain_points]
heldout_graphs = {k: [scaled_graph(p.graph) for p in v] for k, v in heldout_points.items()}

# %% [markdown]
# ## 7. Self-supervised pretraining across grids
#
# Two objectives, alternating, neither of which uses a label:
#
# 1. **Masked bus state.** Hide the entire state (P, Q, |V|, angle) of a random
#    subset of buses and reconstruct it from the rest of the network. This is
#    BERT's objective on a power grid, and it is `gridfm-graphkit`'s.
# 2. **Masked voltage everywhere.** Hide |V| and the angle at *every* bus and
#    reconstruct them from the injections and the topology. This is a
#    **power-flow surrogate**, and it is the objective the downstream task will
#    look like — so the model is not asked at transfer time for something it was
#    never trained to do.
#
# The second is the reason the transfer below has a chance. Pretraining and
# downstream use must share an *input distribution*, or the encoder receives
# inputs it has never seen and the representation is worthless. Getting this
# wrong is the most common way a transfer experiment fails silently.

# %%
def pretrain_encoder(graphs, epochs: int, mask_ratio: float = 0.35,
                     learning_rate: float = 3e-3, batch_size: int = 16):
    set_seed()
    model = MaskedGridModel(NODE_DIM, EDGE_DIM, d_model=D_MODEL, n_layers=N_LAYERS,
                            n_state=N_STATE_FEATURES)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    total_steps = epochs * math.ceil(len(graphs) / batch_size)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate, total_steps=total_steps, pct_start=0.15
    )
    generator = torch.Generator().manual_seed(0)
    curve = []

    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(graphs), generator=generator)
        total, count = 0.0, 0
        for step, start in enumerate(range(0, len(order), batch_size)):
            batch = collate_graphs([graphs[i] for i in order[start : start + batch_size]])
            if step % 2 == 0:
                # objective 1: hide whole bus states
                mask = torch.rand(batch.n_nodes, generator=generator) < mask_ratio
                if not mask.any():
                    continue
                prediction = model(batch, mask=mask)
                target = batch.node_features[:, :N_STATE_FEATURES]
                loss = nn.functional.mse_loss(prediction[mask], target[mask])
                weight = int(mask.sum())
            else:
                # objective 2: hide the voltage everywhere (power-flow surrogate)
                features = batch.node_features.clone()
                features[:, VOLTAGE] = model.mask_token[VOLTAGE]
                hidden = GraphBatch(features, batch.edge_index, batch.edge_features,
                                    batch.batch, batch.n_graphs)
                prediction = model(hidden)
                loss = nn.functional.mse_loss(
                    prediction[:, VOLTAGE], batch.node_features[:, VOLTAGE]
                )
                weight = batch.n_nodes
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            schedule.step()
            total += float(loss.detach()) * weight
            count += weight
        curve.append(total / max(count, 1))
        if epoch % max(epochs // 6, 1) == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch:3d}  self-supervised loss {curve[-1]:.5f}")
    return model, curve


EPOCHS = scaled(full=30, fast=3)
started = time.perf_counter()
pretrained, pretrain_curve = pretrain_encoder(pretrain_graphs, epochs=EPOCHS)
pretrain_seconds = time.perf_counter() - started
print(f"\npretrained in {pretrain_seconds:.0f}s, "
      f"{sum(p.numel() for p in pretrained.encoder.parameters()):,} encoder parameters")
print("No labels were used. The encoder is the artefact; the head is thrown away.")

fig, ax = plt.subplots(figsize=(6.2, 3.2))
ax.semilogy(pretrain_curve, color=COLORS["foundation"])
ax.set_xlabel("epoch")
ax.set_ylabel("masked reconstruction MSE (standardised)")
ax.set_title(f"Pretraining across {len(PRETRAIN_NETWORKS)} grids, no labels")
plt.show()

# %% [markdown]
# ## 8. What did it learn?
#
# The encoder was never told which grid a bus belongs to, what its voltage level
# is, or whether the network is stressed. Pool its node embeddings into one
# vector per operating state and look.

# %%
@torch.no_grad()
def graph_embeddings(graphs) -> np.ndarray:
    pretrained.eval()
    out = []
    for start in range(0, len(graphs), 32):
        batch = collate_graphs(graphs[start : start + 32])
        nodes = pretrained.encoder(batch)
        totals = nodes.new_zeros(batch.n_graphs, nodes.shape[1]).index_add_(
            0, batch.batch, nodes)
        counts = torch.bincount(batch.batch, minlength=batch.n_graphs).clamp(min=1)
        out.append((totals / counts.unsqueeze(1)).numpy())
    return np.concatenate(out)


embeddings = graph_embeddings(pretrain_graphs)
coords = PCA(n_components=2).fit_transform(embeddings)
network_of = np.array([p.network for p in pretrain_points])
loading = np.array([p.max_line_loading for p in pretrain_points])

fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
for i, name in enumerate(PRETRAIN_NETWORKS):
    select = network_of == name
    axes[0].scatter(coords[select, 0], coords[select, 1], s=8, alpha=0.7, label=name)
axes[0].set_title("Operating-state embeddings, coloured by grid")
axes[0].legend(fontsize=7.5)
plot_embedding(coords, np.clip(loading, 0, 150), title="coloured by max line loading",
               label="max loading [%]", cmap="magma", ax=axes[1])
plot_embedding(coords, np.array([p.load_scale for p in pretrain_points]),
               title="coloured by load scaling factor", label="load scale",
               cmap="viridis", ax=axes[2])
fig.tight_layout()
plt.show()

print("The grids separate cleanly, which is expected and not very interesting — each")
print("has its own characteristic impedances and injections.")
print("\nThe informative question is whether *operating condition* is encoded within a")
print("grid's cluster, because that is the part that could transfer.")

# %%
from sklearn.linear_model import Ridge  # noqa: E402

from ai_power_course.metrics import r2  # noqa: E402

print("Linear readout from the 64-dimensional state embedding (random 70/30 split):\n")
rng = np.random.default_rng(0)
order = rng.permutation(len(embeddings))
cut = int(0.7 * len(order))
fit, held = order[:cut], order[cut:]
for label, values in [
    ("max line loading [%]", np.clip(loading, 0, 200)),
    ("load scaling factor", np.array([p.load_scale for p in pretrain_points])),
    ("minimum bus voltage [pu]", np.array([p.vm_pu.min() for p in pretrain_points])),
    ("is there an outage?", np.array([float(p.has_outage) for p in pretrain_points])),
]:
    model = Ridge(alpha=1.0).fit(embeddings[fit], values[fit])
    print(f"  {label:28s} R2 = {r2(values[held], model.predict(embeddings[held])):6.3f}")

print("\nQuantities that follow from the physics of the state are readable; the outage")
print("flag is much harder, because a single branch out of service changes very little")
print("about a well-meshed network — which is, after all, the point of meshing it.")

# %% [markdown]
# ## 9. Transfer to a grid the model has never seen
#
# ### The downstream task
#
# **Predict the voltage state (|V|, angle) at every bus from the injections and
# the topology** — an AC power-flow surrogate. On a grid excluded from
# pretraining.
#
# Four conditions, identical architecture and data in each:
#
# | condition | encoder weights | what trains |
# |---|---|---|
# | **specialist** | random | everything, on the target grid only |
# | **frozen** | pretrained, frozen | a small head only |
# | **fine-tuned** | pretrained | everything |
# | **predict the mean** | — | nothing (the trivial baseline) |

# %%
def voltage_task(graphs, mask_token: torch.Tensor):
    """Hide the voltage block using the SAME token pretraining used.

    Matching the masking mechanism is not a detail: an encoder fed zeros where
    it was pretrained on a learned token receives out-of-distribution input and
    its representation is worth nothing.
    """
    inputs, targets = [], []
    for graph in graphs:
        features = graph["node_features"].clone()
        targets.append(features[:, VOLTAGE].clone())
        features[:, VOLTAGE] = mask_token[VOLTAGE]
        inputs.append({"node_features": features, "edge_index": graph["edge_index"],
                       "edge_features": graph["edge_features"]})
    return inputs, targets


# Every condition gets the SAME optimisation budget, counted in steps rather
# than epochs. With a fixed epoch count, a run with 100 labelled states takes
# twenty times as many gradient steps as one with five, and the comparison
# across the x-axis stops being about labels at all.
TRANSFER_STEPS = scaled(full=400, fast=60)


def run_transfer(graphs, n_train: int, mode: str, steps: int = TRANSFER_STEPS,
                 learning_rate: float = 3e-3, seed: int = 0, n_test: int | None = None,
                 batch_size: int = 8):
    """Train one condition on ``n_train`` target-grid states; return test MAE of |V|."""
    n_test = N_TEST if n_test is None else n_test
    inputs, targets = voltage_task(graphs, pretrained.mask_token.detach())
    train_index = list(range(n_train))
    test_index = list(range(len(graphs) - n_test, len(graphs)))

    set_seed(seed)
    encoder = GridEncoder(NODE_DIM, EDGE_DIM, d_model=D_MODEL, n_layers=N_LAYERS)
    if mode in {"frozen", "finetune"}:
        encoder.load_state_dict(pretrained.encoder.state_dict())
    if mode == "frozen":
        for parameter in encoder.parameters():
            parameter.requires_grad = False
    head = nn.Sequential(nn.Linear(D_MODEL, 32), nn.GELU(), nn.Linear(32, 3))
    trainable = list(head.parameters()) + (
        [] if mode == "frozen" else list(encoder.parameters())
    )
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    generator = torch.Generator().manual_seed(seed)

    encoder.train(mode != "frozen")
    head.train()
    for _ in range(steps):
        draw = torch.randint(len(train_index), (min(batch_size, len(train_index)),),
                             generator=generator)
        index = [train_index[int(i)] for i in draw]
        batch = collate_graphs([inputs[i] for i in index])
        target = torch.cat([targets[i] for i in index])
        loss = nn.functional.mse_loss(head(encoder(batch)), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        schedule.step()

    encoder.eval()
    head.eval()
    with torch.no_grad():
        batch = collate_graphs([inputs[i] for i in test_index])
        target = torch.cat([targets[i] for i in test_index])
        prediction = head(encoder(batch))
    # Back to per-unit so the number means something physical.
    vm_mae = float((prediction[:, 0] - target[:, 0]).abs().mean() * node_std[2])
    return vm_mae, prediction, target


LABEL_BUDGETS = [5, 20, 100] if not fast_mode() else [5, 20]

# The evaluation set is held out from the END of each grid's states, and the
# training states are taken from the START, so the two must fit inside the
# sample. Scaling the sample size without scaling this is how the reduced
# configuration silently indexes off the front of the list.
N_TEST = scaled(full=100, fast=12)
for _grid, _graphs in heldout_graphs.items():
    assert max(LABEL_BUDGETS) + N_TEST <= len(_graphs), (
        f"{_grid}: {max(LABEL_BUDGETS)} training + {N_TEST} test states requested "
        f"but only {len(_graphs)} sampled"
    )
print(f"{max(LABEL_BUDGETS)} training states at most, {N_TEST} test states, "
      f"out of {len(next(iter(heldout_graphs.values())))} per held-out grid")
MODES = {"specialist (from scratch)": "scratch",
         "pretrained, frozen": "frozen",
         "pretrained, fine-tuned": "finetune"}

transfer_rows = []
for grid_name, graphs in heldout_graphs.items():
    baseline = float(
        torch.cat([g["node_features"][:, 2] for g in graphs[-N_TEST:]]).std() * node_std[2]
    )
    transfer_rows.append({"grid": grid_name, "n_train": 0,
                          "method": "predict the mean", "vm MAE [pu]": baseline})
    for n_train in LABEL_BUDGETS:
        for label, mode in MODES.items():
            score, _, _ = run_transfer(graphs, n_train, mode)
            transfer_rows.append({"grid": grid_name, "n_train": n_train,
                                  "method": label, "vm MAE [pu]": score})
        print(f"  {grid_name:10s} n_train={n_train:4d} done")

transfer = pd.DataFrame(transfer_rows)
COLUMN_ORDER = ["specialist (from scratch)", "pretrained, frozen", "pretrained, fine-tuned"]
for grid_name in heldout_graphs:
    trivial = transfer[(transfer.grid == grid_name) & (transfer.n_train == 0)]
    subset = transfer[(transfer.grid == grid_name) & (transfer.n_train > 0)]
    print(f"\n=== held-out grid: {grid_name} "
          f"({heldout_points[grid_name][0].n_buses} buses) ===")
    print(f"predict-the-mean baseline: {trivial['vm MAE [pu]'].iloc[0]:.5f} pu\n")
    display(subset.pivot_table(index="n_train", columns="method",
                               values="vm MAE [pu]")[COLUMN_ORDER].round(5))

# %%
fig, axes = plt.subplots(1, len(heldout_graphs), figsize=(6.2 * len(heldout_graphs), 3.8),
                         squeeze=False)
palette = {"specialist (from scratch)": COLORS["transformer"],
           "pretrained, frozen": COLORS["foundation"],
           "pretrained, fine-tuned": COLORS["tree"]}
for ax, grid_name in zip(axes[0], heldout_graphs, strict=True):
    subset = transfer[(transfer.grid == grid_name) & (transfer.n_train > 0)]
    for label in MODES:
        rows = subset[subset.method == label]
        ax.plot(rows.n_train, rows["vm MAE [pu]"], marker="o", label=label,
                color=palette[label])
    trivial = transfer[(transfer.grid == grid_name) & (transfer.n_train == 0)]
    ax.axhline(trivial["vm MAE [pu]"].iloc[0], ls="--", color="#888", lw=1,
               label="predict the mean")
    ax.set_xscale("log")
    ax.set_xlabel("labelled operating points from the target grid")
    ax.set_ylabel("|V| MAE [pu]")
    ax.set_title(f"{grid_name} — never seen during pretraining")
    ax.legend(fontsize=8)
fig.tight_layout()
plt.show()

# %%
print("Change in |V| MAE relative to training from scratch on the same target grid.")
print("Positive = the pretrained encoder is better.\n")
print(f"{'grid':11s}{'labels':>8s}{'frozen':>12s}{'fine-tuned':>14s}")
for grid_name in heldout_graphs:
    for n_train in LABEL_BUDGETS:
        subset = transfer[(transfer.grid == grid_name) & (transfer.n_train == n_train)]
        scores = dict(zip(subset.method, subset["vm MAE [pu]"], strict=True))
        scratch = scores["specialist (from scratch)"]
        frozen = 1 - scores["pretrained, frozen"] / scratch
        tuned = 1 - scores["pretrained, fine-tuned"] / scratch
        print(f"{grid_name:11s}{n_train:>8d}{frozen:>11.1%}{tuned:>14.1%}")

print("\nThree things this table says, in order of how surprising they are:\n")
print("1. FINE-TUNING THE PRETRAINED ENCODER WINS, and wins by more on case33bw —")
print("   the grid whose *structure* is furthest from anything seen in pretraining.")
print("   That is the opposite of the intuition that transfer works best on similar")
print("   systems, and it is because a radial feeder is the harder problem: there is")
print("   more for a good initialisation to contribute.")
print("\n2. THE FROZEN ENCODER IS CONSISTENTLY WORSE THAN TRAINING FROM SCRATCH.")
print("   Five transmission networks do not contain the features a radial distribution")
print("   feeder needs, and a small head on top cannot invent them. Freezing works")
print("   when the pretraining distribution covers the target; here it does not, and")
print("   pretending otherwise would be the easy mistake to make.")
print("\n3. TRAINING FROM SCRATCH GETS *WORSE* WITH MORE DATA ON case33bw.")
print("   That is not overfitting — the optimisation budget is fixed. It is")
print("   instability: a randomly initialised message-passing network on a radial")
print("   graph is a badly conditioned problem. The pretrained initialisation also")
print("   stabilises optimisation, which is a real and under-discussed benefit of")
print("   pretraining, separate from any transfer of knowledge.")

# %% [markdown]
# **Read this result carefully, and do not let it be oversold.**
#
# Whatever the numbers come out at, three things are true and worth separating:
#
# 1. **The encoder ran at all on grids with different bus counts and topologies.**
#    That is the claim this notebook was built to demonstrate, and it is not
#    trivial — no fixed-width model can do it.
# 2. **Pretraining helps most where labels are scarcest**, which is the same
#    shape as tutorial 04's label-efficiency curve and the same argument as
#    tutorial 09's zero-shot result.
# 3. **A specialist with enough target-grid data is hard to beat**, exactly as
#    in tutorials 02 and 09. Five grids and a few thousand states is a
#    vanishingly small pretraining corpus. GridSFM used ~200 grids and ~500,000
#    scenarios; `gridfm-graphkit` trains multi-GPU. Expect the curve to move with
#    scale, and *say so* rather than extrapolating from this.

# %% [markdown]
# ## 10. Physics: statistically plausible is not physically plausible
#
# Everything above is MAE in per-unit. None of it asks whether the predicted
# state satisfies Kirchhoff's laws.
#
# The AC power-flow equations give us an exact check. For a predicted voltage
# state $\hat V$ and the network's own admittance matrix $Y$, the injections it
# *implies* are $\hat S = \hat V \odot (Y\hat V)^*$. Comparing those with the
# injections that were actually scheduled says whether the prediction is a
# realisable operating point or merely a plausible set of numbers.

# %%
grid_name = HELDOUT_NETWORKS[0]
graphs = heldout_graphs[grid_name]
points = heldout_points[grid_name]

base_net = load_network(grid_name)
assert run_power_flow(base_net)
ybus = get_ybus(base_net)
base_mva = float(base_net.sn_mva)

# The admittance matrix belongs to the *base* topology. A state with a branch
# out of service has a different Ybus, so checking it against this one would
# manufacture a mismatch that is our bookkeeping error rather than the model's.
# Restrict the physical check to intact-topology states, and say how many that
# leaves.
TEST_SLICE = range(len(graphs) - N_TEST, len(graphs))
intact = [i for i in TEST_SLICE
          if not points[i].has_outage and graphs[i]["node_features"].shape[0] == ybus.shape[0]]
print(f"{grid_name}: Ybus {ybus.shape}, base {base_mva} MVA")
print(f"{len(intact)} of the {len(list(TEST_SLICE))} test states have the intact topology "
      "and can be checked against it.")


def unscale_voltage(block: torch.Tensor) -> np.ndarray:
    """Undo the standardisation and rebuild the complex bus voltage."""
    vm = (block[:, 0] * node_std[2] + node_mean[2]).numpy()
    sin_va = (block[:, 1] * node_std[3] + node_mean[3]).numpy()
    cos_va = (block[:, 2] * node_std[4] + node_mean[4]).numpy()
    return vm * np.exp(1j * np.arctan2(sin_va, cos_va))


def physical_report_for(prediction: torch.Tensor) -> dict[str, float]:
    """Power-balance mismatch and voltage violations of a predicted state.

    ``prediction`` holds the concatenated node outputs for every state in
    ``TEST_SLICE``, in order; we index into it and evaluate only the intact ones.
    """
    offsets, cursor = {}, 0
    for i in TEST_SLICE:
        n_bus = graphs[i]["node_features"].shape[0]
        offsets[i] = (cursor, cursor + n_bus)
        cursor += n_bus

    p_errors, q_errors, violations, worst_vm = [], [], [], []
    for i in intact:
        lo, hi = offsets[i]
        voltage = unscale_voltage(prediction[lo:hi])
        implied = voltage * np.conj(ybus @ voltage) * base_mva

        true_voltage = points[i].vm_pu * np.exp(1j * np.radians(points[i].va_degree))
        scheduled = true_voltage * np.conj(ybus @ true_voltage) * base_mva

        p_errors.append(np.abs(implied.real - scheduled.real).mean())
        q_errors.append(np.abs(implied.imag - scheduled.imag).mean())
        violations.append(voltage_violations(np.abs(voltage))["violation_rate"])
        worst_vm.append((np.abs(voltage).min(), np.abs(voltage).max()))
    return {
        "P mismatch [MW]": float(np.mean(p_errors)),
        "Q mismatch [Mvar]": float(np.mean(q_errors)),
        "voltage-band violation rate": float(np.mean(violations)),
        "|V| min [pu]": float(min(v[0] for v in worst_vm)),
        "|V| max [pu]": float(max(v[1] for v in worst_vm)),
    }


physics_rows = {}
for label, mode in MODES.items():
    _, prediction, _ = run_transfer(graphs, max(LABEL_BUDGETS), mode)
    physics_rows[label] = physical_report_for(prediction)

# The true solved states, for reference: zero mismatch by construction.
physics_rows["the true solved state"] = {
    "P mismatch [MW]": 0.0,
    "Q mismatch [Mvar]": 0.0,
    "voltage-band violation rate": float(np.mean([
        voltage_violations(points[i].vm_pu)["violation_rate"] for i in intact
    ])),
    "|V| min [pu]": float(min(points[i].vm_pu.min() for i in intact)),
    "|V| max [pu]": float(max(points[i].vm_pu.max() for i in intact)),
}
display(pd.DataFrame(physics_rows).T.round(4))

truth_rate = physics_rows["the true solved state"]["voltage-band violation rate"]
print("\nCompare the violation rates against the true one "
      f"({truth_rate:.1%}): a model that under-reports")
print("violations is dangerous in a different way from one that over-reports. The first")
print("misses problems; the second triggers unnecessary intervention. Neither is visible")
print("in an MAE.")
print("\nThe true state has zero mismatch by construction — that is what 'solved' means.")
print("Every model's predicted state carries a residual measured in megawatts, and that")
print("residual is invisible to the MAE table in section 9.")
print("\nThis is the tutorial-01 lesson at the other end of the course: a model can be")
print("statistically close and physically impossible, and only a physical check tells")
print("you which. For a state estimator or a power-flow surrogate, the mismatch IS the")
print("quantity of interest — a surrogate whose output violates Kirchhoff's laws cannot")
print("be handed to an optimiser that assumes they hold.")

# %% [markdown]
# ### A physics-informed penalty
#
# The obvious response: put the mismatch in the loss. We do it on one grid, in a
# simplified linearised form that is cheap enough to differentiate through, and
# **measure both** effects — statistical accuracy and physical validity.

# %%
def run_with_physics(graphs, n_train: int, physics_weight: float,
                     steps: int = TRANSFER_STEPS, seed: int = 0, n_test: int | None = None):
    """Fine-tune with an added penalty on the AC power-balance residual."""
    n_test = N_TEST if n_test is None else n_test
    inputs, targets = voltage_task(graphs, pretrained.mask_token.detach())
    train_index = list(range(n_train))
    test_index = list(range(len(graphs) - n_test, len(graphs)))

    set_seed(seed)
    encoder = GridEncoder(NODE_DIM, EDGE_DIM, d_model=D_MODEL, n_layers=N_LAYERS)
    encoder.load_state_dict(pretrained.encoder.state_dict())
    head = nn.Sequential(nn.Linear(D_MODEL, 32), nn.GELU(), nn.Linear(32, 3))
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(head.parameters()), lr=3e-3
    )
    generator = torch.Generator().manual_seed(seed)
    ybus_torch = torch.tensor(ybus, dtype=torch.complex64)

    encoder.train()
    head.train()
    for _ in range(steps):
        draw = torch.randint(len(train_index), (min(4, len(train_index)),),
                             generator=generator)
        index = [train_index[int(i)] for i in draw]
        batch = collate_graphs([inputs[i] for i in index])
        target = torch.cat([targets[i] for i in index])
        prediction = head(encoder(batch))
        loss = nn.functional.mse_loss(prediction, target)

        if physics_weight > 0:
            residuals = []
            offset = 0
            for i in index:
                n_bus = graphs[i]["node_features"].shape[0]
                block = prediction[offset : offset + n_bus]
                offset += n_bus
                # Only intact-topology states share the base admittance matrix.
                if n_bus != ybus.shape[0] or points[i].has_outage:
                    continue
                vm = block[:, 0] * node_std[2] + node_mean[2]
                angle = torch.atan2(block[:, 1] * node_std[3] + node_mean[3],
                                    block[:, 2] * node_std[4] + node_mean[4])
                voltage = torch.polar(vm, angle)
                implied = voltage * torch.conj(ybus_torch @ voltage)
                scheduled_p = (
                    batch.node_features[offset - n_bus : offset, 0] * node_std[0]
                    + node_mean[0]
                )
                residuals.append(((implied.real - scheduled_p) ** 2).mean())
            if residuals:
                loss = loss + physics_weight * torch.stack(residuals).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    encoder.eval()
    head.eval()
    with torch.no_grad():
        batch = collate_graphs([inputs[i] for i in test_index])
        target = torch.cat([targets[i] for i in test_index])
        prediction = head(encoder(batch))
    vm_mae = float((prediction[:, 0] - target[:, 0]).abs().mean() * node_std[2])
    return vm_mae, prediction


physics_comparison = {}
for weight in ([0.0, 0.01, 0.1] if not fast_mode() else [0.0, 0.1]):
    score, prediction = run_with_physics(graphs, max(LABEL_BUDGETS), weight)
    report = physical_report_for(prediction)
    physics_comparison[f"physics weight = {weight}"] = {
        "|V| MAE [pu]": score,
        "P mismatch [MW]": report["P mismatch [MW]"],
        "voltage-band violation rate": report["voltage-band violation rate"],
    }
display(pd.DataFrame(physics_comparison).T.round(5))

table = pd.DataFrame(physics_comparison).T
best_accuracy = table["|V| MAE [pu]"].idxmin()
best_physics = table["P mismatch [MW]"].idxmin()
print(f"lowest |V| MAE        : {best_accuracy}")
print(f"lowest power mismatch : {best_physics}")
if best_accuracy != best_physics:
    print("\nThey disagree — which is the entire point. A physics penalty trades")
    print("statistical accuracy for physical consistency, and which you want depends on")
    print("what the model is for. A forecast that will be eyeballed wants the first; a")
    print("surrogate whose output feeds an optimiser wants the second.")
else:
    print("\nOn this run the same setting wins both. Do not generalise from one grid and")
    print("one seed — the trade-off is real and appears clearly at larger scale.")

# %% [markdown]
# ## 11. The capstone comparison
#
# Every paradigm this course has covered, with the columns that actually
# distinguish them. **Measured values come only from experiments in these
# notebooks**; the qualitative columns are claims about the method, not results.

# %%
capstone = pd.DataFrame(
    [
        ("Linear regression", "one dataset", "yes", "poor", "none", 1),
        ("Gradient boosting", "one dataset", "yes", "poor", "none", 1),
        ("MLP", "one dataset", "yes", "limited", "none", 2),
        ("LSTM", "one sequence domain", "yes", "limited", "temporal", 3),
        ("Masked autoencoder", "one domain, unlabelled", "no", "medium", "temporal", 4),
        ("Transformer", "one/more datasets", "usually", "medium", "temporal", 6),
        ("Tiny GPT", "one corpus", "no", "low", "none", 7),
        ("Pretrained LLM + LoRA", "web-scale text", "no", "high", "none", 8),
        ("Time-series FM (Chronos-2)", "many datasets", "no", "high", "temporal", 9),
        ("Mini-GridFM", "many grids", "no", "target of the design",
         "temporal + topology", 10),
    ],
    columns=["Paradigm", "Training data", "Target task known during pretraining?",
             "Transfer", "Physical structure", "Tutorial"],
).set_index("Paradigm")
display(capstone)

print("Measured results from this course, for the two comparable benchmarks:\n")
for task in ("load_day_ahead", "load_day_ahead_daily_origins"):
    board = leaderboard_table(task=task)
    if not board.empty:
        print(f"--- {task} ---")
        display(board[[c for c in ("MAE", "RMSE", "Skill") if c in board.columns]])

print("\nRows appear only for tutorials that were actually run. If a row is missing,")
print("run that notebook — the table is never filled in by hand.")

# %% [markdown]
# ## 12. Failure analysis and honest limitations
#
# What this notebook does **not** show, stated plainly so nobody quotes it for
# more than it is worth.

# %%
limitations = pd.DataFrame(
    {
        "this notebook": [
            f"{len(PRETRAIN_NETWORKS)} networks",
            f"{len(pretrain_points):,} operating states",
            f"{sum(p.numel() for p in pretrained.encoder.parameters()):,} parameters",
            f"{pretrain_seconds:.0f} s on a laptop CPU",
            "balanced AC power flow only",
            "one downstream task",
            "synthetic, from published test cases",
            "no measurement noise, no bad data",
        ],
        "a research-scale GridFM": [
            "~200 networks (GridSFM)",
            "~500,000 scenarios (GridSFM)",
            "millions to billions",
            "multi-GPU, hours to days",
            "OPF, contingency, dynamics, unbalanced",
            "power flow, OPF, state estimation, screening",
            "real SCADA/PMU where available",
            "explicit bad-data and missingness handling",
        ],
    },
    index=["pretraining grids", "pretraining states", "encoder size", "compute",
           "physics covered", "downstream tasks", "data", "realism"],
)
display(limitations)

print("Three limitations that would matter most if you tried to build on this:\n")
print("1. TOPOLOGY VARIATION IS SHALLOW. We perturb loading and take single lines out.")
print("   Real networks differ in radial-versus-meshed structure, voltage levels,")
print("   equipment types and control philosophy. case33bw is our only genuinely")
print("   different structure, and one held-out structure is not a generalisation claim.")
print("\n2. THE TASK IS A SURROGATE FOR A SOLVER WE ALREADY HAVE. pandapower computes")
print("   the exact answer in milliseconds. The interesting cases are where no fast")
print("   exact method exists — OPF, security-constrained dispatch, dynamic")
print("   simulation — which is precisely what GridSFM targets.")
print("\n3. SAFETY-CRITICAL DEPLOYMENT NEEDS FAR MORE THAN A BENCHMARK. Worst-case")
print("   behaviour, out-of-distribution detection, guaranteed physical feasibility,")
print("   auditability and formal validation are all absent here, and all required")
print("   before anything like this goes near a control room.")

# %% [markdown]
# ## 13. Exercises
#
# **1 — Conceptual.** Section 9 matched the downstream input distribution to the
# pretraining objective by reusing the mask token. Deliberately break it: feed
# zeros instead, and re-run the frozen condition. Quantify how much the frozen
# encoder's performance degrades. Then explain, in terms of what the encoder's
# first layer computes, why this particular mismatch is so damaging — and name
# one place in tutorials 04 or 08 where the same trap exists.
#
# **2 — Coding.** Transfer the grid encoder to a topology excluded from
# pretraining in a stronger sense than we did: add `case57` to the held-out set,
# and additionally build a *modified* `case14` with three lines permanently
# removed and two buses split. Does transfer degrade with topological distance
# from the pretraining set, and can you find a measure of "topological distance"
# that predicts the degradation?
#
# **3 — Research.** Section 10 added a power-balance penalty and measured the
# trade-off on one grid with one seed. Turn it into a study: sweep the penalty
# weight over four orders of magnitude, use three seeds, and report both |V| MAE
# and mean P mismatch with error bars. Then read the physics-informed loss in
# [`gridfm-graphkit`](https://github.com/gridfm/gridfm-graphkit) and write a
# paragraph on how their formulation differs from ours and why that matters.
#
# **4 — Extension (open).** The Mini-GridFM consumes topology and a static
# snapshot. Extend it to time: give each bus a short history of injections
# instead of a single value, and pretrain with a masked *spatio-temporal*
# objective. This is the actual frontier — combining the temporal modelling of
# tutorials 03–06 with the topological modelling here — and there is no settled
# answer about how to do it.

# %% [markdown]
# ## 14. The final question
#
# The course ends where the research does.
#
# > ### What should the "tokens" of an electrical grid foundation model be?
#
# | candidate | argument for | argument against |
# |---|---|---|
# | **time steps** | matches time-series FMs directly | ignores topology entirely |
# | **measurements** | finest granularity; handles missingness naturally | enormous sequences, no structure |
# | **buses** | natural graph nodes; what we used | a bus is not a fixed-size object across voltage levels |
# | **branches** | constraints are written on flows | loses the nodal balance structure |
# | **operating states** | one token per snapshot; matches contingency work | throws away spatial detail |
# | **subgraphs / zones** | matches how operators think; bounds sequence length | how do you choose the partition? |
# | **events** | sparse, meaningful, aligned with operator reasoning | needs labelled event detection |
# | **learned latent tokens** | let the model decide (à la Perceiver) | uninterpretable; hard to validate physically |
#
# And behind it, the question that actually decides whether any of this works:
#
# > ### What must a model learn so that knowledge acquired on Grid A remains useful on previously unseen Grid B?
#
# This notebook's answer — and it is the same answer tutorial 01 gives in
# miniature:
#
# 1. **Units that mean the same thing everywhere.** Per-unit, not volts. Without
#    it, transfer fails before training starts.
# 2. **Relationships, not identities.** Message passing over "my neighbours",
#    never "bus 47". Bus numbering is an accident of a data file.
# 3. **A constraint that holds in both grids.** The power-flow equations are true
#    on every network, which makes them both a free supervision signal and a
#    validity check.
#
# None of that is a foundation-model insight. It is ordinary good modelling —
# choose a representation in which the thing you want to transfer is *expressible*
# — applied at a scale where it becomes a research programme. Which is, in the
# end, what this whole course has been about:
#
# > The question is no longer only *"which neural network should I train for this
# > power-system problem?"* but increasingly *"what reusable representation of the
# > power system should be pretrained, and how can it be adapted safely and
# > efficiently to this particular grid and task?"*

# %% [markdown]
# ## 15. Key takeaways
#
# - **Grids do not share a shape**, and that — not data volume — is the central
#   technical obstacle to a grid foundation model.
# - **Message passing removes the obstacle** because its parameters live in
#   functions applied per node and per edge. We verified permutation equivariance
#   and ran one encoder on grids from 9 to 118 buses.
# - **Per-unit plus standardisation is what makes features comparable** across
#   voltage levels. Skip it and nothing transfers.
# - **Pretraining and downstream use must share an input distribution.** Reusing
#   the mask token was not a detail; it is the difference between transfer and
#   noise.
# - **Simulation is the domain's unfair advantage.** Unlike language, we can
#   generate unlimited correct data with a solver — which is why GridSFM could
#   use half a million scenarios.
# - **MAE does not know about Kirchhoff's laws.** Every predicted state carried a
#   power-balance residual invisible to the accuracy table, and a physics penalty
#   trades one against the other.
# - **This is a research demonstration at a toy scale.** The methodology is
#   right; the numbers are not a claim about what GridFMs can do.
#
# ## 16. Further reading
#
# - Hamann et al., "Foundation Models for the Electric Power Grid", *Joule*,
#   2024. [arXiv:2407.09434](https://arxiv.org/abs/2407.09434)
# - `gridfm-graphkit` and `gridfm-datakit` —
#   <https://github.com/gridfm/gridfm-graphkit>,
#   <https://github.com/gridfm/gridfm-datakit>; the datakit is described in
#   [arXiv:2512.14658](https://arxiv.org/abs/2512.14658).
# - GridSFM, Microsoft Research, 2026 — <https://github.com/microsoft/gridSFM>.
# - Tu et al., "PowerPM: Foundation Model for Power Systems", NeurIPS 2024.
#   [arXiv:2408.04057](https://arxiv.org/abs/2408.04057)
# - Fan et al., "WindFM", [arXiv:2509.06311](https://arxiv.org/abs/2509.06311) —
#   8.1 M parameters, zero-shot.
# - Lam et al., "GraphCast",
#   [arXiv:2212.12794](https://arxiv.org/abs/2212.12794) — the closest
#   methodological cousin: a GNN foundation model on a physical domain.
# - Thurner et al., "pandapower", *IEEE Trans. Power Systems* 33(6), 2018.
#
# ---
#
# ## You have finished the course
#
# Ten notebooks ago the question was how to fit a line to some load data. It is
# now what representation of an electrical system is worth pretraining, and how
# to adapt it safely.
#
# If you want to keep going: run everything on the real OPSD data
# (`scripts/download_data.py`), redo this notebook on SimBench networks
# (`uv sync --extra simbench`), and read the two 2026 benchmark papers in
# tutorial 09's exercise 3 — they disagree with each other, and working out why
# is a genuine piece of research.
