"""Turning a solved pandapower network into tensors a GNN can eat.

The representation is the interesting design decision of tutorial 10, and it is
where the "what are the tokens of a grid?" question first becomes concrete.
The choices made here:

* **One node per bus, one edge per branch.** Lines and transformers become the
  same kind of object, distinguished by a feature rather than by a type.
* **Per-unit everything.** A 380 kV transmission line and a 20 kV feeder have
  impedances three orders of magnitude apart in ohms and comparable values in
  per-unit. Without this normalisation, transfer between grids is hopeless
  before the model even starts.
* **Features split into state and static.** Only the *state* (P, Q, |V|, angle)
  is masked during pretraining; the static descriptors (bus type, voltage
  level) stay visible, exactly as a real operator always knows the equipment
  even when a measurement is missing.
* **Angles as (sin, cos).** The angle is circular; feeding degrees directly
  would tell the network that -179 deg and +179 deg are far apart.

Everything is read from pandapower's *internal* ``_ppc["internal"]`` structure
(PYPOWER's ``ppci``). That view already carries the per-unit conversion, treats
transformers as ordinary branches, contains only the energised part of the
network, and is indexed identically to the ``Ybus`` used by
:mod:`ai_power_course.grid.physics` — so a physics check and a model prediction
refer to the same buses without a lookup table.

The energised-subnetwork detail matters once branch outages are sampled: taking
a line out of a radial feeder islands everything behind it, and the graph then
legitimately has fewer nodes. A message-passing model copes with that; a
fixed-width model could not even be evaluated.
"""

from __future__ import annotations

import numpy as np
import pandapower as pp
import torch
from pandapower.pypower.idx_brch import BR_B, BR_R, BR_STATUS, BR_X, F_BUS, T_BUS, TAP
from pandapower.pypower.idx_bus import BASE_KV, BUS_TYPE, VA, VM

__all__ = [
    "NODE_FEATURE_NAMES",
    "STATE_FEATURE_NAMES",
    "EDGE_FEATURE_NAMES",
    "N_STATE_FEATURES",
    "net_to_graph",
    "graph_feature_frame",
]

#: Maskable state: what a state estimator would be trying to recover.
STATE_FEATURE_NAMES: tuple[str, ...] = ("p_pu", "q_pu", "vm_pu", "va_sin", "va_cos")

#: Full node feature vector. The state features come first so that masking is a
#: simple slice and the reconstruction head has a fixed output width.
NODE_FEATURE_NAMES: tuple[str, ...] = (
    *STATE_FEATURE_NAMES,
    "is_slack",
    "is_pv",
    "is_pq",
    "log_base_kv",
)

EDGE_FEATURE_NAMES: tuple[str, ...] = ("r_pu", "x_pu", "b_pu", "is_transformer", "tap_ratio")

N_STATE_FEATURES = len(STATE_FEATURE_NAMES)


def net_to_graph(net: pp.pandapowerNet, add_reverse_edges: bool = True) -> dict[str, torch.Tensor]:
    """Convert a **solved** pandapower network into graph tensors.

    Returns a dict with

    ``node_features`` ``(n_bus, 9)``, ``edge_index`` ``(2, n_edges)``,
    ``edge_features`` ``(n_edges, 5)``, and the solved state ``vm_pu`` /
    ``va_degree`` ``(n_bus,)`` aligned with the nodes.

    ``n_bus`` is the number of *energised* buses, which after a branch outage
    can be smaller than ``len(net.bus)``.

    Reverse edges are added by default because power flows both ways along a
    branch and a message-passing layer is directional: without them, a bus
    would never hear from its downstream neighbours.
    """
    ppc = net.get("_ppc")
    if ppc is None or "internal" not in ppc:
        raise RuntimeError("run a power flow first: the ppc structures are built by runpp")
    internal = ppc["internal"]
    missing = {"bus", "branch", "Ybus"} - set(internal)
    if missing:
        # pandapower skips building the full internal model when the energised
        # remainder is degenerate (e.g. a branch outage islanded everything but
        # the slack bus). That is not a usable operating point.
        raise ValueError(
            f"internal ppc is incomplete (missing {sorted(missing)}); the solved network is "
            "degenerate, most likely because an outage islanded it"
        )

    bus = np.real(internal["bus"])
    branch = np.real(internal["branch"])
    n_bus = bus.shape[0]

    # --- state ---------------------------------------------------------------
    ybus = np.asarray(internal["Ybus"].todense())
    vm = bus[:, VM]
    va_deg = bus[:, VA]
    va_rad = np.radians(va_deg)
    voltage = vm * np.exp(1j * va_rad)
    injection = voltage * np.conj(ybus @ voltage)  # per-unit on the system base

    # --- static --------------------------------------------------------------
    bus_type = bus[:, BUS_TYPE]
    base_kv = np.clip(bus[:, BASE_KV], 1e-3, None)

    node_features = np.stack(
        [
            injection.real,
            injection.imag,
            vm,
            np.sin(va_rad),
            np.cos(va_rad),
            (bus_type == 3).astype(float),  # REF / slack
            (bus_type == 2).astype(float),  # PV / voltage-controlled
            (bus_type == 1).astype(float),  # PQ
            np.log10(base_kv) / 3.0,  # ~0.3 at 20 kV, ~0.85 at 380 kV
        ],
        axis=1,
    ).astype(np.float32)

    # --- edges ---------------------------------------------------------------
    in_service = branch[:, BR_STATUS] > 0
    active = branch[in_service]
    from_bus = active[:, F_BUS].astype(int)
    to_bus = active[:, T_BUS].astype(int)
    tap = np.where(active[:, TAP] == 0.0, 1.0, active[:, TAP])
    edge_features = np.stack(
        [
            active[:, BR_R],
            active[:, BR_X],
            active[:, BR_B],
            (np.abs(tap - 1.0) > 1e-9).astype(float),  # transformer vs. line
            tap,
        ],
        axis=1,
    ).astype(np.float32)

    edge_index = np.stack([from_bus, to_bus], axis=0)
    if add_reverse_edges:
        edge_index = np.concatenate([edge_index, edge_index[::-1]], axis=1)
        edge_features = np.concatenate([edge_features, edge_features], axis=0)

    if edge_index.size and edge_index.max() >= n_bus:
        raise ValueError("branch references a bus outside the internal ppc bus table")
    if not np.isfinite(node_features).all():
        raise ValueError("non-finite node features; the power flow did not solve cleanly")

    return {
        "node_features": torch.from_numpy(node_features),
        "edge_index": torch.from_numpy(edge_index.astype(np.int64)),
        "edge_features": torch.from_numpy(edge_features),
        "vm_pu": torch.from_numpy(vm.astype(np.float32)),
        "va_degree": torch.from_numpy(va_deg.astype(np.float32)),
    }


def graph_feature_frame(graph: dict[str, torch.Tensor]):
    """Node features as a labelled DataFrame — for looking at, not for training."""
    import pandas as pd

    return pd.DataFrame(
        graph["node_features"].numpy(), columns=list(NODE_FEATURE_NAMES)
    ).rename_axis("node")
