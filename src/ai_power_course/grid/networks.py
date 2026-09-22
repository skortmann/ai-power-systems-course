"""The catalogue of grids the Mini-GridFM is pretrained on and transferred to.

The split is the whole experiment. Pretraining sees five transmission networks
of different sizes; evaluation happens on grids the encoder has never touched,
including one (``case33bw``) that is radial distribution rather than meshed
transmission, so "unseen grid" means genuinely unseen *structure* and not just
a different load level.

All networks ship with pandapower, so nothing is downloaded and the tutorial
runs offline.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import pandapower as pp
import pandapower.networks as pn
import pandas as pd

__all__ = [
    "NETWORK_CATALOGUE",
    "PRETRAIN_NETWORKS",
    "HELDOUT_NETWORKS",
    "load_network",
    "run_power_flow",
    "network_summary",
]

NETWORK_CATALOGUE: dict[str, Callable[[], pp.pandapowerNet]] = {
    "case9": pn.case9,
    "case14": pn.case14,
    "case24_ieee_rts": pn.case24_ieee_rts,
    "case30": pn.case30,
    "case39": pn.case39,
    "case57": pn.case57,
    "case118": pn.case118,
    "case33bw": pn.case33bw,
}

#: Grids the encoder is pretrained on.
PRETRAIN_NETWORKS: tuple[str, ...] = ("case9", "case14", "case24_ieee_rts", "case30", "case39")

#: Grids held out entirely from pretraining.
#:
#: ``case118`` tests scale (4x more buses than the largest pretraining grid);
#: ``case33bw`` tests structure (radial medium-voltage distribution, no
#: generators, no transformers) and is the harder of the two by a wide margin.
HELDOUT_NETWORKS: tuple[str, ...] = ("case118", "case33bw")


def load_network(name: str) -> pp.pandapowerNet:
    """Instantiate a catalogued network by name. Always returns a fresh copy."""
    if name not in NETWORK_CATALOGUE:
        raise KeyError(
            f"unknown network {name!r}; available: {', '.join(sorted(NETWORK_CATALOGUE))}"
        )
    return NETWORK_CATALOGUE[name]()


def run_power_flow(net: pp.pandapowerNet, **kwargs) -> bool:
    """Run a balanced AC power flow, quietly, returning whether it converged.

    ``numba=False`` is passed explicitly: numba is not a course dependency and
    pandapower is otherwise noisy about it on every single call. Non-convergence
    is *data*, not an error — a sampled operating point that the Newton-Raphson
    solver cannot solve is usually one the real grid could not run either, and
    tutorial 10 counts how many there are.
    """
    # The networks shipped with pandapower already carry ``converged = True``
    # from when they were built. If ``runpp`` raises before it updates the flag,
    # trusting it would silently accept an unsolved network — and the resulting
    # half-built internal structures then fail much later, somewhere confusing.
    net["converged"] = False
    options = {"numba": False, "init": "auto", **kwargs}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            pp.runpp(net, **options)
        except Exception:
            return False
    return bool(net.get("converged", False))


def network_summary(names: tuple[str, ...] | None = None) -> pd.DataFrame:
    """A table of size and composition for the catalogued grids.

    Printed at the start of tutorial 10 to make the transfer problem concrete:
    the grids do not share a bus count, a topology, or even a voltage level.
    """
    rows = []
    for name in names or tuple(NETWORK_CATALOGUE):
        net = load_network(name)
        converged = run_power_flow(net)
        rows.append(
            {
                "network": name,
                "buses": len(net.bus),
                "lines": len(net.line),
                "trafos": len(net.trafo),
                "gens": len(net.gen),
                "sgens": len(net.sgen),
                "loads": len(net.load),
                "kV_levels": net.bus.vn_kv.nunique(),
                "base_MVA": net.sn_mva,
                "converged": converged,
                "vm_min_pu": round(net.res_bus.vm_pu.min(), 4) if converged else float("nan"),
                "vm_max_pu": round(net.res_bus.vm_pu.max(), 4) if converged else float("nan"),
            }
        )
    return pd.DataFrame(rows).set_index("network")
