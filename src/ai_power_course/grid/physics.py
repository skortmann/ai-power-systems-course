"""Physical validation of power-system predictions.

Tutorial 10's central claim is that a model can score well on MAE and still be
useless, because MAE does not know about Kirchhoff's laws. These functions
provide the second opinion:

* :func:`power_balance_residual` — does the predicted state satisfy the AC
  power-flow equations on the given network?
* :func:`voltage_violations`, :func:`line_loading_violations` — does it respect
  the operational envelope?

The residual is computed from the *network's own* admittance matrix, so it is a
genuine physical check and not another statistic.
"""

from __future__ import annotations

import numpy as np
import pandapower as pp

__all__ = [
    "get_ybus",
    "internal_state",
    "complex_voltage",
    "power_balance_residual",
    "voltage_violations",
    "line_loading_violations",
    "physical_report",
]


def get_ybus(net: pp.pandapowerNet) -> np.ndarray:
    """Dense bus admittance matrix in per-unit, in *internal* ppc bus order.

    Requires a completed power flow: pandapower builds the internal ``_ppc``
    structures during ``runpp``. The internal ordering covers only the
    energised subnetwork and matches the nodes produced by
    :func:`ai_power_course.grid.graphs.net_to_graph`, so predictions and
    physics checks index the same buses.

    Dense is fine here — the course's grids have at most 118 buses. A real
    GridFM would keep it sparse.
    """
    ppc = net.get("_ppc")
    if ppc is None or "internal" not in ppc or "Ybus" not in ppc["internal"]:
        raise RuntimeError("run a power flow before requesting Ybus")
    return np.asarray(ppc["internal"]["Ybus"].todense())


def internal_state(net: pp.pandapowerNet) -> tuple[np.ndarray, np.ndarray]:
    """Solved ``(vm_pu, va_degree)`` in the same order as :func:`get_ybus`."""
    from pandapower.pypower.idx_bus import VA, VM

    ppc = net.get("_ppc")
    if ppc is None or "internal" not in ppc:
        raise RuntimeError("run a power flow before requesting the internal state")
    bus = np.real(ppc["internal"]["bus"])
    return bus[:, VM].astype(float), bus[:, VA].astype(float)


def complex_voltage(vm_pu: np.ndarray, va_degree: np.ndarray) -> np.ndarray:
    """Build the complex bus voltage phasor from magnitude and angle."""
    vm = np.asarray(vm_pu, dtype=float).reshape(-1)
    va = np.radians(np.asarray(va_degree, dtype=float).reshape(-1))
    return vm * np.exp(1j * va)


def power_balance_residual(
    net: pp.pandapowerNet,
    vm_pu: np.ndarray,
    va_degree: np.ndarray,
    reference: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, float]:
    r"""How badly does a predicted voltage state violate :math:`S = V \odot (YV)^*`?

    Given a *predicted* voltage profile, the nodal injections it implies are
    recovered exactly from the network model. Comparing them with the injections
    that were actually scheduled tells you whether the prediction is a physically
    realisable operating point or merely a plausible-looking set of numbers.

    Parameters
    ----------
    reference:
        ``(p_mw, q_mvar)`` injections to compare against. Defaults to the
        injections implied by the network's own converged solution.

    Returns mean and maximum absolute mismatch in MW / Mvar.
    """
    ybus = get_ybus(net)
    base_mva = float(net.sn_mva)
    voltage = complex_voltage(vm_pu, va_degree)
    if voltage.shape[0] != ybus.shape[0]:
        raise ValueError(
            f"voltage vector has {voltage.shape[0]} entries but Ybus has {ybus.shape[0]} buses"
        )
    implied = voltage * np.conj(ybus @ voltage) * base_mva

    if reference is None:
        solved = complex_voltage(*internal_state(net))
        target = solved * np.conj(ybus @ solved) * base_mva
        p_ref, q_ref = target.real, target.imag
    else:
        p_ref, q_ref = (np.asarray(a, dtype=float).reshape(-1) for a in reference)

    p_mismatch = np.abs(implied.real - p_ref)
    q_mismatch = np.abs(implied.imag - q_ref)
    return {
        "p_mismatch_mean_mw": float(p_mismatch.mean()),
        "p_mismatch_max_mw": float(p_mismatch.max()),
        "q_mismatch_mean_mvar": float(q_mismatch.mean()),
        "q_mismatch_max_mvar": float(q_mismatch.max()),
    }


def voltage_violations(
    vm_pu: np.ndarray, v_min: float = 0.95, v_max: float = 1.05
) -> dict[str, float]:
    """Share of buses outside the admissible band, and the worst excursion.

    The default 0.95-1.05 pu band is the usual transmission planning envelope
    (EN 50160 allows +/-10% at the point of common coupling in LV networks, so
    tighten or relax this consciously rather than treating it as universal).
    """
    values = np.asarray(vm_pu, dtype=float).reshape(-1)
    below, above = values < v_min, values > v_max
    return {
        "undervoltage_rate": float(below.mean()),
        "overvoltage_rate": float(above.mean()),
        "violation_rate": float((below | above).mean()),
        "vm_min_pu": float(values.min()),
        "vm_max_pu": float(values.max()),
    }


def line_loading_violations(
    net: pp.pandapowerNet, limit_percent: float = 100.0
) -> dict[str, float]:
    """Share of branches loaded beyond ``limit_percent`` in the solved network."""
    loadings = [net.res_line.loading_percent.to_numpy()] if len(net.res_line) else []
    if len(net.get("res_trafo", [])):
        loadings.append(net.res_trafo.loading_percent.to_numpy())
    if not loadings:
        return {"overload_rate": 0.0, "max_loading_percent": 0.0, "n_branches": 0}
    values = np.concatenate(loadings)
    values = values[np.isfinite(values)]
    return {
        "overload_rate": float((values > limit_percent).mean()),
        "max_loading_percent": float(values.max()),
        "n_branches": int(values.size),
    }


def physical_report(
    net: pp.pandapowerNet,
    vm_pu: np.ndarray,
    va_degree: np.ndarray,
    v_min: float = 0.95,
    v_max: float = 1.05,
) -> dict[str, float]:
    """Physical checks on a PREDICTED voltage state, plus the network's own loading.

    Read the key names: they say which quantity is which, and the distinction
    matters more than it looks.

    ``voltage_violations`` and ``power_balance_residual`` are computed FROM
    ``vm_pu``/``va_degree`` -- they are checks on the prediction, and a bad
    prediction moves them. ``line_loading_violations`` is not: it reads
    ``net.res_line.loading_percent``, the converged solution already stored on
    the network, and is therefore identical for every prediction you pass.

    Returning both under one flat dict previously made three columns of a table
    headed "physical checks on the prediction" into the answer key: feeding a
    nonsense state (all 0.80 pu, 0 deg) to case14 moves ``p_mismatch_max_mw``
    from 0.0 to 232.4 and ``violation_rate`` from 0.0 to 1.0, while
    ``max_loading_percent`` stays bit-identical at 1.5076. The ``truth_`` prefix
    below says so at the point of use.

    Computing loading from the predicted state is the better answer and is
    straightforward in principle -- ``ppc["internal"]["Yf"]`` gives branch
    currents from any voltage vector, and that route reproduces pandapower's
    ``loading_percent`` to 1e-12 on case14, case30 and case118 -- but the
    internal-branch-to-element mapping does not hold on every catalogued
    network (case33bw disagrees, 32 rows against 37 elements), so it is not
    done here rather than done unreliably.
    """
    truth_loading = {
        f"truth_{key}": value for key, value in line_loading_violations(net).items()
    }
    return {
        **voltage_violations(vm_pu, v_min, v_max),
        **power_balance_residual(net, vm_pu, va_degree),
        **truth_loading,
    }
