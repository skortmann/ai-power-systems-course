"""Course-wide numerical tolerances, in one place with reasons.

Scattering magic numbers like ``1e-6`` through ten notebooks invites two
failures: a check too tight for the arithmetic that produced the number, and a
check so loose it passes on a wrong answer. Both are easy to write and neither
announces itself.

Machine learning needs a wider spread of tolerances than most numerical work,
because the quantities being compared have genuinely different precision. A
causal-mask check is exact arithmetic and must be tested at float precision; a
Monte Carlo coverage estimate has sampling error of several per cent and
testing it at 1e-6 would only measure the seed.

Each constant below says what it guards and why that magnitude.
"""

from __future__ import annotations

__all__ = [
    "EXACT_FLOAT32",
    "EXACT_FLOAT64",
    "METRIC_RELATIVE",
    "GRADIENT_RELATIVE",
    "PROBABILITY_ABSOLUTE",
    "COVERAGE_ABSOLUTE",
    "PHYSICS_RELATIVE",
    "LOSS_RELATIVE",
    "describe",
    "close",
]

#: Two float32 tensors that should be bit-identical, allowing only for
#: non-deterministic reduction order. The causal-mask test uses this: if
#: perturbing a future token changes a past position's logits by MORE than
#: this, the mask leaks, and no amount of "it's just numerical noise" applies
#: -- a correct mask removes the future from the computation entirely, so the
#: difference should be exactly zero and is allowed to be 1e-6 only because
#: float32 matmul reductions are not order-stable.
EXACT_FLOAT32 = 1e-6

#: The same idea in double precision, for NumPy reference implementations.
EXACT_FLOAT64 = 1e-12

#: A metric recomputed independently (by hand, or by scikit-learn) against the
#: course's own implementation. Both are simple float64 reductions over the
#: same array, so they should agree to near machine precision; 1e-10 leaves
#: room for a different summation order.
METRIC_RELATIVE = 1e-10

#: An analytical gradient against a finite-difference estimate. Deliberately
#: loose: central differences carry O(h^2) truncation error on top of
#: cancellation in the subtraction, and 1e-4 is the usual working threshold.
GRADIENT_RELATIVE = 1e-4

#: An empirical rate from N samples against its nominal value. The standard
#: error of a proportion is sqrt(p(1-p)/N); at p=0.5 and N=2000 that is 0.011,
#: so 0.05 is several standard errors and will not flag ordinary sampling
#: noise.
PROBABILITY_ABSOLUTE = 5e-2

#: Empirical coverage of a prediction interval against its nominal level.
#: Looser than PROBABILITY_ABSOLUTE because interval coverage is estimated
#: from fewer effective samples on a correlated time series than the raw count
#: suggests.
COVERAGE_ABSOLUTE = 1e-1

#: A power-system identity such as generation == load + losses, relative to
#: the total. pandapower's Newton-Raphson converges to ~1e-8 per unit, so 1e-6
#: sits well above the solver and well below anything physically meaningful.
PHYSICS_RELATIVE = 1e-6

#: Two training runs that claim to be reproducible, compared on their loss.
#: Not zero: torch's deterministic mode still permits different reduction
#: orders across thread counts, and the course does not pin thread count.
LOSS_RELATIVE = 1e-5


def describe() -> str:
    """A printable summary, for the audit report and for notebooks."""
    rows = [
        ("exact (float32)", EXACT_FLOAT32, "causal masking, frozen weights"),
        ("exact (float64)", EXACT_FLOAT64, "NumPy reference implementations"),
        ("metric (relative)", METRIC_RELATIVE, "our metric vs an independent one"),
        ("gradient (relative)", GRADIENT_RELATIVE, "analytic vs finite difference"),
        ("probability (absolute)", PROBABILITY_ABSOLUTE, "empirical rate vs nominal"),
        ("coverage (absolute)", COVERAGE_ABSOLUTE, "interval coverage vs nominal"),
        ("physics (relative)", PHYSICS_RELATIVE, "generation == load + losses"),
        ("loss (relative)", LOSS_RELATIVE, "reproducibility of training"),
    ]
    width = max(len(name) for name, _, _ in rows)
    lines = [f"{'quantity'.ljust(width)}  tolerance   guards"]
    lines += [f"{name.ljust(width)}  {value:<10.1e}  {why}" for name, value, why in rows]
    return "\n".join(lines)


def close(a: float, b: float, tolerance: float = METRIC_RELATIVE) -> bool:
    """Relative comparison with an absolute floor, so zero compares sanely."""
    return abs(a - b) <= tolerance * max(1.0, abs(a), abs(b))
