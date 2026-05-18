"""Z9 Growth T4D — Bayesian verdict statistics for the hypothesis loop.

Pure, deterministic, no-LLM math. Given a hypothesis's predicted metric
impact and the actually-measured value, decide whether the prediction
*held* (``confirmed``), was *contradicted* (``refuted``) or is *unclear*
(``inconclusive``).

Statistical model
-----------------
We use a **normal-approximation posterior** on the *relative metric lift*
``r = (actual - baseline) / |baseline|``.

  - The prediction asserts a direction (``up`` / ``down``) and a magnitude
    (a fractional lift, e.g. 0.12 for "+12%").
  - The measurement carries sampling noise. We model the observed lift as
    drawn from ``Normal(r, sigma^2)`` where ``sigma`` is a relative
    standard error (default 0.05 = 5%; callers may pass a tighter/looser
    value derived from sample size).
  - ``P(prediction held)`` = posterior mass on the half-line consistent
    with the predicted direction *and* at least a usable fraction of the
    predicted magnitude. ``P(opposite)`` = posterior mass strictly on the
    contradicting side of the baseline.

A one-tailed normal CDF gives both probabilities in closed form — no
sampling, fully reproducible, trivially unit-testable. Threshold is the
founder-decided **0.95** posterior (see docs/i2p-evolution/09-growth-v2.md
"A/B significance → Bayesian posterior > 95%").

This same engine is intended to back the T5 A/B "winner" call.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# Founder-decided posterior threshold for a decisive verdict.
POSTERIOR_THRESHOLD: float = 0.95

# Default relative standard error of a metric measurement when the caller
# supplies no sample-size-derived sigma. 5% is a deliberately conservative
# "we can't see tiny moves" floor.
DEFAULT_REL_SIGMA: float = 0.05

# A prediction "held" if the realised lift reaches at least this fraction
# of the predicted magnitude in the predicted direction. Partial credit:
# a "+12%" prediction that lands at +7% still counts as directionally
# correct and materially so.
MAGNITUDE_CREDIT_FRACTION: float = 0.5


@dataclass(frozen=True)
class VerdictResult:
    """Outcome of a Bayesian verdict computation."""

    verdict: str  # confirmed | refuted | inconclusive
    p_held: float  # posterior P(prediction held)
    p_opposite: float  # posterior P(metric moved the opposite way)
    observed_lift: float  # measured relative lift
    predicted_lift: float  # signed predicted relative lift
    model: str = "normal-approx posterior on relative metric lift"


def _normal_cdf(z: float) -> float:
    """Standard-normal CDF via erf — no scipy dependency."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _signed_predicted_lift(direction: str, magnitude: float) -> float:
    """Resolve a (direction, magnitude) prediction to a signed lift.

    ``direction`` is normalised: up/increase/+ → positive, down/decrease/-
    → negative. ``magnitude`` is treated as a fraction (0.12 == +12%); a
    value >1 is read as a percentage and divided by 100.
    """
    mag = abs(float(magnitude or 0.0))
    if mag > 1.0:
        mag = mag / 100.0
    d = str(direction or "").strip().lower()
    if d in ("down", "decrease", "lower", "reduce", "-", "neg", "negative"):
        return -mag
    # default / explicit up
    return mag


def compute_verdict(
    *,
    baseline: float,
    actual: float,
    direction: str,
    magnitude: float,
    rel_sigma: float = DEFAULT_REL_SIGMA,
) -> VerdictResult:
    """Decide a hypothesis verdict from measured vs predicted metric impact.

    Parameters
    ----------
    baseline:
        The metric value *before* the feature shipped (pre-launch baseline).
    actual:
        The metric value measured after the verdict window closed.
    direction / magnitude:
        The prediction — e.g. ``direction="up", magnitude=0.12`` for
        "+12% conversion".
    rel_sigma:
        Relative standard error of the measurement. Smaller → more
        confident verdicts; callers with a known sample size can tighten
        it. Clamped to a small positive floor.

    Returns
    -------
    VerdictResult with ``verdict`` in {confirmed, refuted, inconclusive}.
    """
    rel_sigma = max(1e-4, float(rel_sigma or DEFAULT_REL_SIGMA))
    predicted = _signed_predicted_lift(direction, magnitude)

    # Relative lift; guard a zero/near-zero baseline.
    denom = abs(float(baseline)) if abs(float(baseline)) > 1e-9 else 1.0
    observed = (float(actual) - float(baseline)) / denom

    # Posterior on the true lift is Normal(observed, rel_sigma^2).
    # "held" = true lift is on the predicted side of the magnitude credit
    # line; "opposite" = true lift is strictly past zero the other way.
    if predicted >= 0:
        credit_line = predicted * MAGNITUDE_CREDIT_FRACTION
        # P(true lift >= credit_line)
        p_held = 1.0 - _normal_cdf((credit_line - observed) / rel_sigma)
        # P(true lift < 0)
        p_opposite = _normal_cdf((0.0 - observed) / rel_sigma)
    else:
        credit_line = predicted * MAGNITUDE_CREDIT_FRACTION
        # P(true lift <= credit_line)
        p_held = _normal_cdf((credit_line - observed) / rel_sigma)
        # P(true lift > 0)
        p_opposite = 1.0 - _normal_cdf((0.0 - observed) / rel_sigma)

    if p_held >= POSTERIOR_THRESHOLD:
        verdict = "confirmed"
    elif p_opposite >= POSTERIOR_THRESHOLD:
        verdict = "refuted"
    else:
        verdict = "inconclusive"

    return VerdictResult(
        verdict=verdict,
        p_held=round(p_held, 6),
        p_opposite=round(p_opposite, 6),
        observed_lift=round(observed, 6),
        predicted_lift=round(predicted, 6),
    )


__all__ = ["VerdictResult", "compute_verdict", "POSTERIOR_THRESHOLD"]
