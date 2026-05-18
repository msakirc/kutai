"""Z9 Growth T5D — A/B experiment result evaluation.

Pure, deterministic, no-LLM math. Given a closed hypothesis verdict and the
two measured variant arms (control + treatment), decide which arm won by a
Bayesian posterior — reusing the normal-approximation engine in
``src/growth/verdict_stats.py``.

Why this module exists
----------------------
``verdict_stats.compute_verdict`` answers "did the *prediction* hold". An
A/B test asks a sibling question: "did *treatment* beat *control*". Both
are normal-approx posteriors on a relative lift, so we reuse
``_normal_cdf`` rather than reinvent the math — the A/B call is
``compute_verdict`` with the control arm as the baseline and the treatment
arm as the actual, predicting an "up" move of the observed magnitude.

A winner needs posterior **> 0.95** (``verdict_stats.POSTERIOR_THRESHOLD``)
that treatment's lift is positive (treatment wins) or negative (control
wins). Below the threshold the result is ``inconclusive`` — no auto-ship.

Hook point
----------
This is called from ``mr_roboto.executors.record_verdict`` once a
hypothesis verdict closes: if the mission carried ``experiment_variants``
rows, the verdict task also measures each arm and writes an ``ab_result``
``growth_events`` row. Auto-rollback of a confident loser still requires a
founder gate — this module only *computes*, it never retires a variant.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.growth.verdict_stats import (
    POSTERIOR_THRESHOLD,
    _normal_cdf,
    DEFAULT_REL_SIGMA,
)


@dataclass(frozen=True)
class ABResult:
    """Outcome of an A/B variant comparison."""

    winner: str  # control | treatment | inconclusive
    p_treatment_better: float  # posterior P(treatment lift > 0)
    p_control_better: float  # posterior P(treatment lift < 0)
    control_metric: float
    treatment_metric: float
    relative_lift: float  # (treatment - control) / |control|
    confident: bool  # True iff a winner cleared the 0.95 posterior
    model: str = "normal-approx posterior on treatment-vs-control lift"


def evaluate_ab(
    *,
    control_metric: float,
    treatment_metric: float,
    rel_sigma: float = DEFAULT_REL_SIGMA,
) -> ABResult:
    """Decide an A/B winner from the two measured arm metrics.

    Parameters
    ----------
    control_metric / treatment_metric:
        The metric value measured for each arm over the experiment window.
    rel_sigma:
        Relative standard error of the measurement (default 5%). Callers
        with a known per-arm sample size can tighten it.

    Returns
    -------
    ABResult — ``winner`` is ``treatment`` / ``control`` only when the
    posterior cleared ``POSTERIOR_THRESHOLD`` (0.95); otherwise
    ``inconclusive`` and ``confident=False``.
    """
    rel_sigma = max(1e-4, float(rel_sigma or DEFAULT_REL_SIGMA))
    denom = (
        abs(float(control_metric))
        if abs(float(control_metric)) > 1e-9
        else 1.0
    )
    rel_lift = (float(treatment_metric) - float(control_metric)) / denom

    # Posterior on the true lift is Normal(rel_lift, rel_sigma^2).
    # P(treatment better) = P(true lift > 0); P(control better) = mirror.
    p_treatment_better = 1.0 - _normal_cdf((0.0 - rel_lift) / rel_sigma)
    p_control_better = _normal_cdf((0.0 - rel_lift) / rel_sigma)

    if p_treatment_better >= POSTERIOR_THRESHOLD:
        winner, confident = "treatment", True
    elif p_control_better >= POSTERIOR_THRESHOLD:
        winner, confident = "control", True
    else:
        winner, confident = "inconclusive", False

    return ABResult(
        winner=winner,
        p_treatment_better=round(p_treatment_better, 6),
        p_control_better=round(p_control_better, 6),
        control_metric=round(float(control_metric), 6),
        treatment_metric=round(float(treatment_metric), 6),
        relative_lift=round(rel_lift, 6),
        confident=confident,
    )


__all__ = ["ABResult", "evaluate_ab", "POSTERIOR_THRESHOLD"]
