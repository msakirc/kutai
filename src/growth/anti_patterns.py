"""Z9 T2D — growth anti-pattern detectors.

Three deterministic, pure-Python detectors the weekly digest pipeline runs
over the analytics pull. The ``growth_digest_synthesizer`` agent calls
:func:`detect_all` (via its tooling) and narrates the warnings; the
mechanical executor can also pre-compute them so an offline test has a
testable surface that never needs the LLM.

Detectors
---------
* **Vanity metric guard** — a north-star metric whose name is an absolute
  count (DAU/MAU absolute, page views, total signups) is a vanity metric:
  it goes up even when the product is dying. Warn: tie it to revenue or
  retention.
* **Engagement vampire** — high event volume paired with flat or declining
  retention. The product is *busy* but not *sticky*: users churn while
  generating noise. Warn before the founder reads activity as health.
* **Insufficient N** — any A/B test / experiment with fewer than 100
  daily-active samples has no statistical power. Warn before any verdict
  is computed off it.

Every detector is a pure function: given a plain dict it returns a
``Finding`` (or ``None``). No I/O, no LLM, no global state — trivially
unit-testable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Minimum daily-active sample size for an experiment to carry signal.
MIN_EXPERIMENT_N: int = 100

# Substrings that mark a north-star metric name as an absolute-count vanity
# metric. Matched case-insensitively against the metric name.
_VANITY_PATTERNS: tuple[str, ...] = (
    "dau",
    "mau",
    "wau",
    "daily active users",
    "monthly active users",
    "page view",
    "pageview",
    "page_view",
    "total signup",
    "total sign-up",
    "total sign up",
    "signups",
    "sign-ups",
    "registered users",
    "total users",
    "downloads",
    "impressions",
)

# Metric names that *contain* a vanity substring but are legitimately
# retention/revenue-tied — never flag these.
_VANITY_EXEMPT: tuple[str, ...] = (
    "dau/mau",  # the DAU/MAU ratio is a retention metric, not a vanity count
    "dau / mau",
    "stickiness",
    "paid mau",
    "activated mau",
)


@dataclass
class Finding:
    """One anti-pattern hit.

    ``code`` is a stable machine key; ``severity`` is ``"warn"`` (v1 never
    blocks — the digest is read-only); ``message`` is founder-facing prose.
    ``detail`` carries the raw numbers for the digest's appendix.
    """

    code: str
    severity: str
    message: str
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "detail": dict(self.detail),
        }


def _norm(name: Any) -> str:
    return str(name or "").strip().lower()


# ── detector 1: vanity metric guard ────────────────────────────────────────


def detect_vanity_metric(north_star: dict[str, Any] | None) -> Finding | None:
    """Flag an absolute-count north-star metric.

    ``north_star`` is the ``success_metrics.north_star_metric`` block:
    ``{"name": ..., "justification": ...}``. Returns ``None`` when the
    metric is healthy, missing, or an exempt ratio.
    """
    if not isinstance(north_star, dict):
        return None
    name = _norm(north_star.get("name"))
    if not name:
        return None
    if any(exempt in name for exempt in _VANITY_EXEMPT):
        return None
    matched = next((p for p in _VANITY_PATTERNS if p in name), None)
    if matched is None:
        return None
    return Finding(
        code="vanity_metric",
        severity="warn",
        message=(
            f"North-star metric '{north_star.get('name')}' is an absolute "
            f"count — it can climb while the product is dying. Tie it to "
            f"revenue or retention (e.g. paying retention, revenue per "
            f"cohort, week-2 retention)."
        ),
        detail={"metric_name": north_star.get("name"), "matched_pattern": matched},
    )


# ── detector 2: engagement vampire ─────────────────────────────────────────


def _retention_slope(curve: list[Any]) -> float:
    """Return retention trend: positive=improving, <=0=flat/declining.

    ``curve`` is a list of retention values (day-0 first). Slope is the
    average step-to-step delta over the tail (day-1 onward) — day-0 is
    always 100% so it would dominate a naive slope.
    """
    nums = [float(v) for v in curve if isinstance(v, (int, float))]
    if len(nums) < 3:
        return 0.0
    tail = nums[1:]
    deltas = [tail[i + 1] - tail[i] for i in range(len(tail) - 1)]
    if not deltas:
        return 0.0
    return sum(deltas) / len(deltas)


def detect_engagement_vampire(
    event_count: Any,
    retention_curve: list[Any] | None,
    *,
    event_count_floor: int = 500,
    slope_tolerance: float = 0.5,
) -> Finding | None:
    """Flag high activity paired with flat / declining retention.

    ``event_count`` is the last-7d total event volume. ``retention_curve``
    is a list of cohort retention values (day-0 first). A vampire is high
    volume (>= ``event_count_floor``) plus a non-positive retention slope
    (<= ``slope_tolerance`` — a tiny positive tolerance absorbs noise).
    """
    try:
        events = int(event_count or 0)
    except (TypeError, ValueError):
        return None
    if events < event_count_floor:
        return None
    if not isinstance(retention_curve, list) or len(retention_curve) < 3:
        return None
    slope = _retention_slope(retention_curve)
    if slope > slope_tolerance:
        return None
    return Finding(
        code="engagement_vampire",
        severity="warn",
        message=(
            f"High event volume ({events:,} events / 7d) but retention is "
            f"flat or declining (trend {slope:+.1f}/day). Activity is not "
            f"the same as stickiness — check whether engagement converts "
            f"to repeat use or revenue."
        ),
        detail={"event_count": events, "retention_slope": round(slope, 3)},
    )


# ── detector 3: insufficient N ─────────────────────────────────────────────


def detect_insufficient_n(
    experiments: list[dict[str, Any]] | None,
    *,
    min_n: int = MIN_EXPERIMENT_N,
) -> list[Finding]:
    """Flag every experiment with < ``min_n`` daily-active samples.

    ``experiments`` is a list of dicts shaped like
    ``{"name": ..., "daily_active_samples": int}`` (``daily_active``,
    ``samples`` and ``n`` are also accepted as the sample-count key).
    Returns one ``Finding`` per under-powered experiment — possibly empty.
    """
    if not isinstance(experiments, list):
        return []
    findings: list[Finding] = []
    for exp in experiments:
        if not isinstance(exp, dict):
            continue
        n_raw = (
            exp.get("daily_active_samples")
            if exp.get("daily_active_samples") is not None
            else exp.get("daily_active")
            if exp.get("daily_active") is not None
            else exp.get("samples")
            if exp.get("samples") is not None
            else exp.get("n")
        )
        try:
            n = int(n_raw)
        except (TypeError, ValueError):
            continue
        if n >= min_n:
            continue
        name = exp.get("name") or exp.get("variant_name") or "unnamed experiment"
        findings.append(
            Finding(
                code="insufficient_n",
                severity="warn",
                message=(
                    f"Experiment '{name}' has only {n} daily-active samples "
                    f"(< {min_n}). It has no statistical power — do not act "
                    f"on its verdict yet; let it accumulate exposure."
                ),
                detail={"experiment": name, "daily_active_samples": n, "min_n": min_n},
            )
        )
    return findings


# ── aggregate ──────────────────────────────────────────────────────────────


def detect_all(digest_input: dict[str, Any]) -> list[dict[str, Any]]:
    """Run all three detectors over a ``digest_input`` bundle.

    Reads (all optional):
      * ``digest_input["north_star"]`` → vanity guard
      * ``digest_input["event_count"]`` + ``["retention_curve"]`` → vampire
      * ``digest_input["experiments"]`` → insufficient-N

    Returns a flat list of finding dicts (see :meth:`Finding.as_dict`),
    suitable to drop straight into the digest's warnings section.
    """
    di = digest_input or {}
    findings: list[Finding] = []

    vanity = detect_vanity_metric(di.get("north_star"))
    if vanity is not None:
        findings.append(vanity)

    vampire = detect_engagement_vampire(
        di.get("event_count"), di.get("retention_curve")
    )
    if vampire is not None:
        findings.append(vampire)

    findings.extend(detect_insufficient_n(di.get("experiments")))

    return [f.as_dict() for f in findings]


__all__ = [
    "Finding",
    "MIN_EXPERIMENT_N",
    "detect_vanity_metric",
    "detect_engagement_vampire",
    "detect_insufficient_n",
    "detect_all",
]
