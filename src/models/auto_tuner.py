# auto_tuner.py
"""
Auto-tuner: blends profile scores, benchmark scores, and empirical
grading data to keep per-model capability vectors up to date.

Runs periodically (every 6 h by default) as a batch recalibration.
"""

from __future__ import annotations

import time
from typing import Any

from src.infra.logging_config import get_logger
from src.models.capabilities import TASK_PROFILES, Cap

log = get_logger("auto_tuner")

# ─── Constants ───────────────────────────────────────────────────────────────

TUNING_INTERVAL_SECONDS = 6 * 3600  # 6 hours
MIN_CHANGE_THRESHOLD = 0.2          # ignore noisy sub-0.2 diffs

_last_run_ts: float = 0.0

# ─── Blend Weight Schedule ───────────────────────────────────────────────────
# (min_calls, grading_weight, benchmark_weight, profile_weight)

_BLEND_SCHEDULE: list[tuple[int, float, float, float]] = [
    (50,  0.50, 0.30, 0.20),
    (20,  0.35, 0.39, 0.26),
    (5,   0.20, 0.48, 0.32),
    (0,   0.00, 0.60, 0.40),
]


def _get_blend_weights(call_count: int) -> tuple[float, float, float]:
    """Return (grading_w, benchmark_w, profile_w) for a given call count."""
    for min_calls, gw, bw, pw in _BLEND_SCHEDULE:
        if call_count >= min_calls:
            return gw, bw, pw
    # fallback (should not be reached)
    return 0.0, 0.60, 0.40


# ─── Core Blend ──────────────────────────────────────────────────────────────

def blend_capability_scores(
    profile_scores: dict[str, float],
    benchmark_scores: dict[str, float],
    grading_scores: dict[str, float],
    grading_call_count: int,
) -> dict[str, float]:
    """
    Blend three signal sources into a single capability vector.

    For each dimension:
    - If grading data exists AND call_count >= 5 → 3-way blend
    - Otherwise → 2-way blend (60% benchmark + 40% profile)

    Returns dict[str, float] with all values clamped to [0, 10].
    """
    all_caps = set(profile_scores) | set(benchmark_scores) | set(grading_scores)
    grading_w, bench_w, profile_w = _get_blend_weights(grading_call_count)

    result: dict[str, float] = {}
    for cap in all_caps:
        p = profile_scores.get(cap, 0.0)
        b = benchmark_scores.get(cap, 0.0)
        g = grading_scores.get(cap, 0.0)

        if cap in grading_scores and grading_call_count >= 5:
            # 3-way blend
            score = grading_w * g + bench_w * b + profile_w * p
        else:
            # 2-way fallback
            score = 0.60 * b + 0.40 * p

        result[cap] = round(min(10.0, max(0.0, score)), 2)

    return result


# ─── Grading Score Computation ───────────────────────────────────────────────

def compute_grading_scores(
    model_name: str,
    stats_rows: list[dict],
) -> tuple[dict[str, float], int]:
    """
    Convert model_stats rows into a capability score dict.

    Each row has: model, agent_type, avg_grade (1-5), success_rate (0-1),
    total_calls.

    Steps:
    1. Map agent_type → TASK_PROFILES → find dominant capabilities (weight >= 0.3)
    2. Convert grade to quality: avg_grade * 2.0 * (0.5 + 0.5 * success_rate)
    3. Distribute quality across capabilities weighted by profile weights.
    4. Average per-capability across all agent_types that contributed.

    Returns (scores_dict, total_call_count).
    """
    cap_accum: dict[str, list[float]] = {}
    total_calls = 0

    for row in stats_rows:
        agent_type = row.get("agent_type", "")
        avg_grade = row.get("avg_grade", 0.0)
        success_rate = row.get("success_rate", 0.0)
        calls = row.get("total_calls", 0)

        profile = TASK_PROFILES.get(agent_type)
        if not profile:
            log.debug(f"no task profile for agent_type={agent_type}, skipping")
            continue

        total_calls += calls

        # quality on 0-10 scale
        quality = avg_grade * 2.0 * (0.5 + 0.5 * success_rate)

        for cap_key, weight in profile.items():
            cap_name = cap_key.value if isinstance(cap_key, Cap) else cap_key
            if weight < 0.3:
                continue
            weighted_quality = quality * weight
            if cap_name not in cap_accum:
                cap_accum[cap_name] = []
            cap_accum[cap_name].append(weighted_quality)

    # Average contributions per capability
    scores: dict[str, float] = {}
    for cap_name, values in cap_accum.items():
        scores[cap_name] = round(min(10.0, max(0.0, sum(values) / len(values))), 2)

    return scores, total_calls


# ─── Tuning Cycle ────────────────────────────────────────────────────────────

async def run_tuning_cycle(force: bool = False) -> dict[str, Any]:
    """
    Batch recalibration of all model capability vectors.

    1. Read model_stats from DB.
    2. For each model: get profile_scores, benchmark_scores, compute
       grading_scores, blend.
    3. Only update capabilities where |change| > 0.2.

    Returns a report dict.
    """
    from src.infra.db import get_model_stats
    from src.models.model_registry import get_registry

    global _last_run_ts

    registry = get_registry()
    all_stats = await get_model_stats()

    # Group stats by model
    stats_by_model: dict[str, list[dict]] = {}
    for row in all_stats:
        model_name = row.get("model", "")
        if model_name:
            stats_by_model.setdefault(model_name, []).append(row)

    tuned: dict[str, dict] = {}
    skipped: list[str] = []

    for model_name, model_info in registry.models.items():
        # Profile scores — from ModelInfo (may not be populated yet)
        profile_scores = getattr(model_info, "profile_scores", None) or {}
        if not profile_scores:
            # Fall back to current capabilities as profile baseline
            profile_scores = dict(model_info.capabilities)

        # Benchmark scores — from ModelInfo (may not be populated yet)
        benchmark_scores = getattr(model_info, "benchmark_scores", None) or {}
        if not benchmark_scores:
            # Fall back to current capabilities as benchmark baseline
            benchmark_scores = dict(model_info.capabilities)

        # Grading scores from DB
        model_stats = stats_by_model.get(model_name, [])
        grading_scores, grading_calls = compute_grading_scores(
            model_name, model_stats
        )

        if not grading_scores and not benchmark_scores:
            skipped.append(model_name)
            continue

        blended = blend_capability_scores(
            profile_scores, benchmark_scores, grading_scores, grading_calls
        )

        # Only apply changes above threshold
        changes: dict[str, dict[str, float]] = {}
        for cap, new_val in blended.items():
            old_val = model_info.capabilities.get(cap, 0.0)
            if abs(new_val - old_val) > MIN_CHANGE_THRESHOLD:
                changes[cap] = {"old": old_val, "new": new_val}
                model_info.capabilities[cap] = new_val

        grading_w, _, _ = _get_blend_weights(grading_calls)

        if changes:
            tuned[model_name] = {
                "changes": changes,
                "grading_calls": grading_calls,
                "grading_weight": grading_w,
            }
            log.info(
                f"tuned {model_name}: {len(changes)} caps changed "
                f"(grading_calls={grading_calls}, grading_w={grading_w:.2f})"
            )
        else:
            skipped.append(model_name)

    _last_run_ts = time.time()
    report = {
        "tuned_models": tuned,
        "skipped": skipped,
        "timestamp": _last_run_ts,
    }
    log.info(f"tuning cycle complete: {len(tuned)} tuned, {len(skipped)} skipped")
    return report


async def maybe_run_tuning() -> dict[str, Any] | None:
    """Only runs if 6+ hours since last run."""
    global _last_run_ts
    if time.time() - _last_run_ts < TUNING_INTERVAL_SECONDS:
        return None
    return await run_tuning_cycle()


# ─── Prometheus Metrics ──────────────────────────────────────────────────────

def get_prometheus_lines() -> list[str]:
    """
    Generate Prometheus-format gauge lines for all model capabilities.

    Metrics emitted:
    - kutay_model_capability{model="...",capability="..."} <value>
    - kutay_model_quality_avg{model="..."} <value>
    - kutay_autotuner_last_run_timestamp <value>
    - kutay_autotuner_interval_seconds <value>
    """
    from src.models.model_registry import get_registry

    registry = get_registry()
    lines: list[str] = []

    for model_name, model_info in registry.models.items():
        caps = model_info.capabilities
        for cap_name, cap_val in caps.items():
            lines.append(
                f'kutay_model_capability{{model="{model_name}",'
                f'capability="{cap_name}"}} {cap_val}'
            )
        # Average quality across all caps
        if caps:
            avg = round(sum(caps.values()) / len(caps), 2)
            lines.append(
                f'kutay_model_quality_avg{{model="{model_name}"}} {avg}'
            )

    lines.append(f"kutay_autotuner_last_run_timestamp {_last_run_ts}")
    lines.append(f"kutay_autotuner_interval_seconds {TUNING_INTERVAL_SECONDS}")

    return lines
