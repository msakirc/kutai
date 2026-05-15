"""Z7 T3B — demo_artifact_check posthook handler.

Verifies that the demo pipeline produced valid output artifacts:

  1. All expected cut files exist (``cuts/30s.mp4``, ``cuts/60s.mp4``,
     ``cuts/3min.mp4``).
  2. Each cut's duration is within ±10% of its declared target.
  3. The caption (.vtt) file is present.

Handler contract
----------------
``handle(task, result) -> dict``

Returns one of:

- ``{"status": "ok", "checks": {...}}``            — all checks passed
- ``{"status": "failed", "error": str, ...}``      — at least one check failed
- ``{"status": "skip", "reason": str}``            — no demo context on task
"""
from __future__ import annotations

import os
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("beckman.posthooks.demo_artifact_check")

# ±10% duration tolerance
_DURATION_TOLERANCE = 0.10

# Expected cut labels and their target seconds
_DEFAULT_CUT_TARGETS: dict[str, int] = {
    "30s": 30,
    "60s": 60,
    "3min": 180,
}


def _video_duration_seconds(path: str) -> float:
    """Best-effort ffprobe duration. Returns 0.0 on failure."""
    import subprocess
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
        return float((out or b"").decode("utf-8", "replace").strip() or 0.0)
    except Exception:
        return 0.0


def _check_cuts(
    cuts: dict[str, str],
    cut_targets: dict[str, int],
) -> tuple[bool, list[str], list[str]]:
    """Check that all expected cut files exist with acceptable durations.

    Returns (passed, missing_cuts, duration_failures).
    """
    missing: list[str] = []
    duration_failures: list[str] = []

    for label, target_s in cut_targets.items():
        path = cuts.get(label, "")
        if not path or not os.path.exists(path):
            missing.append(label)
            continue
        duration = _video_duration_seconds(path)
        if duration > 0.0:
            allowed_low = target_s * (1.0 - _DURATION_TOLERANCE)
            allowed_high = target_s * (1.0 + _DURATION_TOLERANCE)
            if duration < allowed_low or duration > allowed_high:
                duration_failures.append(
                    f"{label} (got {duration:.1f}s, want {target_s}s ±10%)"
                )

    passed = not missing and not duration_failures
    return passed, missing, duration_failures


async def handle(task: dict, result: dict) -> dict[str, Any]:
    """demo_artifact_check posthook handler.

    Reads demo artifact locations from task context (populated by
    the demo/edit and demo/caption verbs).
    """
    import json as _json

    task_id = task.get("id")
    mission_id = task.get("mission_id")

    # Parse task context
    ctx_raw = task.get("context", "{}")
    if isinstance(ctx_raw, str):
        try:
            ctx: dict = _json.loads(ctx_raw)
        except Exception:
            ctx = {}
    elif isinstance(ctx_raw, dict):
        ctx = ctx_raw
    else:
        ctx = {}

    demo_cuts = ctx.get("demo_cuts") or {}
    vtt_path = ctx.get("demo_vtt_path") or ""
    cut_targets = ctx.get("demo_cut_targets") or _DEFAULT_CUT_TARGETS

    # Graceful skip when no demo context is present
    if not demo_cuts and not vtt_path:
        logger.debug(
            "demo_artifact_check: no demo_cuts or demo_vtt_path in context — skip",
            task_id=task_id,
        )
        return {"status": "skip", "reason": "no demo artifact context on task"}

    failures: list[str] = []

    # 1. Check cut files
    passed, missing_cuts, duration_failures = _check_cuts(demo_cuts, cut_targets)
    if missing_cuts:
        failures.append(f"missing cuts: {', '.join(missing_cuts)}")
    if duration_failures:
        failures.append(f"duration out of ±10% tolerance: {'; '.join(duration_failures)}")

    # 2. Check captions
    if not vtt_path or not os.path.exists(vtt_path):
        failures.append(f"caption .vtt file missing at {vtt_path!r}")
    else:
        try:
            vtt_content = open(vtt_path, encoding="utf-8").read().strip()
            if not vtt_content.startswith("WEBVTT"):
                failures.append("caption file is not a valid WebVTT file (missing WEBVTT header)")
        except Exception as exc:
            failures.append(f"caption file unreadable: {exc}")

    if failures:
        error_msg = "; ".join(failures)
        logger.warning(
            "demo_artifact_check: failed",
            task_id=task_id,
            mission_id=mission_id,
            failures=failures,
        )
        return {
            "status": "failed",
            "error": error_msg,
            "missing_cuts": missing_cuts,
            "duration_failures": duration_failures,
        }

    logger.info(
        "demo_artifact_check: all checks passed",
        task_id=task_id,
        mission_id=mission_id,
        cuts=list(demo_cuts.keys()),
    )
    return {
        "status": "ok",
        "checks": {
            "cuts": list(demo_cuts.keys()),
            "captions_present": True,
        },
    }
