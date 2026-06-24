"""Z7 T3B A3.r1 — demo_accessibility_check posthook handler.

Validates that the demo accessibility manifest (produced by
``demo/accessibility_pass``) is complete:

  1. ``alt_texts`` is a non-empty list, each entry has ``scene_id`` and
     ``alt_text``.
  2. ``audio_description_track`` is present (may be empty if no visual-only
     scenes exist, but the key must exist).
  3. ``keyboard_nav_variant`` is a non-empty list, each entry has
     ``scene_id`` and ``step_description``.

Handler contract
----------------
``handle(task, result) -> dict``

Returns one of:

- ``{"status": "ok", "checks": {...}}``        — manifest is complete
- ``{"status": "failed", "error": str, ...}``  — at least one check failed
- ``{"status": "skip", "reason": str}``        — no manifest path in context
"""
from __future__ import annotations

import json
import os
from typing import Any

from yazbunu import get_logger

logger = get_logger("beckman.posthooks.demo_accessibility_check")


async def handle(task: dict, result: dict) -> dict[str, Any]:
    """demo_accessibility_check posthook handler."""
    task_id = task.get("id")
    mission_id = task.get("mission_id")

    # Parse task context
    ctx_raw = task.get("context", "{}")
    if isinstance(ctx_raw, str):
        try:
            ctx: dict = json.loads(ctx_raw)
        except Exception:
            ctx = {}
    elif isinstance(ctx_raw, dict):
        ctx = ctx_raw
    else:
        ctx = {}

    manifest_path = ctx.get("demo_accessibility_manifest_path") or ""

    # Graceful skip when no manifest path in context
    if not manifest_path:
        logger.debug(
            "demo_accessibility_check: no manifest path in context — skip",
            task_id=task_id,
        )
        return {"status": "skip", "reason": "no demo_accessibility_manifest_path in task context"}

    # Check manifest exists
    if not os.path.exists(manifest_path):
        return {
            "status": "failed",
            "error": f"accessibility manifest not found at {manifest_path!r}",
        }

    # Parse manifest
    try:
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as exc:
        return {
            "status": "failed",
            "error": f"accessibility manifest parse error: {exc}",
        }

    failures: list[str] = []

    # 1. alt_texts — must be a non-empty list with scene_id + alt_text
    alt_texts = manifest.get("alt_texts")
    if not isinstance(alt_texts, list) or len(alt_texts) == 0:
        failures.append(
            "alt_texts is missing or empty; every scene must have an alt_text entry"
        )
    else:
        for entry in alt_texts:
            if not isinstance(entry, dict):
                failures.append("alt_texts contains non-dict entry")
                break
            if not entry.get("alt_text"):
                failures.append(
                    f"alt_text is empty for scene_id={entry.get('scene_id')!r}"
                )

    # 2. audio_description_track — key must exist (may be empty list)
    if "audio_description_track" not in manifest:
        failures.append(
            "audio_description_track key is missing from manifest; "
            "it must be present (empty list is acceptable if no visual-only scenes exist)"
        )

    # 3. keyboard_nav_variant — must be a non-empty list
    knv = manifest.get("keyboard_nav_variant")
    if not isinstance(knv, list) or len(knv) == 0:
        failures.append(
            "keyboard_nav_variant is missing or empty; "
            "a keyboard navigation walkthrough variant is required"
        )
    else:
        for step in knv:
            if not isinstance(step, dict):
                failures.append("keyboard_nav_variant contains non-dict entry")
                break
            if not step.get("step_description"):
                failures.append(
                    f"step_description is empty for scene_id={step.get('scene_id')!r}"
                )

    if failures:
        error_msg = "; ".join(failures)
        logger.warning(
            "demo_accessibility_check: failed",
            task_id=task_id,
            mission_id=mission_id,
            failures=failures,
        )
        return {
            "status": "failed",
            "error": error_msg,
            "failures": failures,
        }

    logger.info(
        "demo_accessibility_check: manifest complete",
        task_id=task_id,
        mission_id=mission_id,
        alt_text_count=len(alt_texts or []),
        audio_desc_count=len(manifest.get("audio_description_track") or []),
        keyboard_nav_steps=len(knv or []),
    )
    return {
        "status": "ok",
        "checks": {
            "alt_text_count": len(alt_texts or []),
            "audio_description_track_present": True,
            "keyboard_nav_steps": len(knv or []),
        },
    }
