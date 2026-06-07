"""Z7 T3B — ``demo/storyboard`` mr_roboto verb.

Mechanical sink: reads the producer's raw storyboard file, normalizes scenes,
and writes the final ``demo/storyboard.json``.  No LLM call is made here — the
LLM draft is produced by the ``13.demo_storyboard_draft`` workflow step
(agent:reviewer) and materialized to ``<workspace>/demo/storyboard_raw.json``
before this verb runs.

The storyboard is a structured JSON with ordered scenes, each containing:

  - id: str                 — e.g. "scene_1", "scene_2"
  - title: str              — short scene label
  - target_seconds: int     — how long this scene should run in the demo
  - viewport_state: str     — e.g. "home_page", "dashboard", "feature_xyz"
  - narrator_text: str      — script text for caption / audio-description
  - visual_only: bool       — True when scene has no narrator_text

Reversibility: ``full`` — writes storyboard.json to workspace; git-reversible.
"""
from __future__ import annotations

import json
import os
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.demo_storyboard")


# ---------------------------------------------------------------------------
# Subprocess helper (shared with demo_record / demo_edit)
# ---------------------------------------------------------------------------

async def _run_subprocess(cmd: list[str], timeout: float = 300.0) -> tuple[int, str, str]:
    import asyncio
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        return -1, "", f"timeout after {timeout}s"
    return (proc.returncode or 0), (out or b"").decode("utf-8", "replace"), (err or b"").decode("utf-8", "replace")


def _parse_storyboard_response(content: str) -> dict | None:
    """Extract a JSON storyboard dict from LLM response content."""
    if not content:
        return None
    content = content.strip()
    # Strip code fences if present
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(
            l for l in lines
            if not l.strip().startswith("```")
        ).strip()
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict) and "scenes" in parsed:
            return parsed
        return None
    except (json.JSONDecodeError, ValueError):
        # Try to find a JSON object in the text
        import re
        m = re.search(r"\{[\s\S]*\}", content)
        if m:
            try:
                parsed = json.loads(m.group())
                if isinstance(parsed, dict) and "scenes" in parsed:
                    return parsed
            except Exception:
                pass
        return None


def _normalize_scenes(storyboard: dict) -> int:
    """Backfill scene ids + visual_only in place. Returns scene count."""
    scenes = storyboard.get("scenes") or []
    for i, scene in enumerate(scenes):
        scene.setdefault("id", f"scene_{i + 1}")
        scene.setdefault(
            "visual_only", not bool(scene.get("narrator_text", "").strip())
        )
    return len(scenes)


async def run(
    *,
    mission_id: int,
    workspace_path: str,
    raw_filename: str = "demo/storyboard_raw.json",
) -> dict[str, Any]:
    """Mechanical sink: read the producer's raw storyboard, normalize, write.

    The LLM draft is produced by the `13.demo_storyboard_draft` workflow step
    (agent:reviewer) and materialized to ``<workspace>/<raw_filename>``. This
    verb makes NO LLM call.

    Returns::
        {"ok": True, "storyboard": {...}, "storyboard_path": str, "scene_count": int}
        {"ok": False, "error": str}
    """
    raw_path = os.path.join(workspace_path, raw_filename)
    try:
        with open(raw_path, encoding="utf-8") as fh:
            content = fh.read()
    except OSError as exc:
        return {"ok": False, "error": f"raw storyboard file missing at {raw_path}: {exc}"}

    storyboard = _parse_storyboard_response(content)
    if storyboard is None:
        return {"ok": False, "error": f"raw storyboard unparseable: {content[:200]!r}"}

    scene_count = _normalize_scenes(storyboard)

    demo_dir = os.path.join(workspace_path, "demo")
    os.makedirs(demo_dir, exist_ok=True)
    storyboard_path = os.path.join(demo_dir, "storyboard.json")
    with open(storyboard_path, "w", encoding="utf-8") as f:
        json.dump(storyboard, f, indent=2, ensure_ascii=False)

    logger.info(
        "demo_storyboard sink: written",
        mission_id=mission_id,
        scene_count=scene_count,
        storyboard_path=storyboard_path,
    )
    return {
        "ok": True,
        "storyboard": storyboard,
        "storyboard_path": storyboard_path,
        "scene_count": scene_count,
    }
