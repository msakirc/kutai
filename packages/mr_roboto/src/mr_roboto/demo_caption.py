"""Z7 T3B — ``demo/caption`` mr_roboto verb.

Generates a WebVTT caption file from the storyboard's ``narrator_text``
and scene ``target_seconds``.  **No speech-to-text is involved** — this
is a script-driven captioning pass:

  1. Read storyboard scenes in order.
  2. Compute start/end timestamps from cumulative ``target_seconds``.
  3. Write one VTT cue per scene whose ``narrator_text`` is non-empty.
  4. Write ``demo/demo.vtt`` to workspace.

WebVTT format reference: https://www.w3.org/TR/webvtt1/

Reversibility: ``full`` — writes a local .vtt file; git-reversible.
"""
from __future__ import annotations

import json
import os
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.demo_caption")


def _seconds_to_vtt_time(seconds: float) -> str:
    """Convert float seconds to VTT timestamp format ``HH:MM:SS.mmm``."""
    seconds = max(0.0, float(seconds))
    total_ms = round(seconds * 1000)
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _build_vtt(scenes: list[dict]) -> str:
    """Build WebVTT content from storyboard scenes.

    Each scene with non-empty narrator_text becomes one cue.
    """
    lines = ["WEBVTT", ""]
    t = 0.0
    for i, scene in enumerate(scenes, start=1):
        narrator = (scene.get("narrator_text") or "").strip()
        target = float(scene.get("target_seconds") or 5)
        start = t
        end = t + target
        t = end

        if not narrator:
            continue

        start_ts = _seconds_to_vtt_time(start)
        end_ts = _seconds_to_vtt_time(end)
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(narrator)
        lines.append("")

    return "\n".join(lines)


async def run(
    mission_id: int,
    workspace_path: str,
    storyboard_path: str,
) -> dict[str, Any]:
    """Generate WebVTT captions from storyboard narrator_text.

    Returns::

        {"ok": True, "vtt_path": str, "cue_count": int}
        {"ok": False, "error": str}
    """
    if not os.path.exists(storyboard_path):
        return {
            "ok": False,
            "error": f"storyboard.json not found at {storyboard_path}",
        }

    try:
        with open(storyboard_path, encoding="utf-8") as f:
            storyboard = json.load(f)
    except Exception as exc:
        return {"ok": False, "error": f"storyboard.json parse error: {exc}"}

    scenes = storyboard.get("scenes") or []
    if not scenes:
        return {"ok": False, "error": "storyboard has no scenes"}

    vtt_content = _build_vtt(scenes)
    cue_count = vtt_content.count(" --> ")

    demo_dir = os.path.join(workspace_path, "demo")
    os.makedirs(demo_dir, exist_ok=True)
    vtt_path = os.path.join(demo_dir, "demo.vtt")

    with open(vtt_path, "w", encoding="utf-8") as f:
        f.write(vtt_content)

    logger.info(
        "demo_caption: VTT written",
        mission_id=mission_id,
        cue_count=cue_count,
        vtt_path=vtt_path,
    )

    return {
        "ok": True,
        "vtt_path": vtt_path,
        "cue_count": cue_count,
    }
