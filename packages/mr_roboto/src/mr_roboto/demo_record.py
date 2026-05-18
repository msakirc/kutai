"""Z7 T3B — ``demo/record`` mr_roboto verb.

Records each storyboard scene using Playwright ``--video on`` mode, producing
raw per-scene MP4 files.

Playwright is invoked via subprocess. Tests mock ``_run_subprocess``.

Reversibility: ``full`` — additive artifact writes; git-reversible.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.demo_record")


async def _run_subprocess(cmd: list[str], timeout: float = 300.0) -> tuple[int, str, str]:
    """Run a subprocess; return (rc, stdout, stderr)."""
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


async def _record_scene(
    *,
    scene: dict,
    raw_dir: str,
    workspace_path: str,
    base_url: str,
    timeout: float = 300.0,
) -> dict[str, Any]:
    """Record a single scene with Playwright.

    Playwright is invoked as:
        npx playwright test --grep <scene_id> --video on --output <raw_dir>

    Returns a dict with scene_id, path, and duration_s (0.0 when undetectable).
    """
    scene_id = scene.get("id", "scene_unknown")
    viewport_state = scene.get("viewport_state", "")
    target_seconds = int(scene.get("target_seconds") or 30)

    # Playwright records to an auto-named .webm under --output dir.
    # We ask playwright to run for exactly target_seconds worth of interaction;
    # ffmpeg trim is done later in demo/edit.
    #
    # Command: npx playwright test --grep <scene_id> --video on
    #          --output <raw_dir>/<scene_id>
    scene_output_dir = os.path.join(raw_dir, scene_id)
    os.makedirs(scene_output_dir, exist_ok=True)

    cmd = [
        "npx", "playwright", "test",
        "--grep", scene_id,
        "--video", "on",
        "--output", scene_output_dir,
    ]

    if base_url:
        cmd += ["--base-url", base_url]

    rc, stdout, stderr = await _run_subprocess(cmd, timeout=timeout)
    if rc != 0:
        combined = (stdout + "\n" + stderr).lower()
        hint = ""
        if "playwright" in combined and ("not found" in combined or "no such" in combined):
            hint = " (playwright not installed — run: npx playwright install)"
        elif "npx: not found" in combined or "command not found" in combined:
            hint = " (npx/node not available)"
        return {
            "ok": False,
            "scene_id": scene_id,
            "error": f"playwright failed (rc={rc}){hint}: {stderr.strip()[:300]}",
        }

    # Find the newest .webm in scene_output_dir
    webm_path: str | None = None
    newest_mtime = -1.0
    for root, _, files in os.walk(scene_output_dir):
        for fname in files:
            if fname.endswith(".webm"):
                p = os.path.join(root, fname)
                try:
                    mtime = os.path.getmtime(p)
                except OSError:
                    continue
                if mtime > newest_mtime:
                    newest_mtime = mtime
                    webm_path = p

    if webm_path is None:
        return {
            "ok": False,
            "scene_id": scene_id,
            "error": (
                f"no .webm produced under {scene_output_dir}; "
                "ensure playwright config enables video recording"
            ),
        }

    # Copy the raw .webm for this scene into raw_dir (un-trimmed).
    # No transcode here — demo_edit re-encodes the assembled cut downstream.
    out_webm = os.path.join(raw_dir, f"{scene_id}.webm")
    if webm_path != out_webm:
        import shutil
        shutil.copy2(webm_path, out_webm)

    duration_s = _video_duration_seconds(out_webm)

    return {
        "ok": True,
        "scene_id": scene_id,
        "path": out_webm,
        "target_seconds": target_seconds,
        "duration_s": duration_s,
        "viewport_state": viewport_state,
    }


async def run(
    mission_id: int,
    workspace_path: str,
    storyboard_path: str,
    base_url: str = "",
    playwright_timeout: float = 300.0,
) -> dict[str, Any]:
    """Record each storyboard scene.  Returns result dict.

    Returns::

        {"ok": True, "scene_recordings": [...], "raw_dir": str}
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
        return {"ok": False, "error": "storyboard has no scenes to record"}

    raw_dir = os.path.join(workspace_path, "demo", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    recordings: list[dict] = []
    errors: list[str] = []

    for scene in scenes:
        rec = await _record_scene(
            scene=scene,
            raw_dir=raw_dir,
            workspace_path=workspace_path,
            base_url=base_url,
            timeout=playwright_timeout,
        )
        if rec.get("ok"):
            recordings.append(rec)
        else:
            errors.append(f"{rec.get('scene_id')}: {rec.get('error')}")

    if errors:
        return {
            "ok": False,
            "error": f"playwright recording failed for scenes: {'; '.join(errors)}",
            "scene_recordings": recordings,
        }

    logger.info(
        "demo_record: all scenes recorded",
        mission_id=mission_id,
        scene_count=len(recordings),
        raw_dir=raw_dir,
    )

    return {
        "ok": True,
        "scene_recordings": recordings,
        "raw_dir": raw_dir,
    }
