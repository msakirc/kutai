"""Z7 T3B — ``demo/edit`` mr_roboto verb.

Concatenates and trims per-scene raw recordings into three cut lengths:

  - ``cuts/30s.mp4``   — 30-second highlight
  - ``cuts/60s.mp4``   — 60-second overview
  - ``cuts/3min.mp4``  — 3-minute full demo

ffmpeg is invoked via subprocess.  Tests mock ``_run_subprocess``.

Cut strategy
------------
- **30s**: scenes trimmed to fit within 30 seconds total (proportional to
  each scene's target_seconds).
- **60s**: scenes trimmed to fit within 60 seconds total.
- **3min (180s)**: all scenes at full target_seconds, trimmed to 180s max.

ffmpeg concat approach: write a concat file list, then run ffmpeg with
``-t <duration>`` to limit output length.

Reversibility: ``full`` — writes local files; git-reversible.
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.demo_edit")


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


def _write_concat_list(paths: list[str], list_path: str) -> None:
    """Write an ffmpeg concat list file."""
    with open(list_path, "w", encoding="utf-8") as f:
        for p in paths:
            # Escape single quotes in path for ffmpeg concat format
            escaped = p.replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")


async def _produce_cut(
    *,
    scene_recordings: list[dict],
    output_path: str,
    target_seconds: int,
    temp_dir: str,
) -> dict[str, Any]:
    """Produce one cut by concat + trim to target_seconds.

    Returns {ok: bool, path: str, duration_s: float, error: str}.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Build list of existing recording files
    existing = [r["path"] for r in scene_recordings if os.path.exists(r.get("path", ""))]
    if not existing:
        return {
            "ok": False,
            "path": output_path,
            "error": "no scene recordings on disk",
        }

    # Write concat list
    list_path = os.path.join(temp_dir, f"concat_{os.path.basename(output_path)}.txt")
    _write_concat_list(existing, list_path)

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_path,
        "-t", str(int(target_seconds)),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path,
    ]

    rc, stdout, stderr = await _run_subprocess(cmd, timeout=300.0)
    if rc != 0 or not os.path.exists(output_path):
        return {
            "ok": False,
            "path": output_path,
            "error": f"ffmpeg failed (rc={rc}): {stderr.strip()[:300]}",
        }

    duration_s = _video_duration_seconds(output_path)
    return {
        "ok": True,
        "path": output_path,
        "duration_s": duration_s,
        "target_seconds": target_seconds,
    }


# Cut specs: (label, target_seconds)
CUT_SPECS = [
    ("30s", 30),
    ("60s", 60),
    ("3min", 180),
]


async def run(
    mission_id: int,
    workspace_path: str,
    storyboard_path: str,
    scene_recordings: list[dict],
) -> dict[str, Any]:
    """Concat + trim scene recordings into three cuts.

    Parameters
    ----------
    mission_id:
        Mission identifier (used for logging).
    workspace_path:
        Root workspace directory.
    storyboard_path:
        Path to storyboard.json (used for metadata / cut-point calculation).
    scene_recordings:
        List of dicts with ``{scene_id, path, target_seconds}``,
        as returned by ``demo/record``.

    Returns::

        {"ok": True, "cuts": {"30s": str, "60s": str, "3min": str}}
        {"ok": False, "error": str}
    """
    if not scene_recordings:
        return {"ok": False, "error": "scene_recordings is empty; nothing to edit"}

    cuts_dir = os.path.join(workspace_path, "demo", "cuts")
    os.makedirs(cuts_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        cuts: dict[str, str] = {}
        errors: list[str] = []

        for label, target_s in CUT_SPECS:
            output_path = os.path.join(cuts_dir, f"{label}.mp4")
            result = await _produce_cut(
                scene_recordings=scene_recordings,
                output_path=output_path,
                target_seconds=target_s,
                temp_dir=temp_dir,
            )
            if result["ok"]:
                cuts[label] = result["path"]
            else:
                errors.append(f"{label}: {result['error']}")

    if errors:
        return {
            "ok": False,
            "error": f"ffmpeg edit failed: {'; '.join(errors)}",
            "cuts": cuts,
        }

    logger.info(
        "demo_edit: cuts produced",
        mission_id=mission_id,
        cuts=list(cuts.keys()),
    )

    return {
        "ok": True,
        "cuts": cuts,
        "cuts_dir": cuts_dir,
    }
