"""Z10 T4A — ``verify_demo_artifact`` mechanical verb.

Asserts the mission's ``data/missions/{id}/demo.mp4`` exists, is a real
mp4 (mime check), is at least ``min_bytes`` in size, and runs for at
least ``min_duration_s`` seconds. Used as a sibling gate after
``record_demo`` to fail-fast when Playwright/ffmpeg produced garbage.

Reversibility: ``full`` — pure check.
"""
from __future__ import annotations

import mimetypes
import os
from typing import Any


DEFAULT_MIN_BYTES = 1024 * 1024  # 1 MB
DEFAULT_MIN_DURATION_S = 5.0


def _project_root() -> str:
    here = os.path.abspath(__file__)
    return os.path.abspath(os.path.join(here, "..", "..", "..", "..", ".."))


def _default_path(mission_id: int) -> str:
    return os.path.join(
        _project_root(), "data", "missions", f"{int(mission_id)}", "demo.mp4"
    )


def _ffprobe_duration(path: str) -> float:
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


def run(
    mission_id: int,
    video_path: str | None = None,
    min_bytes: int = DEFAULT_MIN_BYTES,
    min_duration_s: float = DEFAULT_MIN_DURATION_S,
) -> dict[str, Any]:
    """Verify the demo.mp4 artifact. Returns dict with ``ok`` + diagnostics.

    Does NOT raise; caller (mr_roboto.run) maps ``ok=False`` to
    ``Action(status='failed', error=res['reason'])``.
    """
    path = video_path or _default_path(int(mission_id))

    if not os.path.exists(path):
        return {
            "ok": False,
            "path": path,
            "reason": f"demo.mp4 missing at {path}",
        }
    size = os.path.getsize(path)
    if size < int(min_bytes):
        return {
            "ok": False,
            "path": path,
            "size_bytes": size,
            "reason": f"demo.mp4 too small: {size} < {int(min_bytes)} bytes",
        }

    mime, _ = mimetypes.guess_type(path)
    if mime != "video/mp4":
        return {
            "ok": False,
            "path": path,
            "size_bytes": size,
            "mime": mime,
            "reason": f"demo.mp4 mime mismatch: got {mime!r}, want video/mp4",
        }

    duration = _ffprobe_duration(path)
    if duration < float(min_duration_s):
        return {
            "ok": False,
            "path": path,
            "size_bytes": size,
            "mime": mime,
            "duration_s": duration,
            "reason": (
                f"demo.mp4 too short: {duration:.2f}s < {float(min_duration_s):.2f}s"
            ),
        }

    return {
        "ok": True,
        "path": path,
        "size_bytes": size,
        "mime": mime,
        "duration_s": duration,
    }
