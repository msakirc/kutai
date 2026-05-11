"""Z10 T4A — ``verify_demo_artifact`` mechanical verb.

Asserts the mission's demo artifact exists, has acceptable mime, is at
least ``min_bytes`` in size, and (for video) runs for at least
``min_duration_s`` seconds. Used as a sibling gate after ``record_demo``
to fail-fast when Playwright/ffmpeg produced garbage.

Accepted artifact types (z10-wire-fixes F3 — additive):
  - ``demo.mp4``  (video/mp4)             — primary, web/Playwright demos
  - ``demo.gif``  (image/gif)             — animated screencast fallback
  - ``demo.cast`` (asciinema, text/plain) — CLI / TUI mission demos

Reversibility: ``full`` — pure check.
"""
from __future__ import annotations

import mimetypes
import os
from typing import Any


DEFAULT_MIN_BYTES = 1024 * 1024  # 1 MB
DEFAULT_MIN_DURATION_S = 5.0

# F3 — non-video artifacts skip the duration check entirely. .cast files
# are usually well under the 1 MB threshold; size floor is dropped to 1 KB
# for them so a real recorded asciinema session passes the gate.
_CLI_MIN_BYTES = 1024  # asciinema typically <100 KB
_NON_VIDEO_EXTS = {".gif", ".cast"}


def _project_root() -> str:
    here = os.path.abspath(__file__)
    return os.path.abspath(os.path.join(here, "..", "..", "..", "..", ".."))


def _mission_dir(mission_id: int) -> str:
    return os.path.join(
        _project_root(), "data", "missions", f"{int(mission_id)}"
    )


def _default_path(mission_id: int) -> str:
    """Pick the first existing demo artifact, in preference order."""
    md = _mission_dir(int(mission_id))
    for name in ("demo.mp4", "demo.gif", "demo.cast"):
        cand = os.path.join(md, name)
        if os.path.exists(cand):
            return cand
    # Fall through to mp4 (the original default) so the "missing" reason
    # mentions the canonical web/Playwright artifact.
    return os.path.join(md, "demo.mp4")


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


def _classify_artifact(path: str) -> tuple[str, str | None]:
    """Return (kind, mime) where kind ∈ {video, gif, cast, unknown}."""
    lower = path.lower()
    mime, _ = mimetypes.guess_type(path)
    if lower.endswith(".mp4"):
        return ("video", mime or "video/mp4")
    if lower.endswith(".gif"):
        return ("gif", mime or "image/gif")
    if lower.endswith(".cast"):
        # asciinema cast files are JSON Lines under the hood.
        return ("cast", mime or "application/json")
    return ("unknown", mime)


def run(
    mission_id: int,
    video_path: str | None = None,
    min_bytes: int = DEFAULT_MIN_BYTES,
    min_duration_s: float = DEFAULT_MIN_DURATION_S,
) -> dict[str, Any]:
    """Verify a mission demo artifact (.mp4 / .gif / .cast).

    Does NOT raise; caller (mr_roboto.run) maps ``ok=False`` to
    ``Action(status='failed', error=res['reason'])``.
    """
    path = video_path or _default_path(int(mission_id))

    if not os.path.exists(path):
        return {
            "ok": False,
            "path": path,
            "reason": f"demo artifact missing at {path}",
        }

    kind, mime = _classify_artifact(path)
    # Per-kind effective minimum size — .cast files are tiny by nature.
    effective_min = int(min_bytes)
    if kind in _NON_VIDEO_EXTS - {".gif"} | {"cast"}:
        # Apply CLI minimum only when caller stuck with the default video
        # floor; respect explicit overrides.
        if int(min_bytes) == DEFAULT_MIN_BYTES:
            effective_min = _CLI_MIN_BYTES

    size = os.path.getsize(path)
    if size < effective_min:
        return {
            "ok": False,
            "path": path,
            "size_bytes": size,
            "reason": (
                f"demo artifact too small: {size} < {effective_min} bytes "
                f"(kind={kind})"
            ),
        }

    if kind == "video":
        if mime != "video/mp4":
            return {
                "ok": False,
                "path": path,
                "size_bytes": size,
                "mime": mime,
                "reason": (
                    f"demo.mp4 mime mismatch: got {mime!r}, want video/mp4"
                ),
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
                    f"demo.mp4 too short: {duration:.2f}s < "
                    f"{float(min_duration_s):.2f}s"
                ),
            }
        return {
            "ok": True,
            "kind": kind,
            "path": path,
            "size_bytes": size,
            "mime": mime,
            "duration_s": duration,
        }

    if kind in ("gif", "cast"):
        # No duration check (gif/cast tooling varies); size is the signal.
        return {
            "ok": True,
            "kind": kind,
            "path": path,
            "size_bytes": size,
            "mime": mime,
        }

    return {
        "ok": False,
        "path": path,
        "size_bytes": size,
        "mime": mime,
        "reason": (
            f"demo artifact unrecognised kind for {path}; expected .mp4, "
            ".gif, or .cast"
        ),
    }
