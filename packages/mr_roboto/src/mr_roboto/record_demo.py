"""Z10 T4A — ``record_demo`` mechanical verb.

Records an end-of-mission demo video by running a Playwright spec inside
the mission's per-mission Docker container (T3B), then trimming the
resulting `.webm` to `<= max_seconds` with ffmpeg, transcoding to mp4 at
``data/missions/{mission_id}/demo.mp4``.

Trigger contract
----------------
Wired as a mechanical step at the end of phase 15 in `i2p_v3.json`. The
step's ``skip_when: no_e2e_specs`` clause lets missions without e2e
coverage skip silently; the strict path posts a ``[blocker]`` mission
event (D4) when ``missions.demo_required = 1`` (default).

Reversibility: ``full`` — pure additive artifact write.

NOTE: This verb covers web/Playwright demos only. A Maestro/Detox mobile
recorder is owned by the Z5 mobile track — drop a TODO referring to it.

TODO(Z5 mobile track): add a sibling ``record_demo_mobile`` verb that
shells into the per-mission container and drives Maestro/Detox to capture
a mobile e2e demo. Should share the ffmpeg trim/transcode tail with this
verb.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import shlex
import time
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.record_demo")


DEFAULT_MAX_SECONDS = 90


def _project_root() -> str:
    here = os.path.abspath(__file__)
    # packages/mr_roboto/src/mr_roboto/record_demo.py → up 4 to repo root
    return os.path.abspath(os.path.join(here, "..", "..", "..", "..", ".."))


def _demo_dir(mission_id: int) -> str:
    return os.path.join(
        _project_root(), "data", "missions", f"{int(mission_id)}"
    )


def _container_name(mission_id: int) -> str:
    return f"kutai-mission-{int(mission_id)}"


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
    return proc.returncode or 0, (out or b"").decode("utf-8", "replace"), (err or b"").decode("utf-8", "replace")


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_newest_webm(workspace_root: str) -> str | None:
    """Walk workspace test-results dir; return newest `.webm` path or None."""
    test_results = os.path.join(workspace_root, "test-results")
    if not os.path.isdir(test_results):
        return None
    newest_path: str | None = None
    newest_mtime: float = -1.0
    for root, _, files in os.walk(test_results):
        for fname in files:
            if fname.endswith(".webm"):
                p = os.path.join(root, fname)
                try:
                    mtime = os.path.getmtime(p)
                except OSError:
                    continue
                if mtime > newest_mtime:
                    newest_mtime = mtime
                    newest_path = p
    return newest_path


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


async def _e2e_specs_present(workspace_root: str) -> bool:
    """True when ``tests/e2e/*.spec.ts`` glob finds anything in the workspace."""
    import glob
    if not os.path.isdir(workspace_root):
        return False
    matches = (
        glob.glob(os.path.join(workspace_root, "tests", "e2e", "*.spec.ts"))
        + glob.glob(os.path.join(workspace_root, "tests", "e2e", "*.spec.js"))
    )
    return len(matches) > 0


async def _mission_demo_required(mission_id: int) -> bool:
    """Read ``missions.demo_required`` for the mission (default True)."""
    try:
        from src.infra.db import get_db
        db = await get_db()
        cur = await db.execute(
            "SELECT demo_required FROM missions WHERE id = ?", (int(mission_id),),
        )
        row = await cur.fetchone()
        if row is None:
            return True
        val = row[0]
        if val is None:
            return True
        return bool(int(val))
    except Exception:
        # Column might not exist in legacy DBs — strict default.
        return True


async def _post_no_e2e_blocker(mission_id: int) -> None:
    """Best-effort [blocker] mission event when missions lack e2e specs."""
    try:
        from src.app.mission_events import post_event
        from src.app.telegram_bot import get_bot
        bot = await get_bot()
        await post_event(
            bot, int(mission_id), "blocker",
            {
                "reason": (
                    "Mission completed without e2e demo; founder action "
                    "required to add tests/e2e/*.spec.ts for next mission."
                ),
                "source": "record_demo",
                "mission_id": int(mission_id),
            },
        )
    except Exception as e:
        logger.warning(
            "no_e2e_blocker post failed",
            mission_id=mission_id,
            error=str(e),
        )


async def run(
    mission_id: int,
    scenario_path: str,
    max_seconds: int = DEFAULT_MAX_SECONDS,
    workspace_root: str | None = None,
) -> dict[str, Any]:
    """Run Playwright in mission container, trim to mp4 demo. Returns result dict.

    Raises RuntimeError with a clear message on each step failure — caller
    (mr_roboto.run) maps that to ``Action(status='failed', error=...)``.

    D4 — when no e2e specs are present in the workspace:
      - ``demo_required=1`` (default): post a ``[blocker]`` mission event
        and return ``{"skipped": True, "reason": "no_e2e_specs", ...}``
        so the workflow records the gap without crashing the mission.
      - ``demo_required=0``: skip silently (no blocker).
    """
    if mission_id is None:
        raise RuntimeError("record_demo requires mission_id (per-mission container)")
    if not scenario_path or os.path.isabs(scenario_path):
        raise RuntimeError(
            f"record_demo requires a workspace-relative scenario_path; got {scenario_path!r}"
        )

    if workspace_root is None:
        from src.tools.workspace import get_mission_workspace
        workspace_root = get_mission_workspace(int(mission_id))

    # D4 — gracefully handle missions without e2e specs.
    if not await _e2e_specs_present(workspace_root):
        strict = await _mission_demo_required(int(mission_id))
        if strict:
            await _post_no_e2e_blocker(int(mission_id))
            return {
                "skipped": True,
                "reason": "no_e2e_specs",
                "demo_required": True,
                "blocker_posted": True,
                "workspace_root": workspace_root,
            }
        return {
            "skipped": True,
            "reason": "no_e2e_specs",
            "demo_required": False,
            "blocker_posted": False,
            "workspace_root": workspace_root,
        }

    container = _container_name(int(mission_id))

    # 1. Verify container exists + running.
    rc, stdout, stderr = await _run_subprocess(
        ["docker", "inspect", "-f", "{{.State.Running}}", container],
        timeout=15,
    )
    if rc != 0 or "true" not in stdout.lower():
        raise RuntimeError(
            f"mission container {container} not running: rc={rc} {stderr.strip()}"
        )

    # 2. Run Playwright spec inside the container.
    inner_cmd = (
        f"cd /workspace && npx playwright test {shlex.quote(scenario_path)} "
        f"--reporter=line --output=/workspace/test-results"
    )
    rc, stdout, stderr = await _run_subprocess(
        ["docker", "exec", container, "bash", "-c", inner_cmd],
        timeout=600.0,
    )
    if rc != 0:
        # Be specific when Playwright/npm is missing — operators care.
        combined = (stdout + "\n" + stderr).lower()
        hint = ""
        if "playwright" in combined and ("not found" in combined or "no such" in combined):
            hint = " (playwright not installed in container — check package.json devDependencies)"
        elif "npx: not found" in combined or "command not found" in combined:
            hint = " (npx/node not available in container image)"
        raise RuntimeError(
            f"playwright run failed (rc={rc}){hint}: {stderr.strip()[:500]}"
        )

    # 3. Find the newest .webm in workspace test-results.
    webm = _find_newest_webm(workspace_root)
    if webm is None:
        raise RuntimeError(
            f"no .webm produced under {workspace_root}/test-results "
            "(playwright config must enable video recording)"
        )

    # 4. Trim/transcode with ffmpeg → mp4.
    dst_dir = _demo_dir(int(mission_id))
    os.makedirs(dst_dir, exist_ok=True)
    out_mp4 = os.path.join(dst_dir, "demo.mp4")
    # `-t` BEFORE `-i` would be a seek; place after to trim duration.
    # Use libx264 + faststart for broad compatibility (Telegram + browsers).
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-i", webm,
        "-t", str(int(max_seconds)),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        out_mp4,
    ]
    rc, stdout, stderr = await _run_subprocess(ffmpeg_cmd, timeout=300.0)
    if rc != 0 or not os.path.exists(out_mp4):
        raise RuntimeError(
            f"ffmpeg trim failed (rc={rc}): {stderr.strip()[:500]}"
        )

    duration_s = _video_duration_seconds(out_mp4)
    sha = _sha256_file(out_mp4)
    size_bytes = os.path.getsize(out_mp4)

    return {
        "video_path": out_mp4,
        "duration_s": duration_s,
        "sha256": sha,
        "size_bytes": size_bytes,
        "source_webm": webm,
        "scenario_path": scenario_path,
        "max_seconds": int(max_seconds),
        "captured_at": time.time(),
    }
