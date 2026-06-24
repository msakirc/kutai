"""Run a shell-free subprocess inside the mission workspace, with timeout.

Mechanical executor. No LLM. The argv list is passed straight to
``asyncio.create_subprocess_exec`` — no shell, no string splitting, so no
injection. ``cwd`` is resolved under the mission workspace and refused if it
escapes.

Caller decides whether a non-zero exit is fatal via ``require_exit_zero``.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from src.tools.workspace import get_mission_workspace
from yazbunu import get_logger

logger = get_logger("mr_roboto.run_cmd")


_OUTPUT_TAIL_BYTES = 8 * 1024  # keep last 8 KB of each stream


def _resolve_cwd(workspace_root: str, rel_cwd: str | None) -> str | None:
    """Return absolute cwd inside workspace_root, or None on rejection."""
    if not rel_cwd:
        return os.path.realpath(workspace_root)
    if os.path.isabs(rel_cwd):
        return None
    joined = os.path.normpath(os.path.join(workspace_root, rel_cwd))
    root_real = os.path.realpath(workspace_root)
    joined_real = os.path.realpath(joined)
    if not (joined_real == root_real or joined_real.startswith(root_real + os.sep)):
        return None
    return joined_real


def _tail_decode(data: bytes) -> str:
    if len(data) > _OUTPUT_TAIL_BYTES:
        data = data[-_OUTPUT_TAIL_BYTES:]
    return data.decode("utf-8", errors="replace")


async def run_cmd(
    mission_id: int | None,
    cmd: list[str],
    cwd: str | None = None,
    timeout_s: float = 60.0,
    env: dict[str, str] | None = None,
    require_exit_zero: bool = False,
    workspace_path: str | None = None,
    reversibility_intent: str | None = None,
) -> dict[str, Any]:
    """Run ``cmd`` (argv list) under the mission workspace.

    Returns
    -------
    dict
        ``{"exit", "stdout_tail", "stderr_tail", "duration_s", "timed_out", "ok"}``.
        ``ok`` is True iff the process finished and (if ``require_exit_zero``)
        exit was 0.
    """
    if not isinstance(cmd, list) or not cmd or not all(isinstance(a, str) for a in cmd):
        return {
            "exit": -1,
            "stdout_tail": "",
            "stderr_tail": "",
            "duration_s": 0.0,
            "timed_out": False,
            "ok": False,
            "error": "cmd must be a non-empty list of strings",
        }

    if workspace_path is None:
        if mission_id is None:
            return {
                "exit": -1,
                "stdout_tail": "",
                "stderr_tail": "",
                "duration_s": 0.0,
                "timed_out": False,
                "ok": False,
                "error": "no mission_id and no workspace_path",
            }
        workspace_path = get_mission_workspace(mission_id)

    abs_cwd = _resolve_cwd(workspace_path, cwd)
    if abs_cwd is None:
        return {
            "exit": -1,
            "stdout_tail": "",
            "stderr_tail": "",
            "duration_s": 0.0,
            "timed_out": False,
            "ok": False,
            "error": "cwd rejected (absolute or traversal)",
        }

    proc_env = None
    if env is not None:
        proc_env = {**os.environ, **{k: str(v) for k, v in env.items()}}

    # Z10-T1B: record caller intent in the structured log line so the
    # audit trail shows whether this invocation was self-declared as
    # destructive (irreversible) or safe (full).
    logger.info(
        "run_cmd dispatch",
        cmd0=(cmd[0] if cmd else ""),
        argc=len(cmd),
        reversibility_intent=reversibility_intent,
    )

    loop = asyncio.get_event_loop()
    started = loop.time()
    timed_out = False
    proc: asyncio.subprocess.Process | None = None
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=abs_cwd,
            env=proc_env,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_s
            )
        except asyncio.TimeoutError:
            timed_out = True
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            try:
                stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=5.0)
            except asyncio.TimeoutError:
                stdout_b, stderr_b = b"", b""
    except FileNotFoundError as e:
        return {
            "exit": -1,
            "stdout_tail": "",
            "stderr_tail": "",
            "duration_s": loop.time() - started,
            "timed_out": False,
            "ok": False,
            "error": f"executable not found: {e}",
        }
    except OSError as e:
        return {
            "exit": -1,
            "stdout_tail": "",
            "stderr_tail": "",
            "duration_s": loop.time() - started,
            "timed_out": False,
            "ok": False,
            "error": f"spawn failed: {e}",
        }

    exit_code = -1 if timed_out else (proc.returncode if proc else -1)
    duration = loop.time() - started
    ok = (not timed_out) and ((not require_exit_zero) or exit_code == 0)

    return {
        "exit": exit_code,
        "stdout_tail": _tail_decode(stdout_b),
        "stderr_tail": _tail_decode(stderr_b),
        "duration_s": round(duration, 3),
        "timed_out": timed_out,
        "ok": ok,
    }
