"""Z1 Tier 4 (C10+A19) — emit a tunneled preview URL per mission. EMIT-ONLY.

Z1 surface only: writes ``preview_url.txt`` into the mission workspace.
Z2 owns the actual hosting / browser viewer. When the operator opts into
``KUTAI_PREVIEW_PROVIDER=cloudflared`` AND a ``cloudflared`` binary is on
PATH, this module spawns a tunnel subprocess against
``mission_{mission_id}/.prototype/`` and persists the captured URL +
subprocess PID. Otherwise we fail soft with a ``pending:`` placeholder so
the mission still works without preview hosting.

Idempotent: any prior tunnel (PID in ``.tunnel.pid``) is terminated before
re-emitting.
"""
from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
import sys
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.emit_preview_url")

# trycloudflare URLs look like: https://<random-words>.trycloudflare.com
_URL_RE = re.compile(rb"https://[a-z0-9-]+\.trycloudflare\.com", re.IGNORECASE)
# Generic fallback for other providers that print a https:// URL on stdout.
_GENERIC_URL_RE = re.compile(rb"https://[^\s]+", re.IGNORECASE)


def _resolve_workspace(mission_id: int, workspace_path: str | None) -> str:
    if workspace_path:
        return workspace_path
    from src.tools.workspace import get_mission_workspace
    return get_mission_workspace(int(mission_id))


def _kill_prior_tunnel(mission_id: int, workspace_path: str) -> dict[str, Any]:
    """If a stale pidfile exists, try to terminate the prior tunnel."""
    from mr_roboto.kill_preview_url import kill_preview_url
    # Sync wrapper around the async kill — we are already inside an
    # async function. Caller must await this.
    return {"ok": True, "skipped": False, "_async": kill_preview_url(
        mission_id=mission_id, workspace_path=workspace_path, _silent=True,
    )}


async def _await_kill_prior(mission_id: int, workspace_path: str) -> dict[str, Any]:
    pid_file = os.path.join(workspace_path, ".tunnel.pid")
    url_file = os.path.join(workspace_path, "preview_url.txt")
    if not os.path.exists(pid_file) and not os.path.exists(url_file):
        return {"ok": True, "skipped": True}
    # Tests patch `_kill_prior_tunnel` to a sync stub returning a dict;
    # honour that contract first.
    stub = _kill_prior_tunnel(mission_id, workspace_path)
    inner = stub.get("_async") if isinstance(stub, dict) else None
    if asyncio.iscoroutine(inner):
        try:
            await inner
        except Exception as e:  # pragma: no cover — diagnostic only
            logger.warning(f"prior tunnel kill failed: {e}")
    return {"ok": True}


def _spawn_cloudflared(prototype_dir: str) -> subprocess.Popen:
    """Spawn cloudflared tunnel against the prototype dir. Windows-aware."""
    cmd = [
        "cloudflared", "tunnel", "--url",
        f"file://{os.path.abspath(prototype_dir)}",
    ]
    kwargs: dict[str, Any] = dict(
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )
    if sys.platform == "win32":
        # CREATE_NEW_PROCESS_GROUP lets us send CTRL_BREAK_EVENT later.
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    return subprocess.Popen(cmd, **kwargs)


def _read_url_from_proc(proc: subprocess.Popen, max_lines: int = 200) -> str | None:
    """Drain stdout until a recognizable https URL appears or stream ends."""
    if proc.stdout is None:
        return None
    for _ in range(max_lines):
        line = proc.stdout.readline()
        if not line:
            return None
        m = _URL_RE.search(line) or _GENERIC_URL_RE.search(line)
        if m:
            return m.group(0).decode("utf-8", errors="replace").strip()
    return None


async def emit_preview_url(
    mission_id: int,
    workspace_path: str | None = None,
) -> dict[str, Any]:
    """Emit a preview URL surface for ``mission_{mission_id}``.

    Returns a dict with at least: ``ok``, ``pending``, ``provider``, ``url``,
    ``pid``, and ``path``.
    """
    workspace_path = _resolve_workspace(mission_id, workspace_path)
    os.makedirs(workspace_path, exist_ok=True)

    # Idempotency: kill any prior tunnel before re-emitting.
    await _await_kill_prior(mission_id, workspace_path)

    prototype_dir = os.path.join(workspace_path, ".prototype")
    url_file = os.path.join(workspace_path, "preview_url.txt")

    provider = os.environ.get("KUTAI_PREVIEW_PROVIDER", "").strip().lower() or None

    # Fail-soft: env unset OR binary missing → write a pending placeholder.
    if provider != "cloudflared":
        body = (
            f"pending: hosting deferred to Z2 (path: {prototype_dir})\n"
        )
        with open(url_file, "w", encoding="utf-8") as f:
            f.write(body)
        logger.info(
            f"preview_url pending (provider unset) for mission {mission_id} "
            f"at {url_file}"
        )
        return {
            "ok": True,
            "pending": True,
            "provider": provider,
            "url": None,
            "pid": None,
            "path": url_file,
        }

    binary = shutil.which("cloudflared")
    if not binary:
        body = (
            f"pending: cloudflared binary missing on PATH (path: {prototype_dir})\n"
        )
        with open(url_file, "w", encoding="utf-8") as f:
            f.write(body)
        logger.warning(
            f"preview_url pending (cloudflared not on PATH) for mission "
            f"{mission_id}"
        )
        return {
            "ok": True,
            "pending": True,
            "provider": provider,
            "url": None,
            "pid": None,
            "path": url_file,
        }

    # Spawn the tunnel.
    try:
        proc = _spawn_cloudflared(prototype_dir)
    except Exception as e:
        logger.warning(f"failed to spawn cloudflared: {e}")
        body = f"pending: spawn failed ({e}) (path: {prototype_dir})\n"
        with open(url_file, "w", encoding="utf-8") as f:
            f.write(body)
        return {
            "ok": True,
            "pending": True,
            "provider": provider,
            "url": None,
            "pid": None,
            "path": url_file,
            "error": str(e),
        }

    url = _read_url_from_proc(proc)
    pid = getattr(proc, "pid", None)

    if not url:
        # Couldn't capture a URL — surface pending and let a follow-up
        # re-emit pick it up. Don't kill the proc; Z2 may still surface it.
        body = (
            f"pending: cloudflared started but no URL captured "
            f"(pid={pid}, path: {prototype_dir})\n"
        )
        with open(url_file, "w", encoding="utf-8") as f:
            f.write(body)
        if pid:
            with open(os.path.join(workspace_path, ".tunnel.pid"), "w", encoding="utf-8") as f:
                f.write(f"{pid}\n")
        return {
            "ok": True,
            "pending": True,
            "provider": provider,
            "url": None,
            "pid": pid,
            "path": url_file,
        }

    with open(url_file, "w", encoding="utf-8") as f:
        f.write(f"{url}\n")
    if pid:
        with open(os.path.join(workspace_path, ".tunnel.pid"), "w", encoding="utf-8") as f:
            f.write(f"{pid}\n")

    logger.info(
        f"preview_url emitted for mission {mission_id}: {url} (pid={pid})"
    )
    # Persist to DB (best-effort; not all callers run inside an event loop
    # with DB access). Z1 keeps this side-effect optional.
    try:
        await _persist_to_db(mission_id, action="emit", url=url, exit_code=None)
    except Exception as e:  # pragma: no cover — best-effort
        logger.debug(f"preview_log DB persist skipped: {e}")

    return {
        "ok": True,
        "pending": False,
        "provider": provider,
        "url": url,
        "pid": pid,
        "path": url_file,
    }


async def _persist_to_db(
    mission_id: int,
    action: str,
    url: str | None,
    exit_code: int | None,
) -> None:
    from src.infra.db import get_db
    db = await get_db()
    await db.execute(
        "INSERT INTO preview_log (mission_id, action, url, exit_code) "
        "VALUES (?, ?, ?, ?)",
        (int(mission_id), action, url, exit_code),
    )
    if action == "emit" and url:
        await db.execute(
            "UPDATE missions SET preview_url = ?, "
            "preview_started_at = CURRENT_TIMESTAMP WHERE id = ?",
            (url, int(mission_id)),
        )
    elif action == "kill":
        await db.execute(
            "UPDATE missions SET preview_url = NULL, "
            "preview_started_at = NULL WHERE id = ?",
            (int(mission_id),),
        )
    await db.commit()
