"""Z1 Tier 4 (C10+A19) — emit a tunneled preview URL per mission. EMIT-ONLY.

Z1 surface only: writes ``preview_url.txt`` into the mission workspace.
Z2 owns the actual hosting / browser viewer. When the operator opts into
``KUTAI_PREVIEW_PROVIDER=cloudflared`` AND a ``cloudflared`` binary is on
PATH, this module spawns a local static HTTP server over the resolved preview
root, then tunnels it via cloudflared. Persists the captured URL + subprocess
PIDs. Otherwise we fail soft with a ``pending:`` placeholder so the mission
still works without preview hosting.

Preview root resolution (fixes wrong-dir defect):
- ``.prototype/index.html`` exists → serve ``.prototype/`` (mobile/Expo bundle)
- else ``.web/`` is a non-empty directory → serve ``.web/`` (web HTML prototypes)
- else None → pending placeholder

Cloudflared fix: cloudflared receives ``http://127.0.0.1:<port>`` (an HTTP
origin), not ``file://<dir>`` which cloudflared cannot proxy.

Idempotent: any prior tunnel + http server (PIDs in ``.tunnel.pid`` /
``.httpserver.pid``) are terminated before re-emitting.
"""
from __future__ import annotations

import asyncio
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.emit_preview_url")

# trycloudflare URLs look like: https://<random-words>.trycloudflare.com
_URL_RE = re.compile(rb"https://[a-z0-9-]+\.trycloudflare\.com", re.IGNORECASE)
# Generic fallback for other providers that print a https:// URL on stdout.
_GENERIC_URL_RE = re.compile(rb"https://[^\s]+", re.IGNORECASE)


def _resolve_preview_root(workspace_path: str) -> str | None:
    """Return the directory to serve, or None if nothing is ready.

    Priority:
    1. ``<ws>/.prototype/index.html`` exists → return ``<ws>/.prototype``
    2. ``<ws>/.web`` is a non-empty directory → return ``<ws>/.web``
    3. None

    Plan 3 note: ``<ws>/.web/assets/`` (image-gen output from
    swap_placeholder_images) is served automatically — the static server
    serves the resolved root recursively and ``.web/`` is that root, so the
    rewritten ``<img src="assets/<id>.png">`` references resolve with no
    resolver change required (see test_emit_preview_url_assets.py).
    """
    proto = os.path.join(workspace_path, ".prototype")
    if os.path.isfile(os.path.join(proto, "index.html")):
        return proto
    web = os.path.join(workspace_path, ".web")
    if os.path.isdir(web):
        try:
            if os.listdir(web):
                return web
        except OSError:
            pass
    return None


def _pick_free_port() -> int:
    """Bind a socket to an ephemeral port, read it, close, return the port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_port_ready(port: int, timeout_s: float = 3.0) -> bool:
    """Poll until ``127.0.0.1:<port>`` accepts a TCP connection or timeout.

    Returns True if port became ready, False on timeout. Best-effort — the
    caller proceeds regardless, as cloudflared tolerates a slow origin.
    Made module-level so tests can monkeypatch it.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def _spawn_static_server(root: str, port: int) -> subprocess.Popen:
    """Spawn ``python -m http.server`` over *root* bound to localhost.

    Windows: CREATE_NEW_PROCESS_GROUP so we can send CTRL_BREAK_EVENT later.
    """
    cmd = [
        sys.executable, "-m", "http.server", str(port),
        "--bind", "127.0.0.1",
        "--directory", root,
    ]
    kwargs: dict[str, Any] = dict(
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    return subprocess.Popen(cmd, **kwargs)


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


def _spawn_cloudflared(port: int) -> subprocess.Popen:
    """Spawn cloudflared tunnel against a local HTTP origin. Windows-aware."""
    cmd = [
        "cloudflared", "tunnel", "--url",
        f"http://127.0.0.1:{port}",
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


def _write_pending(url_file: str, body: str) -> None:
    with open(url_file, "w", encoding="utf-8") as f:
        f.write(body)


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

    # Idempotency: kill any prior tunnel + http server before re-emitting.
    await _await_kill_prior(mission_id, workspace_path)

    url_file = os.path.join(workspace_path, "preview_url.txt")

    # Step 2: Resolve preview root.
    root = _resolve_preview_root(workspace_path)
    if root is None:
        body = (
            f"pending: no preview root (.prototype/.web) found "
            f"(workspace: {workspace_path})\n"
        )
        _write_pending(url_file, body)
        logger.info(
            f"preview_url pending (no root) for mission {mission_id}"
        )
        return {
            "ok": True,
            "pending": True,
            "provider": None,
            "url": None,
            "pid": None,
            "path": url_file,
        }

    # Step 3: Check provider.
    provider = os.environ.get("KUTAI_PREVIEW_PROVIDER", "").strip().lower() or None

    # Fail-soft: env unset OR binary missing → write a pending placeholder.
    if provider != "cloudflared":
        body = (
            f"pending: hosting deferred to Z2 (path: {root})\n"
        )
        _write_pending(url_file, body)
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

    # Step 4: Check binary.
    binary = shutil.which("cloudflared")
    if not binary:
        body = (
            f"pending: cloudflared binary missing on PATH (path: {root})\n"
        )
        _write_pending(url_file, body)
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

    # Step 5: Pick a free port.
    port = _pick_free_port()

    # Step 6: Spawn the local static HTTP server.
    try:
        server_proc = _spawn_static_server(root, port)
    except Exception as e:
        logger.warning(f"failed to spawn static server: {e}")
        body = f"pending: spawn failed ({e}) (path: {root})\n"
        _write_pending(url_file, body)
        return {
            "ok": True,
            "pending": True,
            "provider": provider,
            "url": None,
            "pid": None,
            "path": url_file,
            "error": str(e),
        }

    server_pid = getattr(server_proc, "pid", None)
    if server_pid:
        with open(os.path.join(workspace_path, ".httpserver.pid"), "w", encoding="utf-8") as f:
            f.write(f"{server_pid}\n")

    # Step 7: Wait for static server readiness (best-effort).
    _wait_port_ready(port)

    # Step 8: Spawn cloudflared tunnel against the local HTTP origin.
    try:
        proc = _spawn_cloudflared(port)
    except Exception as e:
        # Orphan safety: terminate the http server we already started.
        if server_pid:
            try:
                from mr_roboto.kill_preview_url import _terminate
                _terminate(server_pid)
            except Exception:
                pass
            _remove_pidfile(os.path.join(workspace_path, ".httpserver.pid"))
        logger.warning(f"failed to spawn cloudflared: {e}")
        body = f"pending: spawn failed ({e}) (path: {root})\n"
        _write_pending(url_file, body)
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
        # Step 9 (orphan safety): couldn't capture a URL — tear down the
        # static server so no orphan lingers.
        if server_pid:
            try:
                from mr_roboto.kill_preview_url import _terminate
                _terminate(server_pid)
            except Exception:
                pass
            _remove_pidfile(os.path.join(workspace_path, ".httpserver.pid"))

        body = (
            f"pending: cloudflared started but no URL captured "
            f"(pid={pid}, path: {root})\n"
        )
        _write_pending(url_file, body)
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

    # Success: persist URL + both pids.
    with open(url_file, "w", encoding="utf-8") as f:
        f.write(f"{url}\n")
    if pid:
        with open(os.path.join(workspace_path, ".tunnel.pid"), "w", encoding="utf-8") as f:
            f.write(f"{pid}\n")

    # Z3 T3B follow-up: write a consumer-friendly artifact that downstream
    # post-hooks (e.g. accessibility_review) can read to discover the URL.
    # Written only when a real https:// URL is captured.
    _preview_dir = os.path.join(workspace_path, ".preview")
    os.makedirs(_preview_dir, exist_ok=True)
    _last_url_file = os.path.join(_preview_dir, "last_preview_url.txt")
    with open(_last_url_file, "w", encoding="utf-8") as f:
        f.write(f"{url}\n")

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


def _remove_pidfile(path: str) -> None:
    """Remove a pidfile, ignoring errors."""
    try:
        os.remove(path)
    except OSError:
        pass


async def _persist_to_db(
    mission_id: int,
    action: str,
    url: str | None,
    exit_code: int | None,
) -> None:
    from dabidabi import get_db
    db = await get_db()
    await db.execute(
        "INSERT INTO preview_log (mission_id, action, url, exit_code) "
        "VALUES (?, ?, ?, ?)",
        (int(mission_id), action, url, exit_code),
    )
    await db.commit()
    from general_beckman import update_mission_fields as _umf
    if action == "emit" and url:
        from dabidabi.times import db_now as _db_now
        await _umf(int(mission_id), preview_url=url, preview_started_at=_db_now())
    elif action == "kill":
        await _umf(int(mission_id), preview_url=None, preview_started_at=None)
