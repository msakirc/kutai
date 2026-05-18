"""yalayut.mcp_manager — MCP stdio server process lifecycle.

Owns the on-demand start, JSON-RPC-over-stdio transport, health probing, idle
shutdown, and consecutive-failure handling for ``artifact_type='mcp'``.

KutAI rule ``no_auto_connect``: no MCP server starts at boot. ``ensure_running``
is called only by the mcp plugin when intersect has matched an mcp artifact for
a task. v1 supports the ``stdio`` transport only (``sse`` / ``streamable_http``
are Phase 4 — see plan Task 14).

Process state is mirrored to ``yalayut_mcp_processes`` so ``/yalayut mcp status``
can report across restarts; the in-memory ``_procs`` map holds the live
stdin/stdout handles.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from src.infra.logging_config import get_logger
from yalayut.shell_safety import ShellSafetyError, tokenize_cmd

logger = get_logger("yalayut.mcp")

_PROBE_TIMEOUT_S = 5.0
_CALL_TIMEOUT_S = 30.0
_REPROBE_INTERVAL_S = 60.0
_MAX_FAILS_RESTART = 3
_MAX_FAILS_DISABLE = 5


class McpManager:
    """Manages live MCP stdio subprocesses keyed by artifact id."""

    def __init__(self) -> None:
        # artifact_id -> {proc, stdin, stdout, health, pid, last_used,
        #                 last_probe, fails, idle_timeout, lock, next_id}
        self._procs: dict[int, dict[str, Any]] = {}

    # ----- DB mirror (patched in tests) ------------------------------------

    async def _persist_process(self, artifact_id: int, **fields: Any) -> None:
        """Upsert one yalayut_mcp_processes row."""
        from datetime import datetime

        from src.infra.db import get_db

        db = await get_db()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cols = {
            "pid": fields.get("pid"),
            "port": fields.get("port"),
            "started_at": fields.get("started_at", now),
            "last_used_at": now,
            "idle_timeout_s": fields.get("idle_timeout_s", 300),
            "health": fields.get("health", "starting"),
            "last_probe_at": now,
            "consecutive_probe_fails": fields.get("consecutive_probe_fails", 0),
        }
        try:
            await db.execute(
                "INSERT INTO yalayut_mcp_processes "
                "(artifact_id, pid, port, started_at, last_used_at, idle_timeout_s, "
                " health, last_probe_at, consecutive_probe_fails) "
                "VALUES (:aid, :pid, :port, :started, :used, :idle, :health, "
                ":probe, :fails) "
                "ON CONFLICT(artifact_id) DO UPDATE SET "
                "pid=excluded.pid, port=excluded.port, last_used_at=excluded.last_used_at, "
                "health=excluded.health, last_probe_at=excluded.last_probe_at, "
                "consecutive_probe_fails=excluded.consecutive_probe_fails",
                {"aid": artifact_id, "pid": cols["pid"], "port": cols["port"],
                 "started": cols["started_at"], "used": cols["last_used_at"],
                 "idle": cols["idle_timeout_s"], "health": cols["health"],
                 "probe": cols["last_probe_at"], "fails": cols["consecutive_probe_fails"]},
            )
            await db.commit()
        except Exception as e:
            logger.debug("mcp process persist skipped (schema not ready?)",
                         artifact_id=artifact_id, err=str(e))

    async def _load_process(self, artifact_id: int) -> dict[str, Any] | None:
        from src.infra.db import get_db

        db = await get_db()
        try:
            cur = await db.execute(
                "SELECT pid, port, health, last_probe_at, consecutive_probe_fails, "
                "idle_timeout_s FROM yalayut_mcp_processes WHERE artifact_id = ?",
                (artifact_id,),
            )
            row = await cur.fetchone()
        except Exception as e:
            logger.debug("mcp process load skipped (schema not ready?)",
                         artifact_id=artifact_id, err=str(e))
            return None
        if row is None:
            return None
        return {"pid": row[0], "port": row[1], "health": row[2],
                "last_probe_at": row[3], "consecutive_probe_fails": row[4],
                "idle_timeout_s": row[5]}

    # ----- lifecycle -------------------------------------------------------

    def is_running(self, artifact_id: int) -> bool:
        entry = self._procs.get(artifact_id)
        if not entry:
            return False
        proc = entry.get("proc")
        return proc is not None and proc.returncode is None

    async def ensure_running(
        self, artifact_id: int, mcp: dict[str, Any]
    ) -> dict[str, Any]:
        """Start (or reuse) the MCP server for ``artifact_id``.

        ``mcp`` is the manifest's ``mcp:`` block. Returns a handle dict
        ``{"pid", "health", "artifact_id"}``. On health-probe failure the
        process is killed and ``health='unhealthy'`` returned.
        """
        if self.is_running(artifact_id):
            entry = self._procs[artifact_id]
            entry["last_used"] = time.time()
            return {"artifact_id": artifact_id, "pid": entry["pid"],
                    "health": entry["health"]}

        transport = (mcp.get("transport") or "stdio").lower()
        if transport != "stdio":
            logger.warning("non-stdio MCP transport unsupported in v1",
                            artifact_id=artifact_id, transport=transport)
            return {"artifact_id": artifact_id, "pid": None,
                    "health": "unhealthy"}

        run_cmd = mcp.get("run_cmd")
        if not run_cmd:
            return {"artifact_id": artifact_id, "pid": None, "health": "unhealthy"}
        try:
            # Normalise Windows backslashes to forward-slashes before POSIX
            # shlex tokenisation — forward slashes are valid on Windows and
            # shlex.split(posix=True) would otherwise eat bare backslashes.
            import os as _os
            if _os.sep == "\\":
                run_cmd = run_cmd.replace("\\", "/")
            argv = tokenize_cmd(run_cmd)
        except ShellSafetyError as e:
            logger.warning("MCP run_cmd untokenizable", err=str(e))
            return {"artifact_id": artifact_id, "pid": None, "health": "unhealthy"}

        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except (FileNotFoundError, OSError) as e:
            logger.warning("MCP spawn failed", artifact_id=artifact_id, err=str(e))
            return {"artifact_id": artifact_id, "pid": None, "health": "unhealthy"}

        entry = {
            "proc": proc,
            "stdin": proc.stdin,
            "stdout": proc.stdout,
            "health": "starting",
            "pid": proc.pid,
            "last_used": time.time(),
            "last_probe": time.time(),
            "fails": 0,
            "idle_timeout": float(mcp.get("idle_timeout_s", 300)),
            "lock": asyncio.Lock(),
            "next_id": 1,
            "health_check": mcp.get("health_check", "list_tools"),
        }
        self._procs[artifact_id] = entry
        await self._persist_process(artifact_id, pid=proc.pid, health="starting",
                                    idle_timeout_s=entry["idle_timeout"])

        healthy = await self._probe(artifact_id)
        entry["health"] = "ready" if healthy else "unhealthy"
        await self._persist_process(artifact_id, pid=proc.pid,
                                    health=entry["health"])
        if not healthy:
            await self.shutdown(artifact_id)
            return {"artifact_id": artifact_id, "pid": None, "health": "unhealthy"}
        return {"artifact_id": artifact_id, "pid": proc.pid, "health": "ready"}

    async def _rpc(
        self, artifact_id: int, method: str, params: dict | None, timeout: float
    ) -> dict[str, Any]:
        """Send one JSON-RPC request, read one response line."""
        entry = self._procs.get(artifact_id)
        if not entry or entry["proc"].returncode is not None:
            raise RuntimeError(f"MCP {artifact_id} not running")
        async with entry["lock"]:
            req_id = entry["next_id"]
            entry["next_id"] += 1
            req = {"jsonrpc": "2.0", "id": req_id, "method": method,
                   "params": params or {}}
            entry["stdin"].write((json.dumps(req) + "\n").encode("utf-8"))
            await entry["stdin"].drain()
            line = await asyncio.wait_for(
                entry["stdout"].readline(), timeout=timeout
            )
        if not line:
            raise RuntimeError("MCP closed stdout")
        return json.loads(line.decode("utf-8"))

    async def _probe(self, artifact_id: int) -> bool:
        """Run the health-check call within the probe timeout."""
        entry = self._procs.get(artifact_id)
        if not entry:
            return False
        try:
            resp = await self._rpc(artifact_id, "tools/list", None,
                                    timeout=_PROBE_TIMEOUT_S)
        except (asyncio.TimeoutError, RuntimeError, json.JSONDecodeError) as e:
            logger.warning("MCP health probe failed",
                            artifact_id=artifact_id, err=str(e))
            entry["fails"] = entry.get("fails", 0) + 1
            entry["last_probe"] = time.time()
            return False
        entry["fails"] = 0
        entry["last_probe"] = time.time()
        return "result" in resp and "tools" in (resp.get("result") or {})

    async def list_tools(self, artifact_id: int) -> list[dict[str, Any]]:
        """Return the MCP server's advertised tool list (name/description/schema)."""
        resp = await self._rpc(artifact_id, "tools/list", None,
                               timeout=_PROBE_TIMEOUT_S)
        return (resp.get("result") or {}).get("tools") or []

    async def call_tool(
        self, artifact_id: int, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Call one tool on the MCP server. Returns {"ok", "content", "error"}."""
        entry = self._procs.get(artifact_id)
        if entry:
            entry["last_used"] = time.time()
        try:
            resp = await self._rpc(
                artifact_id, "tools/call",
                {"name": tool_name, "arguments": arguments},
                timeout=_CALL_TIMEOUT_S,
            )
        except Exception as e:
            return {"ok": False, "content": None, "error": str(e)}
        if "error" in resp:
            return {"ok": False, "content": None,
                    "error": str(resp["error"])}
        content_items = (resp.get("result") or {}).get("content") or []
        text = "".join(
            c.get("text", "") for c in content_items if isinstance(c, dict)
        )
        return {"ok": True, "content": text, "error": None}

    async def reprobe_if_due(self, artifact_id: int) -> None:
        """Re-run the health probe if 60s have elapsed; handle fail escalation."""
        entry = self._procs.get(artifact_id)
        if not entry or entry["proc"].returncode is not None:
            return
        if time.time() - entry["last_probe"] < _REPROBE_INTERVAL_S:
            return
        healthy = await self._probe(artifact_id)
        if healthy:
            entry["health"] = "ready"
        else:
            fails = entry.get("fails", 0)
            if fails >= _MAX_FAILS_DISABLE:
                logger.warning("MCP disabled after repeated probe fails",
                                artifact_id=artifact_id)
                await self.shutdown(artifact_id)
                await self._mark_artifact_disabled(artifact_id)
            elif fails >= _MAX_FAILS_RESTART:
                logger.info("MCP restart attempt", artifact_id=artifact_id)
                await self.shutdown(artifact_id)
        await self._persist_process(artifact_id, pid=entry.get("pid"),
                                    health=entry["health"],
                                    consecutive_probe_fails=entry.get("fails", 0))

    async def _mark_artifact_disabled(self, artifact_id: int) -> None:
        from src.infra.db import get_db

        db = await get_db()
        await db.execute(
            "UPDATE yalayut_index SET enabled = 0 WHERE id = ?", (artifact_id,)
        )
        await db.commit()

    async def sweep_idle(self, now: float | None = None) -> list[int]:
        """Kill MCP servers idle longer than their idle_timeout. Returns killed ids."""
        now = time.time() if now is None else now
        killed: list[int] = []
        for artifact_id, entry in list(self._procs.items()):
            if entry["proc"].returncode is not None:
                continue
            if now - entry["last_used"] >= entry["idle_timeout"]:
                await self.shutdown(artifact_id)
                killed.append(artifact_id)
        return killed

    async def shutdown(self, artifact_id: int) -> None:
        """Terminate an MCP server and drop its in-memory handle."""
        entry = self._procs.pop(artifact_id, None)
        if not entry:
            return
        proc = entry.get("proc")
        if proc is not None and proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pass
        from src.infra.db import get_db

        try:
            db = await get_db()
            await db.execute(
                "DELETE FROM yalayut_mcp_processes WHERE artifact_id = ?",
                (artifact_id,),
            )
            await db.commit()
        except Exception as e:
            logger.warning("mcp process row cleanup failed", err=str(e))

    def status(self) -> list[dict[str, Any]]:
        """In-memory snapshot for /yalayut mcp status."""
        out = []
        for artifact_id, entry in self._procs.items():
            out.append({
                "artifact_id": artifact_id,
                "pid": entry.get("pid"),
                "health": entry.get("health"),
                "last_probe": entry.get("last_probe"),
                "fails": entry.get("fails", 0),
            })
        return out


# Module-level singleton — there is exactly one MCP fleet per KutAI process.
_MANAGER = McpManager()


def get_manager() -> McpManager:
    return _MANAGER
