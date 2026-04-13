# app/api.py
"""
Phase 12.1 — FastAPI REST API Server

Endpoints: POST /missions, POST /tasks, GET /missions/{id}, GET /tasks/{id},
GET /queue, GET /stats, GET /models, GET /health,
GET /artifacts/{id}, WebSocket /ws/stream/{task_id}

Requires: pip install fastapi uvicorn
API key auth via X-API-Key header (set API_KEY env var).
"""
from __future__ import annotations

import asyncio
import os
from typing import Any, Optional

from src.infra.logging_config import get_logger

logger = get_logger("app.api")

try:
    from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, Header
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from starlette.responses import PlainTextResponse
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False
    logger.warning("FastAPI not installed — REST API unavailable. Run: pip install fastapi uvicorn")


# ── API Key Auth ─────────────────────────────────────────────────────────────

_API_KEY = os.getenv("API_KEY", "")


if _FASTAPI_AVAILABLE:
    def _check_api_key(x_api_key: str = Header(default="")) -> None:
        if _API_KEY and x_api_key != _API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")
else:
    def _check_api_key(*args, **kwargs) -> None:
        pass


# ── WebSocket connection manager ─────────────────────────────────────────────

class _ConnectionManager:
    def __init__(self):
        self._connections: dict[str, list[WebSocket]] = {}

    async def connect(self, task_id: str, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.setdefault(task_id, []).append(ws)

    def disconnect(self, task_id: str, ws: WebSocket) -> None:
        conns = self._connections.get(task_id, [])
        if ws in conns:
            conns.remove(ws)

    async def broadcast(self, task_id: str, data: dict) -> None:
        for ws in list(self._connections.get(task_id, [])):
            try:
                await ws.send_json(data)
            except Exception:
                self.disconnect(task_id, ws)

    async def broadcast_all(self, data: dict) -> None:
        for task_id in list(self._connections.keys()):
            await self.broadcast(task_id, data)


manager = _ConnectionManager()


def create_app() -> Any:
    """Create and configure the FastAPI application."""
    if not _FASTAPI_AVAILABLE:
        return None

    app = FastAPI(
        title="Orchestrator API",
        description="REST API for the AI agent orchestrator",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request/Response models ──────────────────────────────────────────────

    class MissionCreate(BaseModel):
        title: str
        description: str = ""
        priority: int = 5
        workflow: str | None = None
        repo_path: str | None = None

    class TaskCreate(BaseModel):
        title: str
        description: str = ""
        agent_type: str = "executor"
        mission_id: Optional[int] = None
        priority: int = 5

    # ── Health ──────────────────────────────────────────────────────────────

    @app.get("/health")
    async def health(_: None = Depends(_check_api_key)):
        from src.infra.runtime_state import runtime_state
        return {
            "status": "ok",
            "degraded": runtime_state.get("degraded_capabilities", []),
            "boot_time": runtime_state.get("boot_time"),
        }

    # ── Missions ────────────────────────────────────────────────────────────

    @app.post("/missions", status_code=201)
    async def create_mission(body: MissionCreate, _: None = Depends(_check_api_key)):
        from src.infra.db import get_db
        db = await get_db()
        cursor = await db.execute(
            "INSERT INTO missions (title, description, priority, workflow, repo_path, status) VALUES (?, ?, ?, ?, ?, 'active')",
            (body.title, body.description, body.priority, body.workflow, body.repo_path),
        )
        await db.commit()
        mission_id = cursor.lastrowid
        logger.info(f"API: Created mission #{mission_id}: {body.title}")
        return {"id": mission_id, "title": body.title, "status": "active"}

    @app.get("/missions/{mission_id}")
    async def get_mission(mission_id: int, _: None = Depends(_check_api_key)):
        from src.infra.db import get_mission as _get_mission
        mission = await _get_mission(mission_id)
        if not mission:
            raise HTTPException(status_code=404, detail="Mission not found")
        return dict(mission)

    @app.get("/missions")
    async def list_missions(_: None = Depends(_check_api_key)):
        from src.infra.db import get_db
        db = await get_db()
        cursor = await db.execute(
            "SELECT * FROM missions ORDER BY created_at DESC LIMIT 50"
        )
        rows = await cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row)) for row in rows]

    # ── Tasks ────────────────────────────────────────────────────────────────

    @app.post("/tasks", status_code=201)
    async def create_task(body: TaskCreate, _: None = Depends(_check_api_key)):
        from src.infra.db import add_task
        task_id = await add_task(
            title=body.title,
            description=body.description,
            agent_type=body.agent_type,
            mission_id=body.mission_id,
            priority=body.priority,
        )
        return {"id": task_id, "title": body.title, "status": "pending"}

    @app.get("/tasks/{task_id}")
    async def get_task(task_id: int, _: None = Depends(_check_api_key)):
        from src.infra.db import get_db
        db = await get_db()
        cursor = await db.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Task not found")
        cols = [d[0] for d in cursor.description]
        return dict(zip(cols, row))

    # ── Queue ────────────────────────────────────────────────────────────────

    @app.get("/queue")
    async def get_queue(_: None = Depends(_check_api_key)):
        from src.infra.db import get_db
        db = await get_db()
        cursor = await db.execute(
            "SELECT id, title, agent_type, priority, status, created_at "
            "FROM tasks WHERE status = 'pending' ORDER BY priority DESC, created_at ASC LIMIT 50"
        )
        rows = await cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        return {"queue": [dict(zip(cols, row)) for row in rows], "depth": len(rows)}

    # ── Stats ────────────────────────────────────────────────────────────────

    @app.get("/stats")
    async def get_stats(_: None = Depends(_check_api_key)):
        try:
            from src.infra.metrics import get_all_counters
            counters = get_all_counters()
        except Exception:
            counters = {}
        from src.infra.db import get_db
        db = await get_db()
        c1 = await db.execute("SELECT COUNT(*) FROM missions WHERE status = 'active'")
        c2 = await db.execute("SELECT COUNT(*) FROM tasks WHERE status = 'pending'")
        c3 = await db.execute("SELECT COUNT(*) FROM tasks WHERE status = 'completed'")
        return {
            "active_missions": (await c1.fetchone())[0],
            "pending_tasks": (await c2.fetchone())[0],
            "completed_tasks": (await c3.fetchone())[0],
            "metrics": counters,
        }

    # ── Models ───────────────────────────────────────────────────────────────

    @app.get("/models")
    async def get_models(_: None = Depends(_check_api_key)):
        from src.models.model_registry import get_registry
        registry = get_registry()
        return {
            name: {
                "provider": m.provider,
                "context_length": m.context_length,
                "demoted": m.demoted,
            }
            for name, m in registry.models.items()
        }

    # ── LLM Live Metrics ────────────────────────────────────────────────────

    @app.get("/llm")
    async def get_llm_metrics(_: None = Depends(_check_api_key)):
        """Live llama-server performance: token rates, KV cache, queue depth."""
        from src.models.local_model_manager import get_local_manager
        manager = get_local_manager()
        return await manager.get_metrics()

    # ── Prometheus Metrics ────────────────────────────────────────────────

    @app.get("/metrics", response_class=PlainTextResponse)
    async def prometheus_metrics():
        """Prometheus-compatible metrics. Delegates to NerdHerd."""
        try:
            from src.app.run import get_nerd_herd
            nh = get_nerd_herd()
            if nh is not None:
                return await nh.prometheus_lines()
            return ""
        except Exception as e:
            logger.warning("NerdHerd metrics unavailable", error=str(e))
            return ""

    # ── Artifacts ────────────────────────────────────────────────────────────

    @app.get("/artifacts/{mission_id}")
    async def get_artifacts(mission_id: int, _: None = Depends(_check_api_key)):
        try:
            from src.collaboration.blackboard import read_blackboard
            artifacts = await read_blackboard(mission_id, key="artifacts")
            return {"mission_id": mission_id, "artifacts": artifacts or {}}
        except Exception:
            return {"mission_id": mission_id, "artifacts": {}}

    # ── WebSocket streaming ──────────────────────────────────────────────────

    @app.websocket("/ws/stream/{task_id}")
    async def ws_stream(websocket: WebSocket, task_id: str):
        await manager.connect(task_id, websocket)
        try:
            while True:
                # Keep connection alive, send pings
                await asyncio.sleep(30)
                await websocket.send_json({"type": "ping"})
        except (WebSocketDisconnect, Exception):
            manager.disconnect(task_id, websocket)

    return app


# ── Singleton app instance ────────────────────────────────────────────────────

_app_instance: Any = None


def get_app() -> Any:
    """Get or create the FastAPI app instance."""
    global _app_instance
    if _app_instance is None:
        _app_instance = create_app()
    return _app_instance


async def start_api_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Start the API server as a background asyncio task."""
    if not _FASTAPI_AVAILABLE:
        logger.info("FastAPI not available — skipping API server startup")
        return
    try:
        import uvicorn
        app = get_app()
        config = uvicorn.Config(app, host=host, port=port, log_level="warning")
        server = uvicorn.Server(config)
        logger.info(f"Starting API server on http://{host}:{port}")
        await server.serve()
    except ImportError:
        logger.warning("uvicorn not installed — API server unavailable. Run: pip install uvicorn")
    except Exception as exc:
        logger.error(f"API server error: {exc}")
