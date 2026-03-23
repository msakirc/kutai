# app/api.py
"""
Phase 12.1 — FastAPI REST API Server

Endpoints: POST /goals, POST /tasks, GET /goals/{id}, GET /tasks/{id},
GET /queue, GET /stats, GET /models, GET /health, GET /projects,
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


def _check_api_key(x_api_key: str = Header(default="")) -> None:
    if _API_KEY and x_api_key != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


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

    class GoalCreate(BaseModel):
        title: str
        description: str = ""
        priority: int = 5

    class TaskCreate(BaseModel):
        title: str
        description: str = ""
        agent_type: str = "executor"
        goal_id: Optional[int] = None
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

    # ── Goals ───────────────────────────────────────────────────────────────

    @app.post("/goals", status_code=201)
    async def create_goal(body: GoalCreate, _: None = Depends(_check_api_key)):
        from src.infra.db import get_db
        db = await get_db()
        cursor = await db.execute(
            "INSERT INTO goals (title, description, priority, status) VALUES (?, ?, ?, 'active')",
            (body.title, body.description, body.priority),
        )
        await db.commit()
        goal_id = cursor.lastrowid
        logger.info(f"API: Created goal #{goal_id}: {body.title}")
        return {"id": goal_id, "title": body.title, "status": "active"}

    @app.get("/goals/{goal_id}")
    async def get_goal(goal_id: int, _: None = Depends(_check_api_key)):
        from src.infra.db import get_goal as _get_goal
        goal = await _get_goal(goal_id)
        if not goal:
            raise HTTPException(status_code=404, detail="Goal not found")
        return dict(goal)

    @app.get("/goals")
    async def list_goals(_: None = Depends(_check_api_key)):
        from src.infra.db import get_db
        db = await get_db()
        cursor = await db.execute(
            "SELECT * FROM goals ORDER BY created_at DESC LIMIT 50"
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
            goal_id=body.goal_id,
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
        c1 = await db.execute("SELECT COUNT(*) FROM goals WHERE status = 'active'")
        c2 = await db.execute("SELECT COUNT(*) FROM tasks WHERE status = 'pending'")
        c3 = await db.execute("SELECT COUNT(*) FROM tasks WHERE status = 'completed'")
        return {
            "active_goals": (await c1.fetchone())[0],
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
        """Prometheus-compatible metrics endpoint. No auth — Prometheus needs open access."""
        from src.infra.metrics import get_all_counters
        from src.models.local_model_manager import get_local_manager

        lines = []

        # ── Orchestrator counters ──
        counters = get_all_counters()
        tasks_ok = int(counters.get("tasks_completed", 0))
        tasks_fail = int(counters.get("tasks_failed", 0))
        queue = int(counters.get("queue_depth", 0))
        cost = float(counters.get("cost_total", 0.0))

        lines.append(f"# HELP kutay_tasks_completed_total Total tasks completed")
        lines.append(f"# TYPE kutay_tasks_completed_total counter")
        lines.append(f"kutay_tasks_completed_total {tasks_ok}")

        lines.append(f"# HELP kutay_tasks_failed_total Total tasks failed")
        lines.append(f"# TYPE kutay_tasks_failed_total counter")
        lines.append(f"kutay_tasks_failed_total {tasks_fail}")

        lines.append(f"# HELP kutay_queue_depth Current task queue depth")
        lines.append(f"# TYPE kutay_queue_depth gauge")
        lines.append(f"kutay_queue_depth {queue}")

        lines.append(f"# HELP kutay_cost_total_usd Total inference cost in USD")
        lines.append(f"# TYPE kutay_cost_total_usd counter")
        lines.append(f"kutay_cost_total_usd {cost:.6f}")

        # Per-model call counts and tokens
        model_calls = {k.split(":", 1)[1]: int(v)
                       for k, v in counters.items() if k.startswith("model_calls:")}
        if model_calls:
            lines.append(f"# HELP kutay_model_calls_total Model call count by model")
            lines.append(f"# TYPE kutay_model_calls_total counter")
            for model, count in model_calls.items():
                safe = model.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
                lines.append(f'kutay_model_calls_total{{model="{safe}"}} {count}')

        model_tokens = {k.split(":", 1)[1]: int(v)
                        for k, v in counters.items() if k.startswith("tokens:")}
        if model_tokens:
            lines.append(f"# HELP kutay_tokens_total Token count by model")
            lines.append(f"# TYPE kutay_tokens_total counter")
            for model, tokens in model_tokens.items():
                safe = model.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
                lines.append(f'kutay_tokens_total{{model="{safe}"}} {tokens}')

        # ── Local model manager status ──
        try:
            mgr = get_local_manager()
            status = mgr.get_status()
            loaded = status.get("loaded_model") or ""
            healthy = 1 if status.get("healthy") else 0
            swaps = status.get("total_swaps", 0)
            idle = status.get("idle_seconds", 0)
            busy = 1 if status.get("inference_busy") else 0

            lines.append(f"# HELP kutay_model_healthy Is the local model healthy")
            lines.append(f"# TYPE kutay_model_healthy gauge")
            lines.append(f'kutay_model_healthy{{model="{loaded}"}} {healthy}')

            lines.append(f"# HELP kutay_model_swaps_total Total model swaps")
            lines.append(f"# TYPE kutay_model_swaps_total counter")
            lines.append(f"kutay_model_swaps_total {swaps}")

            lines.append(f"# HELP kutay_model_idle_seconds Seconds since last inference")
            lines.append(f"# TYPE kutay_model_idle_seconds gauge")
            lines.append(f"kutay_model_idle_seconds {idle:.1f}")

            lines.append(f"# HELP kutay_model_inference_busy Is inference currently running")
            lines.append(f"# TYPE kutay_model_inference_busy gauge")
            lines.append(f"kutay_model_inference_busy {busy}")
        except Exception as e:
            logger.debug(f"Model manager metrics unavailable: {e}")
            lines.append("# HELP kutay_model_healthy Is the local model healthy")
            lines.append("# TYPE kutay_model_healthy gauge")
            lines.append('kutay_model_healthy{model=""} 0')

        # ── Auto-tuner quality + model health metrics ──
        try:
            from src.models.auto_tuner import get_prometheus_lines_async
            lines.extend(await get_prometheus_lines_async())
        except Exception as e:
            logger.debug(f"Auto-tuner metrics unavailable: {e}")

        # ── GPU load mode metrics ──
        try:
            from src.infra.load_manager import get_load_mode, get_vram_budget_fraction, is_auto_managed
            mode = await get_load_mode()
            budget = get_vram_budget_fraction()
            auto = 1 if is_auto_managed() else 0

            mode_val = {"minimal": 0, "shared": 1, "heavy": 2, "full": 3}.get(mode, 3)
            lines.append("# HELP kutay_gpu_load_mode Current GPU load mode (0=minimal,1=shared,2=heavy,3=full)")
            lines.append("# TYPE kutay_gpu_load_mode gauge")
            lines.append(f"kutay_gpu_load_mode {mode_val}")

            lines.append("# HELP kutay_gpu_load_mode_info GPU load mode as label")
            lines.append("# TYPE kutay_gpu_load_mode_info gauge")
            lines.append(f'kutay_gpu_load_mode_info{{mode="{mode}"}} 1')

            lines.append("# HELP kutay_gpu_vram_budget_fraction VRAM budget fraction 0.0-1.0")
            lines.append("# TYPE kutay_gpu_vram_budget_fraction gauge")
            lines.append(f"kutay_gpu_vram_budget_fraction {budget:.2f}")

            lines.append("# HELP kutay_gpu_auto_managed Whether GPU mode is auto-managed")
            lines.append("# TYPE kutay_gpu_auto_managed gauge")
            lines.append(f"kutay_gpu_auto_managed {auto}")
        except Exception as e:
            logger.debug(f"Load mode metrics unavailable: {e}")

        # ── External GPU usage metrics ──
        try:
            from src.models.gpu_monitor import get_gpu_monitor
            ext = get_gpu_monitor().detect_external_gpu_usage()
            lines.append("# HELP kutay_gpu_external_vram_mb External process VRAM usage in MB")
            lines.append("# TYPE kutay_gpu_external_vram_mb gauge")
            lines.append(f"kutay_gpu_external_vram_mb {ext.external_vram_mb}")

            lines.append("# HELP kutay_gpu_external_processes Number of external GPU processes")
            lines.append("# TYPE kutay_gpu_external_processes gauge")
            lines.append(f"kutay_gpu_external_processes {ext.external_process_count}")

            lines.append("# HELP kutay_gpu_external_vram_fraction External VRAM as fraction of total")
            lines.append("# TYPE kutay_gpu_external_vram_fraction gauge")
            lines.append(f"kutay_gpu_external_vram_fraction {ext.external_vram_fraction:.4f}")
        except Exception as e:
            logger.debug(f"External GPU metrics unavailable: {e}")

        lines.append("")
        return "\n".join(lines)

    # ── Projects ─────────────────────────────────────────────────────────────

    @app.get("/projects")
    async def get_projects(_: None = Depends(_check_api_key)):
        try:
            from src.infra.projects import list_projects
            return await list_projects()
        except Exception:
            return []

    # ── Artifacts ────────────────────────────────────────────────────────────

    @app.get("/artifacts/{goal_id}")
    async def get_artifacts(goal_id: int, _: None = Depends(_check_api_key)):
        try:
            from src.collaboration.blackboard import read_blackboard
            artifacts = await read_blackboard(goal_id, key="artifacts")
            return {"goal_id": goal_id, "artifacts": artifacts or {}}
        except Exception:
            return {"goal_id": goal_id, "artifacts": {}}

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
