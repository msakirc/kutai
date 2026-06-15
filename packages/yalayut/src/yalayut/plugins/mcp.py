"""yalayut.plugins.mcp — AccessPlugin for the ``mcp`` artifact type.

An mcp artifact is exposed via the ``tool`` class. Unlike api, an MCP server's
tool list is discovered at runtime (``tools/list`` on first start). This plugin:

  * starts the server on demand via :mod:`yalayut.mcp_manager` (lazy — never at
    boot, satisfying KutAI's ``no_auto_connect`` rule);
  * caches discovered tool descriptions + schemas into ``yalayut_mcp_tools``;
  * ranks tools by embedding similarity to the step intent and applies the
    per-server budget cap (``K_mcp`` = 3);
  * the consumer (intersect) applies the per-step total cap (``K_mcp_total`` = 6)
    via :func:`enforce_step_budget`;
  * namespaces each tool ``<artifact_slug>__<tool>`` (double underscore);
  * ``execute_mcp_tool`` is the path an agent's tool-call reaches.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("yalayut.plugin.mcp")

K_MCP_PER_SERVER = 3
K_MCP_PER_STEP = 6


def _slug(name: str) -> str:
    return name.replace("-", "_")


def _embed(text: str):
    """Embed text with the shared multilingual-e5-base model (sync path)."""
    from src.memory.embeddings import _get_st_embedding

    return _get_st_embedding(text or "", is_query=True)


def _cosine(a, b) -> float:
    import math

    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def rank_tools_by_intent(
    tools: list[dict[str, Any]], step_intent: str, k: int = K_MCP_PER_SERVER
) -> list[dict[str, Any]]:
    """Return the top-``k`` tools ranked by description-vs-intent similarity.

    Each returned tool dict gets a ``_score`` field. When ``step_intent`` is
    empty the original order is preserved (still capped to ``k``).
    """
    if not tools:
        return []
    if not step_intent:
        for t in tools:
            t.setdefault("_score", 0.0)
        return tools[:k]
    intent_vec = _embed(step_intent)
    scored = []
    for tool in tools:
        desc = tool.get("description") or tool.get("name") or ""
        score = _cosine(intent_vec, _embed(desc))
        tool["_score"] = score
        scored.append(tool)
    scored.sort(key=lambda t: t["_score"], reverse=True)
    return scored[:k]


def enforce_step_budget(
    apps: list[dict[str, Any]], k_total: int = K_MCP_PER_STEP
) -> list[dict[str, Any]]:
    """Trim a list of mcp SkillApplications so total tool count <= ``k_total``.

    Tools are pooled across servers, sorted by ``_score`` descending, and the
    top ``k_total`` kept; each app's ``payload['tools']`` is rewritten.
    """
    pooled: list[tuple[int, dict[str, Any]]] = []
    for app_idx, app in enumerate(apps):
        for tool in (app.get("payload") or {}).get("tools") or []:
            pooled.append((app_idx, tool))
    pooled.sort(key=lambda pair: pair[1].get("_score", 0.0), reverse=True)
    keep = pooled[:k_total]
    kept_by_app: dict[int, list[dict[str, Any]]] = {}
    for app_idx, tool in keep:
        kept_by_app.setdefault(app_idx, []).append(tool)
    for app_idx, app in enumerate(apps):
        app.setdefault("payload", {})["tools"] = kept_by_app.get(app_idx, [])
    return apps


async def _cache_mcp_tools(artifact_id: int, tools: list[dict[str, Any]]) -> None:
    """Persist discovered tool descriptions + schemas into yalayut_mcp_tools."""
    from dabidabi import get_db

    db = await get_db()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for tool in tools:
        name = tool.get("name")
        if not name:
            continue
        desc = tool.get("description") or ""
        try:
            emb = _embed(desc)
            emb_blob = json.dumps(emb).encode("utf-8")
        except Exception:
            emb_blob = None
        try:
            await db.execute(
                "INSERT INTO yalayut_mcp_tools "
                "(artifact_id, tool_name, description, description_embedding, "
                " input_schema_json, first_seen_at) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(artifact_id, tool_name) DO UPDATE SET "
                "description=excluded.description, "
                "description_embedding=excluded.description_embedding, "
                "input_schema_json=excluded.input_schema_json",
                (artifact_id, name, desc, emb_blob,
                 json.dumps(tool.get("inputSchema") or {}), now),
            )
        except Exception as e:
            logger.debug("mcp tool cache skipped (schema not ready?)",
                         artifact_id=artifact_id, tool_name=name, err=str(e))
    try:
        await db.commit()
    except Exception:
        pass


class McpPlugin:
    """AccessPlugin for mcp artifacts."""

    artifact_type = "mcp"

    def to_application(
        self, row: dict[str, Any], task_ctx: dict[str, Any]
    ) -> dict[str, Any]:
        """Synchronous shell — returns empty tools unless env missing.

        Real tool discovery needs an async server start; intersect calls
        :meth:`to_application_async`. This sync form exists only so the plugin
        protocol's signature is satisfied and the env-missing short-circuit is
        cheap.
        """
        env_status = row.get("env_status", "ready")
        return {
            "artifact_id": row.get("id"),
            "name": row.get("name"),
            "exposure_class": "tool",
            "applies_to": "execution",
            "render": None,
            "payload": {
                "kind": "mcp",
                "tools": [],
                "skipped_reason": None if env_status == "ready" else env_status,
            },
            "confidence": float(task_ctx.get("_confidence", 0.0)),
        }

    async def to_application_async(
        self, row: dict[str, Any], task_ctx: dict[str, Any]
    ) -> dict[str, Any]:
        """Start the MCP server, discover + rank + namespace tools, build payload."""
        env_status = row.get("env_status", "ready")
        base = self.to_application(row, task_ctx)
        if env_status != "ready":
            return base

        manifest = row.get("manifest") or {}
        mcp = manifest.get("mcp") or {}
        artifact_id = row.get("id")
        slug = _slug(row.get("name") or manifest.get("name") or "mcp_unknown")

        from yalayut.mcp_manager import get_manager

        manager = get_manager()
        handle = await manager.ensure_running(artifact_id, mcp)
        if handle.get("health") != "ready":
            base["payload"]["skipped_reason"] = "mcp_unhealthy"
            return base

        try:
            discovered = await manager.list_tools(artifact_id)
        except Exception as e:
            logger.warning("mcp tool discovery failed",
                            artifact_id=artifact_id, err=str(e))
            base["payload"]["skipped_reason"] = "mcp_discovery_failed"
            return base

        await _cache_mcp_tools(artifact_id, discovered)
        step_intent = task_ctx.get("step_intent") or task_ctx.get("intent") or ""
        ranked = rank_tools_by_intent(discovered, step_intent, k=K_MCP_PER_SERVER)

        tools_payload = []
        for tool in ranked:
            tools_payload.append({
                "tool_name": f"{slug}__{tool['name']}",
                "artifact_id": artifact_id,
                "mcp_tool_name": tool["name"],
                "description": tool.get("description", ""),
                "input_schema": tool.get("inputSchema") or {},
                "mcp": mcp,
                "_score": tool.get("_score", 0.0),
            })
        base["payload"]["tools"] = tools_payload
        return base

    def bind_args(self, row: dict[str, Any], task_ctx: dict[str, Any]) -> dict | None:
        return None

    async def execute(
        self, row: dict[str, Any], task_ctx: dict[str, Any], inputs: dict[str, Any]
    ) -> dict[str, Any]:
        """Convenience execute of the first discovered tool (tests/CLI)."""
        app = await self.to_application_async(row, task_ctx)
        tools = app["payload"]["tools"]
        if not tools:
            return {"ok": False, "error": app["payload"].get("skipped_reason")
                    or "no mcp tools"}
        return await execute_mcp_tool(tools[0], inputs)


async def execute_mcp_tool(
    tool_spec: dict[str, Any], arguments: dict[str, Any]
) -> dict[str, Any]:
    """Execute an MCP tool-call. Reached when an agent calls ``mcp_<slug>__<tool>``.

    ``tool_spec`` is one entry from ``McpPlugin.to_application_async()`` tools.
    Ensures the server is running (lazy start), then forwards a ``tools/call``.
    Returns ``{"ok", "response", "error"}``.
    """
    from yalayut.mcp_manager import get_manager

    manager = get_manager()
    artifact_id = tool_spec.get("artifact_id")
    mcp = tool_spec.get("mcp") or {}
    mcp_tool = tool_spec.get("mcp_tool_name")
    if artifact_id is None or not mcp_tool:
        return {"ok": False, "response": None, "error": "bad mcp tool_spec"}

    handle = await manager.ensure_running(artifact_id, mcp)
    if handle.get("health") != "ready":
        return {"ok": False, "response": None, "error": "mcp server unhealthy"}

    res = await manager.call_tool(artifact_id, mcp_tool, arguments or {})
    return {
        "ok": res["ok"],
        "response": res.get("content") if res["ok"] else None,
        "error": res.get("error"),
    }
