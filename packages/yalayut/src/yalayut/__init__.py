"""Yalayut — vetted catalog of external skills, APIs, MCP servers.

Public operational API (spec Public APIs section). The intersect (Phase 3) is
the only hot-path importer of query(); the discovery/scout/recipe functions
are mechanical-executor bodies invoked by mr_roboto shims (Phase 3 wiring).

Phase 1 ships REAL bodies for every function. daily_discovery() actually pulls
github_path sources and populates yalayut_index. source_scout_scan() and
on_demand_discovery() have working bodies for the path Phase 1 owns and
documented seams Phase 3/4 extend (web search / awesome-list adapters) — they
are functional, never empty stubs.
"""
from __future__ import annotations

from yalayut.contracts import Artifact
from yalayut.executor import run_recipe  # noqa: F401  (operational API)

__all__ = [
    "query", "daily_discovery", "source_scout_scan", "on_demand_discovery",
    "capture_hint", "record_demand_signal", "run_recipe", "dispatch_tool",
    "observe_and_propose", "Artifact",
]


async def dispatch_tool(
    tool_name: str, args: dict, registry: dict
) -> dict:
    """Route a namespaced yalayut tool-call to its plugin executor.

    ``registry`` is coulson's per-task tool registry: a dict mapping tool name
    to the tool-spec produced by ``ApiPlugin.to_application`` /
    ``McpPlugin.to_application_async``. Each spec carries ``_yalayut_kind``
    ('api' | 'mcp'). Returns ``{"ok", "response", "error"}``.
    """
    spec = registry.get(tool_name)
    if spec is None:
        return {"ok": False, "response": None,
                "error": f"unknown yalayut tool: {tool_name}"}
    kind = spec.get("_yalayut_kind")
    if kind == "api" or tool_name.startswith("api_"):
        from yalayut.plugins.api import execute_api_tool
        return await execute_api_tool(spec, args)
    if kind == "mcp" or tool_name.startswith("mcp_"):
        from yalayut.plugins.mcp import execute_mcp_tool
        return await execute_mcp_tool(spec, args)
    return {"ok": False, "response": None,
            "error": f"tool {tool_name} has no yalayut kind"}


async def query(task_ctx: dict, top_k: int = 12) -> list[Artifact]:
    """Hot read — vector similarity over the index. The intersect's only
    entry. Returns ranked Artifact dataclasses."""
    from yalayut._query_engine import query as _query
    return await _query(task_ctx, top_k=top_k)


# ── Phase 4 — discovery + autonomy entry points ─────────────────────────


async def daily_discovery() -> dict:
    """Mechanical-executor body: pull trusted cron-mode sources."""
    from yalayut.discovery.cron import daily_discovery as _impl
    return await _impl()


async def on_demand_discovery(demand: dict) -> dict:
    """Need-driven fetch for one DemandSignal."""
    from yalayut.discovery.on_demand import on_demand_discovery as _impl
    return await _impl(demand)


async def source_scout_scan() -> dict:
    """Mechanical-executor body: propose candidate sources."""
    from yalayut.discovery.source_scout import source_scout_scan as _impl
    return await _impl()


async def observe_and_propose() -> int:
    """Scan vetting audit data; write founder policy proposals. Returns the
    count of new proposals written."""
    from yalayut.policy_observer import observe_and_propose as _impl
    return await _impl()


async def capture_hint(task: dict, outcome: dict) -> None:
    """Post-hook body: internal_hint auto-capture."""
    from yalayut.capture import capture_hint as _impl
    return await _impl(task, outcome)


async def record_demand_signal(
    *,
    source_step_pattern: str,
    intent_keywords: list[str],
    signal_type: str,
    confidence: float = 0.3,
) -> int:
    """Public API — record one demand signal. Firing sites call this."""
    from yalayut.discovery.demand import record as _impl
    return await _impl(
        source_step_pattern=source_step_pattern,
        intent_keywords=intent_keywords,
        signal_type=signal_type,
        confidence=confidence,
    )


