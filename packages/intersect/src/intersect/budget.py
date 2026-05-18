"""Per-step budget caps for tool-class exposures.

api  → at most API_CAP surfaced per step
mcp  → at most MCP_PER_SERVER tools per server, MCP_TOTAL per step total

inject / preempt exposures are skill-shaped and never capped. Ranking is
by confidence descending; the lowest-confidence overflow is dropped.
Dropped entries are returned so telemetry can log them as not-exposed.
"""
from __future__ import annotations

API_CAP: int = 3
MCP_PER_SERVER: int = 3
MCP_TOTAL: int = 6


def apply_caps(applications: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return ``(kept, dropped)`` after applying api/mcp budget caps."""
    kept: list[dict] = []
    dropped: list[dict] = []

    # inject / preempt — pass through untouched.
    for app in applications:
        if app.get("exposure_class") in ("inject", "preempt"):
            kept.append(app)

    # api — global per-step cap.
    apis = sorted(
        (a for a in applications
         if a.get("artifact_type") == "api"
         and a.get("exposure_class") == "tool"),
        key=lambda a: a.get("confidence", 0.0),
        reverse=True,
    )
    kept.extend(apis[:API_CAP])
    dropped.extend(apis[API_CAP:])

    # mcp — per-server cap then per-step total cap.
    mcps = sorted(
        (a for a in applications
         if a.get("artifact_type") == "mcp"
         and a.get("exposure_class") == "tool"),
        key=lambda a: a.get("confidence", 0.0),
        reverse=True,
    )
    per_server: dict[str, int] = {}
    mcp_kept: list[dict] = []
    for app in mcps:
        srv = app.get("mcp_server") or "_default"
        if per_server.get(srv, 0) >= MCP_PER_SERVER:
            dropped.append(app)
            continue
        if len(mcp_kept) >= MCP_TOTAL:
            dropped.append(app)
            continue
        per_server[srv] = per_server.get(srv, 0) + 1
        mcp_kept.append(app)
    kept.extend(mcp_kept)

    return kept, dropped
