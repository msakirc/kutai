"""Vendor-call mechanical executor — Z6 T3A.

Drives a real-world vendor API call via the IntegrationRegistry. Designed to
be invoked as a workflow post-hook after the LLM-emitting step has produced
its parameters — keeps the side-effect outside the LLM loop where retries
and idempotency live.

Payload (``task["context"]["post_hook"]`` or ``task["payload"]``)::

    {
        "service": "stripe",
        "action":  "create_product",
        "params":  {...},                # literal params
        "params_from_artifact": "name",  # optional, merged under literal params
    }

Returns ``{"ok": True, "result": ..., "service": ..., "action": ...}`` on
success. On any failure the executor emits a ``founder_action(kind='generic')``
with the failure details and returns ``{"ok": False, "reason": ..., ...}`` —
admission gate should have prevented adapter-missing here, this is defensive.

Cost-cap: if ``task.context.cost_estimate_usd`` is set, we consult the
mission's remaining budget (when the schema exposes one) and refuse if the
projected spend exceeds the remaining envelope. When no budget column is
present we no-op the check.
"""
from __future__ import annotations

import json
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.vendor_call")


def _parse_context(task: dict) -> dict:
    raw = task.get("context")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw or "{}") or {}
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


def _pick_hook_spec(task: dict, ctx: dict) -> dict:
    """post-hook spec lives either in context.post_hook or task.payload."""
    spec = ctx.get("post_hook")
    if isinstance(spec, dict) and spec:
        return spec
    payload = task.get("payload") or {}
    if isinstance(payload, dict):
        return payload
    return {}


async def _load_artifact(mission_id: int, name: str) -> dict | None:
    """Best-effort artifact retrieval (cache-first, blackboard fallback)."""
    try:
        from src.workflows.engine.hooks import get_artifact_store
        store = get_artifact_store()
        raw = await store.retrieve(mission_id, name)
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str) and raw.strip():
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return None
    except Exception as exc:  # noqa: BLE001
        logger.debug("artifact-store retrieve failed", error=str(exc))

    try:
        from src.collaboration.blackboard import read_blackboard
        artifacts = await read_blackboard(int(mission_id), "artifacts")
        if isinstance(artifacts, dict):
            raw2 = artifacts.get(name)
            if isinstance(raw2, dict):
                return raw2
            if isinstance(raw2, str) and raw2.strip():
                try:
                    return json.loads(raw2)
                except json.JSONDecodeError:
                    return None
    except Exception as exc:  # noqa: BLE001
        logger.debug("blackboard read failed", error=str(exc))
    return None


async def _check_cost_cap(
    mission_id: int | None,
    cost_estimate_usd: float | None,
) -> tuple[bool, str]:
    """Return (allowed, reason). True allowed when no enforceable budget."""
    if not cost_estimate_usd or cost_estimate_usd <= 0:
        return True, ""
    if not mission_id:
        return True, ""
    try:
        from src.infra.db import get_db
        db = await get_db()
        # Probe mission row for any plausible remaining-budget column.
        cur = await db.execute("PRAGMA table_info(missions)")
        cols = {row[1] for row in await cur.fetchall()}
        budget_col = None
        for candidate in (
            "remaining_budget_usd", "budget_remaining_usd",
            "budget_usd_remaining", "remaining_usd",
        ):
            if candidate in cols:
                budget_col = candidate
                break
        if budget_col is None:
            return True, "no_budget_column"
        cur = await db.execute(
            f"SELECT {budget_col} FROM missions WHERE id = ?", (mission_id,),
        )
        row = await cur.fetchone()
        if not row or row[0] is None:
            return True, "no_budget_value"
        remaining = float(row[0])
        if cost_estimate_usd > remaining:
            return False, (
                f"cost ${cost_estimate_usd:.2f} exceeds remaining "
                f"${remaining:.2f}"
            )
        return True, "within_budget"
    except Exception as exc:  # noqa: BLE001
        logger.debug("cost_cap probe failed — defaulting allow", error=str(exc))
        return True, "probe_failed"


async def _emit_failure_action(
    mission_id: int | None,
    service: str,
    action: str,
    task_id: int | None,
    step_id: str | None,
    error_msg: str,
) -> None:
    """Best-effort founder_action emission. Failures here are swallowed."""
    if not mission_id:
        return
    try:
        import src.founder_actions as fa
        await fa.create(
            mission_id=int(mission_id),
            kind="generic",
            title=f"vendor_call failed for {service}.{action}",
            why=str(error_msg)[:500],
            instructions=[
                f"Inspect vendor adapter '{service}' configuration.",
                f"Verify credentials for {service} via /credential list.",
                "Check rate limits / status page for transient issues.",
                f"Once fixed, retry the task via /retry {task_id or '?'}.",
            ],
            blocking_task_id=task_id,
            blocking_step_id=step_id,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("founder_action emit failed", error=str(exc))


async def run(task: dict[str, Any]) -> dict[str, Any]:
    """Execute one vendor call. Always returns a dict — never raises."""
    ctx = _parse_context(task)
    spec = _pick_hook_spec(task, ctx)
    mission_id = task.get("mission_id")
    task_id = task.get("id")
    step_id = ctx.get("workflow_step_id") or task.get("workflow_step_id")

    service = str(spec.get("service") or "").strip()
    action = str(spec.get("action") or "").strip()
    if not service or not action:
        return {
            "ok": False,
            "reason": "missing_service_or_action",
            "service": service,
            "action": action,
        }

    # Merge artifact params first, then literal params (literal wins).
    merged_params: dict = {}
    artifact_name = spec.get("params_from_artifact")
    if artifact_name and mission_id:
        artifact = await _load_artifact(int(mission_id), str(artifact_name))
        if isinstance(artifact, dict):
            merged_params.update(artifact)
    literal = spec.get("params") or {}
    if isinstance(literal, dict):
        merged_params.update(literal)

    # Cost-cap (defensive; admission also handles this).
    cost_estimate = ctx.get("cost_estimate_usd")
    try:
        cost_f = float(cost_estimate) if cost_estimate is not None else 0.0
    except (TypeError, ValueError):
        cost_f = 0.0
    allowed, cap_reason = await _check_cost_cap(mission_id, cost_f)
    if not allowed:
        await _emit_failure_action(
            mission_id, service, action, task_id, step_id,
            f"cost cap exceeded: {cap_reason}",
        )
        return {
            "ok": False,
            "reason": "cost_cap_exceeded",
            "detail": cap_reason,
            "service": service,
            "action": action,
        }

    # Resolve adapter.
    try:
        from src.integrations.registry import get_integration_registry
        registry = get_integration_registry()
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "reason": "registry_unavailable",
            "error": f"{type(exc).__name__}: {exc}",
            "service": service,
            "action": action,
        }
    adapter = registry.get(service)
    if adapter is None:
        # Defensive: admission gate should have caught this.
        return {
            "ok": False,
            "reason": "adapter_not_registered",
            "service": service,
            "action": action,
        }

    # Execute.
    try:
        result = await adapter.execute(action, merged_params)
    except Exception as exc:  # noqa: BLE001
        error_msg = f"{type(exc).__name__}: {exc}"
        await _emit_failure_action(
            mission_id, service, action, task_id, step_id, error_msg,
        )
        logger.warning(
            "vendor_call raised", service=service, action=action,
            error=error_msg,
        )
        return {
            "ok": False,
            "reason": "vendor_error",
            "error": error_msg,
            "service": service,
            "action": action,
        }

    if not isinstance(result, dict) or result.get("status") != "ok":
        err = (result or {}).get("error", "unknown") if isinstance(result, dict) else str(result)
        await _emit_failure_action(
            mission_id, service, action, task_id, step_id, str(err),
        )
        return {
            "ok": False,
            "reason": "vendor_error",
            "error": err,
            "service": service,
            "action": action,
            "raw": result if isinstance(result, dict) else None,
        }

    return {
        "ok": True,
        "result": result.get("data"),
        "service": service,
        "action": action,
        "status_code": result.get("status_code"),
    }
