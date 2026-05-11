"""Z6 T3B — LLM-callable ``vendor_call`` tool.

Bridges the agent runtime to the IntegrationRegistry. Same backend as the
mechanical T3A executor, but exposed as a regular tool so an agent can
invoke it inline (e.g. researcher probing a vendor for capabilities, or
implementer checking auth before scaffolding).

**Per-agent allowlist** restricts which agents can call which services. The
default is ``[]`` (deny). The runtime injects the calling agent's type and
mission/task ids via the standard tool-call kwarg path.

**Per-call cost cap** enforces ``MAX_TOOL_CALL_COST_USD`` (env, default 5.0).
Projected cost is read from the call payload's ``cost_estimate_usd`` arg or
falls back to 0. Heavier checks (mission budget) belong to the mechanical
path — this is a coarse spend-circuit-breaker.

Returns a JSON string for the agent (tools speak JSON-as-string).
"""
from __future__ import annotations

import json
import os
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("tools.vendor_call")


# Per-agent allowlist. Default-deny for unlisted agents — to opt-in, add
# the agent name with the services it may touch. Keep this list tight;
# every entry is a credential leakage / spend / blast-radius decision.
AGENT_ALLOWLIST: dict[str, list[str]] = {
    "executor": ["vercel", "railway", "supabase", "cloudflare"],
    "implementer": ["stripe", "sendgrid"],
    "reviewer": ["sentry"],
    "researcher": [],
    "coder": [],
    "planner": [],
    "shopping_advisor": [],
    "product_researcher": [],
    "deal_analyst": [],
}


def _allowed_services(agent: str | None) -> list[str]:
    if not agent:
        return []
    return list(AGENT_ALLOWLIST.get(agent, []))


def _max_cost_cap() -> float:
    raw = os.getenv("MAX_TOOL_CALL_COST_USD", "5.0")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 5.0


async def vendor_call_tool(
    service: str,
    action: str,
    params: dict | str | None = None,
    *,
    cost_estimate_usd: float | int | str = 0,
    mission_id: int | str | None = None,
    task_id: int | str | None = None,
    agent: str | None = None,
) -> str:
    """Execute one vendor API call from an LLM agent.

    Returns a JSON string. Always cooperative — never raises. Refusals
    return ``{"status": "refused", "reason": ...}``.
    """
    service = (service or "").strip()
    action = (action or "").strip()
    if not service or not action:
        return json.dumps({
            "status": "refused",
            "reason": "missing_service_or_action",
        })

    # Parse params if it arrived as a JSON string (common LLM tool shape).
    if isinstance(params, str):
        try:
            params_dict = json.loads(params) if params.strip() else {}
        except json.JSONDecodeError:
            return json.dumps({
                "status": "refused",
                "reason": "params_not_json",
            })
    elif isinstance(params, dict):
        params_dict = params
    else:
        params_dict = {}

    # 1. Agent allowlist.
    allowed = _allowed_services(agent)
    if service not in allowed:
        logger.info(
            "vendor_call refused by allowlist",
            agent=agent, service=service,
        )
        return json.dumps({
            "status": "refused",
            "reason": "agent_not_allowed",
            "agent": agent,
            "service": service,
            "allowed_services": allowed,
        })

    # 2. Cost cap.
    try:
        cost_f = float(cost_estimate_usd) if cost_estimate_usd is not None else 0.0
    except (TypeError, ValueError):
        cost_f = 0.0
    cap = _max_cost_cap()
    if cost_f > cap:
        return json.dumps({
            "status": "refused",
            "reason": "cost_cap_exceeded",
            "cost_estimate_usd": cost_f,
            "cap_usd": cap,
        })

    # 3. Audit log. TODO(T2C): once the credential audit log lands, route
    # via the shared audit channel. For now plain logger.info.
    logger.info(
        "vendor_call invoked",
        agent=agent, service=service, action=action,
        mission_id=mission_id, task_id=task_id, cost_estimate_usd=cost_f,
    )

    # 4. Resolve adapter.
    try:
        from src.integrations.registry import get_integration_registry
        registry = get_integration_registry()
    except Exception as exc:  # noqa: BLE001
        return json.dumps({
            "status": "error",
            "reason": "registry_unavailable",
            "error": f"{type(exc).__name__}: {exc}",
        })
    adapter = registry.get(service)
    if adapter is None:
        available = []
        try:
            available = registry.list_services()
        except Exception:  # noqa: BLE001
            pass
        return json.dumps({
            "status": "error",
            "reason": "adapter_not_registered",
            "service": service,
            "available": available,
        })

    # 5. Execute.
    try:
        result = await adapter.execute(action, params_dict)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "vendor_call raised", service=service, action=action,
            error=str(exc),
        )
        return json.dumps({
            "status": "error",
            "reason": "vendor_raised",
            "error": f"{type(exc).__name__}: {exc}",
            "service": service,
            "action": action,
        })

    if not isinstance(result, dict):
        return json.dumps({
            "status": "error",
            "reason": "bad_adapter_shape",
            "result": str(result),
        })
    return json.dumps(result, default=str)
