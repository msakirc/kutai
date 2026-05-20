"""Z8 T3D — alert_triage mechanical executor.

Routed via ``mr_roboto.run`` when ``payload["action"] == "alert_triage"``.
Reads the inbound webhook payload, runs the rule-based severity classifier,
and stamps the verdict on the task result.

Critical/high severities are routed downstream to the ``oncall_agent``
task (Z8 P0 closeout, 2026-05-18 sweep). Low/medium pass through with
``oncall_routed=False`` and feed the periodic digest path.
"""
from __future__ import annotations

from typing import Any

from src.infra.logging_config import get_logger
from src.ops.severity_classifier import classify

logger = get_logger("mr_roboto.alert_triage")


# Severities that warrant immediate oncall_agent dispatch.
# (low/medium aggregate into the periodic digest instead.)
_ESCALATE_SEVERITIES: frozenset[str] = frozenset({"critical", "high"})


async def run(task: dict[str, Any]) -> dict[str, Any]:
    payload_envelope = task.get("payload") or {}
    integration_id = str(payload_envelope.get("integration_id") or "")
    event_id = str(payload_envelope.get("event_id") or "")
    vendor_payload = payload_envelope.get("payload") or {}

    event_type = (
        vendor_payload.get("type")
        or vendor_payload.get("event")
        or vendor_payload.get("event_type")
        or "unknown"
    )

    severity = classify(integration_id, str(event_type), vendor_payload)
    llm_graded = False
    if severity == "uncertain":
        severity = await _llm_grade(integration_id, str(event_type), vendor_payload)
        llm_graded = True

    oncall_task_id: int | None = None
    if severity in _ESCALATE_SEVERITIES:
        # Z8 P0 (2026-05-18 sweep) — hand off to oncall_agent. Was
        # hardcoded oncall_routed=False with a TODO. The oncall_agent
        # exists and is registered; until this enqueue landed, the
        # webhook→triage→oncall loop was open-circuit.
        oncall_task_id = await _enqueue_oncall_task(
            triage_task=task,
            severity=severity,
            integration_id=integration_id,
            event_id=event_id,
            event_type=str(event_type),
            vendor_payload=vendor_payload,
        )

    result = {
        "severity": severity,
        "integration_id": integration_id,
        "event_id": event_id,
        "event_type": event_type,
        "llm_graded": llm_graded,
        "oncall_routed": oncall_task_id is not None,
        "oncall_task_id": oncall_task_id,
    }
    logger.info(
        "alert_triage classified",
        integration_id=integration_id,
        event_id=event_id,
        event_type=event_type,
        severity=severity,
        llm_graded=llm_graded,
        oncall_task_id=oncall_task_id,
    )
    return result


async def _enqueue_oncall_task(
    *,
    triage_task: dict,
    severity: str,
    integration_id: str,
    event_id: str,
    event_type: str,
    vendor_payload: dict,
) -> int | None:
    """Enqueue an oncall_agent task to handle the classified alert.

    Best-effort: enqueue failures are logged but do not propagate — the
    triage task itself still returns success so the verdict row lands and
    the action_cooldowns ledger can be inspected. Escalation policy +
    Twilio SMS already exist downstream; this just lights the chain.

    Returns the enqueued task id, or ``None`` on failure.
    """
    try:
        from general_beckman import enqueue  # lazy import — avoid cycle
        mission_id = triage_task.get("mission_id")
        title = (
            f"oncall: {severity} {event_type} "
            f"({integration_id}#{event_id})"
        )[:200]
        spec = {
            "title": title,
            "description": (
                f"On-call response to a {severity}-severity {event_type} "
                f"event from {integration_id}. Match against incident "
                f"playbooks, take the least-irreversible whitelisted "
                f"action, log every step, escalate to founder if blocked."
            ),
            "agent_type": "oncall_agent",
            "mission_id": mission_id,
            "context": {
                "domain": "ops",
                "triage_source_task_id": triage_task.get("id"),
                "severity": severity,
                "integration_id": integration_id,
                "event_id": event_id,
                "event_type": event_type,
                "vendor_payload": vendor_payload,
            },
        }
        task_id = await enqueue(spec, parent_id=triage_task.get("id"))
        if isinstance(task_id, int):
            return task_id
        # await_inline=False path — enqueue returns int directly. Anything
        # else is a TaskResult and shouldn't happen here.
        return getattr(task_id, "task_id", None)
    except Exception as exc:
        logger.warning(
            "alert_triage: oncall enqueue failed",
            severity=severity,
            integration_id=integration_id,
            event_id=event_id,
            error=str(exc),
        )
        return None


async def _llm_grade(integration_id: str, event_type: str, payload: dict) -> str:
    """T3 stub for LLM-graded fallback when rule classifier is uncertain.

    T4+ wires a real dispatcher.request() call. For now we return "medium"
    as a safe non-critical default so the task can complete without
    requiring a live model.
    """
    logger.debug(
        "alert_triage LLM-grade stub firing",
        integration_id=integration_id,
        event_type=event_type,
    )
    return "medium"
