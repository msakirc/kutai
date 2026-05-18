"""Z8 T3D — alert_triage mechanical executor.

Routed via ``mr_roboto.run`` when ``payload["action"] == "alert_triage"``.
Reads the inbound webhook payload, runs the rule-based severity classifier,
and stamps the verdict on the task result.

Downstream handoff to ``oncall_agent`` is T4 work — for now we just return
the verdict so the orchestrator can persist it. When the classifier returns
``"uncertain"``, ``_llm_grade`` is the future hook; the T3 stub returns
``"medium"`` with a sentinel flag so we never block a triage on missing
LLM scaffolding.
"""
from __future__ import annotations

from typing import Any

from src.infra.logging_config import get_logger
from src.ops.severity_classifier import classify

logger = get_logger("mr_roboto.alert_triage")


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

    result = {
        "severity": severity,
        "integration_id": integration_id,
        "event_id": event_id,
        "event_type": event_type,
        "llm_graded": llm_graded,
        # TODO T4: route critical/high to oncall_agent; low → digest.
        "oncall_routed": False,
    }
    logger.info(
        "alert_triage classified",
        integration_id=integration_id,
        event_id=event_id,
        event_type=event_type,
        severity=severity,
        llm_graded=llm_graded,
    )
    return result


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
