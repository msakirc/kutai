"""Pre-dispatch gates: human approval, risk assessment, clarification validation.

Returns GateDecision (Allow | Block | Cancel) instead of directly sending Telegram.
Orchestrator remains responsible for the I/O side (approval_fn injected).

Phase 1 keeps the injection seam here; Phase 2b will move this entire module
into the task master package and replace approval_fn with a RequestApproval
decision emission.
"""

from dataclasses import dataclass
from typing import Awaitable, Callable

from src.core.decisions import Allow, Block, Cancel, GateDecision
from src.infra.logging_config import get_logger

logger = get_logger("core.task_gates")


@dataclass
class GateContext:
    """Data needed to evaluate gates. Built by process_task from task + task_ctx."""
    task: dict
    task_ctx: dict


ApprovalFn = Callable[..., Awaitable[bool]]


async def run_gates(
    task: dict,
    task_ctx: dict,
    approval_fn: ApprovalFn,
) -> GateDecision:
    """Evaluate gates in order. First blocking gate wins."""
    is_workflow = task_ctx.get("is_workflow_step", False)

    if task_ctx.get("human_gate"):
        logger.info("human approval gate triggered", task_id=task.get("id"))
        approved = await approval_fn(
            task.get("id"),
            task.get("title", ""),
            task.get("description", "")[:200],
            tier=task.get("tier", "auto"),
            mission_id=task.get("mission_id"),
        )
        if not approved:
            return Cancel(reason="human_gate_rejected")

    if is_workflow or task_ctx.get("human_gate"):
        return Allow()

    try:
        from src.security.risk_assessor import assess_risk, format_risk_assessment
        risk = await assess_risk(
            task_title=task.get("title", ""),
            task_description=task.get("description", ""),
        )
    except Exception as e:
        logger.debug("risk assessment failed (open-circuit allow): %s", e)
        return Allow()

    if not risk.get("needs_approval"):
        return Allow()

    logger.info(
        "risk gate triggered",
        task_id=task.get("id"),
        risk_score=risk.get("score"),
        factors=risk.get("risk_factors"),
    )
    approved = await approval_fn(
        task.get("id"),
        task.get("title", ""),
        format_risk_assessment(risk),
        tier=task.get("tier", "auto"),
        mission_id=task.get("mission_id"),
    )
    if not approved:
        return Cancel(reason="risk_gate_rejected")
    return Allow()
