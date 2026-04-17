import pytest
from unittest.mock import AsyncMock, patch
from src.core.task_gates import run_gates
from src.core.decisions import Allow, Cancel


@pytest.mark.asyncio
async def test_no_gates_returns_allow():
    """Task with no gate flags and low-risk content passes through."""
    task = {"id": 1, "title": "list files", "description": ""}
    ctx = {}
    approval = AsyncMock(return_value=True)
    decision = await run_gates(task, ctx, approval_fn=approval)
    assert isinstance(decision, Allow)
    approval.assert_not_called()


@pytest.mark.asyncio
async def test_human_gate_approved_returns_allow():
    task = {"id": 1, "title": "x", "description": "y", "tier": "auto"}
    ctx = {"human_gate": True}
    approval = AsyncMock(return_value=True)
    decision = await run_gates(task, ctx, approval_fn=approval)
    assert isinstance(decision, Allow)
    approval.assert_called_once()


@pytest.mark.asyncio
async def test_human_gate_rejected_returns_cancel():
    task = {"id": 1, "title": "x", "description": "y"}
    ctx = {"human_gate": True}
    approval = AsyncMock(return_value=False)
    decision = await run_gates(task, ctx, approval_fn=approval)
    assert isinstance(decision, Cancel)
    assert "human" in decision.reason.lower()


@pytest.mark.asyncio
async def test_risk_gate_for_dangerous_task_requires_approval():
    task = {"id": 1, "title": "rm -rf /", "description": "wipe the disk"}
    ctx = {}
    approval = AsyncMock(return_value=False)
    decision = await run_gates(task, ctx, approval_fn=approval)
    assert isinstance(decision, Cancel)
    assert "risk" in decision.reason.lower()


@pytest.mark.asyncio
async def test_workflow_step_skips_risk_gate():
    """Workflow steps are pre-approved via the workflow definition; risk gate skipped."""
    task = {"id": 1, "title": "rm temp files", "description": ""}
    ctx = {"is_workflow_step": True}
    approval = AsyncMock(return_value=False)
    decision = await run_gates(task, ctx, approval_fn=approval)
    assert isinstance(decision, Allow)
    approval.assert_not_called()


@pytest.mark.asyncio
async def test_risk_assessor_exception_opens_circuit():
    """If risk assessment raises, gate defaults to Allow (open-circuit)."""
    task = {"id": 1, "title": "anything", "description": ""}
    ctx = {}
    approval = AsyncMock(return_value=False)
    with patch("src.security.risk_assessor.assess_risk", side_effect=RuntimeError("boom")):
        decision = await run_gates(task, ctx, approval_fn=approval)
    assert isinstance(decision, Allow)
    approval.assert_not_called()
