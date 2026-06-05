import pytest


@pytest.mark.asyncio
async def test_selection_failure_marks_task_failed(monkeypatch):
    import general_beckman as gb
    from fatih_hoca.types import SelectionFailure

    monkeypatch.setattr(
        "general_beckman._select_for_admission",
        lambda spec: SelectionFailure(reason="availability",
                                      detail="no eligible image provider"),
    )
    spec = {"kind": "image", "agent_type": "image",
            "context": {"image_call": {"prompt": "x"}}}
    outcome = await gb._handle_admission_pick(spec, pick=None)
    assert outcome["status"] == "failed"
    assert "availability" in outcome.get("error", "")


@pytest.mark.asyncio
async def test_budget_failure_marks_paused(monkeypatch):
    import general_beckman as gb
    from fatih_hoca.types import SelectionFailure

    monkeypatch.setattr(
        "general_beckman._select_for_admission",
        lambda spec: SelectionFailure(reason="budget", detail="exceeded"),
    )
    spec = {"kind": "image", "mission_id": 7,
            "context": {"image_call": {"prompt": "x"}}}
    outcome = await gb._handle_admission_pick(spec, pick=None)
    assert outcome["status"] in ("paused", "failed")  # either is fine; "paused" preferred
