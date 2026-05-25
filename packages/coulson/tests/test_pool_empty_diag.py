# test_pool_empty_diag.py
"""WS-1 forensics plumbing in coulson (handoff 2026-05-25).

``pick_for_iter`` must forward a caller-supplied ``diag_out`` dict to
fatih_hoca.select() so the empty-pool reason survives, and
``record_pool_empty_forensics`` must fold that diag into the
admission_violations.snapshot_summary instead of writing it blank.
"""
from __future__ import annotations

import asyncio

import fatih_hoca
from coulson.dispatch_helpers import pick_for_iter, record_pool_empty_forensics


def test_pick_for_iter_forwards_diag_out(monkeypatch):
    """When select() empties the pool, the diag it populates must reach the
    caller through diag_out (pick_for_iter returns None unchanged)."""
    def fake_select(**kw):
        d = kw.get("diag_out")
        assert d is not None, "pick_for_iter did not forward diag_out to select()"
        d["empty_stage"] = "eligibility"
        d["filter_reasons"] = {"no_function_calling": 3}
        d["fc_capable_rejected"] = {}
        return None

    monkeypatch.setattr(fatih_hoca, "select", fake_select)

    class _Reqs:
        effective_task = "researcher"
        primary_capability = "researcher"
        agent_type = "researcher"
        difficulty = 5
        needs_thinking = False
        needs_function_calling = True
        needs_vision = False
        local_only = False
        prefer_speed = False
        prefer_quality = False
        prefer_local = False
        estimated_input_tokens = 1000
        estimated_output_tokens = 1000
        priority = 5
        exclude_models: list = []

    diag: dict = {}
    pick = pick_for_iter(
        reqs=_Reqs(), task={}, failures=[], iteration=0,
        remaining_budget=0.0, diag_out=diag,
    )

    assert pick is None
    assert diag["empty_stage"] == "eligibility"
    assert diag["filter_reasons"]["no_function_calling"] == 3


def test_record_pool_empty_forensics_writes_diag_into_summary(monkeypatch):
    """The diag's fc_capable_rejected + histogram must land in the
    admission_violations snapshot_summary string."""
    captured: dict = {}

    async def fake_record(**kw):
        captured.update(kw)

    monkeypatch.setattr(
        "src.infra.admission_forensics.record_admission_violation",
        fake_record,
    )

    diag = {
        "empty_stage": "eligibility",
        "eligible_count": 0,
        "filter_reasons": {"no_function_calling": 30, "daily_exhausted(gemini)": 2},
        "fc_capable_rejected": {"gemini/flash": "daily_exhausted(gemini)"},
    }

    asyncio.run(record_pool_empty_forensics(
        task={"id": 166110, "agent_type": "researcher"},
        failures=[], difficulty=5, iteration_n=0, diag=diag,
    ))

    summary = captured.get("snapshot_summary", "")
    assert "eligibility" in summary
    assert "daily_exhausted" in summary
    assert "gemini/flash" in summary
    assert captured.get("task_id") == 166110
