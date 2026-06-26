"""Tier-2 refuter wiring at the apply layer (2026-06-26).

`_apply_review_verdict` defers a `fail` with Tier-2 candidates to an admitted
refuter child instead of routing immediately. The refuter resume drops the
findings the refuter could not support and re-derives: survivors route to
producers; none survive → the reviewer completes (pass). A refuter outage keeps
all candidates (the safety halt must not be silently disabled).
"""
from __future__ import annotations

import pytest

import general_beckman.apply as apply
import general_beckman.posthook_continuations as pc
from general_beckman.result_router import PostHookVerdict


# ── _apply_review_verdict defers to the refuter when candidates exist ─────────

@pytest.mark.asyncio
async def test_fail_with_tier2_spawns_refuter_and_does_not_route(monkeypatch):
    enq = {}
    routed = {"called": False}

    async def fake_resolve(mid, name):
        return "# artifact\nreal content here\n"

    async def fake_enqueue(spec, **kw):
        enq["spec"] = spec
        enq["kw"] = kw

    import mr_roboto.verify_review_verdict as vv
    monkeypatch.setattr(vv, "_resolve_artifact_content", fake_resolve)
    monkeypatch.setattr(apply, "enqueue", fake_enqueue)

    async def fake_route(**kw):
        routed["called"] = True
        return {"routed": [], "escalated": False}
    monkeypatch.setattr("general_beckman.review_routing.route_review_failure", fake_route)

    source = {"id": 77, "mission_id": 90, "context": "{}", "result": "{}"}
    ctx = {"workflow_step_id": "1.13", "_pending_posthooks": ["verify_review_verdict"]}
    raw = {
        "verdict_class": "fail",
        "issues": [{"target_artifact": "market_research_report.md",
                    "severity": "major", "problem": "Check 1 - lacks citations."}],
        "tier2_candidates": [{"target_artifact": "market_research_report.md",
                              "severity": "major", "problem": "Check 1 - lacks citations."}],
    }
    verdict = PostHookVerdict(source_task_id=77, kind="verify_review_verdict",
                              passed=False, raw=raw)
    await apply._apply_review_verdict(source=source, ctx=ctx,
                                      pending=["verify_review_verdict"], verdict=verdict)

    assert "spec" in enq, "must spawn a refuter child"
    assert enq["kw"]["on_complete"] == "posthook.verdict_verify.resume"
    assert enq["kw"]["on_error"] == "posthook.verdict_verify.resume_err"
    assert routed["called"] is False, "must NOT route while the refuter is in flight"


# ── refuter resume — drop unsupported, route survivors ───────────────────────

@pytest.mark.asyncio
async def test_resume_drops_unsupported_keeps_supported(monkeypatch):
    finished = {}

    async def fake_finish(source_task_id, reviewer_id, mission_id, final_issues):
        finished["final"] = final_issues

    async def fake_resolve(mid, name):
        return "# report\nTAM large no citations\n"

    monkeypatch.setattr(pc, "_finish_review_after_refuter", fake_finish)
    import mr_roboto.verify_review_verdict as vv
    monkeypatch.setattr(vv, "_resolve_artifact_content", fake_resolve)

    state = {
        "source_task_id": 77, "reviewer_id": "1.13", "mission_id": 90,
        "kept_issues": [
            {"target_artifact": "a.md", "problem": "Check 1 - real gap."},
            {"target_artifact": "b.md", "problem": "Check 8 - confabulated."},
        ],
        "candidates": [
            {"target_artifact": "a.md", "problem": "Check 1 - real gap."},
            {"target_artifact": "b.md", "problem": "Check 8 - confabulated."},
        ],
    }
    # Refuter supports Check 1 (index 0), refutes Check 8 (index 1).
    result = {"result": {"content":
        '{"verdicts": [{"index": 0, "status": "SUPPORTED"},'
        ' {"index": 1, "status": "UNSUPPORTED"}]}'}}
    await pc._verdict_verify_resume(999, result, state)

    probs = [i["problem"] for i in finished["final"]]
    assert "Check 1 - real gap." in probs
    assert "Check 8 - confabulated." not in probs


@pytest.mark.asyncio
async def test_resume_all_unsupported_yields_empty_final(monkeypatch):
    finished = {}

    async def fake_finish(source_task_id, reviewer_id, mission_id, final_issues):
        finished["final"] = final_issues

    async def fake_resolve(mid, name):
        return "content"
    monkeypatch.setattr(pc, "_finish_review_after_refuter", fake_finish)
    import mr_roboto.verify_review_verdict as vv
    monkeypatch.setattr(vv, "_resolve_artifact_content", fake_resolve)

    state = {
        "source_task_id": 77, "reviewer_id": "1.13", "mission_id": 90,
        "kept_issues": [{"target_artifact": "b.md", "problem": "Check 8 - confab."}],
        "candidates": [{"target_artifact": "b.md", "problem": "Check 8 - confab."}],
    }
    result = {"result": {"content": '{"verdicts": [{"index": 0, "status": "UNSUPPORTED"}]}'}}
    await pc._verdict_verify_resume(999, result, state)
    assert finished["final"] == []


@pytest.mark.asyncio
async def test_resume_garbage_output_keeps_all(monkeypatch):
    """A refuter that returns garbage (outage) must keep ALL candidates."""
    finished = {}

    async def fake_finish(source_task_id, reviewer_id, mission_id, final_issues):
        finished["final"] = final_issues

    async def fake_resolve(mid, name):
        return "content"
    monkeypatch.setattr(pc, "_finish_review_after_refuter", fake_finish)
    import mr_roboto.verify_review_verdict as vv
    monkeypatch.setattr(vv, "_resolve_artifact_content", fake_resolve)

    state = {
        "source_task_id": 77, "reviewer_id": "1.13", "mission_id": 90,
        "kept_issues": [{"target_artifact": "b.md", "problem": "Check 8."}],
        "candidates": [{"target_artifact": "b.md", "problem": "Check 8."}],
    }
    result = {"result": {"content": "the model rambled, no json"}}
    await pc._verdict_verify_resume(999, result, state)
    assert len(finished["final"]) == 1


@pytest.mark.asyncio
async def test_resume_err_keeps_all(monkeypatch):
    finished = {}

    async def fake_finish(source_task_id, reviewer_id, mission_id, final_issues):
        finished["final"] = final_issues
    monkeypatch.setattr(pc, "_finish_review_after_refuter", fake_finish)

    state = {
        "source_task_id": 77, "reviewer_id": "1.13", "mission_id": 90,
        "kept_issues": [{"target_artifact": "b.md", "problem": "Check 8."}],
        "candidates": [{"target_artifact": "b.md", "problem": "Check 8."}],
    }
    await pc._verdict_verify_resume_err(999, {"error": "no candidates"}, state)
    assert len(finished["final"]) == 1
