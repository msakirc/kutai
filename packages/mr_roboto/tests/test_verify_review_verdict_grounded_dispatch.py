"""mr_roboto dispatch wires Tier-1 grounding into verify_review_verdict.

A fail verdict whose findings are all confabulated downgrades to a completed
(pass) Action — the mission is NOT halted. A fail verdict with a real survivor
stays failed and carries tier2_candidates for the apply-layer refuter. The
KUTAI_VERDICT_VERIFY=off kill switch restores the pure classifier (no grounding).
"""
from __future__ import annotations

import asyncio

import mr_roboto.verify_review_verdict as vv
from mr_roboto import run as mr_run

_CONTENT = {
    "reverse_pitch.md": "## Headline\nHabitHub Turns Your Tasks Into a Game\n",
    "competitive_positioning.md": (
        "## Landscape\na\n## Value Thesis\nb\n## Strengths / Weaknesses\nc\n"
        "## Our Differentiators\nd\n## Switching Costs & Risks\nreal prose\n## Notes\ne\n"
    ),
    "market_research_report.md": "# Report\nTAM large\n",
}


async def _fake_resolve(mission_id, target_artifact):
    return _CONTENT.get(target_artifact)


def _task(review_result):
    return {"id": 0, "mission_id": 90,
            "payload": {"action": "verify_review_verdict", "review_result": review_result}}


def test_all_confab_downgrades_to_completed(monkeypatch):
    monkeypatch.setattr(vv, "_resolve_artifact_content", _fake_resolve)
    verdict = {"verdict": "fail", "issues": [
        {"target_artifact": "reverse_pitch.md", "severity": "blocker",
         "problem": 'headline promises "completely free forever".'},
        {"target_artifact": "competitive_positioning.md", "severity": "major",
         "problem": "does not contain all six required sections (Landscape, Value "
                    "Thesis, Strengths-Weaknesses, Our Differentiators, Switching "
                    "Costs & Risks, Notes)."}]}
    res = asyncio.run(mr_run(_task(verdict)))
    assert res.status == "completed"
    assert res.result["verdict_class"] == "pass"
    assert len(res.result["dropped"]) == 2


def test_real_survivor_stays_failed_with_tier2(monkeypatch):
    monkeypatch.setattr(vv, "_resolve_artifact_content", _fake_resolve)
    verdict = {"verdict": "fail", "issues": [
        {"target_artifact": "reverse_pitch.md", "severity": "major",
         "problem": 'headline promises "completely free forever".'},
        {"target_artifact": "market_research_report.md", "severity": "major",
         "problem": "Check 1 - figures presented without any source citations."}]}
    res = asyncio.run(mr_run(_task(verdict)))
    assert res.status == "failed"
    assert res.result["verdict_class"] == "fail"
    assert len(res.result["tier2_candidates"]) == 1


def test_opt_out_restores_pure_classifier(monkeypatch):
    monkeypatch.setattr(vv, "_resolve_artifact_content", _fake_resolve)
    monkeypatch.setenv("KUTAI_VERDICT_VERIFY", "off")
    verdict = {"verdict": "fail", "issues": [
        {"target_artifact": "reverse_pitch.md", "severity": "blocker",
         "problem": 'headline promises "completely free forever".'}]}
    res = asyncio.run(mr_run(_task(verdict)))
    assert res.status == "failed"
    # No grounding ran: the confab issue is NOT dropped.
    assert "dropped" not in res.result or res.result.get("dropped") == []
    assert len(res.result["issues"]) == 1
