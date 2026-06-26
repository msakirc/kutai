"""Tier-1 filter over a whole reviewer verdict (2026-06-26).

`ground_review_verdict` runs the per-finding grounding classifier over every
issue in a `fail` verdict, drops the confabulated ones, re-derives the verdict
from survivors, and surfaces the unverifiable-but-blocking survivors as Tier-2
refuter candidates. A resolver callable maps `target_artifact` -> content
(injected here; disk-backed in production).
"""
from __future__ import annotations

import asyncio

from mr_roboto.verify_review_verdict import ground_review_verdict

# Compact real mission-90 artifact content keyed by target_artifact filename.
_ARTIFACTS = {
    "competitive_positioning.md": """---
named_competitors: ["Habitica", "Todoist", "TickTick"]
---
## Landscape
foo
## Value Thesis
bar
## Strengths / Weaknesses
baz
## Our Differentiators
HabitHub incorporates **Daily Errand Tracker** solution.
## Switching Costs & Risks
Migrating involves transferring streak data — real prose here.
## Notes
- a note
""",
    "reverse_pitch.md": "## Headline\nHabitHub Turns Your Tasks Into a Game\n",
    "product_charter.md": "### Solution\n**Boundaries:** per-habit.\n**Guiding principles:** hero metric.\n",
    "market_research_report.md": "# Market Report\nTAM is large, CAGR healthy.\n",
}


def _resolve(name):
    return _ARTIFACTS.get(name)


# The real 14-finding mission-90 verdict, trimmed to the discriminating subset.
_VERDICT = {
    "verdict": "fail",
    "issues": [
        {"target_artifact": "market_research_report.md", "severity": "major",
         "problem": "Check 1 - Claims without evidence: TAM, CAGR, and venture-funding "
                    "figures are presented without any source citations or links."},
        {"target_artifact": "product_charter.md", "severity": "major",
         "problem": 'Check 9 - Solutions completeness: some sections contain placeholder '
                    'text such as "TODO: define boundaries".'},
        {"target_artifact": "reverse_pitch.md", "severity": "major",
         "problem": 'Check 10 - The press-release headline promises a "completely free '
                    'forever" experience, while the charter defines a paid tier.'},
        {"target_artifact": "competitive_positioning.md", "severity": "major",
         "problem": "Check 11 - The document does not contain all six required sections "
                    "(Landscape, Value Thesis, Strengths-Weaknesses, Our Differentiators, "
                    "Switching Costs & Risks, Notes)."},
        {"target_artifact": "competitive_positioning.md", "severity": "major",
         "problem": 'Check 13 - Switching Costs & Risks section is empty or contains only '
                    'a stub such as "[to be added]".'},
    ],
}


def _run(verdict):
    return asyncio.run(ground_review_verdict(review_result=verdict, resolve_artifact=_resolve))


def _checks(issues):
    return [i["problem"].split(" -", 1)[0].strip() for i in issues]


def test_confabulated_findings_dropped():
    res = _run(_VERDICT)
    dropped = _checks(res["dropped"])
    assert "Check 9" in dropped
    assert "Check 10" in dropped
    assert "Check 11" in dropped
    assert "Check 13" in dropped


def test_real_finding_survives():
    res = _run(_VERDICT)
    assert "Check 1" in _checks(res["issues"])


def test_verdict_stays_fail_while_real_major_survives():
    """A real major survivor keeps the verdict fail (routes to producer, not halt)."""
    res = _run(_VERDICT)
    assert res["verdict_class"] == "fail"


def test_unverifiable_survivor_is_tier2_candidate():
    res = _run(_VERDICT)
    assert "Check 1" in _checks(res["tier2_candidates"])


def test_all_confab_dropped_downgrades_to_pass():
    """If every surviving issue is dropped, the verdict downgrades to pass —
    the mission is NOT halted on confabulation alone."""
    only_confab = {
        "verdict": "fail",
        "issues": [
            {"target_artifact": "reverse_pitch.md", "severity": "blocker",
             "problem": 'Check 10 - headline promises "completely free forever".'},
            {"target_artifact": "competitive_positioning.md", "severity": "major",
             "problem": "Check 11 - does not contain all six required sections "
                        "(Landscape, Value Thesis, Strengths-Weaknesses, Our "
                        "Differentiators, Switching Costs & Risks, Notes)."},
        ],
    }
    res = _run(only_confab)
    assert res["verdict_class"] == "pass"
    assert res["ok"] is True
    assert res["issues"] == []


def test_pass_verdict_untouched():
    """A pass verdict skips grounding entirely (only fail halts)."""
    res = asyncio.run(ground_review_verdict(
        review_result={"verdict": "pass", "issues": []}, resolve_artifact=_resolve))
    assert res["verdict_class"] == "pass"
    assert res["dropped"] == []
    assert res["tier2_candidates"] == []


def test_resolver_exception_keeps_issue():
    """A resolver that raises must not drop or crash — the finding is kept."""
    def boom(_name):
        raise RuntimeError("store down")
    res = asyncio.run(ground_review_verdict(
        review_result={"verdict": "fail", "issues": [
            {"target_artifact": "x.md", "severity": "major",
             "problem": 'headline promises "completely free forever".'}]},
        resolve_artifact=boom))
    assert res["verdict_class"] == "fail"
    assert len(res["issues"]) == 1
