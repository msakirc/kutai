"""Tier-1 deterministic grounding of reviewer findings (2026-06-26).

A single reviewer LLM confabulates findings — invented verbatim quotes,
rubric-example echoes, and false "missing section" claims. Before a `fail`
halts the mission, each finding is checked against its target artifact and
DROPPED when its cited evidence is provably not present/absent as claimed.

Fixtures are the REAL mission-90 `research_review_result.md` finding strings
checked against the REAL on-disk artifacts (`workspace/mission_90/...`).
"""
from __future__ import annotations

from mr_roboto.verify_review_verdict import classify_issue_grounding

# ── Real artifact snippets (verbatim from workspace/mission_90/) ─────────────

COMPETITIVE_POSITIONING = """---
_schema_version: "1"
mission_id: "90"
named_competitors: ["Habitica", "HabitForge", "Hero", "Habits Garden", "Habitify", "Todoist", "TickTick"]
---

## Landscape
The habit-tracking and productivity market spans several overlapping categories.

## Value Thesis
**Habitica** turns daily life into an RPG.

## Strengths / Weaknesses
**Habitica** — strong social features.

## Our Differentiators
HabitHub incorporates **Daily Errand Tracker** solution and a **Gamification Engine**.

## Switching Costs & Risks
Migrating from dedicated habit apps like Habitica involves transferring streak data.

## Notes
- Direct competitor research synthesized from 1.3 direct_competitors_list
"""

CHARTER_SNIPPET = """# HabitHub Product Charter

### Habit Tracking with Streaks
**Boundaries:** Streaks are per-habit, not global.
**Guiding principles:** Make streaks the hero metric.

### Daily Errand Tracking
**Boundaries:** Errands are one-time items.
**Guiding principles:** Keep logging to two taps or fewer.
"""

REVERSE_PITCH = """## Headline
HabitHub Turns Your Daily Tasks and Errands Into a Game You Actually Want to Play

## Sub-head
Stop relying on willpower alone.
"""

MARKET_REPORT = """# Market Research Report

## Market Size
TAM is large and the CAGR is healthy. Venture funding is flowing.

## Competitive Landscape
No competitor offers a unified habit-and-errand flow.
"""


# ── Rule B · fabricated quote (positive evidence claim, span absent) ─────────

def test_check10_free_forever_quote_absent_dropped():
    problem = ('Check 10 - Reverse-pitch alignment: The press-release headline '
               'promises a "completely free forever" experience, while the charter '
               'defines a paid premium tier, creating a direct contradiction.')
    assert classify_issue_grounding(problem, REVERSE_PITCH) == "drop"


def test_check9_todo_placeholder_quote_absent_dropped():
    problem = ('Check 9 - Solutions completeness: Several Solution entries are missing '
               'required `Boundaries` or `Guiding principles` fields, and some sections '
               'contain placeholder text such as "TODO: define boundaries".')
    assert classify_issue_grounding(problem, CHARTER_SNIPPET) == "drop"


def test_check12_generic_benefit_quotes_absent_dropped():
    problem = ('Check 12 - Differentiators lack charter references: The "Our '
               'Differentiators" paragraph lists generic benefits (e.g., "better UI", '
               '"more gamification") but does not cite any Solution ID or name.')
    assert classify_issue_grounding(problem, COMPETITIVE_POSITIONING) == "drop"


def test_check13_stub_quote_absent_dropped():
    problem = ('Check 13 - Switching Costs & Risks section is empty or contains only a '
               'stub such as "[to be added]", providing no substantive prose.')
    assert classify_issue_grounding(problem, COMPETITIVE_POSITIONING) == "drop"


# ── Rule A · false-absence (enumerated sections all present) ─────────────────

def test_check11_all_six_sections_present_dropped():
    problem = ('Check 11 - Missing sections: The document does not contain all six '
               'required sections (Landscape, Value Thesis, Strengths-Weaknesses, '
               'Our Differentiators, Switching Costs & Risks, Notes) and the '
               '`named_competitors` list is empty.')
    assert classify_issue_grounding(problem, COMPETITIVE_POSITIONING) == "drop"


# ── KEEP · real findings with no checkable quote/section go to Tier 2 ────────

def test_check1_missing_citations_unverifiable_kept():
    problem = ('Check 1 - Claims without evidence: TAM, CAGR, and venture-funding '
               'figures are presented without any source citations or links.')
    assert classify_issue_grounding(problem, MARKET_REPORT) == "keep_unverifiable"


def test_check6_no_pricing_column_unverifiable_kept():
    problem = ('Check 6 - Feature matrix and pricing completeness: '
               'competitor_feature_matrix.md is present but only contains raw data; '
               'it lacks a synthesized pricing column.')
    assert classify_issue_grounding(problem, MARKET_REPORT) == "keep_unverifiable"


# ── No over-drop · a genuinely-absent section must SURVIVE ────────────────────

def test_genuinely_missing_section_not_dropped():
    """Rule A must NOT fire when an enumerated section is truly absent."""
    incomplete = "## Landscape\nfoo\n\n## Value Thesis\nbar\n"  # missing Notes etc.
    problem = ('The document does not contain all six required sections '
               '(Landscape, Value Thesis, Strengths-Weaknesses, Our Differentiators, '
               'Switching Costs & Risks, Notes).')
    assert classify_issue_grounding(problem, incomplete) != "drop"


def test_real_quote_present_kept_confirmed():
    """A finding quoting text that IS present is evidenced — keep, no refute."""
    content = 'The headline promises a completely free forever experience to all users.'
    problem = 'The headline promises a "completely free forever" experience but the app charges.'
    assert classify_issue_grounding(problem, content) == "keep_confirmed"


# ── Fail-safe · never drop when the artifact cannot be loaded ─────────────────

def test_unresolvable_artifact_kept():
    problem = 'The headline promises "completely free forever".'
    assert classify_issue_grounding(problem, None) == "keep_unverifiable"


def test_empty_problem_kept():
    assert classify_issue_grounding("", COMPETITIVE_POSITIONING) == "keep_unverifiable"
