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

def test_config_enum_quote_not_evidence_kept():
    """Review MAJOR #1 / mission-90 Check 14: the only quoted span is a mission
    CONFIG enum (`"public_launch"`), not text the model claims to have read in
    the artifact. A single identifier token must NOT be treated as a fabricated
    evidence quote — that would drop a genuine structured blocker."""
    problem = ('Check 14 - Interview-script grounding: No interview transcripts exist '
               '(interview_count == 0) while the mission ambition is set to '
               '"public_launch", triggering a hard-reject condition.')
    content = "# Interview Script\nQ1. Tell me about your day.\nQ2. ...\n"
    assert classify_issue_grounding(problem, content) != "drop"


def test_snake_case_field_quotes_not_evidence_kept():
    """Check 17: quotes are JSON field/enum identifiers (graveyard_count,
    inconclusive) — references, not prose evidence. Must not drop."""
    problem = ('Check 17 - verdict field must NOT be "inconclusive" when '
               '"graveyard_count" >= 3; the report does not provide a '
               '"graveyard_count" field.')
    content = '{"verdict": "graveyard_well_populated", "attempted_solutions": []}'
    assert classify_issue_grounding(problem, content) != "drop"


def test_rule_a_word_collision_does_not_false_present():
    """Review MAJOR #2: unrelated headers that merely CONTAIN the section words
    must not count as 'all sections present'. Loose word-subset matching would
    over-drop a real missing-section finding."""
    doc = ("## Important Landscape Overview Notes\nx\n"
           "## Value Thesis Differentiators Our Edge\ny\n"
           "## Strengths Weaknesses Switching Costs Risks\nz\n")
    problem = ('does not contain all six required sections (Landscape, Value Thesis, '
               'Strengths-Weaknesses, Our Differentiators, Switching Costs & Risks, Notes).')
    assert classify_issue_grounding(problem, doc) != "drop"


def test_unresolvable_artifact_kept():
    problem = 'The headline promises "completely free forever".'
    assert classify_issue_grounding(problem, None) == "keep_unverifiable"


def test_empty_problem_kept():
    assert classify_issue_grounding("", COMPETITIVE_POSITIONING) == "keep_unverifiable"


# ── Rule C · false "missing falsification triple" (m90 567426) ───────────────
# The reviewer (3.11) re-derives triple PRESENCE from prose and confabulates
# "missing" for requirements that provably carry all three fields. Presence is
# a mechanical fact already hard-gated on the producers (verify_falsification_
# present on 3.1/3.2/3.3/3.7); Rule C grounds the reviewer's absence claim
# against the requirements_spec.md table and drops it when the triple columns
# are populated. Quality axes (vague critical validation, contradictions) are
# untouched — only the false ABSENCE claim is dropped.

REQ_SPEC_FULL = """## Functional Requirements

| ID | Title | Description | Priority | Category | Risk | Validation Method | Falsification Signal |
|-----|-------|-------------|----------|----------|------|-------------------|----------------------|
| FR-001 | Core Habit Tracking | Users track habits | High | Core | medium | Manual testing of habit CRUD operations | Habits not saving correctly |
| FR-003 | Task Management | Users manage tasks | Medium | Core | high | Integration tests of task lifecycle | Unable to create or manage tasks |
| FR-008 | Upsell Prompts | Power-user upsell | Low | Growth | low | Analytics verification of upsell triggers | Upsell not shown to power users |
"""


def test_ruleC_drops_false_missing_triple_when_columns_populated():
    problem = ("Missing falsification triples (risk_if_wrong, validation_method, "
               "falsification_signal) for multiple requirements (FR-003, FR-008).")
    assert classify_issue_grounding(problem, REQ_SPEC_FULL) == "drop"


def test_ruleC_drops_false_empty_table_claim_when_rows_present():
    problem = ("Functional Requirements table is empty (no rows). All FRs are "
               "missing the falsification triple.")
    assert classify_issue_grounding(problem, REQ_SPEC_FULL) == "drop"


def test_ruleC_keeps_when_a_triple_cell_is_genuinely_empty():
    spec = REQ_SPEC_FULL.replace(
        "| Integration tests of task lifecycle |", "|  |")
    problem = "Missing falsification triple (validation_method) for FR-003."
    assert classify_issue_grounding(problem, spec) != "drop"


def test_ruleC_keeps_nfr_finding_when_only_fr_table_parsed():
    # D1: NFRs render as prose (no triple table); the FR table proving out is
    # NOT evidence about NFRs. A finding naming NFR IDs absent from the parsed
    # table must NOT be dropped.
    problem = ("Non-functional requirements (NFR-001, NFR-002) are missing the "
               "falsification triple (risk_if_wrong, validation_method, falsification_signal).")
    assert classify_issue_grounding(problem, REQ_SPEC_FULL) != "drop"


def test_ruleC_keeps_unrelated_absence_with_incidental_falsification_word():
    # D2: the absence marker binds to a DIFFERENT section; the falsification
    # word is incidental. Must NOT drop.
    p1 = ("Missing a traceability matrix section; several falsification signals "
          "could also be stronger.")
    p2 = ("The spec is missing a revision-history section. Falsification triples "
          "themselves look complete.")
    assert classify_issue_grounding(p1, REQ_SPEC_FULL) != "drop"
    assert classify_issue_grounding(p2, REQ_SPEC_FULL) != "drop"


def test_ruleC_does_not_fire_without_a_triple_table():
    # A falsification-absence claim against an artifact that has no requirement
    # triple table → no mechanical proof of presence → must NOT drop.
    problem = "Missing falsification triples for the requirements."
    assert classify_issue_grounding(problem, COMPETITIVE_POSITIONING) != "drop"


def test_ruleC_does_not_drop_specificity_quality_claim():
    # "lack SPECIFIC falsification signals" judges QUALITY (specificity), not
    # presence — a legit reviewer axis. Rule C must not drop it even though the
    # signals are present (m90 567426 finding [4] over-drop regression).
    problem = ("Falsification signals for the requirements lack specific, "
               "measurable thresholds to be adequately testable.")
    assert classify_issue_grounding(problem, REQ_SPEC_FULL) != "drop"


def test_ruleC_leaves_non_falsification_findings_to_other_rules():
    # A vague-critical-validation QUALITY finding is not an absence claim about
    # presence → Rule C must not touch it (other rules / Tier-2 handle it).
    problem = ("Validation method for FR-003 is too generic to test a "
               "critical-risk requirement adequately.")
    assert classify_issue_grounding(problem, REQ_SPEC_FULL) != "drop"
