"""Z1 Tier 1 — verify_charter_shape contract tests.

Locks the paraflow-shape product_charter.md validator. Pure unit tests:
no LLM, no DB, no async.
"""
from __future__ import annotations

import asyncio

from mr_roboto.verify_charter_shape import verify_charter_shape
from mr_roboto import run as mr_roboto_run


_GOOD_CHARTER = """\
# Product Charter — TruthRate

## 1) Product Positioning

TruthRate is a universal review platform for everyday consumers and business
owners seeking transparent information about anything in the world. Unlike
traditional category-bounded review platforms, we let users review and share
verified facts about any product, service, or business — separating
subjective opinions from objective facts.

## 2) Brand Keywords

- **Universal** — Users can review anything without category limitations.
- **Transparent** — Clear separation between opinions and factual claims.
- **Credible** — Reputation systems for reviewers and fact-checking.
- **Privacy-First** — Username-based authentication, strong data protection.
- **Fair** — Business owners can claim listings and respond.

## 3) Core Problem / JTBD

When people need to make purchasing decisions, they struggle to find
trustworthy information that distinguishes opinions from verifiable facts.
Existing platforms blur the line between subjective and objective signals.

## 4) Goals & Mission

- **Mission:** Create a universal platform where truth and transparency
  empower better decisions.

- **Desired Outcomes:**
  - Users feel confident making decisions
  - Business owners engage with feedback
  - Contributors build reputation
  - Platform maintains information quality

## 5) Solutions We Own

### Universal Search & Discovery
- **What it solves:** Users can find and review literally anything.
- **Typical path:** Search by name → existing reviews/facts → contribute.
- **Outcome for the user:** Credible information about anything.
- **Boundaries:** No category restrictions; admin moderates abuse manually.
- **Guiding principles:** Flexibility / Comprehensive coverage

### Dual-Layer Information System
- **What it solves:** Separates opinions from objective facts.
- **Typical path:** Submit review or contribute fact with evidence.
- **Outcome for the user:** Quickly assess sentiment AND verifiable info.
- **Boundaries:** Facts require user guarantee of truth; no auto verification.
- **Guiding principles:** Clarity / Truth-seeking / Transparency

### Credibility & Reputation System
- **What it solves:** Builds trust through helpfulness metrics.
- **Typical path:** Contribute → community votes helpfulness → reputation.
- **Outcome for the user:** Trust high-reputation contributors.
- **Boundaries:** Reputation is non-transferrable; no badges-for-money.
- **Guiding principles:** Meritocratic / Quality-focused
"""


_BAD_CHARTER_MISSING_BOUNDARIES = """\
# Product Charter

## Product Positioning

A platform.

## Brand Keywords

- **One** — first
- **Two** — second
- **Three** — third
- **Four** — fourth
- **Five** — fifth

## Core Problem / JTBD

Some pain.

## Goals & Mission

- **Mission:** Win.

- **Desired Outcomes:**
  - Outcome A
  - Outcome B

## Solutions We Own

### Solution Alpha
- **What it solves:** A problem.
- **Typical path:** A path.
- **Outcome for the user:** A win.
- **Guiding principles:** Speed

### Solution Beta
- **What it solves:** Another problem.
- **Typical path:** Another path.
- **Outcome for the user:** Another win.
- **Boundaries:** Some boundary.
- **Guiding principles:** Quality

### Solution Gamma
- **What it solves:** Third problem.
- **Typical path:** Third path.
- **Outcome for the user:** Third win.
- **Boundaries:** Another boundary.
- **Guiding principles:** Trust
"""


_BAD_CHARTER_PLACEHOLDER = _GOOD_CHARTER.replace(
    "Universal Search & Discovery",
    "TODO: Universal Search & Discovery",
)


_BAD_CHARTER_TOO_FEW_SOLUTIONS = """\
## Product Positioning
A.
## Brand Keywords
- **A** — a
- **B** — b
- **C** — c
- **D** — d
- **E** — e
## Core Problem / JTBD
P.
## Goals & Mission
- **Mission:** M.
- **Desired Outcomes:**
  - x
## Solutions We Own
### Sole
- **What it solves:** s
- **Typical path:** t
- **Outcome for the user:** o
- **Boundaries:** b
- **Guiding principles:** g
### Second
- **What it solves:** s
- **Typical path:** t
- **Outcome for the user:** o
- **Boundaries:** b
- **Guiding principles:** g
"""


def test_good_charter_passes():
    res = verify_charter_shape(charter_text=_GOOD_CHARTER)
    assert res["ok"] is True, res
    assert res["solution_count"] == 3
    assert res["brand_keyword_count"] >= 5
    assert res["missing_sections"] == []
    assert res["solution_problems"] == []
    assert res["placeholders"] == []


def test_charter_missing_boundaries_rejected():
    res = verify_charter_shape(charter_text=_BAD_CHARTER_MISSING_BOUNDARIES)
    assert res["ok"] is False
    # Solution Alpha lacks Boundaries.
    names_with_problems = {p["name"] for p in res["solution_problems"]}
    assert "Solution Alpha" in names_with_problems
    alpha = next(p for p in res["solution_problems"] if p["name"] == "Solution Alpha")
    assert "Boundaries" in alpha["missing_subfields"]


def test_charter_with_placeholder_rejected():
    res = verify_charter_shape(charter_text=_BAD_CHARTER_PLACEHOLDER)
    assert res["ok"] is False
    assert res["placeholders"], res
    assert any("TODO" in p for p in res["placeholders"])


def test_charter_too_few_solutions_rejected():
    res = verify_charter_shape(
        charter_text=_BAD_CHARTER_TOO_FEW_SOLUTIONS, min_solutions=3
    )
    assert res["ok"] is False
    assert res["solution_count"] == 2


def test_empty_charter_rejected():
    res = verify_charter_shape(charter_text="")
    assert res["ok"] is False
    assert res["error"] == "empty charter"
    assert set(res["missing_sections"]) == {
        "Product Positioning",
        "Brand Keywords",
        "Core Problem",
        "Goals & Mission",
        "Solutions We Own",
    }


def test_dispatch_via_mr_roboto_run():
    task = {
        "id": 1,
        "mission_id": 1,
        "payload": {
            "action": "verify_charter_shape",
            "charter_text": _GOOD_CHARTER,
        },
    }
    result = asyncio.run(mr_roboto_run(task))
    assert result.status == "completed", result
    assert result.result["ok"] is True


def test_dispatch_failure_via_mr_roboto_run():
    task = {
        "id": 2,
        "mission_id": 1,
        "payload": {
            "action": "verify_charter_shape",
            "charter_text": _BAD_CHARTER_PLACEHOLDER,
        },
    }
    result = asyncio.run(mr_roboto_run(task))
    assert result.status == "failed"
    assert "verify_charter_shape" in (result.error or "")
