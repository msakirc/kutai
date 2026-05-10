"""Z1 Tier 2 (C2) — verify_competitive_positioning_shape contract tests."""
from __future__ import annotations

import asyncio

from mr_roboto.verify_competitive_positioning_shape import (
    verify_competitive_positioning_shape,
)
from mr_roboto import run as mr_roboto_run


_GOOD = """\
---
_schema_version: "1"
mission_id: 42
named_competitors: [Yelp, TripAdvisor, Google Reviews, Amazon Reviews, Trustpilot]
---

## Landscape
Yelp serves restaurant and local business reviews for general consumers; TripAdvisor focuses on travel and hospitality reviews for travelers; Google Reviews provides cross-category business reviews integrated with search; Amazon Reviews covers product reviews for online shoppers; Trustpilot serves e-commerce business reviews for online purchasers.

## Value Thesis
- **Yelp** offers deep local business coverage with strong community engagement.
- **TripAdvisor** provides travel-specific expertise and booking integration.
- **Google Reviews** leverages search dominance and broad coverage.
- **Amazon Reviews** offers verified purchase validation but only on-platform.
- **Trustpilot** focuses on company reputation management.

## Strengths / Weaknesses
- **Yelp** strong community, weak outside categories.
- **TripAdvisor** deep travel expertise, no general reviews.
- **Google Reviews** search-integrated, lacks fact-checking nuance.
- **Amazon Reviews** purchase-verified, marketplace-only.
- **Trustpilot** subscription-based, e-commerce focus.

## Our Differentiators
TruthRate embodies the **Universal Search & Listings** solution from the charter — review anything regardless of category — combined with the **Dual-Layer Contribution** solution that separates objective facts from subjective opinions. The charter's **Reputation System** solution further sets us apart.

## Switching Costs & Risks
Users accustomed to category-specific platforms may initially feel uncertain navigating without familiar structures. Business owners may resist paying for yet another platform when they already manage multiple sites. The dual-layer system requires learning new mental models for distinguishing facts from opinions.

## Notes
- Yelp business owner dashboard patterns
- Google Reviews fact-checking UI concepts
- Trustpilot subscription tier models
"""


_BAD_EMPTY_COMPETITORS = """\
---
_schema_version: "1"
mission_id: 42
named_competitors: []
---

## Landscape
Some review platforms exist.

## Value Thesis
They offer reviews.

## Strengths / Weaknesses
Pros and cons.

## Our Differentiators
We are universal.

## Switching Costs & Risks
Some friction.

## Notes
Various references.
"""


_BAD_MISSING_SECTIONS = """\
---
_schema_version: "1"
mission_id: 42
named_competitors: [Yelp]
---

## Landscape
Yelp is everywhere.

## Value Thesis
They aggregate reviews.
"""


_BAD_PLACEHOLDER = """\
---
_schema_version: "1"
mission_id: 42
named_competitors: [Yelp]
---

## Landscape
TODO

## Value Thesis
Yelp provides reviews.

## Strengths / Weaknesses
Pros and cons.

## Our Differentiators
We win on universal coverage.

## Switching Costs & Risks
Some friction.

## Notes
References.
"""


_BAD_SCHEMA_VERSION = """\
---
_schema_version: "2"
mission_id: 42
named_competitors: [Yelp]
---

## Landscape
Yelp.

## Value Thesis
Reviews.

## Strengths / Weaknesses
Mixed.

## Our Differentiators
Universal.

## Switching Costs & Risks
Friction.

## Notes
Refs.
"""


def test_good_positioning_passes():
    res = verify_competitive_positioning_shape(positioning_text=_GOOD)
    assert res["ok"] is True, res
    assert len(res["named_competitors"]) == 5
    assert res["schema_version"] == "1"
    assert res["mission_id"] in ("42", 42)
    assert res["placeholders"] == []
    assert res["empty_sections"] == []


def test_empty_named_competitors_rejected():
    res = verify_competitive_positioning_shape(
        positioning_text=_BAD_EMPTY_COMPETITORS
    )
    assert res["ok"] is False
    assert res["named_competitors"] == []


def test_missing_sections_rejected():
    res = verify_competitive_positioning_shape(
        positioning_text=_BAD_MISSING_SECTIONS
    )
    assert res["ok"] is False
    assert "Strengths" in res["missing_sections"]
    assert "Our Differentiators" in res["missing_sections"]
    assert "Switching Costs" in res["missing_sections"]


def test_placeholder_text_rejected():
    res = verify_competitive_positioning_shape(positioning_text=_BAD_PLACEHOLDER)
    assert res["ok"] is False
    assert res["placeholders"], res


def test_wrong_schema_version_rejected():
    res = verify_competitive_positioning_shape(positioning_text=_BAD_SCHEMA_VERSION)
    assert res["ok"] is False
    assert res["schema_version"] == "2"


def test_empty_input_rejected():
    res = verify_competitive_positioning_shape(positioning_text="")
    assert res["ok"] is False
    assert res["named_competitors"] == []


def test_dispatch_via_mr_roboto_run_pass():
    task = {
        "id": 1,
        "mission_id": 42,
        "payload": {
            "action": "verify_competitive_positioning_shape",
            "positioning_text": _GOOD,
        },
    }
    result = asyncio.run(mr_roboto_run(task))
    assert result.status == "completed", result
    assert result.result["ok"] is True


def test_dispatch_via_mr_roboto_run_fail():
    task = {
        "id": 2,
        "mission_id": 42,
        "payload": {
            "action": "verify_competitive_positioning_shape",
            "positioning_text": _BAD_EMPTY_COMPETITORS,
        },
    }
    result = asyncio.run(mr_roboto_run(task))
    assert result.status == "failed", result


def test_schema_version_is_one():
    """Acceptance criterion: artifacts produced by 1.4a carry _schema_version == '1'."""
    res = verify_competitive_positioning_shape(positioning_text=_GOOD)
    assert res["schema_version"] == "1"
