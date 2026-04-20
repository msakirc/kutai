"""Tests for shopping pipeline v2."""
from __future__ import annotations

import pytest

from src.workflows.shopping.pipeline_v2 import (
    Candidate,
    ProductGroup,
    ReviewSynthesis,
)


def test_dataclass_shapes():
    c = Candidate(
        title="Siemens EQ.6 Plus",
        site="hepsiburada",
        site_rank=1,
        price=24745.0,
        original_price=None,
        url="https://example.com",
        rating=4.5,
        review_count=312,
        review_snippets=["Köpük güzel"],
    )
    assert c.site_rank == 1

    g = ProductGroup(
        representative_title="Siemens EQ.6 Plus",
        member_indices=[0, 2],
        is_accessory_or_part=False,
        prominence=1.5,
    )
    assert g.member_indices == [0, 2]

    s = ReviewSynthesis(
        praise=["Köpük güzel"],
        complaints=[],
        red_flags=[],
        insufficient_data=False,
    )
    assert s.praise == ["Köpük güzel"]
