"""Tests for shopping pipeline v2."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

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


@pytest.mark.asyncio
async def test_step_resolve_preserves_site_order_and_caps_per_site_n():
    """Per-site top-N only; no token-overlap filter; site_rank is 1-based."""
    from src.workflows.shopping.pipeline_v2 import step_resolve

    # Fake scraper returns 4 products from site A, 2 from site B
    def _fake_product(name: str, site: str, url: str, price: float):
        # Mimic the scraper dataclass shape (see src/shopping/models.py)
        from types import SimpleNamespace
        return SimpleNamespace(
            name=name, site=site, url=url, price=price,
            original_price=None, rating=None, review_count=None,
            review_snippets=[],
        )

    fake_products = [
        _fake_product("A1", "trendyol", "u1", 100),
        _fake_product("A2", "trendyol", "u2", 110),
        _fake_product("A3", "trendyol", "u3", 120),
        _fake_product("A4", "trendyol", "u4", 130),
        _fake_product("B1", "hepsiburada", "u5", 140),
        _fake_product("B2", "hepsiburada", "u6", 150),
    ]

    with patch(
        "src.workflows.shopping.pipeline_v2._fetch_products",
        new=AsyncMock(return_value=fake_products),
    ):
        cands = await step_resolve("test query", per_site_n=2)

    # Top-2 from each site kept, site_rank is 1-based per site
    trendyol = [c for c in cands if c.site == "trendyol"]
    hepsi = [c for c in cands if c.site == "hepsiburada"]
    assert [c.title for c in trendyol] == ["A1", "A2"]
    assert [c.site_rank for c in trendyol] == [1, 2]
    assert [c.title for c in hepsi] == ["B1", "B2"]
    assert [c.site_rank for c in hepsi] == [1, 2]
