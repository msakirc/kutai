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
            name=name, source=site, url=url, discounted_price=price,
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


@pytest.mark.asyncio
async def test_step_resolve_works_with_real_product_dataclass():
    """Regression: real Product model (source/discounted_price) must flow through."""
    from src.shopping.models import Product
    from src.workflows.shopping.pipeline_v2 import step_resolve

    products = [
        Product(name="EQ6", url="u1", source="hepsiburada",
                discounted_price=24745.0, original_price=26000.0,
                rating=4.5, review_count=312),
        Product(name="EQ3 part", url="u2", source="amazon_tr",
                discounted_price=4800.0, original_price=None,
                rating=5.0, review_count=3),
    ]
    with patch(
        "src.workflows.shopping.pipeline_v2._fetch_products",
        new=AsyncMock(return_value=products),
    ):
        cands = await step_resolve("siemens", per_site_n=3)

    assert len(cands) == 2
    assert {c.site for c in cands} == {"hepsiburada", "amazon_tr"}
    eq6 = next(c for c in cands if c.title == "EQ6")
    assert eq6.price == 24745.0
    assert eq6.original_price == 26000.0
    assert eq6.rating == 4.5
    assert eq6.site_rank == 1


def test_prompt_templates_exist_and_have_required_placeholders():
    from src.workflows.shopping import prompts_v2

    # GROUPING_PROMPT must accept the candidates JSON block
    assert "{candidates_json}" in prompts_v2.GROUPING_PROMPT
    # It must instruct the model to flag accessories
    assert "accessor" in prompts_v2.GROUPING_PROMPT.lower() or "part" in prompts_v2.GROUPING_PROMPT.lower()

    # SYNTHESIS_PROMPT must accept the representative title and snippets
    assert "{representative_title}" in prompts_v2.SYNTHESIS_PROMPT
    assert "{review_snippets_json}" in prompts_v2.SYNTHESIS_PROMPT
    # Must instruct output JSON schema
    assert "praise" in prompts_v2.SYNTHESIS_PROMPT
    assert "complaints" in prompts_v2.SYNTHESIS_PROMPT
    assert "insufficient_data" in prompts_v2.SYNTHESIS_PROMPT
