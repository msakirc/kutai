"""Tests for shopping pipeline v2."""
from __future__ import annotations

import asyncio
import json
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


@pytest.mark.asyncio
async def test_step_group_parses_llm_response_and_filters_accessories():
    """Grouping LLM output is parsed; accessory-flagged groups kept with flag set."""
    from src.workflows.shopping.pipeline_v2 import Candidate, step_group

    cands = [
        Candidate(title="Siemens EQ.6 S100 coffee machine", site="hepsiburada",
                  site_rank=1, price=24745.0, original_price=None,
                  url="h1", rating=4.5, review_count=100, review_snippets=[]),
        Candidate(title="DL-Pro demleme ünitesi Siemens EQ.3 S100",
                  site="amazon_tr", site_rank=1, price=4800.0,
                  original_price=None, url="a1", rating=5.0,
                  review_count=3, review_snippets=[]),
    ]
    fake_llm_response = {
        "content": (
            '{"groups": ['
            '  {"representative_title": "Siemens EQ.6 S100", '
            '   "member_indices": [0], "is_accessory_or_part": false},'
            '  {"representative_title": "Siemens EQ.3 brewing unit", '
            '   "member_indices": [1], "is_accessory_or_part": true}'
            ']}'
        ),
        "model": "fake-model",
        "cost": 0.0,
    }
    with patch(
        "src.workflows.shopping.pipeline_v2._grouping_llm_call",
        new=AsyncMock(return_value=fake_llm_response),
    ):
        groups = await step_group(cands)

    assert len(groups) == 2
    assert groups[0].representative_title.startswith("Siemens EQ.6")
    assert groups[0].is_accessory_or_part is False
    assert groups[1].is_accessory_or_part is True


@pytest.mark.asyncio
async def test_step_group_fallback_on_llm_failure():
    """LLM failure → each site's top-1 becomes its own group (trust-sites fallback)."""
    from src.workflows.shopping.pipeline_v2 import Candidate, step_group

    cands = [
        Candidate(title="A1", site="trendyol", site_rank=1, price=100,
                  original_price=None, url="u1", rating=None,
                  review_count=None, review_snippets=[]),
        Candidate(title="A2", site="trendyol", site_rank=2, price=110,
                  original_price=None, url="u2", rating=None,
                  review_count=None, review_snippets=[]),
        Candidate(title="B1", site="hepsiburada", site_rank=1, price=140,
                  original_price=None, url="u3", rating=None,
                  review_count=None, review_snippets=[]),
    ]
    with patch(
        "src.workflows.shopping.pipeline_v2._grouping_llm_call",
        new=AsyncMock(side_effect=RuntimeError("LLM boom")),
    ):
        groups = await step_group(cands)

    # One group per site, containing that site's rank-1 candidate only
    assert len(groups) == 2
    titles = {g.representative_title for g in groups}
    assert titles == {"A1", "B1"}


@pytest.mark.asyncio
async def test_step_group_fallback_on_malformed_json():
    from src.workflows.shopping.pipeline_v2 import Candidate, step_group

    cands = [
        Candidate(title="X1", site="s1", site_rank=1, price=1, original_price=None,
                  url="u", rating=None, review_count=None, review_snippets=[]),
    ]
    bad = {"content": "not json at all", "model": "m", "cost": 0}
    with patch(
        "src.workflows.shopping.pipeline_v2._grouping_llm_call",
        new=AsyncMock(return_value=bad),
    ):
        groups = await step_group(cands)

    assert len(groups) == 1
    assert groups[0].representative_title == "X1"


@pytest.mark.asyncio
async def test_step_synthesize_parses_full_response():
    from src.workflows.shopping.pipeline_v2 import (
        Candidate, ProductGroup, step_synthesize_reviews,
    )

    cands = [
        Candidate(title="T", site="s", site_rank=1, price=100, original_price=None,
                  url="u", rating=None, review_count=None,
                  review_snippets=["köpük harika", "sessiz"]),
    ]
    group = ProductGroup(
        representative_title="T",
        member_indices=[0],
        is_accessory_or_part=False,
        prominence=1.0,
    )
    fake_llm = {
        "content": (
            '{"praise": ["köpük iyi"], '
            ' "complaints": ["pahalı"], '
            ' "red_flags": [], '
            ' "insufficient_data": false}'
        ),
        "model": "m", "cost": 0,
    }
    with patch(
        "src.workflows.shopping.pipeline_v2._synthesis_llm_call",
        new=AsyncMock(return_value=fake_llm),
    ):
        syn = await step_synthesize_reviews(group, cands)

    assert syn.insufficient_data is False
    assert syn.praise == ["köpük iyi"]
    assert syn.complaints == ["pahalı"]
    assert syn.red_flags == []


@pytest.mark.asyncio
async def test_step_synthesize_short_circuits_when_no_snippets():
    """No snippets at all → return insufficient_data without calling LLM."""
    from src.workflows.shopping.pipeline_v2 import (
        Candidate, ProductGroup, step_synthesize_reviews,
    )

    cands = [
        Candidate(title="T", site="s", site_rank=1, price=1, original_price=None,
                  url="u", rating=None, review_count=None, review_snippets=[]),
    ]
    group = ProductGroup("T", [0], False, 1.0)

    called = AsyncMock()
    with patch("src.workflows.shopping.pipeline_v2._synthesis_llm_call", new=called):
        syn = await step_synthesize_reviews(group, cands)

    assert syn.insufficient_data is True
    assert syn.praise == [] and syn.complaints == [] and syn.red_flags == []
    called.assert_not_called()


@pytest.mark.asyncio
async def test_step_synthesize_insufficient_data_flag_from_llm():
    from src.workflows.shopping.pipeline_v2 import (
        Candidate, ProductGroup, step_synthesize_reviews,
    )

    cands = [
        Candidate(title="T", site="s", site_rank=1, price=1, original_price=None,
                  url="u", rating=None, review_count=None,
                  review_snippets=["tek yorum"]),
    ]
    group = ProductGroup("T", [0], False, 1.0)
    fake_llm = {
        "content": ('{"praise":[],"complaints":[],"red_flags":[],"insufficient_data":true}'),
        "model": "m", "cost": 0,
    }
    with patch(
        "src.workflows.shopping.pipeline_v2._synthesis_llm_call",
        new=AsyncMock(return_value=fake_llm),
    ):
        syn = await step_synthesize_reviews(group, cands)

    assert syn.insufficient_data is True


@pytest.mark.asyncio
async def test_step_synthesize_failure_returns_insufficient_data():
    from src.workflows.shopping.pipeline_v2 import (
        Candidate, ProductGroup, step_synthesize_reviews,
    )

    cands = [
        Candidate(title="T", site="s", site_rank=1, price=1, original_price=None,
                  url="u", rating=None, review_count=None,
                  review_snippets=["x"]),
    ]
    group = ProductGroup("T", [0], False, 1.0)
    with patch(
        "src.workflows.shopping.pipeline_v2._synthesis_llm_call",
        new=AsyncMock(side_effect=RuntimeError("LLM boom")),
    ):
        syn = await step_synthesize_reviews(group, cands)
    assert syn.insufficient_data is True


def test_select_groups_drops_accessories_and_applies_50pct_rule():
    from src.workflows.shopping.pipeline_v2 import ProductGroup, select_groups

    groups = [
        ProductGroup("Dominant",  [0, 1], False, prominence=2.0),
        ProductGroup("Close runner", [2], False, prominence=1.2),   # 60% of top → keep
        ProductGroup("Weak runner", [3], False, prominence=0.8),    # 40% of top → drop
        ProductGroup("Accessory", [4], True, prominence=99.0),      # always drop
    ]

    kept_named = select_groups(groups, max_groups=2)
    assert [g.representative_title for g in kept_named] == ["Dominant", "Close runner"]

    kept_cat = select_groups(groups, max_groups=3)
    assert [g.representative_title for g in kept_cat] == ["Dominant", "Close runner"]


def test_select_groups_empty_input():
    from src.workflows.shopping.pipeline_v2 import select_groups
    assert select_groups([], max_groups=2) == []


def test_format_group_card_full_output():
    from src.workflows.shopping.pipeline_v2 import (
        Candidate, ProductGroup, ReviewSynthesis, format_group_card,
    )
    cands = [
        Candidate(title="Siemens EQ.6 Plus S100", site="hepsiburada",
                  site_rank=1, price=24745.0, original_price=None,
                  url="https://h.com/1", rating=4.5, review_count=312,
                  review_snippets=[]),
        Candidate(title="Siemens EQ.6 Plus S100", site="akakce",
                  site_rank=1, price=25499.0, original_price=None,
                  url="https://a.com/2", rating=None, review_count=None,
                  review_snippets=[]),
    ]
    group = ProductGroup("Siemens EQ.6 Plus S100", [0, 1], False, prominence=2.0)
    syn = ReviewSynthesis(
        praise=["Köpük kalitesi iyi", "Sessiz çalışıyor"],
        complaints=["Fiyat yüksek"],
        red_flags=["Şikayetvar'da 47 şikayet"],
        insufficient_data=False,
    )
    card = format_group_card(group, syn, cands, community_counts={})
    assert "Siemens EQ.6 Plus S100" in card
    assert "⭐ 4.5" in card and "312" in card
    assert "👍" in card and "Köpük kalitesi iyi" in card
    assert "👎" in card and "Fiyat yüksek" in card
    assert "⚠️" in card and "Şikayetvar" in card
    assert "Hepsiburada" in card.title() or "hepsiburada" in card
    assert "24.745" in card or "24745" in card
    assert "Yeterli inceleme verisi yok" not in card


def test_format_group_card_insufficient_data_shows_footer():
    from src.workflows.shopping.pipeline_v2 import (
        Candidate, ProductGroup, ReviewSynthesis, format_group_card,
    )
    cands = [
        Candidate(title="X", site="trendyol", site_rank=1, price=100.0,
                  original_price=None, url="u", rating=None, review_count=None,
                  review_snippets=[]),
    ]
    group = ProductGroup("X", [0], False, 1.0)
    syn = ReviewSynthesis([], [], [], insufficient_data=True)
    card = format_group_card(group, syn, cands, community_counts={})
    assert "Yeterli inceleme verisi yok" in card
    assert "👍" not in card
    assert "👎" not in card


def test_format_group_card_community_line_when_present():
    from src.workflows.shopping.pipeline_v2 import (
        Candidate, ProductGroup, ReviewSynthesis, format_group_card,
    )
    cands = [Candidate(title="X", site="t", site_rank=1, price=1,
                       original_price=None, url="u", rating=None,
                       review_count=None, review_snippets=[])]
    group = ProductGroup("X", [0], False, 1.0)
    syn = ReviewSynthesis([], [], [], insufficient_data=True)
    card = format_group_card(group, syn, cands,
                             community_counts={"Technopat": 12, "Ekşi": 8})
    assert "💬" in card and "Technopat (12" in card and "Ekşi (8" in card


def test_format_response_joins_cards():
    from src.workflows.shopping.pipeline_v2 import format_response
    out = format_response(["CARD1", "CARD2"])
    assert "CARD1" in out and "CARD2" in out
    assert out.index("CARD1") < out.index("CARD2")


@pytest.mark.asyncio
async def test_pipeline_v2_full_run_siemens_style():
    """End-to-end with patched scraper + patched LLMs.

    Emulates the 2026-04-20 Siemens S100 bug: ensures the accessory is dropped
    and the output contains the real machine, not the brewing unit part.
    """
    from types import SimpleNamespace
    from src.workflows.shopping.pipeline_v2 import (
        step_resolve, step_group, step_synthesize_reviews,
        select_groups, format_group_card, format_response,
    )

    fake_products = [
        SimpleNamespace(
            name="Siemens EQ.6 Plus S100", source="hepsiburada",
            url="https://h.com/1", discounted_price=24745.0, original_price=None,
            rating=4.5, review_count=312,
            review_snippets=["köpük iyi", "sessiz", "fiyat yüksek"],
        ),
        SimpleNamespace(
            name="Siemens EQ.6 Plus S100", source="akakce",
            url="https://a.com/2", discounted_price=25499.0, original_price=None,
            rating=None, review_count=None, review_snippets=[],
        ),
        SimpleNamespace(
            name="DL-Pro demleme ünitesi Siemens EQ.3 S100",
            source="amazon_tr", url="https://am.tr/3", discounted_price=4800.0,
            original_price=None, rating=5.0, review_count=3,
            review_snippets=[],
        ),
    ]

    grouping_resp = {
        "content": (
            '{"groups": ['
            '  {"representative_title": "Siemens EQ.6 Plus S100", '
            '   "member_indices": [0, 1], "is_accessory_or_part": false},'
            '  {"representative_title": "Siemens EQ.3 brewing unit", '
            '   "member_indices": [2], "is_accessory_or_part": true}'
            ']}'
        ),
        "model": "m", "cost": 0.0,
    }
    synth_resp = {
        "content": (
            '{"praise":["köpük iyi","sessiz"],'
            ' "complaints":["fiyat yüksek"],'
            ' "red_flags":[], "insufficient_data": false}'
        ),
        "model": "m", "cost": 0.0,
    }

    with patch(
        "src.workflows.shopping.pipeline_v2._fetch_products",
        new=AsyncMock(return_value=fake_products),
    ), patch(
        "src.workflows.shopping.pipeline_v2._grouping_llm_call",
        new=AsyncMock(return_value=grouping_resp),
    ), patch(
        "src.workflows.shopping.pipeline_v2._synthesis_llm_call",
        new=AsyncMock(return_value=synth_resp),
    ):
        cands = await step_resolve("Siemens s100", per_site_n=3)
        groups = await step_group(cands)
        kept = select_groups(groups, max_groups=2)
        cards: list[str] = []
        for g in kept:
            syn = await step_synthesize_reviews(g, cands)
            cards.append(format_group_card(g, syn, cands, community_counts={}))
        response = format_response(cards)

    # Accessory is dropped
    assert "demleme ünitesi" not in response
    assert "EQ.3" not in response
    # Real machine is present
    assert "EQ.6" in response
    assert "24.745" in response
    assert "köpük" in response
    assert "fiyat yüksek" in response


@pytest.mark.asyncio
async def test_shopping_pipeline_v2_unknown_step_returns_failed():
    from src.workflows.shopping.pipeline_v2 import ShoppingPipelineV2
    task = {
        "id": 1,
        "title": "[0.1] nonsense_step",
        "context": {"step_name": "nonsense_step"},
    }
    result = await ShoppingPipelineV2().run(task)
    assert result["status"] == "failed"
    assert "nonsense_step" in result["result"].lower() or "unknown" in result["result"].lower()


@pytest.mark.asyncio
async def test_shopping_pipeline_v2_resolve_candidates_step():
    from types import SimpleNamespace
    from src.workflows.shopping.pipeline_v2 import ShoppingPipelineV2

    fake = [SimpleNamespace(name="X", source="trendyol", url="u", discounted_price=100,
                             original_price=None, rating=None, review_count=None,
                             review_snippets=[])]
    task = {
        "id": 2,
        "title": "[0.1] resolve_candidates",
        "context": {
            "step_name": "resolve_candidates",
            "input_artifacts": [],
            "per_site_n": 3,
        },
        "description": "X",
    }
    with patch(
        "src.workflows.shopping.pipeline_v2._fetch_products",
        new=AsyncMock(return_value=fake),
    ):
        result = await ShoppingPipelineV2().run(task)
    assert result["status"] == "completed"
    payload = json.loads(result["result"])
    assert payload["candidates"][0]["title"] == "X"
