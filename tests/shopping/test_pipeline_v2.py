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


def test_product_group_accepts_label_fields():
    g = ProductGroup(
        representative_title="Samsung Galaxy S25",
        member_indices=[0, 1],
        is_accessory_or_part=False,
        prominence=2.0,
        product_type="authentic_product",
        base_model="Samsung Galaxy S25",
        variant=None,
        authenticity_confidence=0.95,
        matches_user_intent=True,
    )
    assert g.product_type == "authentic_product"
    assert g.variant is None
    assert g.authenticity_confidence == 0.95


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


def test_select_groups_drops_accessories_and_applies_50pct_rule():
    from src.workflows.shopping.pipeline_v2 import ProductGroup, select_groups

    groups = [
        ProductGroup("Dominant",  [0, 1], False, prominence=2.0),
        ProductGroup("Close runner", [2], False, prominence=1.2),   # 60% of top → keep
        ProductGroup("Weak runner", [3], False, prominence=0.8),    # 40% of top → drop
        ProductGroup("Accessory", [4], True, prominence=99.0),      # always drop
    ]

    kept_named = select_groups(groups, max_groups=2, query="")
    assert [g.representative_title for g in kept_named] == ["Dominant", "Close runner"]

    kept_cat = select_groups(groups, max_groups=3, query="")
    assert [g.representative_title for g in kept_cat] == ["Dominant", "Close runner"]


def test_select_groups_empty_input():
    from src.workflows.shopping.pipeline_v2 import select_groups
    assert select_groups([], max_groups=2) == []


def test_select_groups_query_match_penalises_fe_variant():
    """Samsung S25 query should rank S25 above S25 FE despite equal prominence."""
    from src.workflows.shopping.pipeline_v2 import ProductGroup, select_groups

    # S25 FE appears on more sites → higher raw prominence, but query says "s25"
    groups = [
        ProductGroup("Samsung Galaxy S25 FE", [0, 1, 2], False, prominence=3.0),
        ProductGroup("Samsung Galaxy S25",    [3, 4],    False, prominence=2.0),
    ]
    kept = select_groups(groups, max_groups=2, query="Samsung s25")
    # S25 (exact match) must be ranked first
    assert kept[0].representative_title == "Samsung Galaxy S25"
    assert kept[1].representative_title == "Samsung Galaxy S25 FE"


def test_select_groups_query_match_no_penalty_when_fe_in_query():
    """When the user asks for 'S25 FE', FE variant should NOT be penalised."""
    from src.workflows.shopping.pipeline_v2 import ProductGroup, select_groups

    groups = [
        ProductGroup("Samsung Galaxy S25 FE", [0], False, prominence=2.0),
        ProductGroup("Samsung Galaxy S25",    [1], False, prominence=3.0),
    ]
    kept = select_groups(groups, max_groups=2, query="Samsung S25 FE")
    # S25 FE is an exact match to query → must win despite lower prominence
    assert kept[0].representative_title == "Samsung Galaxy S25 FE"


def test_query_match_score_variant_penalty():
    """_query_match_score penalises unsolicited FE/Plus/Ultra etc."""
    from src.workflows.shopping.pipeline_v2 import _query_match_score

    # "S25 FE" vs query "s25" — FE not in query → penalty applied
    score_fe = _query_match_score("Samsung Galaxy S25 FE", "Samsung s25")
    # "S25" exact → no penalty
    score_exact = _query_match_score("Samsung Galaxy S25", "Samsung s25")
    assert score_exact > score_fe

    # When FE is in query, no penalty
    score_fe_asked = _query_match_score("Samsung Galaxy S25 FE", "Samsung s25 FE")
    assert score_fe_asked > score_fe


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


@pytest.mark.asyncio
async def test_candidate_passes_sku_and_category_path():
    from src.shopping.models import Product
    from src.workflows.shopping.pipeline_v2 import step_resolve

    p = Product(name="Galaxy S25", url="https://trendyol.com/p-1", source="trendyol",
                sku="TY-123", category_path="Elektronik > Telefon")
    with patch("src.workflows.shopping.pipeline_v2._fetch_products",
               new=AsyncMock(return_value=[p])), \
         patch("src.workflows.shopping.pipeline_v2._fetch_reviews",
               new=AsyncMock(return_value={})):
        cands = await step_resolve("s25", per_site_n=3)
    assert cands[0].sku == "TY-123"
    assert cands[0].category_path == "Elektronik > Telefon"


def test_compare_all_table_format():
    from src.workflows.shopping.pipeline_v2 import step_compare_all, Candidate, ProductGroup
    cands = [
        Candidate(title="Galaxy S25", site="trendyol", site_rank=1,
                  price=32500, original_price=None, url="u",
                  rating=4.7, review_count=1200),
        Candidate(title="Galaxy S25", site="hepsiburada", site_rank=1,
                  price=34800, original_price=None, url="u",
                  rating=4.7, review_count=800),
        Candidate(title="Galaxy S25 Ultra", site="trendyol", site_rank=1,
                  price=48000, original_price=None, url="u",
                  rating=4.9, review_count=2100),
    ]
    vanilla = ProductGroup(representative_title="Galaxy S25",
                           member_indices=[0, 1], is_accessory_or_part=False,
                           prominence=2.0, base_model="Samsung Galaxy S25",
                           variant=None)
    ultra = ProductGroup(representative_title="Galaxy S25 Ultra",
                         member_indices=[2], is_accessory_or_part=False,
                         prominence=1.0, base_model="Samsung Galaxy S25",
                         variant="Ultra")
    md = step_compare_all([vanilla, ultra], cands, base_label="Samsung Galaxy S25")
    assert "Galaxy S25" in md
    assert "Ultra" in md
    assert "32.500" in md
    assert "34.800" in md
    assert "48.000" in md
    assert "⭐" in md
