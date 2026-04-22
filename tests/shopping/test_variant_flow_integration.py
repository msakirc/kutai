import json
import pytest
from unittest.mock import AsyncMock, patch
from src.shopping.models import Product


def _p(name, source, sku=None, cat=None, price=30000, rank=0):
    p = Product(name=name, url=f"https://{source}/{sku or name}", source=source,
                original_price=price, sku=sku, category_path=cat)
    p.site_rank = rank
    return p


@pytest.mark.asyncio
async def test_end_to_end_phone_ambiguous_triggers_clarify():
    from src.workflows.shopping.pipeline_v2 import (
        _handler_resolve_candidates, _handler_group_label_filter_gate,
    )
    products = [
        _p("Samsung Galaxy S25 128GB",     "trendyol",    sku="TY-1", cat="Telefon"),
        _p("Samsung Galaxy S25 128GB",     "hepsiburada", sku="HB-1", cat="Telefon"),
        _p("Samsung Galaxy S25 FE 128GB",  "trendyol",    sku="TY-2", cat="Telefon"),
        _p("Samsung Galaxy S25 Ultra 256", "amazon_tr",   sku="AM-3", cat="Telefon"),
        _p("Samsung S25 Silicone Case",    "trendyol",    sku="TY-9", cat="Aksesuar"),
    ]
    # Groups are enumerated in candidate order from step_group (sku-bucketed, insertion order):
    # cands: trendyol TY-1(rank1), TY-2(rank2), TY-9(rank3); hepsiburada HB-1(rank1); amazon_tr AM-3(rank1)
    # sku_buckets iteration order → group 0=TY-1, 1=TY-2, 2=TY-9, 3=HB-1, 4=AM-3
    label_resp = json.dumps({"groups": [
        {"group_id": 0, "product_type": "authentic_product", "base_model": "Samsung Galaxy S25", "variant": None,    "authenticity_confidence": 0.95, "matches_user_intent": True},
        {"group_id": 1, "product_type": "authentic_product", "base_model": "Samsung Galaxy S25", "variant": "FE",    "authenticity_confidence": 0.95, "matches_user_intent": True},
        {"group_id": 2, "product_type": "accessory",         "base_model": "Samsung Galaxy S25", "variant": None,    "authenticity_confidence": 0.95, "matches_user_intent": False},
        {"group_id": 3, "product_type": "authentic_product", "base_model": "Samsung Galaxy S25", "variant": None,    "authenticity_confidence": 0.95, "matches_user_intent": True},
        {"group_id": 4, "product_type": "authentic_product", "base_model": "Samsung Galaxy S25", "variant": "Ultra", "authenticity_confidence": 0.95, "matches_user_intent": True},
    ]})
    with patch("src.workflows.shopping.pipeline_v2._fetch_products",
               new=AsyncMock(return_value=products)), \
         patch("src.workflows.shopping.pipeline_v2._fetch_reviews",
               new=AsyncMock(return_value={})), \
         patch("src.workflows.shopping.labels._label_llm_call",
               new=AsyncMock(return_value={"content": label_resp})):
        r1 = await _handler_resolve_candidates(
            task={"id": 1}, artifacts={"user_query": "samsung s25"},
            ctx={"per_site_n": 3},
        )
        r2 = await _handler_group_label_filter_gate(
            task={"id": 2}, artifacts={"search_results": json.dumps(r1)}, ctx={},
        )
    assert r2["gate"]["kind"] == "clarify"
    labels = " ".join(opt["label"] for opt in r2["clarify_options"])
    assert "FE" in labels or "Ultra" in labels
    assert "Case" not in labels


@pytest.mark.asyncio
async def test_end_to_end_single_variant_skips_clarify():
    from src.workflows.shopping.pipeline_v2 import (
        _handler_resolve_candidates, _handler_group_label_filter_gate,
    )
    products = [_p("Samsung Galaxy S25 Ultra 256GB", "trendyol", sku="TY-3", cat="Telefon")]
    label_resp = json.dumps({"groups": [
        {"group_id": 0, "product_type": "authentic_product", "base_model": "Samsung Galaxy S25", "variant": "Ultra", "authenticity_confidence": 0.95, "matches_user_intent": True},
    ]})
    with patch("src.workflows.shopping.pipeline_v2._fetch_products",
               new=AsyncMock(return_value=products)), \
         patch("src.workflows.shopping.pipeline_v2._fetch_reviews",
               new=AsyncMock(return_value={})), \
         patch("src.workflows.shopping.labels._label_llm_call",
               new=AsyncMock(return_value={"content": label_resp})):
        r1 = await _handler_resolve_candidates(
            task={"id": 1}, artifacts={"user_query": "samsung s25 ultra 256gb"},
            ctx={"per_site_n": 3},
        )
        r2 = await _handler_group_label_filter_gate(
            task={"id": 2}, artifacts={"search_results": json.dumps(r1)}, ctx={},
        )
    assert r2["gate"]["kind"] == "chosen"
    assert "Ultra" in r2["chosen_group"]["representative_title"]


@pytest.mark.asyncio
async def test_end_to_end_all_filtered_escalates():
    from src.workflows.shopping.pipeline_v2 import (
        _handler_resolve_candidates, _handler_group_label_filter_gate,
    )
    products = [_p("S25 Knockoff",  "aliexpress", sku="AE-1", cat="Telefon", rank=0)]
    label_resp = json.dumps({"groups": [
        {"group_id": 0, "product_type": "knockoff", "base_model": "Samsung Galaxy S25", "variant": None, "authenticity_confidence": 0.3, "matches_user_intent": False},
    ]})
    with patch("src.workflows.shopping.pipeline_v2._fetch_products",
               new=AsyncMock(return_value=products)), \
         patch("src.workflows.shopping.pipeline_v2._fetch_reviews",
               new=AsyncMock(return_value={})), \
         patch("src.workflows.shopping.labels._label_llm_call",
               new=AsyncMock(return_value={"content": label_resp})):
        r1 = await _handler_resolve_candidates(
            task={"id": 1}, artifacts={"user_query": "samsung s25"},
            ctx={"per_site_n": 3},
        )
        r2 = await _handler_group_label_filter_gate(
            task={"id": 2}, artifacts={"search_results": json.dumps(r1)}, ctx={},
        )
    assert r2["gate"]["kind"] == "escalation"
    assert r2["gate"]["reason"] == "all_filtered"
