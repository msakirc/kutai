import pytest
from unittest.mock import AsyncMock, patch
from src.workflows.shopping.pipeline_v2 import ProductGroup, Candidate


@pytest.mark.asyncio
async def test_step_label_parses_taxonomy():
    from src.workflows.shopping.labels import step_label
    groups = [
        ProductGroup(representative_title="Samsung Galaxy S25",
                     member_indices=[0], is_accessory_or_part=False, prominence=1.0),
        ProductGroup(representative_title="Galaxy S25 Silicone Case",
                     member_indices=[1], is_accessory_or_part=False, prominence=0.5),
    ]
    cands = [
        Candidate(title="Samsung Galaxy S25", site="trendyol", site_rank=1,
                  price=30000, original_price=None, url="u1",
                  rating=None, review_count=None,
                  category_path="Telefon > Akıllı Telefon"),
        Candidate(title="Galaxy S25 Silicone Case", site="trendyol", site_rank=2,
                  price=200, original_price=None, url="u2",
                  rating=None, review_count=None,
                  category_path="Aksesuar > Kılıf"),
    ]
    fake_resp = {"content": '{"groups": [\
        {"group_id": 0, "product_type": "authentic_product", "base_model": "Samsung Galaxy S25", "variant": null, "authenticity_confidence": 0.95, "matches_user_intent": true},\
        {"group_id": 1, "product_type": "accessory", "base_model": "Samsung Galaxy S25", "variant": null, "authenticity_confidence": 0.8, "matches_user_intent": false}\
    ]}'}
    with patch("src.workflows.shopping.labels._label_llm_call",
               new=AsyncMock(return_value=fake_resp)):
        labelled = await step_label(groups, cands, query="samsung s25")
    assert labelled[0].product_type == "authentic_product"
    assert labelled[0].matches_user_intent is True
    assert labelled[1].product_type == "accessory"
    assert labelled[1].matches_user_intent is False


@pytest.mark.asyncio
async def test_step_label_falls_back_on_llm_error():
    from src.workflows.shopping.labels import step_label
    groups = [ProductGroup(representative_title="x", member_indices=[0],
                           is_accessory_or_part=False, prominence=1.0)]
    cands = [Candidate(title="x", site="s", site_rank=1, price=None,
                       original_price=None, url="u",
                       rating=None, review_count=None)]
    with patch("src.workflows.shopping.labels._label_llm_call",
               new=AsyncMock(side_effect=RuntimeError("boom"))):
        labelled = await step_label(groups, cands, query="x")
    assert labelled[0].product_type == "authentic_product"
    assert labelled[0].matches_user_intent is True
    assert labelled[0].authenticity_confidence == 0.5
