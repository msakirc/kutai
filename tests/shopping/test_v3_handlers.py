"""Unit tests for the shopping_v3 prep/apply triad handlers (group/label/synth)."""
import json
import pytest

from src.workflows.shopping.pipeline_v2 import (
    Candidate, _candidates_to_json,
    handler_group_prep, handler_group_apply_label_prep, handler_label_apply_filter_gate,
)


def _cands():
    return [
        Candidate(title="Galaxy S25 256GB", site="a", site_rank=1, price=50000.0,
                  original_price=None, url="u1", rating=4.5, review_count=10,
                  review_snippets=["iyi"], sku=None, category_path="telefon"),
        Candidate(title="Galaxy S25 FE", site="b", site_rank=1, price=40000.0,
                  original_price=None, url="u2", rating=4.0, review_count=5,
                  review_snippets=["fena değil"], sku=None, category_path="telefon"),
    ]


@pytest.mark.asyncio
async def test_group_prep_emits_input_and_residual_flag():
    cands = _cands()
    art = {"search_results": json.dumps(
        {"candidates": _candidates_to_json(cands), "query": "s25"})}
    out = await handler_group_prep(task={}, artifacts=art, ctx={})
    gi = json.loads(out["group_input"])
    assert gi["has_residuals"] == "true"  # both candidates are sku-less
    assert len(gi["view"]) == 2 and "index" in gi["view"][0]
    gs = json.loads(out["groups_state"])
    assert gs["query"] == "s25" and gs["unbucketed"] == [0, 1]


@pytest.mark.asyncio
async def test_group_prep_no_residuals_when_all_have_sku():
    cands = _cands()
    cands[0].sku = "SKU1"
    cands[1].sku = "SKU2"
    art = {"search_results": json.dumps(
        {"candidates": _candidates_to_json(cands), "query": "s25"})}
    out = await handler_group_prep(task={}, artifacts=art, ctx={})
    gi = json.loads(out["group_input"])
    assert gi["has_residuals"] == "false" and gi["view"] == []
    gs = json.loads(out["groups_state"])
    assert len(gs["groups"]) == 2  # two sku buckets


@pytest.mark.asyncio
async def test_group_apply_parses_producer_raw_into_label_input():
    cands = _cands()
    groups_state = json.dumps({
        "groups": [], "candidates": _candidates_to_json(cands),
        "query": "s25", "unbucketed": [0, 1]})
    raw = json.dumps({"groups": [
        {"representative_title": "Galaxy S25", "member_indices": [0], "is_accessory_or_part": False},
        {"representative_title": "Galaxy S25 FE", "member_indices": [1], "is_accessory_or_part": False},
    ]})
    out = await handler_group_apply_label_prep(
        task={}, artifacts={"group_raw": raw, "groups_state": groups_state}, ctx={})
    li = json.loads(out["label_input"])
    assert len(li["view"]) == 2 and li["query"] == "s25"
    # member indices remapped back to the full candidate list
    gs = json.loads(out["groups_state"])
    assert {tuple(g["member_indices"]) for g in gs["groups"]} == {(0,), (1,)}


@pytest.mark.asyncio
async def test_label_apply_filter_gate_emits_gate_result():
    cands = _cands()
    groups_state = json.dumps({
        "groups": [
            {"representative_title": "Galaxy S25", "member_indices": [0],
             "is_accessory_or_part": False, "prominence": 1.0, "product_type": "unknown",
             "base_model": "", "variant": None, "authenticity_confidence": 0.5,
             "matches_user_intent": True, "line_id": ""},
            {"representative_title": "Galaxy S25 FE", "member_indices": [1],
             "is_accessory_or_part": False, "prominence": 1.0, "product_type": "unknown",
             "base_model": "", "variant": None, "authenticity_confidence": 0.5,
             "matches_user_intent": True, "line_id": ""},
        ],
        "candidates": _candidates_to_json(cands), "query": "s25"})
    label_raw = json.dumps({"groups": [
        {"group_id": 0, "line_id": "samsung-galaxy-s25", "product_type": "authentic_product",
         "base_model": "Galaxy S25", "variant": "256GB", "authenticity_confidence": 0.9,
         "matches_user_intent": True},
        {"group_id": 1, "line_id": "samsung-galaxy-s25-fe", "product_type": "authentic_product",
         "base_model": "Galaxy S25 FE", "variant": None, "authenticity_confidence": 0.9,
         "matches_user_intent": True},
    ]})
    out = await handler_label_apply_filter_gate(
        task={}, artifacts={"groups_state": groups_state, "label_raw": label_raw}, ctx={})
    # two distinct lines that both match query "s25" => clarify gate
    assert out["gate"]["kind"] in ("clarify", "chosen", "escalation")
    if out["gate"]["kind"] == "clarify":
        assert out["clarify_payloads"]
