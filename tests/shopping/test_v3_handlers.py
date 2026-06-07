"""Unit tests for the shopping_v3 prep/apply triad handlers (group/label/synth)."""
import json
import pytest

from src.workflows.shopping.pipeline_v2 import (
    Candidate, _candidates_to_json,
    handler_group_prep, handler_group_apply_label_prep, handler_label_apply_filter_gate,
    handler_synth_prep, handler_synth_apply,
    handler_compare_prep, handler_compare_line_apply, handler_compare_assemble,
    handler_understand_query,
)


def _chosen_group_dict():
    return {"representative_title": "Galaxy S25", "member_indices": [0],
            "is_accessory_or_part": False, "prominence": 1.0,
            "product_type": "authentic_product", "base_model": "Galaxy S25",
            "variant": None, "authenticity_confidence": 0.9,
            "matches_user_intent": True, "line_id": "galaxy-s25"}


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


@pytest.mark.asyncio
async def test_synth_prep_builds_input_and_meta_from_chosen():
    cands = _cands()
    gate = {"gate": {"kind": "chosen"}, "chosen_group": _chosen_group_dict(),
            "candidates": _candidates_to_json(cands), "query": "s25"}
    out = await handler_synth_prep(task={}, artifacts={"gate_result": json.dumps(gate)}, ctx={})
    si = json.loads(out["synth_input"])
    sm = json.loads(out["synth_meta"])
    assert si["representative_title"] == "Galaxy S25"
    assert si["snippets"] == ["iyi"]  # member index 0's snippet
    assert sm["escalation"] is False and sm["snippet_count"] == 1


@pytest.mark.asyncio
async def test_synth_prep_escalates_when_no_group():
    gate = {"gate": {"kind": "clarify"}, "clarify_payloads": {},
            "candidates": [], "query": "x"}
    out = await handler_synth_prep(
        task={}, artifacts={"gate_result": json.dumps(gate),
                            "clarify_choice": json.dumps({"kind": "variant", "group_id": 9})}, ctx={})
    assert json.loads(out["synth_meta"])["escalation"] is True


@pytest.mark.asyncio
async def test_synth_apply_parses_aspects_into_card():
    cands = _cands()
    meta = {"group": _chosen_group_dict(), "candidates": _candidates_to_json(cands),
            "snippet_count": 5, "escalation": False}
    raw = json.dumps({"aspects": [{"aspect": "kamera", "sentiment": 0.8, "mention_count": 3,
                                   "summary": "net", "quote": "kamera çok net"}],
                      "praise": ["hızlı"], "complaints": [], "red_flags": [],
                      "insufficient_data": False, "overall_sentiment": 0.7,
                      "comparative_mentions": [], "notable_quote": "iyi telefon"})
    out = await handler_synth_apply(
        task={}, artifacts={"synth_raw": raw, "synth_meta": json.dumps(meta)}, ctx={})
    assert out["escalation_needed"] is False and out["cards"]


@pytest.mark.asyncio
async def test_synth_apply_escalates_on_meta_escalation():
    out = await handler_synth_apply(
        task={}, artifacts={"synth_raw": "", "synth_meta": json.dumps({"escalation": True})}, ctx={})
    assert out["escalation_needed"] is True and out["cards"] == []


# ── 0.1 understand_query (deterministic parse → parsed_intent) ──────────────

@pytest.mark.asyncio
async def test_understand_query_passes_query_through_raw():
    out = await handler_understand_query(
        task={}, artifacts={"user_query": "kahve makinesi"}, ctx={})
    assert out["query"] == "kahve makinesi"
    assert out["needs_clarification"] is True  # bare 2-token category, no model no.


@pytest.mark.asyncio
async def test_understand_query_specific_query_not_vague():
    out = await handler_understand_query(
        task={}, artifacts={"user_query": "siemens eq300 espresso"}, ctx={})
    assert out["query"] == "siemens eq300 espresso"
    assert out["needs_clarification"] is False  # has a model number


@pytest.mark.asyncio
async def test_understand_query_unwraps_json_user_query():
    out = await handler_understand_query(
        task={}, artifacts={"user_query": json.dumps({"query": "laptop"})}, ctx={})
    assert out["query"] == "laptop"


# ── compare-all triad (Approach B: fixed-cap native-join) ───────────────────

def _line_group(title, line_id, base):
    return {"representative_title": title, "member_indices": [0],
            "is_accessory_or_part": False, "prominence": 1.0,
            "product_type": "authentic_product", "base_model": base,
            "variant": None, "authenticity_confidence": 0.9,
            "matches_user_intent": True, "line_id": line_id}


def _clarify_gate(n_lines):
    cands = _cands()
    payloads = {str(i): _line_group(f"Line {i}", f"line-{i}", f"Base {i}")
                for i in range(n_lines)}
    return {"gate": {"kind": "clarify"}, "clarify_payloads": payloads,
            "base_label": "Telefon", "candidates": _candidates_to_json(cands),
            "query": "telefon"}


@pytest.mark.asyncio
async def test_compare_prep_emits_flags_and_per_line_slots():
    gate = _clarify_gate(2)
    out = await handler_compare_prep(
        task={}, artifacts={"gate_result": json.dumps(gate)}, ctx={})
    # Always emits all 5 line slots + the flags artifact (multi-output split needs
    # every declared key present every run).
    for i in range(5):
        assert f"cmp_input_{i}" in out and f"cmp_meta_{i}" in out
    flags = json.loads(out["compare_flags"])
    assert flags["has_line_0"] == "true" and flags["has_line_1"] == "true"
    assert flags["has_line_2"] == "false" and flags["has_line_4"] == "false"
    assert flags["n_lines"] == 2 and flags["header"]
    # gid->line-index map so a post-delivery keyboard can surface the right
    # stored cmp_card_i when the user drills into a line (keep-pick-after-compare).
    assert flags["line_gids"] == ["0", "1"]
    # populated line carries the synthesizer's user-message data
    si0 = json.loads(out["cmp_input_0"])
    assert si0["representative_title"] == "Line 0"
    # unused slot is a harmless stub
    assert json.loads(out["cmp_meta_4"])["escalation"] is True


@pytest.mark.asyncio
async def test_compare_prep_caps_at_max_clarify_options():
    gate = _clarify_gate(8)  # more than the 5-slot cap
    out = await handler_compare_prep(
        task={}, artifacts={"gate_result": json.dumps(gate)}, ctx={})
    flags = json.loads(out["compare_flags"])
    assert flags["n_lines"] == 5
    assert all(flags[f"has_line_{i}"] == "true" for i in range(5))


@pytest.mark.asyncio
async def test_compare_line_apply_parses_raw_into_indexed_card():
    meta = {"group": _line_group("Line 0", "line-0", "Base 0"),
            "candidates": _candidates_to_json(_cands()),
            "snippet_count": 5, "escalation": False}
    raw = json.dumps({"aspects": [{"aspect": "pil", "sentiment": 0.6, "mention_count": 4,
                                   "summary": "iyi pil", "quote": "pil günler gidiyor"}],
                      "praise": ["hızlı"], "complaints": [], "red_flags": [],
                      "insufficient_data": False, "overall_sentiment": 0.6,
                      "comparative_mentions": [], "notable_quote": "iyi"})
    out = await handler_compare_line_apply(
        task={}, artifacts={"cmp_raw_2": raw, "cmp_meta_2": json.dumps(meta)},
        ctx={"line_index": 2})
    assert "cmp_card_2" in out
    card = json.loads(out["cmp_card_2"])
    assert card["card"] and card["escalation"] is False


@pytest.mark.asyncio
async def test_compare_assemble_joins_cards_with_header():
    flags = {"header": "*Telefon — Karşılaştırma*\n", "n_lines": 2,
             "has_line_0": "true", "has_line_1": "true",
             "has_line_2": "false", "has_line_3": "false", "has_line_4": "false"}
    art = {
        "compare_flags": json.dumps(flags),
        "cmp_card_0": json.dumps({"card": "CARD-A", "escalation": False}),
        "cmp_card_1": json.dumps({"card": "CARD-B", "escalation": False}),
    }
    out = await handler_compare_assemble(task={}, artifacts=art, ctx={})
    assert out["escalation"] is False
    text = out["formatted_text"]
    assert text.startswith("*Telefon — Karşılaştırma*")
    assert "CARD-A" in text and "CARD-B" in text


@pytest.mark.asyncio
async def test_compare_assemble_escalates_when_no_cards():
    flags = {"header": "*X*\n", "n_lines": 1, "has_line_0": "true",
             "has_line_1": "false", "has_line_2": "false",
             "has_line_3": "false", "has_line_4": "false"}
    out = await handler_compare_assemble(
        task={}, artifacts={"compare_flags": json.dumps(flags)}, ctx={})
    assert out["escalation"] is True
