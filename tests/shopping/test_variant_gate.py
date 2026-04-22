from src.workflows.shopping.pipeline_v2 import ProductGroup


def _g(title, product_type="authentic_product", conf=0.9, intent=True,
       variant=None, base_model=None, prom=1.0):
    return ProductGroup(
        representative_title=title,
        member_indices=[0],
        is_accessory_or_part=False,
        prominence=prom,
        product_type=product_type,
        base_model=base_model or title,
        variant=variant,
        authenticity_confidence=conf,
        matches_user_intent=intent,
    )


def test_step_filter_drops_accessories_parts_knockoffs_refurbished():
    from src.workflows.shopping.variant_gate import step_filter
    groups = [
        _g("Samsung S25", product_type="authentic_product"),
        _g("S25 Case", product_type="accessory"),
        _g("S25 Battery", product_type="replacement_part"),
        _g("Galaxy S25 Clone", product_type="knockoff"),
        _g("Galaxy S25 Refurb", product_type="refurbished"),
    ]
    survivors = step_filter(groups)
    assert [g.representative_title for g in survivors] == ["Samsung S25"]


def test_step_filter_drops_intent_mismatch():
    from src.workflows.shopping.variant_gate import step_filter
    groups = [
        _g("Samsung S25", intent=True),
        _g("Samsung S25 Case", intent=False),
    ]
    survivors = step_filter(groups)
    assert [g.representative_title for g in survivors] == ["Samsung S25"]


def test_step_filter_drops_low_confidence():
    from src.workflows.shopping.variant_gate import step_filter, FILTER_AUTHENTICITY_MIN
    groups = [
        _g("Samsung S25", conf=0.95),
        _g("Samsung S25 Fake", conf=0.4),
    ]
    survivors = step_filter(groups)
    assert len(survivors) == 1
    assert survivors[0].authenticity_confidence >= FILTER_AUTHENTICITY_MIN


def test_variant_gate_zero_survivors_signals_escalation():
    from src.workflows.shopping.variant_gate import step_variant_gate
    out = step_variant_gate(survivors=[], all_groups=[])
    assert out["kind"] == "escalation"
    assert out["reason"] == "all_filtered"


def test_variant_gate_single_variant_returns_chosen():
    from src.workflows.shopping.variant_gate import step_variant_gate
    g = _g("Samsung S25", base_model="Samsung Galaxy S25", variant=None, prom=3.0)
    out = step_variant_gate(survivors=[g], all_groups=[g])
    assert out["kind"] == "chosen"
    assert out["group"] is g


def test_variant_gate_multiple_variants_returns_clarify():
    from src.workflows.shopping.variant_gate import step_variant_gate
    vanilla = _g("Galaxy S25", base_model="Samsung Galaxy S25", variant=None, prom=2.0)
    fe = _g("Galaxy S25 FE", base_model="Samsung Galaxy S25", variant="FE", prom=3.0)
    ultra = _g("Galaxy S25 Ultra", base_model="Samsung Galaxy S25", variant="Ultra", prom=2.5)
    out = step_variant_gate(survivors=[vanilla, fe, ultra], all_groups=[vanilla, fe, ultra])
    assert out["kind"] == "clarify"
    labels = [opt["label"] for opt in out["options"]]
    assert labels[0] == "Galaxy S25 FE"        # highest prominence first
    assert len(out["options"]) == 3
