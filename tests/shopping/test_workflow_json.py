import json
from pathlib import Path

WF_PATHS = [
    # shopping_v2.json retired 2026-06-08 (orphaned after v3 launch). The
    # shared v2-style handlers it used live on in quick_search/product_research.
    "src/workflows/shopping/quick_search_v2.json",
    "src/workflows/shopping/product_research_v2.json",
]


def test_workflows_use_producer_triads():
    # Migrated 2026-06-08 to producer-agent triads (no inline request()). The old
    # fused steps (group_label_filter_gate/synth_one/format_compare) are gone.
    required = {"resolve_candidates", "group_prep", "group_dispatch",
                "group_apply_label_prep", "label_dispatch", "label_apply_filter_gate",
                "synth_prep", "synth_dispatch", "synth_apply", "clarify_variant",
                "compare_prep", "compare_line_apply", "compare_assemble",
                "format_response"}
    for p in WF_PATHS:
        wf = json.loads(Path(p).read_text(encoding="utf-8"))
        names = {s.get("name") for s in wf.get("steps", [])}
        missing = required - names
        assert not missing, f"{p} missing steps: {missing}"
        # the retired fused handlers must NOT reappear
        assert not ({"group_label_filter_gate", "synth_one", "format_compare",
                     "group_and_synthesize"} & names), f"{p} still uses fused v2 steps"


def test_producer_steps_use_agent_types():
    for p in WF_PATHS:
        wf = json.loads(Path(p).read_text(encoding="utf-8"))
        by_name = {}
        for s in wf["steps"]:
            by_name.setdefault(s["name"], s["agent"])
        assert by_name["group_dispatch"] == "shopping_grouper"
        assert by_name["label_dispatch"] == "shopping_labeler"
        assert by_name["synth_dispatch"] == "shopping_synthesizer"


def test_clarify_variant_is_mechanical():
    for p in WF_PATHS:
        wf = json.loads(Path(p).read_text(encoding="utf-8"))
        step = next(s for s in wf["steps"] if s["name"] == "clarify_variant")
        assert step.get("agent") == "mechanical", f"{p} clarify_variant agent wrong"
        # mr_roboto routes on payload.action; expander reads top-level executor +
        # payload. Nesting executor:"clarify" in context (the old shape this test
        # used to assert) produced payload.action=None -> live DLQ.
        assert step.get("executor") == "mechanical", f"{p} clarify_variant needs top-level executor:mechanical"
        payload = step.get("payload", {})
        assert payload.get("action") == "clarify", f"{p} clarify_variant payload.action"
        assert payload.get("kind") == "variant_choice", f"{p} clarify_variant payload.kind"


def test_branches_discriminate_on_clarify_choice():
    # chosen/variant/compare branches all key skip_when on the seeded clarify_choice
    # (always present) — never on a maybe-absent artifact (the spurious-run bug).
    for p in WF_PATHS:
        wf = json.loads(Path(p).read_text(encoding="utf-8"))
        by_id = {s["id"]: s for s in wf["steps"]}
        assert by_id["2.0a"]["skip_when"] == "clarify_choice.kind != 'chosen'"
        assert by_id["2.2a"]["skip_when"] == "clarify_choice.kind != 'variant'"
        assert by_id["2.3a"]["skip_when"] == "clarify_choice.kind != 'compare_all'"
        assert by_id["3.0"]["skip_when"] == "clarify_choice.kind == 'compare_all'"
        # compare_assemble must be the last step (highest id) for delivery
        assert wf["steps"][-1]["id"] == "2.3z"
