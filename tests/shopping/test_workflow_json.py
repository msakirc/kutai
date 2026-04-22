import json
from pathlib import Path

WF_PATHS = [
    "src/workflows/shopping/shopping_v2.json",
    "src/workflows/shopping/quick_search_v2.json",
    "src/workflows/shopping/product_research_v2.json",
]


def test_workflows_reference_new_steps():
    required = {"resolve_candidates", "group_label_filter_gate",
                "synth_one", "clarify_variant", "format_compare", "format_response"}
    for p in WF_PATHS:
        wf = json.loads(Path(p).read_text(encoding="utf-8"))
        names = {s.get("name") for s in wf.get("steps", [])}
        missing = required - names
        assert not missing, f"{p} missing steps: {missing}"


def test_clarify_variant_is_mechanical():
    for p in WF_PATHS:
        wf = json.loads(Path(p).read_text(encoding="utf-8"))
        step = next(s for s in wf["steps"] if s["name"] == "clarify_variant")
        assert step.get("agent") == "mechanical", f"{p} clarify_variant agent wrong"
        ctx = step.get("context", {})
        assert ctx.get("executor") == "clarify"
        assert ctx.get("kind") == "variant_choice"


def test_synth_one_skipped_when_not_chosen():
    for p in WF_PATHS:
        wf = json.loads(Path(p).read_text(encoding="utf-8"))
        synth_steps = [s for s in wf["steps"] if s["name"] == "synth_one"]
        assert any("chosen" in str(s.get("skip_when", "")) for s in synth_steps), \
            f"{p} no synth_one step with 'chosen' skip_when"
