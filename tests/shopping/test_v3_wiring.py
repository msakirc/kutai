"""shopping_v3.json structural wiring tests."""
from src.workflows.engine.loader import load_workflow


def test_v3_loads_with_plan_id():
    wf = load_workflow("shopping_v3")
    assert wf.plan_id == "shopping_v3"


def test_v3_producer_steps_use_agent_types():
    wf = load_workflow("shopping_v3")
    agents = {s["id"]: s["agent"] for s in wf.steps}
    assert agents["1.1b"] == "shopping_grouper"
    assert agents["1.1d"] == "shopping_labeler"
    assert agents["2.0b"] == "shopping_synthesizer"
    assert agents["2.2b"] == "shopping_synthesizer"


def test_v3_group_residual_skips_when_no_residuals():
    wf = load_workflow("shopping_v3")
    s11b = next(s for s in wf.steps if s["id"] == "1.1b")
    assert s11b.get("skip_when") == "group_input.has_residuals == 'false'"


def test_v3_synth_paths_are_mutually_exclusive():
    wf = load_workflow("shopping_v3")
    by_id = {s["id"]: s for s in wf.steps}
    # all three branches discriminate on the seeded clarify_choice (always present)
    assert by_id["2.0a"]["skip_when"] == "clarify_choice.kind != 'chosen'"
    assert by_id["2.2a"]["skip_when"] == "clarify_choice.kind != 'variant'"
    assert by_id["2.3a"]["skip_when"] == "clarify_choice.kind != 'compare_all'"
    assert by_id["3.0"]["skip_when"] == "clarify_choice.kind == 'compare_all'"


def test_v3_prep_apply_steps_route_to_pipeline_handler():
    wf = load_workflow("shopping_v3")
    by_id = {s["id"]: s for s in wf.steps}
    for sid in ("1.1a", "1.1c", "1.1e", "2.0a", "2.0c", "2.2a", "2.2c", "3.0"):
        assert by_id[sid]["agent"] == "shopping_pipeline_v2"
        # the pipeline dispatches by step.name -> _STEP_HANDLERS_V2 key
        assert by_id[sid]["name"] in {
            "group_prep", "group_apply_label_prep", "label_apply_filter_gate",
            "synth_prep", "synth_apply", "format_response",
        }


def test_v3_compare_all_fans_into_synth_producers():
    wf = load_workflow("shopping_v3")
    by_id = {s["id"]: s for s in wf.steps}
    # prep gated on the compare-all tap, depends on the clarify step
    prep = by_id["2.3a"]
    assert prep["agent"] == "shopping_pipeline_v2" and prep["name"] == "compare_prep"
    assert prep["skip_when"] == "clarify_choice.kind != 'compare_all'"
    assert prep["depends_on"] == ["2.1"]
    # five synthesizer producer dispatch steps, each per-index skip-gated
    for i in range(5):
        d = by_id[f"2.3_{i}_d"]
        assert d["agent"] == "shopping_synthesizer"
        assert d["depends_on"] == ["2.3a"]
        # discriminate on the always-present clarify_choice, NOT compare_flags
        # (which is absent when compare_prep skips -> missing-artifact defaults
        # to run -> the spurious-compare-on-variant-path bug).
        assert d["skip_when"] == "clarify_choice.kind != 'compare_all'"
        assert d["input_artifacts"] == [f"cmp_input_{i}"]
        assert d["output_artifacts"] == [f"cmp_raw_{i}"]
        a = by_id[f"2.3_{i}_a"]
        assert a["name"] == "compare_line_apply"
        assert a["context"]["line_index"] == i
        assert a["output_artifacts"] == [f"cmp_card_{i}"]


def test_v3_compare_assemble_is_native_join_and_last():
    wf = load_workflow("shopping_v3")
    steps = wf.steps
    by_id = {s["id"]: s for s in steps}
    asm = by_id["2.3z"]
    assert asm["name"] == "compare_assemble"
    # native depends_on join over all five line applies
    assert asm["depends_on"] == [f"2.3_{i}_a" for i in range(5)]
    assert asm["output_artifacts"] == ["shopping_response"]
    assert asm["skip_when"] == "clarify_choice.kind != 'compare_all'"
    # must be the highest-id (last) step so mission-completion delivers it
    assert steps[-1]["id"] == "2.3z"


def test_v3_every_pipeline_step_has_a_registered_handler():
    # Regression guard: step 0.1 understand_query_check_clarity shipped with NO
    # _STEP_HANDLERS_V2 entry and DLQ'd live the moment the category path ran
    # ("Unknown step"). Every shopping_pipeline_v2 step's name must dispatch.
    from src.workflows.shopping.pipeline_v2 import _STEP_HANDLERS_V2
    wf = load_workflow("shopping_v3")
    missing = [
        s["id"] for s in wf.steps
        if s["agent"] == "shopping_pipeline_v2"
        and s["name"] not in _STEP_HANDLERS_V2
    ]
    assert not missing, f"steps with no handler: {missing}"


def test_v3_every_agent_type_is_registered():
    # get_agent() silently falls back to the generic "executor" for unknown
    # types — an unregistered producer would run as a no-op agent producing
    # garbage, not error. Assert every v3 agent_type resolves to itself.
    from src.agents import AGENT_REGISTRY
    wf = load_workflow("shopping_v3")
    # mechanical -> mr_roboto; shopping_pipeline_v2 -> orchestrator special-case
    # (deterministic dispatcher, covered by the handler guard above). Both bypass
    # get_agent/AGENT_REGISTRY by design.
    builtin = {"mechanical", "shopping_pipeline_v2"}
    missing = [
        s["agent"] for s in wf.steps
        if s["agent"] not in AGENT_REGISTRY and s["agent"] not in builtin
    ]
    assert not missing, f"unregistered agent types: {sorted(set(missing))}"


def test_v3_mechanical_steps_have_top_level_executor_and_payload_action():
    # The expander reads step.executor (top-level) + step.payload (top-level);
    # mr_roboto routes on payload.action. 2.1 originally nested executor:"clarify"
    # inside context with no payload -> "unknown mechanical action: None" DLQ.
    wf = load_workflow("shopping_v3")
    for s in wf.steps:
        if s["agent"] == "mechanical":
            assert s.get("executor") == "mechanical", f"{s['id']} missing top-level executor:mechanical"
            assert (s.get("payload") or {}).get("action"), f"{s['id']} missing payload.action"


def test_v3_clarify_variant_shape():
    wf = load_workflow("shopping_v3")
    s = next(x for x in wf.steps if x["id"] == "2.1")
    assert s["payload"]["action"] == "clarify"
    assert s["payload"]["kind"] == "variant_choice"
    assert s["payload"]["payload_from"] == "gate_result"


def test_all_shopping_workflows_mechanical_shape():
    # The clarify_variant DLQ ("unknown mechanical action: None") existed in ALL
    # FOUR shopping workflows — mechanical steps need top-level executor +
    # payload.action, not a context.executor nesting. Guard every shopping wf.
    for name in ("shopping_v3", "quick_search_v2", "product_research_v2"):
        wf = load_workflow(name)
        for s in wf.steps:
            ctx = s.get("context") or {}
            assert not (isinstance(ctx, dict) and ctx.get("executor")), (
                f"{name}[{s['id']}] nests executor in context (wrong)"
            )
            if s.get("agent") == "mechanical" or s.get("executor") == "mechanical":
                assert s.get("executor") == "mechanical", (
                    f"{name}[{s['id']}] missing top-level executor:mechanical"
                )
                assert (s.get("payload") or {}).get("action"), (
                    f"{name}[{s['id']}] missing payload.action"
                )


def test_v3_declares_no_file_produces():
    # Producer/prep/apply steps emit plain JSON artifacts, NOT declared file
    # produces — so materialize_produces is a no-op for them (spec risk 6).
    wf = load_workflow("shopping_v3")
    for s in wf.steps:
        assert "produces" not in s, f"{s['id']} unexpectedly declares produces"
