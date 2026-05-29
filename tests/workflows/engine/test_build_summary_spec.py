"""SP3 Task 3 - build_summary_spec."""
def test_build_summary_spec_shape():
    from src.workflows.engine.hooks import build_summary_spec
    spec = build_summary_spec("long text " * 500, "user_stories")
    assert spec["agent_type"] == "summarizer"
    assert spec["kind"] == "overhead"
    llm = spec["context"]["llm_call"]
    assert llm["raw_dispatch"] is True
    assert llm["prefer_local"] is True
    assert "user_stories" in llm["messages"][1]["content"]
    # no mission_id / parent leakage on the spec
    assert "mission_id" not in spec
