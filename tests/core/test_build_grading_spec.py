"""SP3 Task 1 - build_grading_spec pure spec-builder."""
import pytest


def test_build_grading_spec_returns_overhead_raw_dispatch_spec():
    from src.core.grading import build_grading_spec
    source = {"id": 7, "title": "T", "description": "D",
              "result": "x" * 200, "context": "{}"}
    spec = build_grading_spec(source, exclusions=["bad-model"])
    assert isinstance(spec, dict)
    assert spec["agent_type"] == "reviewer"
    assert spec["kind"] == "overhead"
    llm = spec["context"]["llm_call"]
    assert llm["raw_dispatch"] is True
    assert llm["call_category"] == "overhead"
    assert "bad-model" in llm["exclude_models"]
    assert "x" * 100 in llm["messages"][1]["content"]


def test_build_grading_spec_auto_fails_trivial_output():
    from src.core.grading import build_grading_spec, GradeResult
    source = {"id": 7, "title": "T", "description": "D", "result": "  ", "context": "{}"}
    out = build_grading_spec(source, exclusions=[])
    assert isinstance(out, GradeResult)
    assert out.passed is False
    assert "auto-fail" in out.raw
