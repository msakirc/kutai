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


def test_build_grading_spec_never_truncates_description():
    """The instruction IS the grading contract. Truncating it severs the tail
    of the requirement list, so the grader sees a partial spec and flags the
    cut-off requirements as 'extra / not requested'. This is the real cause of
    the #289700 charter false-DLQ (2026-06-04): the 1329-char 5-section charter
    instruction was cut at 500 chars, dropping sections 4 (Goals & Mission) and
    5 (Solutions We Own) — the grader then reported the artifact 'added a sixth
    section'. NO TRUNCATION of any grader input, ever."""
    from src.core.grading import build_grading_spec
    # A long instruction whose REQUIRED items live past the old 500/30000 caps.
    desc = (
        "Write the charter with EXACTLY these five `## ` sections in order:\n"
        + "filler line that pushes the tail past the old 500-char cap. " * 12
        + "\n4. **Goals & Mission**\n5. **Solutions We Own**"
    )
    assert len(desc) > 500
    source = {"id": 9, "title": "T", "description": desc,
              "result": "x" * 200, "context": "{}"}
    content = build_grading_spec(source, exclusions=[])["context"]["llm_call"]["messages"][1]["content"]
    # The whole contract reaches the grader — including the tail sections.
    assert desc in content
    assert "Goals & Mission" in content
    assert "Solutions We Own" in content


def test_build_grading_spec_never_truncates_result_or_title():
    """A long artifact truncated mid-document reads as 'incomplete' to the
    grader → false FAIL. The full output and full title must reach the grader."""
    from src.core.grading import build_grading_spec
    big_result = "A" * 40_000 + " TAIL_MARKER_END"
    long_title = "T" * 300 + " TITLE_END"
    source = {"id": 9, "title": long_title, "description": "D" * 10,
              "result": big_result, "context": "{}"}
    spec = build_grading_spec(source, exclusions=[])
    content = spec["context"]["llm_call"]["messages"][1]["content"]
    assert "TAIL_MARKER_END" in content      # full result, no 30000 cap
    assert "TITLE_END" in content            # full title, no 100 cap
    # The input estimate must reflect the real (untruncated) size so selection
    # picks a model whose context window fits — else the call-level cap becomes
    # the new silent-truncation point.
    assert spec["context"]["llm_call"]["estimated_input_tokens"] > 9_000
