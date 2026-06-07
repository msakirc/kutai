"""SP3 Task 12 - no await_inline in migrated post-hook source files."""
FILES = ["src/core/grading.py", "src/core/code_review.py", "src/workflows/engine/hooks.py"]

def test_no_await_inline_true():
    for path in FILES:
        with open(path, encoding="utf-8") as f:
            src = f.read()
        assert "await_inline=True" not in src, f"await_inline=True still in {path}"

def test_parsers_and_builders_survive():
    from src.core.grading import parse_grade_response, build_grading_spec, GradeResult
    from src.core.code_review import parse_code_review_response, build_code_review_spec, CodeReviewResult
    from src.workflows.engine.hooks import build_summary_spec

def test_dead_functions_removed():
    import src.core.grading as g
    import src.core.code_review as cr
    import src.workflows.engine.hooks as h
    assert not hasattr(g, "grade_task")
    assert not hasattr(g, "apply_grade_result")  # deleted 2026-06-07 (dead, zero callers)
    assert not hasattr(cr, "code_review_task")
    assert not hasattr(h, "_llm_summarize")
