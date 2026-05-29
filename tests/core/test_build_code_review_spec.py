"""SP3 Task 2 - build_code_review_spec."""
def test_build_code_review_spec_shape():
    from src.core.code_review import build_code_review_spec
    source = {"id": 9, "title": "T", "description": "D",
              "result": "def f(): pass\n" * 20,
              "context": '{"produces": ["a.py"]}'}
    spec = build_code_review_spec(source, exclusions=[])
    assert spec["agent_type"] == "reviewer"
    assert spec["kind"] == "overhead"
    assert spec["context"]["llm_call"]["raw_dispatch"] is True

def test_build_code_review_spec_auto_fails_trivial():
    from src.core.code_review import build_code_review_spec, CodeReviewResult
    out = build_code_review_spec({"id": 9, "result": "x", "context": "{}"}, exclusions=[])
    assert isinstance(out, CodeReviewResult)
    assert out.passed is False
