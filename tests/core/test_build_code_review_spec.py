"""SP3 Task 2 - build_code_review_spec."""
def test_build_code_review_spec_shape():
    from src.core.code_review import build_code_review_spec
    source = {
        "id": 9, "title": "Add login", "description": "Implement login handler",
        "result": (
            "def login(user, pw):\n"
            "    row = db.query(user)\n"
            "    if row is None:\n"
            "        return None\n"
            "    return verify(row.hash, pw)\n"
        ),
        "context": '{"produces": ["auth.py"]}',
    }
    spec = build_code_review_spec(source, exclusions=[])
    assert isinstance(spec, dict)
    assert spec["agent_type"] == "reviewer"
    assert spec["kind"] == "overhead"
    assert spec["context"]["llm_call"]["raw_dispatch"] is True


def test_build_code_review_spec_auto_fails_trivial():
    from src.core.code_review import build_code_review_spec, CodeReviewResult
    out = build_code_review_spec({"id": 9, "result": "x", "context": "{}"}, exclusions=[])
    assert isinstance(out, CodeReviewResult)
    assert out.passed is False


def test_build_code_review_spec_auto_fails_degenerate():
    from src.core.code_review import build_code_review_spec, CodeReviewResult
    source = {"id": 9, "title": "T", "description": "D",
              "result": "def f(): pass\n" * 30, "context": "{}"}
    out = build_code_review_spec(source, exclusions=[])
    assert isinstance(out, CodeReviewResult)
    assert out.passed is False
