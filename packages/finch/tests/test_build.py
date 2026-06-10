from finch.build import build_messages, register_rubric


def test_build_messages_system_plus_user():
    register_rubric("grading", system="You are a strict SEMANTIC evaluator.",
                    user_template="Task: {title}\nResult: {response}")
    msgs = build_messages("grading", {"title": "T", "response": "R"})
    assert msgs[0] == {"role": "system", "content": "You are a strict SEMANTIC evaluator."}
    assert msgs[1]["role"] == "user"
    assert "Task: T" in msgs[1]["content"]
    assert "Result: R" in msgs[1]["content"]


def test_build_messages_appends_dynamic_blocks():
    register_rubric("r2", system="S", user_template="U")
    msgs = build_messages("r2", {}, extra_blocks=["BLOCK1", "BLOCK2"])
    assert msgs[0]["content"] == "S\n\nBLOCK1\n\nBLOCK2"


def test_literal_braces_survive():
    """Literal JSON braces must pass through untouched; only named placeholders are replaced."""
    register_rubric(
        "grade_json",
        system="You are a grader.",
        user_template='Grade this. Output {"verdict": "PASS"} format.\nResult: {response}',
    )
    msgs = build_messages("grade_json", {"response": "R"})
    user_content = msgs[1]["content"]
    assert '{"verdict": "PASS"}' in user_content, "Literal JSON braces were mangled"
    assert "Result: R" in user_content, "Placeholder {response} was not substituted"
