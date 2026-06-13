from finch.build import build_messages, register_rubric


def test_build_messages_system_plus_user():
    # Use a throwaway key, NOT the live "grading" rubric. register_rubric
    # mutates the process-global _RUBRICS dict; reusing "grading" here clobbers
    # the real YAML-loaded grading rubric for the rest of the pytest session
    # and makes coulson's grading-equivalence tests fail order-dependently.
    register_rubric("_test_generic", system="You are a strict SEMANTIC evaluator.",
                    user_template="Task: {title}\nResult: {response}")
    msgs = build_messages("_test_generic", {"title": "T", "response": "R"})
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


def test_render_no_double_substitution():
    """A field VALUE containing another field's {token} must NOT be re-substituted.

    Sequential str.replace would turn the literal "{response}" inside the
    description into "REAL". Single-pass re.sub leaves it verbatim.
    """
    register_rubric(
        "_test_nodouble",
        system="S",
        user_template="{title} | {description} | {response}",
    )
    msgs = build_messages(
        "_test_nodouble",
        {"title": "T", "description": "Build {response} handler", "response": "REAL"},
    )
    assert msgs[1]["content"] == "T | Build {response} handler | REAL"


def test_render_substring_key_no_shadow():
    """A short key must not shadow a longer token sharing its prefix."""
    register_rubric("_test_substr", system="S", user_template="{a} {ab}")
    msgs = build_messages("_test_substr", {"a": "X", "ab": "Y"})
    assert msgs[1]["content"] == "X Y"
