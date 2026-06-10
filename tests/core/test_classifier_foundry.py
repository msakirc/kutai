"""Test: classifier rubric builds the correct user message via build_messages."""


def test_classifier_uses_foundry_build():
    from finch.build import build_messages

    msgs = build_messages("classifier", {
        "task_description": "build a todo app: create a simple todo list",
    })
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    content = msgs[1]["content"]
    # Core classification prompt content
    assert "task classifier for an AI agent system" in content
    assert "AGENT PICK/REJECT RULES" in content
    assert "build a todo app" in content
    assert '"agent_type"' in content
    assert '"difficulty"' in content


def test_classifier_rubric_has_all_agent_types():
    """Rubric must contain all major agent type rules."""
    from finch.build import build_messages

    msgs = build_messages("classifier", {"task_description": "x"})
    content = msgs[1]["content"]
    for agent in ["coder", "implementer", "fixer", "reviewer", "researcher",
                  "shopping_advisor", "writer", "summarizer", "planner", "architect",
                  "assistant", "executor", "visual_reviewer"]:
        assert f'"{agent}"' in content, f"Missing agent type: {agent}"


def test_classifier_rubric_task_description_injection():
    """task_description field is substituted into the correct position."""
    from finch.build import build_messages

    unique_task = "ZZZ-unique-task-description-XYZ"
    msgs = build_messages("classifier", {"task_description": unique_task})
    content = msgs[1]["content"]
    assert f"Task: {unique_task}" in content


def test_classifier_rubric_json_example_present():
    """The JSON respond-as example must be present with single braces (not double)."""
    from finch.build import build_messages

    msgs = build_messages("classifier", {"task_description": "fix a bug"})
    content = msgs[1]["content"]
    # The example line must be present with single braces
    assert '"agent_type": "coder"' in content
    assert '"search_depth": "none"' in content


def test_classifier_char_exact_vs_old_constant():
    """Rubric output must be character-exact vs old CLASSIFIER_PROMPT.format().

    Reads the old constant from git history (HEAD before classifier migration commit).
    Uses bytes + UTF-8 decode to avoid subprocess encoding issues with em-dashes.
    """
    import sys, pathlib, re, subprocess
    _root = pathlib.Path(__file__).resolve().parents[2]

    result = subprocess.run(
        ["git", "show", "HEAD~1:src/core/task_classifier.py"],
        capture_output=True, cwd=str(_root),
    )
    if result.returncode != 0:
        import pytest
        pytest.skip("Cannot recover old CLASSIFIER_PROMPT from git history")

    old_src = result.stdout.decode("utf-8")
    m = re.search(r'CLASSIFIER_PROMPT = """(.*?)"""', old_src, re.DOTALL)
    if not m:
        import pytest
        pytest.skip("CLASSIFIER_PROMPT not found in HEAD~1")

    old_prompt_raw = m.group(1)

    from finch.build import build_messages
    task_desc = "build a todo app: create a simple todo list"[:500]
    old_content = old_prompt_raw.format(task_description=task_desc)

    msgs = build_messages("classifier", {"task_description": task_desc})
    new_content = msgs[1]["content"]

    assert new_content == old_content, (
        f"Content mismatch: len={len(new_content)} vs {len(old_content)}\n"
        f"First diff at: {next((i for i,(a,b) in enumerate(zip(new_content, old_content)) if a!=b), -1)}"
    )
