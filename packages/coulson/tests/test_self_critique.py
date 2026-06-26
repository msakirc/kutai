"""Z3 T2A — self-critique sub-iter guard tests.

All tests are synchronous (guards.py / self_critique.py are pure sync functions).
"""
from __future__ import annotations

import json
import pytest

from coulson.self_critique import (
    MAX_SELF_CRITIQUE_PASSES,
    SELF_CRITIQUE_OPT_OUT_AGENT_TYPES,
    build_self_critique_message,
    check_self_critique_sub_iter,
)
from coulson.guards import GuardCorrection


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _task(produces=None):
    ctx = {}
    if produces is not None:
        ctx["produces"] = produces
    return {"title": "Test task", "context": json.dumps(ctx)}


def _parsed_final():
    return {"action": "final_answer", "result": "done"}


def _parsed_tool():
    return {"action": "tool_call", "tool": "shell", "args": {}}


def _write_calls(paths):
    return [
        {"name": "write_file", "args": {"filepath": p}, "ok": True}
        for p in paths
    ]


# ────────────────────────────────────────────────────────────────────────────
# Opt-out roles
# ────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("role", sorted(SELF_CRITIQUE_OPT_OUT_AGENT_TYPES))
def test_opt_out_roles_return_none(role):
    result = check_self_critique_sub_iter(
        _parsed_final(),
        task=_task(produces=["src/foo.py"]),
        agent_type=role,
        self_critique_passes=0,
        tool_calls=_write_calls(["src/foo.py"]),
    )
    assert result is None, f"Expected None for opt-out role {role!r}"


# ────────────────────────────────────────────────────────────────────────────
# Empty produces → None
# ────────────────────────────────────────────────────────────────────────────

def test_empty_produces_returns_none():
    result = check_self_critique_sub_iter(
        _parsed_final(),
        task=_task(produces=[]),
        agent_type="coder",
        self_critique_passes=0,
    )
    assert result is None


def test_missing_produces_key_returns_none():
    result = check_self_critique_sub_iter(
        _parsed_final(),
        task=_task(),          # no produces key at all
        agent_type="coder",
        self_critique_passes=0,
    )
    assert result is None


def test_skips_when_write_tools_stripped():
    """Task 567381 ([1.0a] prior_art_query_plan, object schema → write_file
    auto-stripped). The self-critique guard must NOT nag 'Call write_file' when
    the agent physically cannot write (mirrors the grounding guard's skip at
    guards.py). For a write-stripped step the artifact IS the final_answer and
    the engine materializes it — the file-existence premise is moot, and the
    impossible re-emit demand loops the agent to max_iterations."""
    result = check_self_critique_sub_iter(
        _parsed_final(),
        task=_task(produces=["mission_90/.research/prior_art_queries.json"]),
        agent_type="query_planner",
        self_critique_passes=0,
        tool_calls=None,
        allowed_tools=["read_file", "smart_search"],   # no write tools
    )
    assert result is None


def test_fires_when_write_tools_present():
    """Self-critique still fires for a markdown/non-stripped step that keeps
    write_file (the agent CAN re-write the declared path)."""
    result = check_self_critique_sub_iter(
        _parsed_final(),
        task=_task(produces=["mission_90/.charter/non_goals.md"]),
        agent_type="writer",
        self_critique_passes=0,
        tool_calls=_write_calls(["mission_90/.charter/non_goals.md"]),
        allowed_tools=["read_file", "write_file"],
    )
    assert result is not None


def test_all_tools_default_none_still_fires():
    """allowed_tools=None means 'all tools' (write present) — guard unchanged."""
    result = check_self_critique_sub_iter(
        _parsed_final(),
        task=_task(produces=["mission_90/.charter/non_goals.md"]),
        agent_type="writer",
        self_critique_passes=0,
        tool_calls=_write_calls(["mission_90/.charter/non_goals.md"]),
    )
    assert result is not None


# ────────────────────────────────────────────────────────────────────────────
# Non-final_answer action → None
# ────────────────────────────────────────────────────────────────────────────

def test_tool_call_action_returns_none():
    result = check_self_critique_sub_iter(
        _parsed_tool(),
        task=_task(produces=["src/foo.py"]),
        agent_type="coder",
        self_critique_passes=0,
    )
    assert result is None


# ────────────────────────────────────────────────────────────────────────────
# First call returns correction + bumps counter externally
# ────────────────────────────────────────────────────────────────────────────

def test_first_call_returns_correction():
    result = check_self_critique_sub_iter(
        _parsed_final(),
        task=_task(produces=["src/foo.py", "src/bar.py"]),
        agent_type="coder",
        self_critique_passes=0,
        tool_calls=_write_calls(["src/foo.py", "src/bar.py"]),
    )
    assert result is not None
    assert isinstance(result, GuardCorrection)
    assert result.guard_name == "self_critique"
    assert isinstance(result.message, str)
    assert len(result.message) > 0


# ────────────────────────────────────────────────────────────────────────────
# Budget exhausted → None on second call
# ────────────────────────────────────────────────────────────────────────────

def test_second_call_returns_none_budget_exhausted():
    """After one pass the guard must not fire again."""
    assert MAX_SELF_CRITIQUE_PASSES == 1  # sanity

    result = check_self_critique_sub_iter(
        _parsed_final(),
        task=_task(produces=["src/foo.py"]),
        agent_type="coder",
        self_critique_passes=1,  # already consumed the 1 pass
    )
    assert result is None


def test_counter_above_max_also_returns_none():
    result = check_self_critique_sub_iter(
        _parsed_final(),
        task=_task(produces=["src/foo.py"]),
        agent_type="coder",
        self_critique_passes=99,
    )
    assert result is None


# ────────────────────────────────────────────────────────────────────────────
# Counter persists across simulated calls
# ────────────────────────────────────────────────────────────────────────────

def test_counter_persists_across_calls():
    """Simulate two calls as the react loop would do it."""
    passes = 0

    r1 = check_self_critique_sub_iter(
        _parsed_final(),
        task=_task(produces=["src/foo.py"]),
        agent_type="coder",
        self_critique_passes=passes,
        tool_calls=_write_calls(["src/foo.py"]),
    )
    assert r1 is not None
    passes += 1  # caller increments

    r2 = check_self_critique_sub_iter(
        _parsed_final(),
        task=_task(produces=["src/foo.py"]),
        agent_type="coder",
        self_critique_passes=passes,
        tool_calls=_write_calls(["src/foo.py"]),
    )
    assert r2 is None  # budget exhausted


# ────────────────────────────────────────────────────────────────────────────
# Message content checks
# ────────────────────────────────────────────────────────────────────────────

def test_message_contains_produces_paths():
    produces = ["src/models/user.py", "src/api/routes.py"]
    msg = build_self_critique_message(
        diff_summary="wrote user model and routes",
        produces=produces,
        agent_type="coder",
    )
    for p in produces:
        assert p in msg, f"Expected path {p!r} in message"


def test_message_contains_json_schema_instruction():
    msg = build_self_critique_message(
        diff_summary="wrote something",
        produces=["src/foo.py"],
        agent_type="coder",
    )
    assert '"verdict"' in msg
    assert '"clean"' in msg or "clean" in msg
    assert '"issues"' in msg or "issues" in msg
    assert '"findings"' in msg
    assert '"severity"' in msg
    assert '"file"' in msg
    assert '"why"' in msg


def test_message_contains_agent_type():
    msg = build_self_critique_message(
        diff_summary="",
        produces=["src/foo.py"],
        agent_type="implementer",
    )
    assert "implementer" in msg


def test_self_critique_foundry_char_exact():
    """The migrated rubric (rubrics/self_critique.yaml via build_messages) must
    reproduce the ORIGINAL frozen prompt string byte-for-byte. Locks the Phase 3
    Task 12 Batch H migration: any drift in the YAML breaks this."""
    diff_summary = "wrote user model and routes"
    produces = ["src/models/user.py", "src/api/routes.py"]
    agent_type = "coder"
    paths_block = "\n".join(f"  - {p}" for p in produces)
    expected = (
        f"You are acting as your own critic ({agent_type} self-review).\n\n"
        "Review the work you just completed for the following declared output "
        "paths:\n"
        f"{paths_block}\n\n"
        f"Work summary:\n{diff_summary}\n\n"
        "Check for:\n"
        "  1. Missing or empty files that were declared in the path list above\n"
        "  2. Obvious correctness errors visible from the summary alone\n"
        "  3. Incomplete implementations (stubs, TODOs, placeholder content)\n\n"
        "Respond with ONLY a JSON block in this exact schema:\n"
        "```json\n"
        '{"verdict": "clean"|"issues", "findings": ['
        '{"severity": "error"|"warning", "file": "<path>", "why": "<reason>"}]}\n'
        "```\n\n"
        'Use "clean" when all declared paths look correct. Use "issues" and '
        "populate findings when there are real problems that need fixing. "
        "Return ONLY the JSON — no prose before or after it."
    )
    got = build_self_critique_message(diff_summary, produces, agent_type)
    assert got == expected


def test_self_critique_foundry_empty_summary_fallback():
    """The '(no summary provided)' fallback (resolved in the builder, not the
    rubric) must survive the migration."""
    msg = build_self_critique_message("", ["src/foo.py"], "coder")
    assert "Work summary:\n(no summary provided)" in msg


def test_message_with_no_tool_calls_still_works():
    result = check_self_critique_sub_iter(
        _parsed_final(),
        task=_task(produces=["src/foo.py"]),
        agent_type="coder",
        self_critique_passes=0,
        tool_calls=None,
    )
    assert result is not None
    assert "src/foo.py" in result.message


def test_message_with_tool_calls_shows_written_paths():
    result = check_self_critique_sub_iter(
        _parsed_final(),
        task=_task(produces=["src/foo.py"]),
        agent_type="coder",
        self_critique_passes=0,
        tool_calls=_write_calls(["src/foo.py"]),
    )
    assert result is not None
    assert "src/foo.py" in result.message


# ────────────────────────────────────────────────────────────────────────────
# guards.check_sub_iter_guards integration
# ────────────────────────────────────────────────────────────────────────────

def test_guards_check_fires_self_critique():
    """check_sub_iter_guards should delegate and return self_critique."""
    from coulson.guards import check_sub_iter_guards
    from types import SimpleNamespace

    profile = SimpleNamespace(
        name="coder",
        allowed_tools=["write_file"],
        can_create_subtasks=False,
        _suppress_clarification=True,
    )
    task = _task(produces=["src/foo.py"])
    parsed = _parsed_final()
    # tools_used=True so hallucination guard skips; produces set so grounding
    # passes (writes match); critique should fire
    result = check_sub_iter_guards(
        parsed,
        profile=profile,
        iteration=0,
        tools_used=True,
        tools_used_names={"write_file"},
        task=task,
        search_depth="none",
        suppress_guards=False,
        tool_calls=_write_calls(["src/foo.py"]),
        self_critique_passes=0,
    )
    assert result is not None
    assert result.guard_name == "self_critique"


def test_guards_check_skips_self_critique_when_budget_spent():
    from coulson.guards import check_sub_iter_guards
    from types import SimpleNamespace

    profile = SimpleNamespace(
        name="coder",
        allowed_tools=["write_file"],
        can_create_subtasks=False,
        _suppress_clarification=True,
    )
    task = _task(produces=["src/foo.py"])
    result = check_sub_iter_guards(
        _parsed_final(),
        profile=profile,
        iteration=0,
        tools_used=True,
        tools_used_names={"write_file"},
        task=task,
        search_depth="none",
        suppress_guards=False,
        tool_calls=_write_calls(["src/foo.py"]),
        self_critique_passes=1,  # budget spent
    )
    # Search guard should not fire (no web tool, search_depth=none), so None
    assert result is None


def test_guards_check_opt_out_role_skips_self_critique():
    from coulson.guards import check_sub_iter_guards
    from types import SimpleNamespace

    profile = SimpleNamespace(
        name="code_reviewer",
        allowed_tools=None,
        can_create_subtasks=False,
        _suppress_clarification=True,
    )
    task = _task(produces=["src/foo.py"])
    result = check_sub_iter_guards(
        _parsed_final(),
        profile=profile,
        iteration=0,
        tools_used=True,
        tools_used_names={"write_file"},
        task=task,
        search_depth="none",
        suppress_guards=False,
        tool_calls=_write_calls(["src/foo.py"]),
        self_critique_passes=0,
    )
    assert result is None or result.guard_name != "self_critique"
