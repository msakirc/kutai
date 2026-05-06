"""Coverage: tool-call grounding helpers + sub-iter grounding guard.

Layer 1 of G: catches narration-as-completion inside a single outer
iteration. Agent emits final_answer, guard scans tool_calls log for
write_file invocations matching workflow ``produces`` declaration. If
nothing matches, returns a path-specific correction; agent gets a cheap
sub-iter retry with concrete feedback ("call write_file({filepath: X})
before final_answer").
"""
from __future__ import annotations

import json

import pytest

from coulson.grounding import (
    WRITE_TOOLS,
    build_grounding_message,
    extract_written_paths,
    match_produces_entry,
    unmatched_produces,
)
from coulson.guards import (
    GuardCorrection,
    check_grounding_sub_iter,
    check_sub_iter_guards,
)


# ── extract_written_paths ────────────────────────────────────────────────

def test_extract_includes_only_successful_writes():
    calls = [
        {"name": "write_file", "args": {"path": "a.py"}, "ok": True},
        {"name": "write_file", "args": {"path": "b.py"}, "ok": False},   # failed
        {"name": "read_file",  "args": {"path": "c.py"}, "ok": True},    # not write
        {"name": "shell",      "args": {"command": "ls"}, "ok": True},   # no path
    ]
    assert extract_written_paths(calls) == {"a.py"}


def test_extract_recognizes_path_aliases():
    calls = [
        {"name": "write_file", "args": {"filepath": "x.py"}, "ok": True},
        {"name": "edit_file",  "args": {"file": "y.py"},     "ok": True},
        {"name": "patch_file", "args": {"target": "z.py"},   "ok": True},
    ]
    assert extract_written_paths(calls) == {"x.py", "y.py", "z.py"}


def test_extract_normalizes_separators_and_dot_prefix():
    calls = [
        {"name": "write_file", "args": {"path": ".\\backend\\x.py"}, "ok": True},
        {"name": "write_file", "args": {"path": "./frontend/y.ts"},  "ok": True},
    ]
    paths = extract_written_paths(calls)
    assert "backend/x.py" in paths
    assert "frontend/y.ts" in paths


def test_extract_handles_empty_or_malformed_log():
    assert extract_written_paths([]) == set()
    assert extract_written_paths(None) == set()  # type: ignore[arg-type]
    assert extract_written_paths([{}, {"name": None}, "garbage"]) == set()  # type: ignore[list-item]


def test_write_tools_set_includes_canonical_names():
    """If a write tool is renamed, the grounding fix is to add it here."""
    assert "write_file" in WRITE_TOOLS
    assert "edit_file" in WRITE_TOOLS
    assert "patch_file" in WRITE_TOOLS


# ── match_produces_entry ─────────────────────────────────────────────────

def test_match_literal_path():
    written = {"backend/app/x.py"}
    assert match_produces_entry("backend/app/x.py", written) is True
    assert match_produces_entry("backend/app/y.py", written) is False


def test_match_normalizes_pattern_separators():
    written = {"a/b/c.py"}
    assert match_produces_entry("a\\b\\c.py", written) is True
    assert match_produces_entry("./a/b/c.py", written) is True


def test_match_glob_pattern():
    """fnmatch-based matching: ``*`` matches across path separators,
    so ``migrations/*.py`` matches nested dirs too. Permissive direction
    is safe for grounding (false-positive = "agent did write SOMETHING
    matching the intent"; false-negative would be the harmful direction)."""
    written = {"migrations/versions/001_initial.py"}
    assert match_produces_entry("migrations/**/*.py", written) is True
    assert match_produces_entry("migrations/*.py", written) is True
    assert match_produces_entry("backend/**/*.py", written) is False
    # Pure literal is still strict.
    assert match_produces_entry("migrations/versions/001_initial.py", written) is True
    assert match_produces_entry("migrations/versions/002_initial.py", written) is False


def test_match_any_of_satisfied_by_one_alternative():
    written = {"prisma/schema.prisma"}
    entry = ["alembic.ini", "prisma/schema.prisma", "drizzle.config.ts"]
    assert match_produces_entry(entry, written) is True


def test_match_any_of_with_glob_alternative():
    written = {"migrations/v/01.py"}
    entry = ["alembic.ini", "migrations/**/*.py"]
    assert match_produces_entry(entry, written) is True


def test_match_any_of_all_miss():
    written = {"unrelated.py"}
    entry = ["a.py", "b.py", "c.py"]
    assert match_produces_entry(entry, written) is False


def test_match_invalid_entry_returns_false():
    assert match_produces_entry(42, {"x"}) is False  # type: ignore[arg-type]
    assert match_produces_entry("", {"x"}) is False
    assert match_produces_entry([""], {"x"}) is False


# ── unmatched_produces ───────────────────────────────────────────────────

def test_unmatched_returns_only_missing_entries():
    produces = ["a.py", "b.py", ["c.py", "d.py"]]
    written = {"a.py", "d.py"}
    assert unmatched_produces(produces, written) == ["b.py"]


def test_unmatched_empty_when_all_satisfied():
    assert unmatched_produces(["a.py"], {"a.py"}) == []


def test_unmatched_returns_all_when_nothing_written():
    produces = ["a.py", ["b.py", "c.py"]]
    assert unmatched_produces(produces, set()) == produces


def test_unmatched_handles_non_list_input():
    assert unmatched_produces("oops", {"x"}) == []  # type: ignore[arg-type]


# ── build_grounding_message ──────────────────────────────────────────────

def test_message_lists_missing_paths():
    msg = build_grounding_message(
        missing=["backend/x.py", ["a.py", "b.py"]],
        written={"frontend/y.ts"},
        task_title="impl feature foo",
    )
    assert "backend/x.py" in msg
    assert "any of: a.py, b.py" in msg
    assert "frontend/y.ts" in msg
    assert "impl feature foo" in msg
    assert "STOP" in msg


def test_message_handles_no_writes():
    msg = build_grounding_message(missing=["x.py"], written=set())
    assert "no write_file" in msg


def test_message_picks_concrete_example_path():
    msg = build_grounding_message(missing=["concrete/path.py"], written=set())
    # The example block should reference the missing path so the agent has
    # an exact target to copy.
    assert "concrete/path.py" in msg


def test_message_picks_first_alt_for_any_of_example():
    msg = build_grounding_message(missing=[["alembic.ini", "prisma/schema.prisma"]], written=set())
    assert "alembic.ini" in msg


# ── check_grounding_sub_iter ─────────────────────────────────────────────

def _task(produces=None, title="do thing") -> dict:
    return {
        "id": 1,
        "title": title,
        "context": {"produces": produces or [], "is_workflow_step": True},
    }


def test_guard_passes_when_produces_empty():
    correction = check_grounding_sub_iter(
        parsed={"action": "final_answer"},
        task=_task(produces=[]),
        tool_calls=[],
    )
    assert correction is None


def test_guard_passes_when_no_context():
    correction = check_grounding_sub_iter(
        parsed={"action": "final_answer"},
        task={"id": 1, "title": "x", "context": None},
        tool_calls=[],
    )
    assert correction is None


def test_guard_passes_when_action_is_tool_call():
    """Only fires on final_answer; tool_call iters bypass."""
    correction = check_grounding_sub_iter(
        parsed={"action": "tool_call", "tool": "shell"},
        task=_task(produces=["x.py"]),
        tool_calls=[],
    )
    assert correction is None


def test_guard_fires_when_produces_unwritten():
    correction = check_grounding_sub_iter(
        parsed={"action": "final_answer", "result": "done"},
        task=_task(produces=["backend/app/x.py"], title="impl X"),
        tool_calls=[
            {"name": "read_file", "args": {"filepath": "spec.md"}, "ok": True},
        ],
    )
    assert correction is not None
    assert isinstance(correction, GuardCorrection)
    assert correction.guard_name == "grounding"
    assert "backend/app/x.py" in correction.message
    assert "impl X" in correction.message


def test_guard_passes_when_write_matches():
    correction = check_grounding_sub_iter(
        parsed={"action": "final_answer"},
        task=_task(produces=["backend/x.py"]),
        tool_calls=[
            {"name": "write_file", "args": {"path": "backend/x.py"}, "ok": True},
        ],
    )
    assert correction is None


def test_guard_passes_when_glob_satisfied():
    correction = check_grounding_sub_iter(
        parsed={"action": "final_answer"},
        task=_task(produces=["migrations/**/*.py"]),
        tool_calls=[
            {"name": "write_file", "args": {"path": "migrations/v/001.py"}, "ok": True},
        ],
    )
    assert correction is None


def test_guard_passes_when_any_of_satisfied():
    correction = check_grounding_sub_iter(
        parsed={"action": "final_answer"},
        task=_task(produces=[["alembic.ini", "prisma/schema.prisma"]]),
        tool_calls=[
            {"name": "write_file", "args": {"path": "prisma/schema.prisma"}, "ok": True},
        ],
    )
    assert correction is None


def test_guard_ignores_failed_writes():
    """Agent attempted but the call failed — count as ungrounded."""
    correction = check_grounding_sub_iter(
        parsed={"action": "final_answer"},
        task=_task(produces=["x.py"]),
        tool_calls=[
            {"name": "write_file", "args": {"path": "x.py"}, "ok": False},
        ],
    )
    assert correction is not None


def test_guard_respects_suppress_flag():
    correction = check_grounding_sub_iter(
        parsed={"action": "final_answer"},
        task=_task(produces=["x.py"]),
        tool_calls=[],
        suppress_guards=True,
    )
    assert correction is None


def test_guard_decodes_string_context():
    """Beckman sometimes round-trips context as JSON string."""
    task = {
        "id": 1,
        "title": "x",
        "context": json.dumps({"produces": ["x.py"]}),
    }
    correction = check_grounding_sub_iter(
        parsed={"action": "final_answer"},
        task=task,
        tool_calls=[],
    )
    assert correction is not None
    assert "x.py" in correction.message


# ── integration with check_sub_iter_guards ───────────────────────────────

class _Profile:
    name = "coder"
    allowed_tools = ["write_file", "read_file"]
    can_create_subtasks = False
    _suppress_clarification = False


def test_grounding_fires_through_check_sub_iter_guards_dispatch():
    correction = check_sub_iter_guards(
        parsed={"action": "final_answer"},
        profile=_Profile(),
        iteration=0,
        tools_used=True,
        tools_used_names={"read_file"},  # non-empty so hallucination guard skips
        task=_task(produces=["backend/x.py"]),
        search_depth="none",
        suppress_guards=False,
        tool_calls=[
            {"name": "read_file", "args": {"path": "spec.md"}, "ok": True},
        ],
    )
    assert correction is not None
    assert correction.guard_name == "grounding"


def test_legacy_callers_without_tool_calls_kwarg_still_work():
    """check_sub_iter_guards must remain backwards compatible."""
    correction = check_sub_iter_guards(
        parsed={"action": "final_answer"},
        profile=_Profile(),
        iteration=0,
        tools_used=True,
        tools_used_names={"read_file"},
        task=_task(produces=["x.py"]),
        search_depth="none",
        suppress_guards=False,
        # tool_calls deliberately omitted
    )
    # No grounding check runs without tool_calls; other guards may or may
    # not fire (none should here). Either way: no crash.
    assert correction is None or correction.guard_name != "grounding"
