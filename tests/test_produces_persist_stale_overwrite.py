"""produces-persist must overwrite a STALE junk file, but preserve rich content.

mission_79 (2026-05-30): step 0.0c interview_script. The producer's write_file
call persisted a fenced, TRUNCATED final_answer envelope to
``.intake/interview_script.md`` (```json {"action":"final_answer","result":
"## Analysis...} cut mid-string, 1087 bytes), while the producer's RESULT held
the good 5093-char rendered script (written to the mission-root path by the
legacy `<name>.md` block). The produces-persist block's "only FILL missing"
guard saw the .intake file already existed and skipped it — locking in the
truncated junk. The shape validator (0.0c.verify) then correctly failed on the
junk file forever -> DLQ.

The guard exists to protect intake #73 (don't clobber a rich agent-written
file with a thin result summary). The fix narrows it: overwrite only when the
existing file is JUNK (a raw/truncated LLM envelope or degenerate/empty) AND
the new value is substantial + clean. Rich/valid artifacts are still preserved.
"""
from __future__ import annotations

import json

from src.workflows.engine.hooks import _produces_file_is_stale


# A substantial, NON-repetitive replacement artifact. Must not itself trip the
# dogru_mu_samet degeneracy check (the source refuses to replace junk with junk),
# so each line is distinct.
_GOOD_SCRIPT = "# Interview Script\n\n" + "\n".join(
    f"## Q{i} — {topic}\nProbe the user about {topic.lower()} and record specifics."
    for i, topic in enumerate(
        [
            "Onboarding", "Daily Habit", "Streak Recovery", "Gamification Appeal",
            "Errand Tracking", "Notification Tolerance", "Social Accountability",
            "Switching Trigger", "Privacy Concerns", "Pricing Sensitivity",
            "Feature Gaps", "Retention Risk",
        ],
        start=1,
    )
)


def test_fenced_truncated_final_answer_envelope_is_stale():
    """The exact mission_79 shape: ```json fence + truncated final_answer."""
    junk = '```json\r\n{\r\n  "action": "final_answer",\r\n  "result": "## Analysis: Interview'
    assert _produces_file_is_stale(junk, _GOOD_SCRIPT) is True


def test_closed_final_answer_envelope_is_stale():
    junk = json.dumps({"action": "final_answer", "result": "## short"})
    assert _produces_file_is_stale(junk, _GOOD_SCRIPT) is True


def test_tool_call_envelope_is_stale():
    junk = json.dumps({"action": "tool_call", "tool": "write_file", "args": {}})
    assert _produces_file_is_stale(junk, _GOOD_SCRIPT) is True


def test_call_result_envelope_is_stale():
    junk = json.dumps({"content": "x", "model": "gemini", "usage": {}, "ran_on": "g"})
    assert _produces_file_is_stale(junk, _GOOD_SCRIPT) is True


def test_empty_existing_is_stale():
    assert _produces_file_is_stale("", _GOOD_SCRIPT) is True
    assert _produces_file_is_stale("   \n ", _GOOD_SCRIPT) is True


def test_rich_markdown_artifact_is_preserved():
    """intake #73: a real agent-written artifact must NEVER be clobbered.

    Body is genuine, varied prose (not repeated filler) so it is neither an
    envelope nor degenerate — exactly the case the preserve guard protects."""
    rich = (
        "# Real Charter\n\n## Positioning\n"
        "HabitFlow targets ambitious professionals who abandon goals within weeks. "
        "We compete in habit-tracking but win on gamified errand management.\n\n"
        "## Brand Keywords\n* Motivating — turns chores into rewarding streaks.\n"
        "* Integrated — habits, tasks, and errands in one workflow.\n\n"
        "## Core Problem\nMost productivity tools treat habits as obligations, "
        "so motivation decays and users churn back to spreadsheets.\n"
    )
    assert _produces_file_is_stale(rich, _GOOD_SCRIPT) is False


def test_legit_json_artifact_is_preserved():
    """A bare JSON artifact (no action/CallResult markers) is real content."""
    legit = json.dumps({"todos": [{"id": 1, "text": "a"}], "summary": "x" * 200})
    assert _produces_file_is_stale(legit, _GOOD_SCRIPT) is False


def test_short_new_value_never_overwrites():
    """Never replace anything with a thin new value, even over junk."""
    junk = json.dumps({"action": "final_answer", "result": "x"})
    assert _produces_file_is_stale(junk, "too short") is False


def test_degenerate_new_value_not_used():
    junk = json.dumps({"action": "final_answer", "result": "x" * 200})
    degenerate_new = "spam spam spam " * 100
    # degenerate new content must not overwrite (would just replace junk w/ junk)
    assert _produces_file_is_stale(junk, degenerate_new) is False
