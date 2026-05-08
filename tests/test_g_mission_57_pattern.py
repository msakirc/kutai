"""End-to-end coverage of the mission-57 fabrication pattern.

Mission 57 (2026-05-04 audit): 3,542 "completed" build tasks claimed
backend/app/X.py was written, but the workspace was empty. The agent
narrated "I wrote X" and emitted final_answer; nothing actually called
write_file.

G's job is to make this exact pattern impossible:
  - L1 sub-iter guard: in-loop, returns GuardCorrection with the
    specific missing path, agent gets a sub-iter retry within the
    same outer iteration (cheap, ~1 LLM call cost per catch)
  - L2 post-hook: after Complete persists, mechanical check matches
    tool_calls audit log against produces; fail -> retry source with
    feedback, honors retry budget, DLQ if exhausted

This file pins both layers against the canonical bad pattern.
"""
from __future__ import annotations

import json

import pytest

from coulson.grounding import (
    extract_written_paths,
    unmatched_produces,
)
from coulson.guards import check_grounding_sub_iter
from mr_roboto.check_grounding import check_grounding


# ── The canonical bad pattern ────────────────────────────────────────────
#
# Workflow step 8.F-001.feat.4 (backend_service template instance, mission
# 57 task 4458): produces=[backend/app/models/F-001.py, backend/app/services/
# F-001_service.py]. Agent emitted final_answer with a confident
# "I implemented the user model and service" but the audit log shows ONLY
# read_file calls — never write_file.

_PRODUCES = [
    "backend/app/models/F-001.py",
    "backend/app/services/F-001_service.py",
]

_NARRATING_TASK = {
    "id": 4458,
    "title": "Implement backend service for F-001",
    "context": json.dumps({
        "is_workflow_step": True,
        "produces": _PRODUCES,
        "post_hooks": ["grounding", "verify_artifacts", "code_review"],
    }),
}

_NARRATING_FINAL_ANSWER = {
    "action": "final_answer",
    "result": (
        "I implemented the User model with email validation and password "
        "hashing, plus the UserService with create/get/update methods. "
        "All wired against the existing database client; ready for tests."
    ),
}

# Audit log of what the agent ACTUALLY did: 3 reads, no writes. Exactly
# the mission-57 shape.
_NARRATING_TOOL_CALLS = [
    {"name": "read_file", "args": {"filepath": "backend/app/models/__init__.py"}, "ok": True},
    {"name": "read_file", "args": {"filepath": "implementation_context.md"}, "ok": True},
    {"name": "read_file", "args": {"filepath": "openapi_spec.json"}, "ok": True},
]


# ── L1 sub-iter guard catches it ─────────────────────────────────────────

class TestLayer1Catches:
    def test_l1_fires_on_narration(self):
        correction = check_grounding_sub_iter(
            parsed=_NARRATING_FINAL_ANSWER,
            task=_NARRATING_TASK,
            tool_calls=_NARRATING_TOOL_CALLS,
        )
        assert correction is not None, (
            "L1 must catch the mission-57 pattern: agent emitted "
            "final_answer with 0 write_file calls"
        )
        assert correction.guard_name == "grounding"

    def test_l1_message_names_both_missing_paths(self):
        correction = check_grounding_sub_iter(
            parsed=_NARRATING_FINAL_ANSWER,
            task=_NARRATING_TASK,
            tool_calls=_NARRATING_TOOL_CALLS,
        )
        for path in _PRODUCES:
            assert path in correction.message, (
                f"L1 retry feedback must name {path!r} so the agent "
                f"knows the exact target to write_file"
            )

    def test_l1_message_steers_to_write_file_call(self):
        correction = check_grounding_sub_iter(
            parsed=_NARRATING_FINAL_ANSWER,
            task=_NARRATING_TASK,
            tool_calls=_NARRATING_TOOL_CALLS,
        )
        # Concrete json example with the missing path baked in.
        assert "write_file" in correction.message
        assert "tool_call" in correction.message

    def test_l1_clears_when_agent_retries_with_real_writes(self):
        """After the L1 correction, agent's next sub-iter attempt SHOULD
        include actual write_file calls. Confirm guard goes silent."""
        good_calls = _NARRATING_TOOL_CALLS + [
            {"name": "write_file", "args": {"path": p}, "ok": True}
            for p in _PRODUCES
        ]
        correction = check_grounding_sub_iter(
            parsed=_NARRATING_FINAL_ANSWER,
            task=_NARRATING_TASK,
            tool_calls=good_calls,
        )
        assert correction is None


# ── L2 post-hook catches escapees ────────────────────────────────────────

class TestLayer2Catches:
    def test_l2_fails_on_narration(self):
        """L1 may be bypassed (suppress_guards, exhausted budget); L2 is
        the floor that catches anyway."""
        verdict = check_grounding(
            tool_calls=_NARRATING_TOOL_CALLS,
            produces=_PRODUCES,
        )
        assert verdict["passed"] is False
        # Both produces slots show up in missing
        for path in _PRODUCES:
            assert path in verdict["missing"]

    def test_l2_passes_after_real_writes(self):
        good_calls = _NARRATING_TOOL_CALLS + [
            {"name": "write_file", "args": {"path": p}, "ok": True}
            for p in _PRODUCES
        ]
        verdict = check_grounding(
            tool_calls=good_calls,
            produces=_PRODUCES,
        )
        assert verdict["passed"] is True
        assert verdict["missing"] == []
        for path in _PRODUCES:
            assert path in verdict["written"]

    def test_l2_partial_writes_still_fail(self):
        """If agent writes ONE of two declared paths, L2 still fails on
        the missing one — no silent pass for half-done work."""
        partial = _NARRATING_TOOL_CALLS + [
            {"name": "write_file", "args": {"path": _PRODUCES[0]}, "ok": True},
        ]
        verdict = check_grounding(tool_calls=partial, produces=_PRODUCES)
        assert verdict["passed"] is False
        assert verdict["missing"] == [_PRODUCES[1]]

    def test_l2_failed_writes_still_fail(self):
        """Agent attempted but the call errored — count as ungrounded."""
        attempted = _NARRATING_TOOL_CALLS + [
            {"name": "write_file", "args": {"path": p}, "ok": False}
            for p in _PRODUCES
        ]
        verdict = check_grounding(tool_calls=attempted, produces=_PRODUCES)
        assert verdict["passed"] is False


# ── Cross-layer invariants ───────────────────────────────────────────────

class TestCrossLayerInvariants:
    def test_layers_agree_on_pass_fail(self):
        """L1 and L2 must reach the same verdict on the same data —
        otherwise either L1 false-positives (wastes sub-iter budget) or
        L2 false-positives (wastes a whole task retry)."""
        scenarios = [
            ([], False),  # no calls
            (_NARRATING_TOOL_CALLS, False),  # reads only
            (
                [{"name": "write_file", "args": {"path": p}, "ok": True}
                 for p in _PRODUCES],
                True,
            ),
            (
                [{"name": "write_file", "args": {"path": _PRODUCES[0]}, "ok": True}],
                False,  # partial
            ),
        ]
        for tool_calls, expected_pass in scenarios:
            l1 = check_grounding_sub_iter(
                parsed={"action": "final_answer"},
                task=_NARRATING_TASK,
                tool_calls=tool_calls,
            )
            l1_pass = l1 is None
            l2 = check_grounding(tool_calls=tool_calls, produces=_PRODUCES)
            l2_pass = l2["passed"]
            assert l1_pass == l2_pass == expected_pass, (
                f"layer disagreement on {tool_calls!r}: "
                f"L1_pass={l1_pass}, L2_pass={l2_pass}, expected={expected_pass}"
            )

    def test_extract_written_paths_consistent_between_layers(self):
        """Both layers compute written-paths via the same helper. Pin
        the contract: any change to extract_written_paths must keep
        L1 + L2 in sync."""
        from coulson.grounding import extract_written_paths
        calls = _NARRATING_TOOL_CALLS + [
            {"name": "write_file", "args": {"path": _PRODUCES[0]}, "ok": True},
        ]
        # L1 path
        l1_written = extract_written_paths(calls)
        # L2 path (via mr_roboto verb result with non-empty produces — empty
        # produces short-circuits to vacuous-pass and skips the scan)
        l2_written = set(
            check_grounding(tool_calls=calls, produces=_PRODUCES)["written"]
        )
        assert l1_written == l2_written
