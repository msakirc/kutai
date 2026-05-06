"""Coverage: L2 grounding post-hook (mechanical) end-to-end.

Layer 2 of G: catches narration-as-completion that escaped the L1
sub-iter guard. Mechanical post-hook reads source task's tool_calls +
produces from ctx, computes pass/fail via salako.check_grounding, applies
verdict via beckman _apply_grounding_verdict.

Auto-wired in expander: any step with declared produces gets "grounding"
prepended to post_hooks (runs before verify_artifacts).
"""
from __future__ import annotations

import json

import pytest


# ── salako.check_grounding verb ──────────────────────────────────────────

class TestSalakoCheckGroundingVerb:
    def test_pass_when_writes_match_produces(self):
        from salako.check_grounding import check_grounding
        res = check_grounding(
            tool_calls=[
                {"name": "write_file", "args": {"path": "x.py"}, "ok": True},
            ],
            produces=["x.py"],
        )
        assert res["passed"] is True
        assert res["missing"] == []
        assert "x.py" in res["written"]

    def test_fail_when_no_writes(self):
        from salako.check_grounding import check_grounding
        res = check_grounding(
            tool_calls=[
                {"name": "read_file", "args": {"path": "x.py"}, "ok": True},
            ],
            produces=["x.py"],
        )
        assert res["passed"] is False
        assert res["missing"] == ["x.py"]
        assert res["written"] == []

    def test_pass_with_any_of_alternative(self):
        from salako.check_grounding import check_grounding
        res = check_grounding(
            tool_calls=[
                {"name": "write_file", "args": {"path": "prisma/schema.prisma"}, "ok": True},
            ],
            produces=[["alembic.ini", "prisma/schema.prisma", "drizzle.config.ts"]],
        )
        assert res["passed"] is True

    def test_pass_with_glob(self):
        from salako.check_grounding import check_grounding
        res = check_grounding(
            tool_calls=[
                {"name": "write_file", "args": {"path": "migrations/v/01.py"}, "ok": True},
            ],
            produces=["migrations/**/*.py"],
        )
        assert res["passed"] is True

    def test_empty_produces_passes_vacuously(self):
        from salako.check_grounding import check_grounding
        res = check_grounding(tool_calls=[], produces=[])
        assert res["passed"] is True

    def test_failed_writes_excluded(self):
        from salako.check_grounding import check_grounding
        res = check_grounding(
            tool_calls=[
                {"name": "write_file", "args": {"path": "x.py"}, "ok": False},
            ],
            produces=["x.py"],
        )
        assert res["passed"] is False


# ── salako.run dispatcher ────────────────────────────────────────────────

class TestSalakoRunDispatch:
    @pytest.mark.asyncio
    async def test_dispatch_check_grounding_pass(self):
        from salako import run
        action = await run({
            "id": 1,
            "mission_id": None,
            "payload": {
                "action": "check_grounding",
                "produces": ["x.py"],
                "tool_calls": [
                    {"name": "write_file", "args": {"path": "x.py"}, "ok": True},
                ],
            },
        })
        assert action.status == "completed"
        assert action.result["passed"] is True

    @pytest.mark.asyncio
    async def test_dispatch_check_grounding_fail(self):
        from salako import run
        action = await run({
            "id": 1,
            "mission_id": None,
            "payload": {
                "action": "check_grounding",
                "produces": ["x.py"],
                "tool_calls": [],
            },
        })
        assert action.status == "failed"
        assert "missing" in action.error
        assert action.result["passed"] is False


# ── beckman determine_posthooks ──────────────────────────────────────────

class TestDeterminePosthooksGrounding:
    def test_grounding_kind_is_known(self):
        from general_beckman.posthooks import _KNOWN_EXTRA_KINDS
        assert "grounding" in _KNOWN_EXTRA_KINDS

    def test_grounding_propagates_when_listed(self):
        from general_beckman.posthooks import determine_posthooks
        kinds = determine_posthooks(
            task={"agent_type": "coder"},
            task_ctx={"post_hooks": ["grounding", "verify_artifacts"]},
            result={},
        )
        assert "grounding" in kinds
        assert "verify_artifacts" in kinds


# ── beckman _posthook_agent_and_payload ──────────────────────────────────

class TestPosthookPayloadGrounding:
    def test_payload_carries_produces_and_tool_calls(self):
        from general_beckman.apply import _posthook_agent_and_payload
        from general_beckman.result_router import RequestPostHook
        a = RequestPostHook(source_task_id=42, kind="grounding", source_ctx={})
        agent_type, payload = _posthook_agent_and_payload(
            a,
            source={"id": 42, "title": "build x"},
            source_ctx={
                "produces": ["backend/x.py", ["a.py", "b.py"]],
                "tool_calls": [
                    {"name": "write_file", "args": {"path": "x.py"}, "ok": True},
                ],
            },
        )
        assert agent_type == "mechanical"
        assert payload["posthook_kind"] == "grounding"
        assert payload["payload"]["action"] == "check_grounding"
        assert payload["payload"]["produces"] == ["backend/x.py", ["a.py", "b.py"]]
        assert len(payload["payload"]["tool_calls"]) == 1


# ── beckman _posthook_title ──────────────────────────────────────────────

class TestPosthookTitleGrounding:
    def test_title_format(self):
        from general_beckman.apply import _posthook_title
        from general_beckman.result_router import RequestPostHook
        a = RequestPostHook(source_task_id=99, kind="grounding", source_ctx={})
        assert _posthook_title(a, source={}) == "Grounding check for #99"


# ── expander auto-wire ───────────────────────────────────────────────────

class TestExpanderAutoWire:
    def _step(self, *, produces=None, post_hooks=None, **overrides) -> dict:
        base = {
            "id": "9.99",
            "phase": "phase_9",
            "name": "build_x",
            "agent": "coder",
            "difficulty": "easy",
            "tools_hint": ["write_file"],
            "depends_on": [],
            "input_artifacts": [],
            "output_artifacts": ["x_result"],
            "instruction": "build it",
            "done_when": "x_result exists",
            "artifact_schema": {"x_result": {"type": "object", "fields": {}}},
        }
        if produces is not None:
            base["produces"] = produces
        if post_hooks is not None:
            base["post_hooks"] = post_hooks
        base.update(overrides)
        return base

    def _ctx(self, task: dict) -> dict:
        from src.workflows.engine.expander import expand_steps_to_tasks
        tasks = expand_steps_to_tasks([task], mission_id=1, initial_context={})
        raw = tasks[0].get("context")
        return json.loads(raw) if isinstance(raw, str) else raw or {}

    def test_grounding_prepended_when_produces_declared(self):
        step = self._step(produces=["x.py"], post_hooks=["verify_artifacts"])
        ctx = self._ctx(step)
        assert ctx["post_hooks"] == ["grounding", "verify_artifacts"]

    def test_grounding_prepended_with_no_explicit_post_hooks(self):
        """produces alone is enough — even without explicit post_hooks."""
        step = self._step(produces=["x.py"])
        ctx = self._ctx(step)
        assert ctx["post_hooks"] == ["grounding"]

    def test_grounding_idempotent_when_already_listed(self):
        step = self._step(
            produces=["x.py"],
            post_hooks=["grounding", "verify_artifacts"],
        )
        ctx = self._ctx(step)
        # Single grounding entry, listed first.
        assert ctx["post_hooks"].count("grounding") == 1
        assert ctx["post_hooks"][0] == "grounding"

    def test_no_grounding_when_no_produces(self):
        step = self._step(post_hooks=["verify_artifacts"])
        ctx = self._ctx(step)
        assert "grounding" not in (ctx.get("post_hooks") or [])

    def test_no_grounding_when_produces_filtered_to_empty(self):
        step = self._step(produces=[42, ""])  # all invalid
        ctx = self._ctx(step)
        assert "grounding" not in (ctx.get("post_hooks") or [])
