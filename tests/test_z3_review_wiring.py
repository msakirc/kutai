"""Z3 review-pass — verdict round-trip wiring for config-only reviewers
and the integration_bisect advisory spawn.

Gaps found during the R1-R4 review pass:

1. integration_reviewer (Z3 T2C) + adr_drift_judge (Z3 R3) are config-only
   LLM agents — they have no execute() override to build a posthook_verdict
   payload the way grader / code_reviewer do. rewrite.py only translated
   grader / artifact_summarizer / code_reviewer completions into
   PostHookVerdict actions, so an integration_reviewer / adr_drift_judge
   task completed and its verdict went NOWHERE — the source stayed
   ``ungraded`` forever. Fixed by rewrite.py Rule 0d.

2. integration_bisect (Z3 R4) was never dispatched from any production
   path — the verb + its mission_lessons emission were dead code. Fixed by
   _maybe_spawn_integration_bisect on integration_replay fail.

3. _apply_posthook_verdict had no branch for kind="integration_review" — even
   a correctly-synthesised verdict fell through to "unknown kind". Fixed by
   adding integration_review to the simple_blocker dispatch tuple.
"""
from __future__ import annotations

import asyncio
import json

import pytest


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Gap 1 — rewrite.py Rule 0d: config-only reviewer → PostHookVerdict
# ---------------------------------------------------------------------------

class TestRule0dVerdictTranslation:
    def _complete(self, agent_type: str, result, posthook_kind="integration_review"):
        from general_beckman.result_router import Complete
        task = {"id": 5, "agent_type": agent_type, "mission_id": 7}
        task_ctx = {"source_task_id": 99, "posthook_kind": posthook_kind}
        a = Complete(task_id=5, result=result, raw={})
        return task, task_ctx, a

    def test_integration_reviewer_pass_translates(self):
        from general_beckman.rewrite import _rewrite_one
        from general_beckman.result_router import PostHookVerdict
        task, ctx, a = self._complete(
            "integration_reviewer",
            json.dumps({"verdict": "pass", "findings": []}),
        )
        out = _rewrite_one(task, ctx, a)
        verdicts = [x for x in out if isinstance(x, PostHookVerdict)]
        assert len(verdicts) == 1
        assert verdicts[0].kind == "integration_review"
        assert verdicts[0].source_task_id == 99
        assert verdicts[0].passed is True

    def test_integration_reviewer_fail_translates(self):
        from general_beckman.rewrite import _rewrite_one
        from general_beckman.result_router import PostHookVerdict
        task, ctx, a = self._complete(
            "integration_reviewer",
            json.dumps({
                "verdict": "fail",
                "findings": [{"severity": "critical", "file": "x.py", "why": "mismatch"}],
            }),
        )
        out = _rewrite_one(task, ctx, a)
        v = next(x for x in out if isinstance(x, PostHookVerdict))
        assert v.passed is False
        assert v.raw["findings"][0]["severity"] == "critical"

    def test_adr_drift_judge_translates(self):
        from general_beckman.rewrite import _rewrite_one
        from general_beckman.result_router import PostHookVerdict
        task, ctx, a = self._complete(
            "adr_drift_judge",
            json.dumps({"verdict": "fail", "findings": [{"adr_id": "ADR-1"}]}),
            posthook_kind="adr_drift_judge",
        )
        out = _rewrite_one(task, ctx, a)
        v = next(x for x in out if isinstance(x, PostHookVerdict))
        assert v.kind == "adr_drift_judge"
        assert v.passed is False

    def test_garbled_result_defaults_to_fail(self):
        """Unparseable reviewer output → passed=False (retry the source)."""
        from general_beckman.rewrite import _rewrite_one
        from general_beckman.result_router import PostHookVerdict
        task, ctx, a = self._complete("integration_reviewer", "not json at all{{{")
        out = _rewrite_one(task, ctx, a)
        v = next(x for x in out if isinstance(x, PostHookVerdict))
        assert v.passed is False

    def test_missing_verdict_field_defaults_to_fail(self):
        from general_beckman.rewrite import _rewrite_one
        from general_beckman.result_router import PostHookVerdict
        task, ctx, a = self._complete(
            "integration_reviewer", json.dumps({"findings": []}),
        )
        out = _rewrite_one(task, ctx, a)
        v = next(x for x in out if isinstance(x, PostHookVerdict))
        assert v.passed is False

    def test_non_posthook_reviewer_task_not_translated(self):
        """An integration_reviewer task WITHOUT posthook ctx keys is not a
        post-hook — must not synthesise a verdict."""
        from general_beckman.rewrite import _rewrite_one
        from general_beckman.result_router import Complete, PostHookVerdict
        task = {"id": 5, "agent_type": "integration_reviewer", "mission_id": 7}
        ctx = {}  # no source_task_id / posthook_kind
        a = Complete(task_id=5, result=json.dumps({"verdict": "pass"}), raw={})
        out = _rewrite_one(task, ctx, a)
        assert not any(isinstance(x, PostHookVerdict) for x in out)

    def test_reviewer_failed_action_synthesises_fail_verdict(self):
        """A config-only reviewer that Failed must still emit passed=False so
        the source is not stranded."""
        from general_beckman.rewrite import _rewrite_one
        from general_beckman.result_router import Failed, PostHookVerdict
        task = {"id": 5, "agent_type": "adr_drift_judge", "mission_id": 7}
        ctx = {"source_task_id": 99, "posthook_kind": "adr_drift_judge"}
        a = Failed(task_id=5, error="model exploded", raw={})
        out = _rewrite_one(task, ctx, a)
        v = next(x for x in out if isinstance(x, PostHookVerdict))
        assert v.passed is False
        assert v.kind == "adr_drift_judge"

    def test_reviewer_exhausted_action_synthesises_fail_verdict(self):
        from general_beckman.rewrite import _rewrite_one
        from general_beckman.result_router import Exhausted, PostHookVerdict
        task = {"id": 5, "agent_type": "integration_reviewer", "mission_id": 7}
        ctx = {"source_task_id": 99, "posthook_kind": "integration_review"}
        a = Exhausted(task_id=5, error="max_iterations", raw={})
        out = _rewrite_one(task, ctx, a)
        v = next(x for x in out if isinstance(x, PostHookVerdict))
        assert v.passed is False

    def test_reviewer_complete_does_not_emit_mission_advance(self):
        """A posthook reviewer task must not recurse into MissionAdvance."""
        from general_beckman.rewrite import _rewrite_one
        from general_beckman.result_router import MissionAdvance
        task, ctx, a = self._complete(
            "integration_reviewer", json.dumps({"verdict": "pass"}),
        )
        out = _rewrite_one(task, ctx, a)
        assert not any(isinstance(x, MissionAdvance) for x in out)


# ---------------------------------------------------------------------------
# Gap 3 — integration_review has a verdict handler
# ---------------------------------------------------------------------------

class TestIntegrationReviewVerdictHandled:
    def test_integration_review_in_simple_blocker_tuple(self):
        """Source-level guard: the apply.py dispatch tuple must list
        integration_review, else its verdict hits 'unknown kind'."""
        import inspect
        from general_beckman import apply as apply_module
        # The dispatch logic lives in the locked impl; the public
        # _apply_posthook_verdict is just the _source_verdict_guard wrapper
        # (lock-split landed in a prior session — this guard previously
        # inspected the wrapper and silently went stale).
        src = inspect.getsource(apply_module._apply_posthook_verdict_locked)
        # The simple-blocker dispatch tuple must include integration_review.
        assert '"integration_review"' in src

    def test_integration_review_verdict_pass_completes_source(self, monkeypatch):
        from general_beckman.apply import _apply_simple_blocker_verdict
        from general_beckman.result_router import PostHookVerdict

        completed: list = []

        async def _fake_update(task_id, **kwargs):
            if kwargs.get("status") == "completed":
                completed.append(task_id)

        async def _fake_advance(*a, **kw):
            return None

        monkeypatch.setattr("src.infra.db.update_task", _fake_update)
        from general_beckman import apply as apply_module
        monkeypatch.setattr(
            apply_module, "_spawn_workflow_advance_if_mission", _fake_advance
        )

        source = {"id": 3, "mission_id": 7, "result": ""}
        ctx = {"_pending_posthooks": ["integration_review"]}
        pending = ["integration_review"]
        v = PostHookVerdict(
            source_task_id=3, kind="integration_review", passed=True,
            raw={"verdict": "pass", "findings": []},
        )
        _run(_apply_simple_blocker_verdict(
            kind="integration_review", source=source, ctx=ctx,
            pending=pending, verdict=v, feedback_prefix="integration_review gate",
        ))
        assert completed == [3]


# ---------------------------------------------------------------------------
# Gap 2 — integration_bisect advisory spawn on integration_replay fail
# ---------------------------------------------------------------------------

class TestIntegrationBisectSpawn:
    def _verdict(self, raw: dict, passed=False):
        from general_beckman.result_router import PostHookVerdict
        return PostHookVerdict(
            source_task_id=1, kind="integration_replay", passed=passed, raw=raw,
        )

    def test_spawn_on_fail_with_commits(self, monkeypatch):
        from general_beckman.apply import _maybe_spawn_integration_bisect

        added: list = []

        async def _fake_add(**kwargs):
            added.append(kwargs)
            return 1

        monkeypatch.setattr("src.infra.db.add_task", _fake_add)

        source = {"id": 9, "mission_id": 7}
        ctx = {"workspace_path": "/ws", "tech_stack_detected": "fastapi+nextjs"}
        v = self._verdict({
            "verdict": "fail",
            "commits_replayed": ["sha0", "sha1", "sha2"],
        })
        spawned = _run(_maybe_spawn_integration_bisect(
            source=source, ctx=ctx, verdict=v,
        ))
        assert spawned is True
        assert len(added) == 1
        kw = added[0]
        assert kw["agent_type"] == "mechanical"
        # mission_id=None deliberately — advisory, not a workflow step.
        assert kw["mission_id"] is None
        payload = kw["context"]["payload"]
        assert payload["action"] == "integration_bisect"
        assert payload["commits"] == ["sha0", "sha1", "sha2"]
        assert payload["mission_id"] == 7  # real mission carried in payload
        assert payload["stack"] == "fastapi+nextjs"
        assert payload["source_task_id"] == 9
        # NOT a post-hook — no source_task_id / posthook_kind at ctx top level.
        assert "posthook_kind" not in kw["context"]

    def test_no_spawn_with_fewer_than_two_commits(self, monkeypatch):
        from general_beckman.apply import _maybe_spawn_integration_bisect

        added: list = []

        async def _fake_add(**kwargs):
            added.append(kwargs)
            return 1

        monkeypatch.setattr("src.infra.db.add_task", _fake_add)

        source = {"id": 9, "mission_id": 7}
        ctx = {"workspace_path": "/ws"}
        v = self._verdict({"verdict": "fail", "commits_replayed": ["sha0"]})
        spawned = _run(_maybe_spawn_integration_bisect(
            source=source, ctx=ctx, verdict=v,
        ))
        assert spawned is False
        assert added == []

    def test_no_spawn_without_workspace_path(self, monkeypatch):
        from general_beckman.apply import _maybe_spawn_integration_bisect

        added: list = []

        async def _fake_add(**kwargs):
            added.append(kwargs)
            return 1

        monkeypatch.setattr("src.infra.db.add_task", _fake_add)

        source = {"id": 9, "mission_id": 7}
        ctx = {}  # no workspace_path
        v = self._verdict({"verdict": "fail", "commits_replayed": ["a", "b"]})
        spawned = _run(_maybe_spawn_integration_bisect(
            source=source, ctx=ctx, verdict=v,
        ))
        assert spawned is False
        assert added == []

    def test_bisect_task_payload_dispatchable_by_mr_roboto(self, monkeypatch):
        """End-to-end: the spawned task's context.payload is the shape
        mr_roboto._run_dispatch expects after the orchestrator lifts
        context.payload → task.payload."""
        from general_beckman.apply import _maybe_spawn_integration_bisect
        from mr_roboto import _run_dispatch

        added: list = []

        async def _fake_add(**kwargs):
            added.append(kwargs)
            return 1

        async def _fake_bisect(**kwargs):
            return {"breaking_pair": None}

        monkeypatch.setattr("src.infra.db.add_task", _fake_add)
        import sys
        monkeypatch.setattr(
            sys.modules["mr_roboto.integration_bisect"],
            "integration_bisect", _fake_bisect,
        )

        source = {"id": 9, "mission_id": 7}
        ctx = {"workspace_path": "/ws"}
        v = self._verdict({"verdict": "fail", "commits_replayed": ["a", "b"]})
        _run(_maybe_spawn_integration_bisect(source=source, ctx=ctx, verdict=v))

        # Simulate the orchestrator lift: context.payload → task.payload.
        spawned_ctx = added[0]["context"]
        task = {"id": 1, "agent_type": "mechanical", "payload": spawned_ctx["payload"]}
        result = _run(_run_dispatch(task))
        assert result.status == "completed"
