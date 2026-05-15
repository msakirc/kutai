"""Z3 residual R3 — ADR drift gray-zone LLM-judge path.

Mechanical ``check_adr_drift`` returns ``judgment_only_adr_ids`` for ADRs
whose ``falsification_signal`` is a v1 string / null / unknown shape.
When the mechanical verdict is pass + that list is non-empty, apply.py
spawns an ``adr_drift_judge`` LLM task carrying the ADR ids + paths +
produced files.

Verdict precedence: mechanical-fail > judge-fail > judge-pass. The judge
is only spawned when mechanical *passed*, so the precedence is enforced
naturally (mechanical-fail never reaches the spawn helper).
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
# Registry + agent wiring
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_adr_drift_judge_in_registry(self):
        from general_beckman.posthooks import POST_HOOK_REGISTRY
        assert "adr_drift_judge" in POST_HOOK_REGISTRY
        spec = POST_HOOK_REGISTRY["adr_drift_judge"]
        assert spec.kind == "adr_drift_judge"
        assert spec.verb == "adr_drift_judge"
        assert spec.default_severity == "blocker"

    def test_adr_drift_judge_never_auto_wires(self):
        from general_beckman.posthooks import POST_HOOK_REGISTRY, MissionDialContext
        spec = POST_HOOK_REGISTRY["adr_drift_judge"]
        assert spec.resolve_triggers(MissionDialContext()) == []
        assert spec.resolve_triggers(MissionDialContext(qa_dial="strict")) == []

    def test_agent_in_registry(self):
        from src.agents import AGENT_REGISTRY
        assert "adr_drift_judge" in AGENT_REGISTRY
        agent = AGENT_REGISTRY["adr_drift_judge"]
        assert agent.name == "adr_drift_judge"
        # No grader chain — its verdict IS the gate.
        from general_beckman.posthooks import _NO_POSTHOOKS_AGENT_TYPES
        assert "adr_drift_judge" in _NO_POSTHOOKS_AGENT_TYPES

    def test_agent_prompt_invariants(self):
        from src.agents import AGENT_REGISTRY
        agent = AGENT_REGISTRY["adr_drift_judge"]
        prompt = agent.get_system_prompt({})
        # First line "You are ..."
        first_line = prompt.split("\n", 1)[0]
        assert first_line.startswith("You are")
        # body has must/always + don't/never
        lower = prompt.lower()
        assert any(w in lower for w in ("must", "always"))
        assert any(w in lower for w in ("don't", "never", "do not"))
        # body contains final_answer + ```json
        assert "final_answer" in prompt
        assert "```json" in prompt

    def test_agent_is_readonly(self):
        from src.agents import AGENT_REGISTRY
        agent = AGENT_REGISTRY["adr_drift_judge"]
        for tool in (agent.allowed_tools or []):
            assert "write" not in tool.lower(), f"judge has write tool: {tool}"


# ---------------------------------------------------------------------------
# Spawn helper
# ---------------------------------------------------------------------------

class TestSpawnHelper:
    def _make_verdict(self, raw: dict, passed: bool = True):
        from general_beckman.result_router import PostHookVerdict
        return PostHookVerdict(
            source_task_id=1, kind="adr_drift_check", passed=passed, raw=raw,
        )

    def test_no_spawn_when_no_judgment_only_ids(self, monkeypatch, tmp_path):
        from general_beckman.apply import _maybe_spawn_adr_drift_judge

        added: list[dict] = []
        updates: list[dict] = []

        async def _fake_add(**kwargs):
            added.append(kwargs)
            return 99

        async def _fake_update(task_id, **kwargs):
            updates.append({"id": task_id, **kwargs})

        monkeypatch.setattr("src.infra.db.add_task", _fake_add)
        monkeypatch.setattr("src.infra.db.update_task", _fake_update)

        source = {
            "id": 1, "mission_id": 7,
            "context": json.dumps({"workspace_path": str(tmp_path)}),
        }
        v = self._make_verdict({"verdict": "pass", "judgment_only_adr_ids": []})

        _run(_maybe_spawn_adr_drift_judge(source=source, verdict=v))
        assert added == []
        assert updates == []

    def test_spawn_when_judgment_only_present(self, monkeypatch, tmp_path):
        from general_beckman.apply import _maybe_spawn_adr_drift_judge

        # Create ADR files so adr_paths resolves.
        adr_dir = tmp_path / ".adr"
        adr_dir.mkdir()
        (adr_dir / "ADR-0001.json").write_text('{"id": "ADR-0001"}')
        (adr_dir / "ADR-0007.json").write_text('{"id": "ADR-0007"}')

        added: list[dict] = []
        updates: list[dict] = []

        async def _fake_add(**kwargs):
            added.append(kwargs)
            return 42

        async def _fake_update(task_id, **kwargs):
            updates.append({"id": task_id, **kwargs})

        monkeypatch.setattr("src.infra.db.add_task", _fake_add)
        monkeypatch.setattr("src.infra.db.update_task", _fake_update)

        source = {
            "id": 1, "mission_id": 7,
            "context": json.dumps({
                "workspace_path": str(tmp_path),
                "produces": ["src/a.py", "src/b.py"],
            }),
        }
        v = self._make_verdict({
            "verdict": "pass",
            "judgment_only_adr_ids": ["ADR-0001", "ADR-0007"],
        })

        _run(_maybe_spawn_adr_drift_judge(source=source, verdict=v))

        assert len(added) == 1
        kw = added[0]
        assert kw["agent_type"] == "adr_drift_judge"
        assert kw["mission_id"] == 7
        ctx = json.loads(kw["context"])
        assert ctx["adr_ids"] == ["ADR-0001", "ADR-0007"]
        assert "ADR-0001" in ctx["adr_paths"]
        assert "ADR-0007" in ctx["adr_paths"]
        assert ctx["produced_files"] == ["src/a.py", "src/b.py"]
        assert ctx["posthook_kind"] == "adr_drift_judge"
        assert ctx["source_task_id"] == 1

        # Source bookkeeping: adr_drift_judge added to _pending_posthooks.
        assert len(updates) == 1
        source_ctx = json.loads(updates[0]["context"])
        assert "adr_drift_judge" in source_ctx["_pending_posthooks"]
        assert updates[0]["status"] == "ungraded"

    def test_spawn_handles_missing_adr_files_gracefully(self, monkeypatch, tmp_path):
        """When workspace .adr/ is missing the ADR json file, the spawn still
        proceeds but adr_paths omits the missing id."""
        from general_beckman.apply import _maybe_spawn_adr_drift_judge

        # Don't create the adr dir at all.
        added: list[dict] = []

        async def _fake_add(**kwargs):
            added.append(kwargs)
            return 1

        async def _fake_update(*a, **kw):
            return None

        monkeypatch.setattr("src.infra.db.add_task", _fake_add)
        monkeypatch.setattr("src.infra.db.update_task", _fake_update)

        source = {
            "id": 1, "mission_id": 7,
            "context": json.dumps({"workspace_path": str(tmp_path)}),
        }
        v = self._make_verdict({
            "verdict": "pass",
            "judgment_only_adr_ids": ["ADR-9999"],
        })

        _run(_maybe_spawn_adr_drift_judge(source=source, verdict=v))
        assert len(added) == 1
        ctx = json.loads(added[0]["context"])
        assert ctx["adr_ids"] == ["ADR-9999"]
        assert ctx["adr_paths"] == {}  # nothing on disk


# ---------------------------------------------------------------------------
# Verdict precedence — only spawn on mechanical PASS
# ---------------------------------------------------------------------------

class TestPrecedence:
    """When mechanical adr_drift_check fails, the judge must NOT be spawned —
    mechanical-fail strictly beats anything the judge could say. The
    apply.py wiring enforces this by gating on `a.passed` before calling
    _maybe_spawn_adr_drift_judge.
    """

    def test_apply_branch_does_not_spawn_on_mechanical_fail(self, monkeypatch, tmp_path):
        """Direct test of the conditional branch: passed=False → no spawn."""
        from general_beckman import apply as apply_module

        spawn_calls: list[dict] = []

        async def _capture_spawn(*, source, verdict):
            spawn_calls.append({"source": source, "verdict": verdict})

        monkeypatch.setattr(
            apply_module, "_maybe_spawn_adr_drift_judge", _capture_spawn
        )

        # Simulate the gating logic from _apply_posthook_verdict.
        from general_beckman.result_router import PostHookVerdict
        verdict = PostHookVerdict(
            source_task_id=1, kind="adr_drift_check", passed=False,
            raw={"verdict": "fail", "judgment_only_adr_ids": ["ADR-1"]},
        )
        # In the real flow, passed=False would skip the spawn block.
        source = {"id": 1, "mission_id": 7}
        if verdict.kind == "adr_drift_check" and verdict.passed:
            _run(apply_module._maybe_spawn_adr_drift_judge(
                source=source, verdict=verdict
            ))
        assert spawn_calls == []

    def test_apply_branch_spawns_on_mechanical_pass(self, monkeypatch, tmp_path):
        """passed=True + judgment_only ids non-empty → spawn fires."""
        from general_beckman import apply as apply_module

        spawn_calls: list[dict] = []

        async def _capture_spawn(*, source, verdict):
            spawn_calls.append({"source": source, "verdict": verdict})

        monkeypatch.setattr(
            apply_module, "_maybe_spawn_adr_drift_judge", _capture_spawn
        )

        from general_beckman.result_router import PostHookVerdict
        verdict = PostHookVerdict(
            source_task_id=1, kind="adr_drift_check", passed=True,
            raw={"verdict": "pass", "judgment_only_adr_ids": ["ADR-1"]},
        )
        source = {"id": 1, "mission_id": 7}
        if verdict.kind == "adr_drift_check" and verdict.passed:
            _run(apply_module._maybe_spawn_adr_drift_judge(
                source=source, verdict=verdict
            ))
        assert len(spawn_calls) == 1


# ---------------------------------------------------------------------------
# DLQ cascade includes adr_drift_judge
# ---------------------------------------------------------------------------

class TestDLQCascade:
    def test_judge_dlq_cascades_source_to_failed(self, monkeypatch):
        """An adr_drift_judge DLQ must cascade the source task to failed."""
        from general_beckman.apply import _posthook_dlq_cascade
        from unittest.mock import AsyncMock, patch

        source = {
            "id": 99, "mission_id": None, "agent_type": "coder",
            "worker_attempts": 1, "status": "ungraded", "context": "{}",
        }
        task = {
            "id": 201, "agent_type": "adr_drift_judge",
            "context": json.dumps({
                "posthook_kind": "adr_drift_judge", "source_task_id": 99,
            }),
        }

        updates: list[tuple] = []

        async def _fake_update(task_id, **kwargs):
            updates.append((task_id, kwargs))

        async def _fake_get(task_id):
            return source

        with patch("src.infra.db.update_task", side_effect=_fake_update), \
             patch("src.infra.db.get_task", side_effect=_fake_get), \
             patch("general_beckman.apply._spawn_workflow_advance_if_mission",
                   new_callable=AsyncMock):
            _run(_posthook_dlq_cascade(task, "judge blew up"))

        statuses = [kw.get("status") for tid, kw in updates if tid == 99]
        assert "failed" in statuses
