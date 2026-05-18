"""Z3 residual R4 — integration_bisect emits richer mission_lessons rows.

Two surface changes:
1. ``integration_bisect`` enriches its result with ``changed_files`` /
   ``file_cluster`` / ``diff_shortstat`` / ``commit_subject`` derived
   from the ``commit_a..commit_b`` git diff.
2. The mr_roboto ``run_dispatch`` action for ``integration_bisect`` upserts
   a ``mission_lessons`` row when a ``breaking_pair`` is identified and a
   ``mission_id`` is in the payload — best-effort, never cascades.
"""
from __future__ import annotations

import asyncio

import pytest

from mr_roboto import integration_bisect as ib_mod
from mr_roboto.integration_bisect import _cluster_files
from mr_roboto.integration_bisect import _gather_breaking_diff


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# _cluster_files
# ---------------------------------------------------------------------------

class TestClusterFiles:
    def test_empty(self):
        assert _cluster_files([]) == []

    def test_single_file_root(self):
        out = _cluster_files(["README.md"])
        assert out == [{"dir": "(root)", "count": 1, "sample": ["README.md"]}]

    def test_groups_by_top_dir(self):
        out = _cluster_files([
            "src/app/run.py", "src/app/bot.py", "src/agents/x.py", "tests/a.py"
        ])
        dirs = [c["dir"] for c in out]
        assert dirs[0] == "src"  # 3 files in src wins
        assert out[0]["count"] == 3
        assert "tests" in dirs

    def test_descending_count_order(self):
        out = _cluster_files([
            "a/1", "a/2", "a/3", "b/1", "b/2", "c/1"
        ])
        counts = [c["count"] for c in out]
        assert counts == sorted(counts, reverse=True)

    def test_sample_capped_at_3(self):
        out = _cluster_files(["x/" + str(i) for i in range(10)])
        assert len(out[0]["sample"]) == 3

    def test_top_n_capped(self):
        out = _cluster_files([f"d{i}/x" for i in range(20)], top_n=5)
        assert len(out) == 5

    def test_windows_path_separators(self):
        out = _cluster_files(["src\\app\\x.py", "src\\agents\\y.py"])
        assert out[0]["dir"] == "src"
        assert out[0]["count"] == 2

    def test_ignores_blank_paths(self):
        out = _cluster_files(["", "  ", "src/a"])
        assert len(out) == 1
        assert out[0]["dir"] == "src"


# ---------------------------------------------------------------------------
# _gather_breaking_diff — non-git path
# ---------------------------------------------------------------------------

class TestGatherBreakingDiff:
    def test_no_git_returns_empty_fields(self, tmp_path):
        result = _run(_gather_breaking_diff("abc", "def", cwd=str(tmp_path)))
        assert result["changed_files"] == []
        assert result["file_cluster"] == []
        assert result["diff_shortstat"] == ""
        assert result["commit_subject"] == ""

    def test_keys_always_present(self, tmp_path):
        result = _run(_gather_breaking_diff("a", "b", cwd=str(tmp_path)))
        assert set(result.keys()) == {
            "changed_files", "file_cluster", "diff_shortstat", "commit_subject"
        }


# ---------------------------------------------------------------------------
# Lesson emission via dispatcher action wrapper
# ---------------------------------------------------------------------------

class TestEmitBisectLesson:
    def test_emit_called_when_breaking_pair_and_mission_id(self, monkeypatch):
        from mr_roboto import _emit_bisect_lesson

        upserts: list[dict] = []

        async def _fake_upsert(**kw):
            upserts.append(kw)
            return 1

        monkeypatch.setattr(
            "src.infra.mission_lessons.upsert_mission_lesson", _fake_upsert
        )

        _run(_emit_bisect_lesson(
            mission_id=42,
            stack="fastapi+nextjs",
            bisect_result={
                "breaking_pair": ["aaaaaaaa", "bbbbbbbb"],
                "failing_test": "FAILED tests/test_thing.py::test_x - assertion",
                "diff_shortstat": "3 files changed, 12 insertions(+)",
                "commit_subject": "feat: add thing",
                "file_cluster": [{"dir": "src", "count": 3, "sample": ["src/a", "src/b", "src/c"]}],
                "changed_files": ["src/a", "src/b", "src/c"],
            },
            source_task_id=999,
        ))

        assert len(upserts) == 1
        kw = upserts[0]
        assert kw["stack"] == "fastapi+nextjs"
        assert kw["domain"] == "integration_replay"
        assert "break at bbbbbbbb" in kw["pattern"]
        assert "test_x" in kw["pattern"]
        assert kw["severity"] == "blocker"
        assert kw["source_kind"] == "bisect_break"
        ref = kw["source_ref"]
        assert ref["mission_id"] == 42
        assert ref["source_task_id"] == 999
        assert ref["breaking_pair"] == ["aaaaaaaa", "bbbbbbbb"]
        assert ref["commit_subject"] == "feat: add thing"
        assert ref["file_cluster"][0]["dir"] == "src"
        assert ref["diff_shortstat"] == "3 files changed, 12 insertions(+)"
        # Fix contains both the cluster hint and shortstat.
        assert "suspect: src" in kw["fix"]
        assert "3 files changed" in kw["fix"]

    def test_fix_falls_back_to_subject_when_no_failing_test(self, monkeypatch):
        from mr_roboto import _emit_bisect_lesson

        upserts: list[dict] = []

        async def _fake_upsert(**kw):
            upserts.append(kw)
            return 1

        monkeypatch.setattr(
            "src.infra.mission_lessons.upsert_mission_lesson", _fake_upsert
        )

        _run(_emit_bisect_lesson(
            mission_id=1,
            stack="unknown",
            bisect_result={
                "breaking_pair": ["a", "deadbeef"],
                "failing_test": "",
                "commit_subject": "fix: regress middleware",
                "file_cluster": [],
                "diff_shortstat": "",
            },
        ))

        assert len(upserts) == 1
        assert "fix: regress middleware" in upserts[0]["pattern"]

    def test_no_emit_when_no_breaking_pair(self, monkeypatch):
        from mr_roboto import _emit_bisect_lesson

        upserts: list[dict] = []

        async def _fake_upsert(**kw):
            upserts.append(kw)
            return 1

        monkeypatch.setattr(
            "src.infra.mission_lessons.upsert_mission_lesson", _fake_upsert
        )

        _run(_emit_bisect_lesson(
            mission_id=1,
            stack="x",
            bisect_result={"breaking_pair": None},
        ))

        assert upserts == []

    def test_no_emit_when_pair_malformed(self, monkeypatch):
        from mr_roboto import _emit_bisect_lesson

        upserts: list[dict] = []

        async def _fake_upsert(**kw):
            upserts.append(kw)
            return 1

        monkeypatch.setattr(
            "src.infra.mission_lessons.upsert_mission_lesson", _fake_upsert
        )

        _run(_emit_bisect_lesson(
            mission_id=1,
            stack="x",
            bisect_result={"breaking_pair": ["only-one"]},
        ))

        assert upserts == []

    def test_changed_files_capped_in_source_ref(self, monkeypatch):
        from mr_roboto import _emit_bisect_lesson

        upserts: list[dict] = []

        async def _fake_upsert(**kw):
            upserts.append(kw)
            return 1

        monkeypatch.setattr(
            "src.infra.mission_lessons.upsert_mission_lesson", _fake_upsert
        )

        many = [f"src/f{i}.py" for i in range(50)]
        _run(_emit_bisect_lesson(
            mission_id=1,
            stack="x",
            bisect_result={
                "breaking_pair": ["a", "b"],
                "changed_files": many,
                "file_cluster": [{"dir": "src", "count": 50, "sample": many[:3]}],
            },
        ))

        assert len(upserts[0]["source_ref"]["changed_files"]) == 20


# ---------------------------------------------------------------------------
# Dispatch action wrapper integration
# ---------------------------------------------------------------------------

class TestDispatchEmits:
    def test_bisect_action_emits_on_breaking_pair(self, monkeypatch):
        """When the dispatcher runs `integration_bisect` and gets a
        breaking_pair back, an upsert call is fired."""
        from mr_roboto import _run_dispatch

        upserts: list[dict] = []

        async def _fake_upsert(**kw):
            upserts.append(kw)
            return 1

        async def _fake_bisect(**kw):
            return {
                "breaking_pair": ["c0", "c1"],
                "failing_test": "FAILED tests/x.py::test_y",
                "changed_files": ["src/a"],
                "file_cluster": [{"dir": "src", "count": 1, "sample": ["src/a"]}],
                "diff_shortstat": "1 file changed",
                "commit_subject": "feat: x",
            }

        monkeypatch.setattr(
            "src.infra.mission_lessons.upsert_mission_lesson", _fake_upsert
        )
        import sys
        monkeypatch.setattr(
            sys.modules["mr_roboto.integration_bisect"],
            "integration_bisect",
            _fake_bisect,
        )

        task = {
            "id": 1,
            "agent_type": "mechanical",
            "payload": {
                "action": "integration_bisect",
                "commits": ["c0", "c1"],
                "workspace_path": "/tmp",
                "mission_id": 7,
                "stack": "fastapi+nextjs",
                "source_task_id": 99,
            },
        }
        result = _run(_run_dispatch(task))
        assert result.status == "completed"
        assert result.result["breaking_pair"] == ["c0", "c1"]
        assert len(upserts) == 1
        assert upserts[0]["source_ref"]["mission_id"] == 7

    def test_bisect_action_silent_on_no_mission_id(self, monkeypatch):
        from mr_roboto import _run_dispatch

        upserts: list[dict] = []

        async def _fake_upsert(**kw):
            upserts.append(kw)
            return 1

        async def _fake_bisect(**kw):
            return {"breaking_pair": ["c0", "c1"]}

        monkeypatch.setattr(
            "src.infra.mission_lessons.upsert_mission_lesson", _fake_upsert
        )
        import sys
        monkeypatch.setattr(
            sys.modules["mr_roboto.integration_bisect"],
            "integration_bisect",
            _fake_bisect,
        )

        task = {
            "id": 1,
            "agent_type": "mechanical",
            "payload": {
                "action": "integration_bisect",
                "commits": [],
                "workspace_path": "/tmp",
            },
        }
        result = _run(_run_dispatch(task))
        assert result.status == "completed"
        assert upserts == []

    def test_bisect_action_silent_on_no_break(self, monkeypatch):
        from mr_roboto import _run_dispatch

        upserts: list[dict] = []

        async def _fake_upsert(**kw):
            upserts.append(kw)
            return 1

        async def _fake_bisect(**kw):
            return {"breaking_pair": None}

        monkeypatch.setattr(
            "src.infra.mission_lessons.upsert_mission_lesson", _fake_upsert
        )
        import sys
        monkeypatch.setattr(
            sys.modules["mr_roboto.integration_bisect"],
            "integration_bisect",
            _fake_bisect,
        )

        task = {
            "id": 1,
            "agent_type": "mechanical",
            "payload": {
                "action": "integration_bisect",
                "mission_id": 5,
                "workspace_path": "/tmp",
            },
        }
        result = _run(_run_dispatch(task))
        assert result.status == "completed"
        assert upserts == []

    def test_lesson_emit_failure_does_not_cascade(self, monkeypatch):
        """A blowup inside upsert must not flip status to failed."""
        from mr_roboto import _run_dispatch

        async def _explosive(**kw):
            raise RuntimeError("db down")

        async def _fake_bisect(**kw):
            return {"breaking_pair": ["a", "b"]}

        monkeypatch.setattr(
            "src.infra.mission_lessons.upsert_mission_lesson", _explosive
        )
        import sys
        monkeypatch.setattr(
            sys.modules["mr_roboto.integration_bisect"],
            "integration_bisect",
            _fake_bisect,
        )

        task = {
            "id": 1,
            "agent_type": "mechanical",
            "payload": {
                "action": "integration_bisect",
                "mission_id": 5,
                "workspace_path": "/tmp",
            },
        }
        result = _run(_run_dispatch(task))
        assert result.status == "completed"
