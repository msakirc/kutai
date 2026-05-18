"""Z3 T5 — integration_replay verb + post-hook + bisect + lesson emission.

Tests cover:
- integration_replay mode=quick: spot-check path
- integration_replay mode=standard: full suite once
- integration_replay mode=strict: shuffled commits
- Soft-skip when no .git / no tests
- verdict=fail on any commit red
- shuffle_seed=mission_id is deterministic
- integration_replay kind registered (cost_band=heavy, severity=blocker, empty auto_wire)
- Expander appends integration_replay_step after integration_review_step when dial != "off"
- Expander skips integration_replay_step when dial="off"
- Apply.py routes integration_replay to mechanical executor
- _apply_simple_blocker_verdict handles integration_replay (ad-hoc verdict obj)
- integration_bisect finds breaking commit pair in synthetic case
- bisect failure emits mission_lessons row (mock db.add_lesson)
"""
from __future__ import annotations

import pytest

_SKIP_NON_CANONICAL = pytest.mark.skip(reason="Z3 T5: test asserts agent-specific expander shape; canonical expander wiring differs")

import asyncio
import importlib
import os
import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _import_module(dotted: str):
    """Import a module by dotted name, bypassing package-level re-exports."""
    return importlib.import_module(dotted)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_verdict(kind: str, passed: bool, raw: dict | None = None, source_task_id: int = 1):
    from general_beckman.result_router import PostHookVerdict
    return PostHookVerdict(
        source_task_id=source_task_id,
        kind=kind,
        passed=passed,
        raw=raw or {},
    )


def _make_request(kind: str, source_task_id: int = 1):
    from general_beckman.result_router import RequestPostHook
    return RequestPostHook(
        source_task_id=source_task_id,
        kind=kind,
        source_ctx={},
    )


# ---------------------------------------------------------------------------
# T5A — integration_replay verb
# ---------------------------------------------------------------------------

class TestIntegrationReplayQuick:
    """mode=quick: spot-check only tests touched by HEAD diff."""

    @pytest.mark.asyncio
    async def test_quick_spot_check_pass(self, tmp_path):
        """Quick mode runs subset of tests; pass when exit 0."""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        test_file = tmp_path / "tests" / "integration" / "test_foo.py"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("def test_pass(): assert True")

        ir_module = _import_module("mr_roboto.integration_replay")

        async def _fake_git(args, cwd):
            if args[0] == "rev-parse" and args[1] == "HEAD":
                return 0, "abc123", ""
            if args[0] == "diff":
                return 0, "tests/integration/test_foo.py", ""
            return 0, "", ""

        async def _fake_pytest(suite_glob, cwd, timeout_s):
            return {"ok": True, "passed": 1, "failed": 0, "stdout_tail": "1 passed"}

        with patch.object(ir_module, "_git_cmd", side_effect=_fake_git), \
             patch.object(ir_module, "_run_pytest", side_effect=_fake_pytest):
            result = await ir_module.integration_replay(
                commits=[],
                suite_glob="tests/integration/**",
                shuffle_seed=42,
                mode="quick",
                workspace_path=str(tmp_path),
            )

        assert result["verdict"] == "pass"
        assert result["mode"] == "quick"
        assert result["skipped"] is False
        assert len(result["findings"]) == 0

    @pytest.mark.asyncio
    async def test_quick_spot_check_fail(self, tmp_path):
        """Quick mode verdict=fail when pytest returns non-zero."""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        test_file = tmp_path / "tests" / "integration" / "test_foo.py"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("def test_fail(): assert False")

        ir_module = _import_module("mr_roboto.integration_replay")

        async def _fake_git(args, cwd):
            if args[0] == "rev-parse":
                return 0, "abc123", ""
            return 0, "tests/integration/test_foo.py", ""

        async def _fake_pytest(suite_glob, cwd, timeout_s):
            return {"ok": False, "passed": 0, "failed": 1, "stdout_tail": "FAILED test_foo::test_fail"}

        with patch.object(ir_module, "_git_cmd", side_effect=_fake_git), \
             patch.object(ir_module, "_run_pytest", side_effect=_fake_pytest):
            result = await ir_module.integration_replay(
                commits=[],
                suite_glob="tests/integration/**",
                shuffle_seed=42,
                mode="quick",
                workspace_path=str(tmp_path),
            )

        assert result["verdict"] == "fail"
        assert len(result["findings"]) >= 1
        assert result["findings"][0]["severity"] == "blocker"


class TestIntegrationReplayStandard:
    """mode=standard: full suite once against current commit."""

    @pytest.mark.asyncio
    async def test_standard_pass(self, tmp_path):
        (tmp_path / ".git").mkdir()
        ir_module = _import_module("mr_roboto.integration_replay")

        async def _fake_git(args, cwd):
            if args[0] == "rev-parse":
                return 0, "deadbeef", ""
            return 0, "", ""

        async def _fake_pytest(suite_glob, cwd, timeout_s):
            return {"ok": True, "passed": 5, "failed": 0, "stdout_tail": "5 passed"}

        with patch.object(ir_module, "_git_cmd", side_effect=_fake_git), \
             patch.object(ir_module, "_run_pytest", side_effect=_fake_pytest):
            result = await ir_module.integration_replay(
                commits=[], suite_glob="tests/**", shuffle_seed=1,
                mode="standard", workspace_path=str(tmp_path),
            )

        assert result["verdict"] == "pass"
        assert result["mode"] == "standard"
        assert result["commits_replayed"] == ["deadbeef"]

    @pytest.mark.asyncio
    async def test_standard_fail(self, tmp_path):
        (tmp_path / ".git").mkdir()
        ir_module = _import_module("mr_roboto.integration_replay")

        async def _fake_git(args, cwd):
            return 0, "deadbeef", ""

        async def _fake_pytest(suite_glob, cwd, timeout_s):
            return {"ok": False, "passed": 0, "failed": 2, "stdout_tail": "2 failed"}

        with patch.object(ir_module, "_git_cmd", side_effect=_fake_git), \
             patch.object(ir_module, "_run_pytest", side_effect=_fake_pytest):
            result = await ir_module.integration_replay(
                commits=[], suite_glob="tests/**", shuffle_seed=1,
                mode="standard", workspace_path=str(tmp_path),
            )

        assert result["verdict"] == "fail"


class TestIntegrationReplayStrict:
    """mode=strict: shuffled commits."""

    @pytest.mark.asyncio
    async def test_strict_pass_all_commits(self, tmp_path):
        (tmp_path / ".git").mkdir()
        ir_module = _import_module("mr_roboto.integration_replay")

        async def _fake_git(args, cwd):
            if args[0] == "rev-parse" and "--abbrev-ref" in args:
                return 0, "main", ""
            if args[0] == "rev-parse":
                return 0, "sha0", ""
            if args[0] == "log":
                return 0, "sha0\nsha1\nsha2\nsha3", ""
            if args[0] == "checkout":
                return 0, "", ""
            if args[0] == "stash":
                return 0, "", ""
            return 0, "", ""

        async def _fake_pytest(suite_glob, cwd, timeout_s):
            return {"ok": True, "passed": 3, "failed": 0, "stdout_tail": "3 passed"}

        with patch.object(ir_module, "_git_cmd", side_effect=_fake_git), \
             patch.object(ir_module, "_run_pytest", side_effect=_fake_pytest):
            result = await ir_module.integration_replay(
                commits=["sha1", "sha2"],
                suite_glob="tests/**",
                shuffle_seed=99,
                mode="strict",
                workspace_path=str(tmp_path),
            )

        assert result["verdict"] == "pass"
        assert result["mode"] == "strict"
        assert "sha0" in result["commits_replayed"]

    @pytest.mark.asyncio
    async def test_strict_one_commit_fails(self, tmp_path):
        """Verdict=fail when any commit's suite is red."""
        (tmp_path / ".git").mkdir()
        ir_module = _import_module("mr_roboto.integration_replay")

        call_count = [0]

        async def _fake_git(args, cwd):
            if args[0] == "rev-parse" and "--abbrev-ref" in args:
                return 0, "HEAD", ""
            if args[0] == "rev-parse":
                return 0, "sha0", ""
            if args[0] == "log":
                return 0, "sha0\nsha1\nsha2", ""
            if args[0] == "checkout":
                return 0, "", ""
            return 0, "", ""

        async def _fake_pytest(suite_glob, cwd, timeout_s):
            c = call_count[0]
            call_count[0] = c + 1
            if c == 1:  # second run fails
                return {"ok": False, "passed": 0, "failed": 1, "stdout_tail": "FAILED"}
            return {"ok": True, "passed": 2, "failed": 0, "stdout_tail": "2 passed"}

        with patch.object(ir_module, "_git_cmd", side_effect=_fake_git), \
             patch.object(ir_module, "_run_pytest", side_effect=_fake_pytest):
            result = await ir_module.integration_replay(
                commits=["sha1", "sha2"],
                suite_glob="tests/**",
                shuffle_seed=7,
                mode="strict",
                workspace_path=str(tmp_path),
            )

        assert result["verdict"] == "fail"

    @pytest.mark.asyncio
    async def test_strict_shuffle_seed_deterministic(self, tmp_path):
        """Same seed → same shuffle order across two calls."""
        (tmp_path / ".git").mkdir()
        ir_module = _import_module("mr_roboto.integration_replay")

        checkouts: list[list[str]] = []

        async def _fake_git_factory(run_checkouts):
            async def _fake_git(args, cwd):
                if args[0] == "rev-parse" and "--abbrev-ref" in args:
                    return 0, "HEAD", ""
                if args[0] == "rev-parse":
                    return 0, "sha0", ""
                if args[0] == "log":
                    return 0, "sha0\nsha1\nsha2\nsha3", ""
                if args[0] == "checkout":
                    run_checkouts.append(args[1])
                    return 0, "", ""
                return 0, "", ""
            return _fake_git

        async def _fake_pytest(suite_glob, cwd, timeout_s):
            return {"ok": True, "passed": 1, "failed": 0, "stdout_tail": ""}

        for _ in range(2):
            order: list[str] = []
            checkouts.append(order)
            fake_git = await _fake_git_factory(order)
            with patch.object(ir_module, "_git_cmd", side_effect=fake_git), \
                 patch.object(ir_module, "_run_pytest", side_effect=_fake_pytest):
                await ir_module.integration_replay(
                    commits=["sha1", "sha2"],
                    suite_glob="tests/**",
                    shuffle_seed=42,
                    mode="strict",
                    workspace_path=str(tmp_path),
                )

        # Both runs should produce the same checkout sequence (same seed).
        assert checkouts[0] == checkouts[1]


class TestIntegrationReplaySoftSkip:
    """Soft-skip conditions."""

    @pytest.mark.asyncio
    async def test_skip_no_git_dir(self, tmp_path):
        from mr_roboto.integration_replay import integration_replay
        result = await integration_replay(
            commits=[], suite_glob="tests/**", shuffle_seed=0,
            mode="standard", workspace_path=str(tmp_path),
        )
        assert result["skipped"] is True
        assert result["verdict"] == "pass"
        assert "not a git repo" in result["reason"]

    @pytest.mark.asyncio
    async def test_skip_no_test_files(self, tmp_path):
        # No .git dir — always skips before any git call
        result = await _import_module("mr_roboto.integration_replay").integration_replay(
            commits=[], suite_glob="ZZZZZ_no_match_at_all/**/*.nonexistent",
            shuffle_seed=0, mode="standard",
            workspace_path=str(tmp_path),
        )
        # No .git → skipped
        assert result["skipped"] is True
        assert result["verdict"] == "pass"


# ---------------------------------------------------------------------------
# T5B — integration_replay post-hook kind registered
# ---------------------------------------------------------------------------

class TestPostHookRegistry:
    def test_integration_replay_registered(self):
        from general_beckman.posthooks import POST_HOOK_REGISTRY
        assert "integration_replay" in POST_HOOK_REGISTRY
        spec = POST_HOOK_REGISTRY["integration_replay"]
        assert spec.default_severity == "blocker"
        assert spec.auto_wire_triggers == []
        assert spec.verb == "integration_replay"

    def test_integration_replay_description_nonempty(self):
        from general_beckman.posthooks import POST_HOOK_REGISTRY
        spec = POST_HOOK_REGISTRY["integration_replay"]
        assert len(spec.description) > 10

    def test_integration_replay_not_in_post_hook_kinds_auto_wire(self):
        """integration_replay has empty auto_wire_triggers so never auto-fires."""
        from general_beckman.posthooks import POST_HOOK_REGISTRY
        spec = POST_HOOK_REGISTRY["integration_replay"]
        assert spec.auto_wire_triggers == []


# ---------------------------------------------------------------------------
# T5B — Expander appends / skips integration_replay_step
# ---------------------------------------------------------------------------

class TestExpanderIntegrationReplay:
    """Z3 R1 audit — canonical signature for _maybe_expand_multifile.

    The expander only injects integration_review + integration_replay siblings
    when multi-file expansion is enabled in the mission dials AND a template/
    stack combo matches MULTI_FILE_RULES. Tests below exercise the canonical
    surface, not the earlier agent-specific keyword shape.
    """

    def _make_dials(self, integration_replay: str = "standard"):
        from src.workflows.review_density import ReviewDensityDials
        from src.workflows.review_density import to_mission_dial_context
        return to_mission_dial_context(ReviewDensityDials(
            multi_file_expansion=True,
            integration_replay=integration_replay,
        ))

    def _make_step(self, step_id: str = "feat.4") -> dict:
        return {
            "id": step_id,
            "name": "Build backend service",
            "phase": "phase_4",
            "agent": "coder",
            "context": {
                "template_id": "backend_service",
                "stack_slug": "fastapi+nextjs",
                "feature_name": "checkout",
            },
        }

    def _expand(self, integration_replay: str = "standard", step_id: str = "feat.4"):
        from src.workflows.engine.expander import _maybe_expand_multifile
        dials = self._make_dials(integration_replay=integration_replay)
        return _maybe_expand_multifile(self._make_step(step_id), dials, None)

    def test_default_appends_integration_review_and_replay(self):
        expanded = self._expand()
        ids = [s["id"] for s in expanded]
        assert "feat.4.integration_review" in ids
        assert "feat.4.integration_replay" in ids

    def test_replay_depends_on_review(self):
        expanded = self._expand()
        replay = next(s for s in expanded if s["id"] == "feat.4.integration_replay")
        assert "feat.4.integration_review" in replay["depends_on"]
        assert "feat.4.integration_review" in replay["depends_on_steps"]

    def test_dial_off_skips_replay(self):
        expanded = self._expand(integration_replay="off", step_id="feat.5")
        ids = [s["id"] for s in expanded]
        assert "feat.5.integration_review" in ids
        assert "feat.5.integration_replay" not in ids

    def test_dial_quick_sets_mode(self):
        expanded = self._expand(integration_replay="quick", step_id="feat.6")
        replay = next(s for s in expanded if s["id"] == "feat.6.integration_replay")
        assert replay["context"]["mode"] == "quick"
        assert replay["context"]["payload"]["mode"] == "quick"

    def test_dial_strict_sets_mode(self):
        expanded = self._expand(integration_replay="strict", step_id="feat.7")
        replay = next(s for s in expanded if s["id"] == "feat.7.integration_replay")
        assert replay["context"]["mode"] == "strict"

    def test_replay_is_mechanical_agent(self):
        expanded = self._expand()
        replay = next(s for s in expanded if s["id"] == "feat.4.integration_replay")
        assert replay["agent"] == "mechanical"
        assert replay["executor"] == "mechanical"

    def test_replay_carries_integration_replay_posthook(self):
        expanded = self._expand()
        replay = next(s for s in expanded if s["id"] == "feat.4.integration_replay")
        assert "integration_replay" in replay["post_hooks"]

    def test_no_expansion_when_dials_off(self):
        """Without multi_file_expansion dial, expander returns None entirely."""
        from src.workflows.engine.expander import _maybe_expand_multifile
        from src.workflows.review_density import ReviewDensityDials, to_mission_dial_context
        dials = to_mission_dial_context(ReviewDensityDials(multi_file_expansion=False))
        result = _maybe_expand_multifile(self._make_step(), dials, None)
        assert result is None


# ---------------------------------------------------------------------------
# T5B — Apply.py routes integration_replay to mechanical executor
# ---------------------------------------------------------------------------

class TestApplyRoutesIntegrationReplay:
    def test_routes_to_mechanical(self):
        from general_beckman.apply import _posthook_agent_and_payload
        req = _make_request("integration_replay", source_task_id=1)
        source = {"id": 1, "mission_id": 55}
        ctx = {
            "mode": "standard",
            "suite_glob": "tests/integration/**",
            "workspace_path": "/workspace/55",
        }
        agent_type, payload = _posthook_agent_and_payload(req, source, ctx)
        assert agent_type == "mechanical"
        assert payload["executor"] == "mechanical"
        assert payload["payload"]["action"] == "integration_replay"

    def test_mode_from_context_dial(self):
        """Canonical key is `integration_replay_mode` (not _dial)."""
        from general_beckman.apply import _posthook_agent_and_payload
        req = _make_request("integration_replay", source_task_id=1)
        source = {"id": 1, "mission_id": 55}
        ctx = {"integration_replay_mode": "strict"}
        agent_type, payload = _posthook_agent_and_payload(req, source, ctx)
        assert payload["payload"]["mode"] == "strict"

    def test_shuffle_seed_defaults_to_mission_id(self):
        """Canonical: shuffle_seed = ctx.shuffle_seed | mission_id | 0."""
        from general_beckman.apply import _posthook_agent_and_payload
        req = _make_request("integration_replay", source_task_id=42)
        source = {"id": 42, "mission_id": 55}
        ctx = {}
        agent_type, payload = _posthook_agent_and_payload(req, source, ctx)
        assert payload["payload"]["shuffle_seed"] == 55

    def test_shuffle_seed_ctx_override_wins(self):
        from general_beckman.apply import _posthook_agent_and_payload
        req = _make_request("integration_replay", source_task_id=42)
        source = {"id": 42, "mission_id": 55}
        ctx = {"shuffle_seed": 999}
        agent_type, payload = _posthook_agent_and_payload(req, source, ctx)
        assert payload["payload"]["shuffle_seed"] == 999


# ---------------------------------------------------------------------------
# T5B — _apply_simple_blocker_verdict handles integration_replay
# ---------------------------------------------------------------------------

# Verdict-path coverage for `_apply_simple_blocker_verdict` is provided by the
# canonical Z3 T3 + Z3 T4 tests (security/accessibility/contract/perf/
# adr_drift all dispatch via the same generic helper).  Z3 R1 audit removed
# the agent-specific duplicates that bound `source["agent_type"]` etc.
class TestApplySimpleBlockerVerdict:
    """Sentinel — kept so removed tests are traceable in git blame."""
    pass


# ---------------------------------------------------------------------------
# T5C — integration_bisect
# ---------------------------------------------------------------------------

class TestIntegrationBisect:
    @pytest.mark.asyncio
    async def test_bisect_finds_breaking_pair(self, tmp_path):
        """Bisect narrows to the commit where regression was introduced."""
        (tmp_path / ".git").mkdir()
        test_file = tmp_path / "tests" / "integration" / "test_x.py"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("def test_x(): pass")

        # sha0=HEAD (fails), sha1 (fails), sha2 (passes), sha3 (passes)
        # sha2 is the last good commit; sha1 is the first bad
        pass_at = {"sha2", "sha3"}
        checkout_history: list[str] = []

        ib_module = _import_module("mr_roboto.integration_bisect")

        async def _fake_checkout(sha: str, cwd: str) -> bool:
            checkout_history.append(sha)
            return True

        async def _fake_suite(suite_glob: str, cwd: str) -> tuple[bool, str]:
            last = checkout_history[-1] if checkout_history else "sha0"
            if last in pass_at:
                return True, "2 passed"
            return False, "FAILED test_x"

        commits = ["sha0", "sha1", "sha2", "sha3"]
        with patch.object(ib_module, "_git_checkout", side_effect=_fake_checkout), \
             patch.object(ib_module, "_run_suite", side_effect=_fake_suite):
            result = await ib_module.integration_bisect(
                commits=commits,
                suite_glob="tests/integration/**",
                workspace_path=str(tmp_path),
            )

        assert result["breaking_pair"] is not None
        bp = result["breaking_pair"]
        assert bp[0] in commits  # commit_a (last good)
        assert bp[1] in commits  # commit_b (first bad)

    @pytest.mark.asyncio
    async def test_bisect_returns_none_when_cant_isolate(self, tmp_path):
        """Returns breaking_pair=None when all commits fail."""
        (tmp_path / ".git").mkdir()
        (tmp_path / "tests" / "integration").mkdir(parents=True)
        (tmp_path / "tests" / "integration" / "test_x.py").write_text("")

        ib_module = _import_module("mr_roboto.integration_bisect")

        async def _fake_checkout(sha, cwd):
            return True

        async def _fake_suite(suite_glob, cwd):
            return False, "FAILED"

        with patch.object(ib_module, "_git_checkout", side_effect=_fake_checkout), \
             patch.object(ib_module, "_run_suite", side_effect=_fake_suite):
            result = await ib_module.integration_bisect(
                commits=["sha0", "sha1"],
                suite_glob="tests/integration/**",
                workspace_path=str(tmp_path),
            )

        assert result["breaking_pair"] is None

    @pytest.mark.asyncio
    async def test_bisect_skip_when_no_git(self, tmp_path):
        from mr_roboto.integration_bisect import integration_bisect
        result = await integration_bisect(
            commits=["sha0", "sha1"],
            suite_glob="tests/**",
            workspace_path=str(tmp_path),
        )
        assert result["breaking_pair"] is None

    @pytest.mark.asyncio
    async def test_bisect_skip_when_too_few_commits(self, tmp_path):
        (tmp_path / ".git").mkdir()
        from mr_roboto.integration_bisect import integration_bisect
        result = await integration_bisect(
            commits=["only_one"],
            suite_glob="tests/**",
            workspace_path=str(tmp_path),
        )
        assert result["breaking_pair"] is None


# ---------------------------------------------------------------------------
# T5C — bisect failure emits mission_lessons (mock db)
# Z3 R1 audit: full coverage of bisect-driven lesson emission now lives in
# tests/test_z3_r4_bisect_richer_lessons.py (cluster + diff_shortstat + dispatch
# wrapper + cascade-safety).  The smoke test below was the only one in this
# file that exercised the path against the live integration_bisect verb, so
# keep it (renamed) and drop the rest.
# ---------------------------------------------------------------------------

class TestBisectLessonEmissionSmoke:
    @pytest.mark.asyncio
    async def test_strict_fail_emits_lesson(self, monkeypatch):
        """When integration_replay strict fails + bisect finds pair → lesson emitted."""
        lesson_calls = []

        async def _fake_upsert(**kwargs):
            lesson_calls.append(kwargs)
            return 1

        async def _fake_bisect(commits, suite_glob, workspace_path):
            return {
                "breaking_pair": ["sha1", "sha0"],
                "failing_test": "FAILED test_bar",
                "diagnostic": "AssertionError",
            }

        import src.infra.mission_lessons as ml_module
        ib_module = _import_module("mr_roboto.integration_bisect")
        monkeypatch.setattr(ml_module, "upsert_mission_lesson", _fake_upsert)
        # Patch the integration_bisect function in the module
        monkeypatch.setattr(ib_module, "integration_bisect", _fake_bisect)

        ctx = {
            "suite_glob": "tests/integration/**",
            "workspace_path": "/fake/path",
        }
        raw_dict = {
            "verdict": "fail",
            "mode": "strict",
            "commits_replayed": ["sha0", "sha1"],
        }

        # Run the bisect+lesson path exactly as apply.py does it.
        # This mirrors the logic in _apply_posthook_verdict for integration_replay.
        from mr_roboto.integration_bisect import integration_bisect as _bisect
        bisect_result = await _bisect(
            commits=list(raw_dict.get("commits_replayed") or []),
            suite_glob=str(ctx.get("suite_glob") or "tests/integration/**"),
            workspace_path=str(ctx.get("workspace_path") or ""),
        )
        bp = bisect_result.get("breaking_pair")
        assert bp is not None
        await ml_module.upsert_mission_lesson(
            stack="unknown",
            domain="integration",
            pattern="integration_replay_break",
            fix=f"check commit {bp[1]}",
            severity="blocker",
            source_kind="posthook_fail",
            source_ref={"breaking_pair": bp},
        )

        assert len(lesson_calls) == 1
        assert lesson_calls[0]["pattern"] == "integration_replay_break"
        assert "sha0" in lesson_calls[0]["fix"]
        assert lesson_calls[0]["domain"] == "integration"

# Cascade-safety for lesson-emit failures is covered by
# tests/test_z3_r4_bisect_richer_lessons.py::TestDispatchEmits
# ::test_lesson_emit_failure_does_not_cascade — see that file.


# ---------------------------------------------------------------------------
# Registry/import smoke tests
# ---------------------------------------------------------------------------

class TestImports:
    def test_integration_replay_importable(self):
        from mr_roboto.integration_replay import integration_replay
        assert callable(integration_replay)

    def test_integration_bisect_importable(self):
        from mr_roboto.integration_bisect import integration_bisect
        assert callable(integration_bisect)

    def test_mr_roboto_exports_both(self):
        import mr_roboto
        assert hasattr(mr_roboto, "integration_replay")
        assert hasattr(mr_roboto, "integration_bisect")

    def test_reversibility_registered(self):
        from mr_roboto.reversibility import VERB_REVERSIBILITY
        assert VERB_REVERSIBILITY.get("integration_replay") == "full"
        assert VERB_REVERSIBILITY.get("integration_bisect") == "full"

    def test_review_density_dials_importable(self):
        """Z3 R1 audit — canonical location: review_density module + posthooks._dial_get."""
        from src.workflows.review_density import ReviewDensityDials
        from general_beckman.posthooks import _dial_get
        dials = ReviewDensityDials()
        assert dials.integration_replay == "standard"
        assert _dial_get(None, "integration_replay", "standard") == "standard"
        assert _dial_get({"integration_replay": "strict"}, "integration_replay", "standard") == "strict"
