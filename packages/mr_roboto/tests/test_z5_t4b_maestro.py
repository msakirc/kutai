"""Z5 T4b — Maestro mobile-QA adapter + mobile_smoke post-hook + recipes 4-6.

Covers:
1.  `maestro` verb — dispatches through mr_roboto.run, parses pass/fail per
    flow, soft-skips when the Maestro CLI is absent, is in VERB_REVERSIBILITY.
2.  `mobile_smoke` post-hook — registry entry present, _posthook_agent_and_payload
    routes to the mechanical `maestro` verb, the verdict handler gates the
    source on the Maestro exit code (monkeypatching the maestro verb).
3.  Recipes 4-6 (mobile_push / mobile_deep_links / mobile_offline_sync) —
    each load_recipe()s clean, match_recipe('fastapi+sqlite+expo', ...)
    returns it, instantiate_recipe round-trips with no unresolved <<...>>.
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

import pytest

import mr_roboto
import mr_roboto.maestro_run  # noqa: F401 — registers the module in sys.modules

# `from mr_roboto import maestro_run` resolves to the FUNCTION (the package
# __init__ re-exports it, shadowing the submodule on the package namespace).
# The real module object — the monkeypatch target for `run_cmd` — lives in
# sys.modules under its dotted name.
_MAESTRO_MODULE = sys.modules["mr_roboto.maestro_run"]


# ===========================================================================
# Helpers
# ===========================================================================

WORKTREE_ROOT = Path(__file__).resolve().parents[3]
RECIPES_DIR = WORKTREE_ROOT / "recipes"

_MOBILE_RECIPES = ("mobile_push", "mobile_deep_links", "mobile_offline_sync")


def _make_run_cmd(exit_seq: list[int], stdout: str = "Flow passed",
                   missing: bool = False, timed_out: bool = False):
    """Return an async run_cmd stand-in.

    ``exit_seq`` — one exit code consumed per invocation (per flow).
    ``missing`` — simulate the Maestro CLI not being installed.
    ``timed_out`` — simulate a per-flow timeout.
    """
    calls: list[list[str]] = []
    seq = list(exit_seq)

    async def _mock(**kwargs):
        calls.append(list(kwargs.get("cmd") or []))
        if missing:
            return {
                "exit": -1, "stdout_tail": "", "stderr_tail": "",
                "duration_s": 0.0, "timed_out": False, "ok": False,
                "error": "executable not found: maestro",
            }
        code = seq.pop(0) if seq else 0
        return {
            "exit": code,
            "stdout_tail": stdout,
            "stderr_tail": "",
            "duration_s": 1.0,
            "timed_out": timed_out,
            "ok": code == 0 and not timed_out,
        }

    _mock.calls = calls  # type: ignore[attr-defined]
    return _mock


# ===========================================================================
# 1.  maestro verb
# ===========================================================================

class TestMaestroVerb:
    def test_maestro_in_reversibility_registry(self):
        from mr_roboto.reversibility import VERB_REVERSIBILITY, get_reversibility
        assert "maestro" in VERB_REVERSIBILITY
        # Read-only test run — fully reversible.
        assert get_reversibility("maestro") == "full"

    @pytest.mark.asyncio
    async def test_maestro_dispatches_and_passes(self, monkeypatch):
        mock = _make_run_cmd(exit_seq=[0, 0])
        monkeypatch.setattr(_MAESTRO_MODULE, "run_cmd", mock)
        task = {
            "id": 1, "mission_id": None,
            "payload": {
                "action": "maestro",
                "flow_paths": ["flows/a.flow.yaml", "flows/b.flow.yaml"],
                "workspace_path": "/ws",
            },
        }
        action = await mr_roboto.run(task)
        assert action.status == "completed"
        res = action.result
        assert res["ok"] is True
        assert res["skipped"] is False
        assert res["flows_run"] == 2
        assert res["passed"] == 2
        assert res["failed"] == 0
        assert res["exit"] == 0
        # One `maestro test <flow>` invocation per flow.
        assert len(mock.calls) == 2
        assert mock.calls[0][:2] == ["maestro", "test"]

    @pytest.mark.asyncio
    async def test_maestro_parses_a_failed_flow(self, monkeypatch):
        # First flow passes (exit 0), second fails (exit 1).
        mock = _make_run_cmd(exit_seq=[0, 1])
        monkeypatch.setattr(_MAESTRO_MODULE, "run_cmd", mock)
        task = {
            "id": 2, "mission_id": None,
            "payload": {
                "action": "maestro",
                "flow_paths": ["flows/a.flow.yaml", "flows/b.flow.yaml"],
                "workspace_path": "/ws",
            },
        }
        action = await mr_roboto.run(task)
        # A failed flow → the verb fails (non-zero exit gates the step).
        assert action.status == "failed"
        res = action.result
        assert res["ok"] is False
        assert res["flows_run"] == 2
        assert res["passed"] == 1
        assert res["failed"] == 1
        assert res["exit"] == 1
        assert "1/2" in (action.error or "")

    @pytest.mark.asyncio
    async def test_maestro_soft_skips_when_cli_absent(self, monkeypatch):
        mock = _make_run_cmd(exit_seq=[], missing=True)
        monkeypatch.setattr(_MAESTRO_MODULE, "run_cmd", mock)
        task = {
            "id": 3, "mission_id": None,
            "payload": {
                "action": "maestro",
                "flow_paths": ["flows/a.flow.yaml"],
                "workspace_path": "/ws",
            },
        }
        action = await mr_roboto.run(task)
        # Soft-skip is a completed (soft pass), never failed.
        assert action.status == "completed"
        res = action.result
        assert res["skipped"] is True
        assert res["ok"] is True
        assert res["flows_run"] == 0

    @pytest.mark.asyncio
    async def test_maestro_soft_skips_with_no_flows(self, monkeypatch):
        # No flow paths → nothing to run → soft-skip (not a failure).
        mock = _make_run_cmd(exit_seq=[])
        monkeypatch.setattr(_MAESTRO_MODULE, "run_cmd", mock)
        task = {
            "id": 4, "mission_id": None,
            "payload": {"action": "maestro", "flow_paths": [],
                        "workspace_path": "/ws"},
        }
        action = await mr_roboto.run(task)
        assert action.status == "completed"
        assert action.result["skipped"] is True
        assert len(mock.calls) == 0

    @pytest.mark.asyncio
    async def test_maestro_timeout_counts_as_failed(self, monkeypatch):
        mock = _make_run_cmd(exit_seq=[-1], timed_out=True)
        monkeypatch.setattr(_MAESTRO_MODULE, "run_cmd", mock)
        task = {
            "id": 5, "mission_id": None,
            "payload": {"action": "maestro",
                        "flow_paths": ["flows/a.flow.yaml"],
                        "workspace_path": "/ws"},
        }
        action = await mr_roboto.run(task)
        assert action.status == "failed"
        assert action.result["failed"] == 1

    @pytest.mark.asyncio
    async def test_maestro_verb_function_direct(self, monkeypatch):
        from mr_roboto.maestro_run import maestro_run
        mock = _make_run_cmd(exit_seq=[0])
        monkeypatch.setattr(_MAESTRO_MODULE, "run_cmd", mock)
        res = await maestro_run(
            mission_id=None,
            flow_paths=["flows/smoke.flow.yaml"],
            workspace_path="/ws",
        )
        assert res["ok"] is True
        assert res["passed"] == 1


# ===========================================================================
# 2.  mobile_smoke post-hook
# ===========================================================================

class TestMobileSmokeRegistry:
    def test_registry_entry_present(self):
        from general_beckman.posthooks import POST_HOOK_REGISTRY, PostHookSpec
        assert "mobile_smoke" in POST_HOOK_REGISTRY
        spec = POST_HOOK_REGISTRY["mobile_smoke"]
        assert isinstance(spec, PostHookSpec)
        assert spec.kind == "mobile_smoke"
        assert spec.verb == "maestro"
        assert spec.default_severity == "blocker"
        assert spec.cost_band == "heavy"

    def test_mobile_smoke_in_post_hook_kinds(self):
        from general_beckman.posthooks import POST_HOOK_KINDS
        assert "mobile_smoke" in POST_HOOK_KINDS

    def test_mobile_smoke_no_autowire(self):
        # Maestro needs a running app + a flow YAML — opt-in only.
        from general_beckman.posthooks import POST_HOOK_REGISTRY
        spec = POST_HOOK_REGISTRY["mobile_smoke"]
        assert spec.resolve_triggers() == []

    def test_determine_posthooks_accepts_mobile_smoke(self):
        from general_beckman.posthooks import determine_posthooks
        task = {"agent_type": "coder"}
        ctx = {"post_hooks": ["mobile_smoke"]}
        kinds = determine_posthooks(task, ctx, {})
        assert "mobile_smoke" in kinds


class TestMobileSmokePayload:
    def test_posthook_agent_and_payload_routes_to_maestro(self):
        from general_beckman.apply import _posthook_agent_and_payload
        from general_beckman.result_router import RequestPostHook

        a = RequestPostHook(
            source_task_id=99, kind="mobile_smoke",
            source_ctx={},
        )
        source_ctx = {
            "maestro_flows": ["flows/push_smoke.flow.yaml"],
            "workspace_path": "/ws",
        }
        agent_type, payload = _posthook_agent_and_payload(a, {}, source_ctx)
        assert agent_type == "mechanical"
        assert payload["posthook_kind"] == "mobile_smoke"
        assert payload["payload"]["action"] == "maestro"
        assert payload["payload"]["flow_paths"] == ["flows/push_smoke.flow.yaml"]
        assert payload["payload"]["workspace_path"] == "/ws"

    def test_payload_falls_back_to_produces_flow_yaml(self):
        from general_beckman.apply import _posthook_agent_and_payload
        from general_beckman.result_router import RequestPostHook

        a = RequestPostHook(source_task_id=7, kind="mobile_smoke", source_ctx={})
        # No explicit maestro_flows → scan produces for *.flow.yaml.
        source_ctx = {
            "produces": ["app/index.tsx", "flows/login.flow.yaml", "README.md"],
        }
        _agent, payload = _posthook_agent_and_payload(a, {}, source_ctx)
        assert payload["payload"]["flow_paths"] == ["flows/login.flow.yaml"]

    def test_posthook_title(self):
        from general_beckman.apply import _posthook_title
        from general_beckman.result_router import RequestPostHook
        a = RequestPostHook(source_task_id=12, kind="mobile_smoke", source_ctx={})
        assert "Mobile smoke" in _posthook_title(a, {})


class TestMobileSmokeVerdictGate:
    """The verdict handler gates the source task on the Maestro exit code.

    The maestro verb is monkeypatched so the test never spawns a process;
    the structured result drives _apply_mobile_smoke_verdict directly.
    """

    @pytest.mark.asyncio
    async def test_passing_maestro_run_drains_pending(self, monkeypatch):
        import general_beckman.apply as apply_mod
        from general_beckman.result_router import PostHookVerdict

        updates: list[dict] = []

        async def _fake_update_task(task_id, **kw):
            updates.append({"task_id": task_id, **kw})

        async def _noop(*a, **k):
            return None

        monkeypatch.setattr("src.infra.db.update_task", _fake_update_task)
        monkeypatch.setattr(apply_mod, "_spawn_workflow_advance_if_mission", _noop)

        source = {"id": 99, "worker_attempts": 0, "result": "{}"}
        ctx: dict = {"_pending_posthooks": ["mobile_smoke"]}
        # Maestro exit 0 → passed verdict (rewrite.py reads `ok`).
        verdict = PostHookVerdict(
            source_task_id=99, kind="mobile_smoke", passed=True,
            raw={"ok": True, "skipped": False, "flows_run": 1,
                 "passed": 1, "failed": 0, "exit": 0},
        )
        await apply_mod._apply_mobile_smoke_verdict(
            source=source, ctx=ctx, pending=["mobile_smoke"], verdict=verdict,
        )
        # Pending drained; source flipped to completed.
        assert ctx["_pending_posthooks"] == []
        assert any(u.get("status") == "completed" for u in updates)

    @pytest.mark.asyncio
    async def test_failing_maestro_run_retries_source(self, monkeypatch):
        import general_beckman.apply as apply_mod
        from general_beckman.result_router import PostHookVerdict

        updates: list[dict] = []

        async def _fake_update_task(task_id, **kw):
            updates.append({"task_id": task_id, **kw})

        monkeypatch.setattr("src.infra.db.update_task", _fake_update_task)

        source = {"id": 88, "worker_attempts": 1, "max_worker_attempts": 15,
                  "result": "prior output"}
        ctx: dict = {"_pending_posthooks": ["mobile_smoke"]}
        # Maestro exit 1 → failed verdict; source must retry with feedback.
        verdict = PostHookVerdict(
            source_task_id=88, kind="mobile_smoke", passed=False,
            raw={"ok": False, "skipped": False, "flows_run": 1,
                 "passed": 0, "failed": 1, "exit": 1,
                 "error": "maestro: 1/1 flow(s) failed (exit 1)"},
        )
        await apply_mod._apply_mobile_smoke_verdict(
            source=source, ctx=ctx, pending=["mobile_smoke"], verdict=verdict,
        )
        retry = [u for u in updates if u.get("status") == "pending"]
        assert retry, "failing mobile_smoke must retry the source"
        assert retry[0]["error_category"] == "quality"
        assert "Maestro" in (ctx.get("_schema_error") or "")

    @pytest.mark.asyncio
    async def test_soft_skipped_maestro_run_passes(self, monkeypatch):
        # Maestro CLI absent → maestro verb returns ok=True, skipped=True →
        # rewrite.py synthesises passed=True → verdict drains without block.
        import general_beckman.apply as apply_mod
        from general_beckman.result_router import PostHookVerdict

        updates: list[dict] = []

        async def _fake_update_task(task_id, **kw):
            updates.append({"task_id": task_id, **kw})

        async def _noop(*a, **k):
            return None

        monkeypatch.setattr("src.infra.db.update_task", _fake_update_task)
        monkeypatch.setattr(apply_mod, "_spawn_workflow_advance_if_mission", _noop)

        source = {"id": 77, "worker_attempts": 0, "result": "{}"}
        ctx: dict = {"_pending_posthooks": ["mobile_smoke"]}
        verdict = PostHookVerdict(
            source_task_id=77, kind="mobile_smoke", passed=True,
            raw={"ok": True, "skipped": True, "flows_run": 0,
                 "passed": 0, "failed": 0, "exit": -1,
                 "reason": "maestro CLI not installed"},
        )
        await apply_mod._apply_mobile_smoke_verdict(
            source=source, ctx=ctx, pending=["mobile_smoke"], verdict=verdict,
        )
        assert ctx["_pending_posthooks"] == []
        assert any(u.get("status") == "completed" for u in updates)


class TestMobileSmokeEndToEndGate:
    """maestro verb → result → rewrite.py PostHookVerdict synthesis.

    Verifies the gate closes end to end: a non-zero Maestro exit produces a
    failed mr_roboto Action, which rewrite.py Rule 0c turns into a
    PostHookVerdict(passed=False).
    """

    @pytest.mark.asyncio
    async def test_failed_flow_yields_failed_verdict(self, monkeypatch):
        from general_beckman.rewrite import _rewrite_one
        from general_beckman.result_router import Failed, PostHookVerdict

        # Simulate the orchestrator handing rewrite.py a Failed action for
        # a mechanical mobile_smoke posthook task (maestro exit non-zero).
        task = {"id": 500, "agent_type": "mechanical"}
        task_ctx = {"source_task_id": 42, "posthook_kind": "mobile_smoke"}
        failed = Failed(task_id=500, error="maestro flows failed", raw={})
        out = _rewrite_one(task, task_ctx, failed)
        verdicts = [a for a in out if isinstance(a, PostHookVerdict)]
        assert verdicts, "failed mobile_smoke posthook must synthesise a verdict"
        assert verdicts[0].kind == "mobile_smoke"
        assert verdicts[0].passed is False

    @pytest.mark.asyncio
    async def test_passing_flow_yields_passed_verdict(self, monkeypatch):
        from general_beckman.rewrite import _rewrite_one
        from general_beckman.result_router import Complete, PostHookVerdict

        # A completed mechanical posthook task carries the maestro result as
        # a JSON string in Complete.raw["result"] — rewrite.py Rule 0b reads
        # `ok` to synthesise the verdict.
        task = {"id": 501, "agent_type": "mechanical"}
        task_ctx = {"source_task_id": 43, "posthook_kind": "mobile_smoke"}
        maestro_result = {"ok": True, "skipped": False, "flows_run": 1,
                          "passed": 1, "failed": 0, "exit": 0}
        complete = Complete(
            task_id=501, result="ok", iterations=1, metadata={},
            raw={"status": "completed", "result": json.dumps(maestro_result)},
        )
        out = _rewrite_one(task, task_ctx, complete)
        verdicts = [a for a in out if isinstance(a, PostHookVerdict)]
        assert verdicts, "completed mobile_smoke posthook must synthesise a verdict"
        assert verdicts[0].kind == "mobile_smoke"
        assert verdicts[0].passed is True


# ===========================================================================
# 3.  Recipes 4-6 — mobile_push / mobile_deep_links / mobile_offline_sync
# ===========================================================================

@pytest.mark.parametrize("recipe_name", _MOBILE_RECIPES)
class TestMobileRecipes:
    def _yaml_path(self, recipe_name: str) -> str:
        return str(RECIPES_DIR / recipe_name / "v1" / "recipe.yaml")

    def test_load_recipe_clean(self, recipe_name):
        from src.infra.recipes import load_recipe
        recipe = load_recipe(self._yaml_path(recipe_name))
        assert recipe.name == recipe_name
        assert recipe.version == "v1"
        # lessons_domain is set to the recipe name (cross-mission lessons key).
        assert recipe.lessons_domain == recipe_name

    def test_tech_stack_includes_expo(self, recipe_name):
        from src.infra.recipes import load_recipe
        recipe = load_recipe(self._yaml_path(recipe_name))
        stacks = recipe.requires.get("tech_stack") or []
        assert "fastapi+sqlite+expo" in stacks
        assert "fastapi+postgres+expo" in stacks

    def test_post_hooks_declared(self, recipe_name):
        from src.infra.recipes import load_recipe
        recipe = load_recipe(self._yaml_path(recipe_name))
        for hook in ("imports_check", "test_run", "pattern_lint"):
            assert hook in recipe.post_hooks, f"{hook} missing from {recipe_name}"

    def test_match_recipe_returns_it_for_expo_stack(self, recipe_name):
        from src.infra.recipes import list_recipes, match_recipe
        recipes = list_recipes(str(RECIPES_DIR))
        ranked = match_recipe("fastapi+sqlite+expo", recipes)
        names = {r.name: score for r, score in ranked}
        assert recipe_name in names, f"{recipe_name} not matched"
        # Exact stack match → fit score 1.0.
        assert names[recipe_name] == 1.0

    def test_instantiate_recipe_round_trips(self, recipe_name):
        from src.infra.recipes import load_recipe, instantiate_recipe
        recipe = load_recipe(self._yaml_path(recipe_name))
        with tempfile.TemporaryDirectory() as td:
            res = instantiate_recipe(recipe, td, params=dict(recipe.param_defaults))
            assert res["ok"] is True
            assert res["files_written"], "no files instantiated"
            assert not res["skipped"], f"templates skipped: {res['skipped']}"
            # No unresolved <<KEY>> tokens anywhere in the output.
            for f in Path(td).rglob("*"):
                if not f.is_file():
                    continue
                text = f.read_text(encoding="utf-8", errors="replace")
                unresolved = re.findall(r"<<[A-Z_][A-Z0-9_]*>>", text)
                assert not unresolved, f"{f.name} has unresolved tokens: {unresolved}"

    def test_prompts_expo_present(self, recipe_name):
        from src.infra.recipes import load_recipe
        recipe = load_recipe(self._yaml_path(recipe_name))
        assert "expo" in recipe.prompts
        prompt_path = RECIPES_DIR / recipe_name / "v1" / recipe.prompts["expo"]
        assert prompt_path.exists(), f"{prompt_path} missing"

    def test_lessons_md_has_real_pitfalls(self, recipe_name):
        # Each recipe ships 8-12 real pitfalls as markdown bullets.
        lessons = RECIPES_DIR / recipe_name / "v1" / "lessons.md"
        assert lessons.exists()
        bullets = [
            ln for ln in lessons.read_text(encoding="utf-8").splitlines()
            if ln.strip().startswith("- **")
        ]
        assert 8 <= len(bullets) <= 14, (
            f"{recipe_name} lessons.md has {len(bullets)} bullets, expected 8-12"
        )

    def test_spec_md_present(self, recipe_name):
        spec = RECIPES_DIR / recipe_name / "v1" / "spec.md"
        assert spec.exists()
        assert spec.read_text(encoding="utf-8").strip()


class TestMaestroFlowTemplates:
    """mobile_push + mobile_deep_links ship a Maestro smoke-flow YAML so the
    mobile_smoke post-hook has a flow to run."""

    @pytest.mark.parametrize("recipe_name", ("mobile_push", "mobile_deep_links"))
    def test_smoke_flow_declared_and_present(self, recipe_name):
        from src.infra.recipes import load_recipe
        recipe = load_recipe(
            str(RECIPES_DIR / recipe_name / "v1" / "recipe.yaml")
        )
        assert "smoke_flow" in recipe.templates
        flow_path = (
            RECIPES_DIR / recipe_name / "v1" / recipe.templates["smoke_flow"]
        )
        assert flow_path.exists(), f"{flow_path} missing"

    @pytest.mark.parametrize("recipe_name", ("mobile_push", "mobile_deep_links"))
    def test_smoke_flow_covers_signin_to_signout(self, recipe_name):
        from src.infra.recipes import load_recipe
        recipe = load_recipe(
            str(RECIPES_DIR / recipe_name / "v1" / "recipe.yaml")
        )
        flow_path = (
            RECIPES_DIR / recipe_name / "v1" / recipe.templates["smoke_flow"]
        )
        text = flow_path.read_text(encoding="utf-8")
        # sign in -> onboard -> core action -> sign out.
        assert "Sign in" in text or "sign in" in text.lower()
        assert "Sign out" in text or "sign out" in text.lower()
        assert "launchApp" in text

    def test_offline_sync_ships_flow(self):
        # Z5 P2 (461a90d0): mobile_offline_sync ships its own Maestro smoke
        # flow so the mobile_smoke post-hook has a flow to run instead of
        # soft-passing. (Was previously asserted absent — that pre-dated P2.)
        from src.infra.recipes import load_recipe
        recipe = load_recipe(
            str(RECIPES_DIR / "mobile_offline_sync" / "v1" / "recipe.yaml")
        )
        assert "smoke_flow" in recipe.templates
        flow_path = (
            RECIPES_DIR / "mobile_offline_sync" / "v1"
            / recipe.templates["smoke_flow"]
        )
        assert flow_path.exists(), f"{flow_path} missing"
