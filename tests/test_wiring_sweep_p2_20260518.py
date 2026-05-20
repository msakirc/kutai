"""Host-path tests for the Z5 P2 + Z10 P2 fixes (2026-05-18 sweep).

Z5 P2 — mobile_smoke / Maestro always soft-skipped because flow_paths
always resolved to []. Two-pronged fix:
  - Ship a smoke flow in every mobile_* core recipe so an instantiated
    recipe leaves its .flow.yaml under the mission workspace.
  - Extend the apply.py mobile_smoke dispatch fallback to auto-discover
    those flows from <workspace>/recipes/**/flows/*.flow.yaml.

Z10 P2 — `require_confirmation` was opt-in only; no workflow set it. The
dispatcher now auto-arms it when ``confirm_policy`` is configured (task
context OR ``KUTAI_CONFIRM_POLICY`` env), preserving current default-off
behaviour for callers that haven't opted in.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestZ5P2RecipesShipSmokeFlows(unittest.TestCase):
    """Each of the 4 previously-missing core mobile recipes now ships a
    smoke flow AND wires it via templates.smoke_flow."""

    CORE_RECIPES = (
        ("mobile_auth", "auth_smoke.flow.yaml"),
        ("mobile_nav", "nav_smoke.flow.yaml"),
        ("mobile_persistence", "persistence_smoke.flow.yaml"),
        ("mobile_offline_sync", "offline_sync_smoke.flow.yaml"),
    )

    def test_every_core_recipe_ships_a_smoke_flow_file(self):
        for recipe, flow_name in self.CORE_RECIPES:
            with self.subTest(recipe=recipe):
                flow = (
                    REPO_ROOT / "recipes" / recipe / "v1" / "flows" / flow_name
                )
                self.assertTrue(
                    flow.exists(), f"missing {flow}",
                )
                text = flow.read_text(encoding="utf-8")
                # Sanity: it actually looks like a Maestro flow
                self.assertIn("appId:", text)
                self.assertIn("launchApp", text)

    def test_every_core_recipe_declares_smoke_flow_template(self):
        for recipe, flow_name in self.CORE_RECIPES:
            with self.subTest(recipe=recipe):
                yaml_path = REPO_ROOT / "recipes" / recipe / "v1" / "recipe.yaml"
                spec = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
                templates = spec.get("templates") or {}
                self.assertIn(
                    "smoke_flow", templates,
                    f"{recipe}/v1/recipe.yaml: templates.smoke_flow missing",
                )
                self.assertTrue(
                    templates["smoke_flow"].endswith(flow_name),
                    f"{recipe}: smoke_flow does not match expected file",
                )


class TestZ5P2MobileSmokeAutoDiscovery(unittest.IsolatedAsyncioTestCase):
    """The mobile_smoke posthook dispatch must surface workspace-scanned
    flows when neither maestro_flows nor produces declare any."""

    async def test_dispatch_discovers_flows_in_workspace_recipes(self):
        from general_beckman.apply import _posthook_agent_and_payload
        from general_beckman.result_router import PostHookVerdict

        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            flows_dir = ws / "recipes" / "mobile_auth" / "v1" / "flows"
            flows_dir.mkdir(parents=True)
            (flows_dir / "auth_smoke.flow.yaml").write_text(
                "appId: com.example.app\n---\n- launchApp: {}\n",
                encoding="utf-8",
            )
            nav_flows = ws / "recipes" / "mobile_nav" / "v1" / "flows"
            nav_flows.mkdir(parents=True)
            (nav_flows / "nav_smoke.flow.yaml").write_text(
                "appId: com.example.app\n---\n- launchApp: {}\n",
                encoding="utf-8",
            )

            verdict = PostHookVerdict(
                source_task_id=1, kind="mobile_smoke",
                passed=True, raw={},
            )
            # Step 14.8 shape: produces only .json files, no maestro_flows.
            ctx = {
                "workspace_path": str(ws),
                "produces": [
                    "mission_1/.store/store_metadata.json",
                    "mission_1/.store/privacy_nutrition_labels.json",
                ],
            }
            source = {"id": 1, "mission_id": 1, "context": "{}"}
            agent_type, wire = _posthook_agent_and_payload(verdict, source, ctx)
            self.assertEqual(agent_type, "mechanical")
            flow_paths = wire["payload"]["flow_paths"]
            # Both flows discovered, deterministic order.
            self.assertEqual(len(flow_paths), 2)
            self.assertTrue(
                any("auth_smoke" in p for p in flow_paths),
                f"auth_smoke missing: {flow_paths}",
            )
            self.assertTrue(
                any("nav_smoke" in p for p in flow_paths),
                f"nav_smoke missing: {flow_paths}",
            )

    async def test_dispatch_respects_explicit_maestro_flows(self):
        """Explicit context.maestro_flows must short-circuit auto-discovery
        — callers that already supplied flows shouldn't get extras tacked on.
        """
        from general_beckman.apply import _posthook_agent_and_payload
        from general_beckman.result_router import PostHookVerdict

        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            flows = ws / "recipes" / "mobile_auth" / "v1" / "flows"
            flows.mkdir(parents=True)
            (flows / "auth_smoke.flow.yaml").write_text(
                "appId: x\n", encoding="utf-8",
            )
            verdict = PostHookVerdict(
                source_task_id=1, kind="mobile_smoke",
                passed=True, raw={},
            )
            ctx = {
                "workspace_path": str(ws),
                "maestro_flows": ["explicit.flow.yaml"],
            }
            source = {"id": 1, "mission_id": 1, "context": "{}"}
            _, wire = _posthook_agent_and_payload(verdict, source, ctx)
            self.assertEqual(
                wire["payload"]["flow_paths"], ["explicit.flow.yaml"],
            )


class TestZ10P2RequireConfirmationAutoArm(unittest.IsolatedAsyncioTestCase):
    """Dispatcher must auto-arm require_confirmation when a confirm_policy
    is set — either via task ctx or KUTAI_CONFIRM_POLICY env.

    Test method: monkeypatch the gate (_await_confirmation) and the
    dispatch path (_run_dispatch, _log_action_event, _log_external_publish)
    so we can directly observe whether the gate was reached. The
    skeleton gate polls for 60s in production — useless to actually run.
    """

    async def asyncSetUp(self):
        import mr_roboto as _m
        self._mod = _m
        # Save originals.
        self._orig_gate = _m._await_confirmation
        self._orig_dispatch = _m._run_dispatch
        self._orig_audit = _m._log_action_event
        self._orig_publish = _m._log_external_publish

        from mr_roboto.actions import Action
        self._gate_calls: list[dict] = []
        self._dispatch_calls: list[dict] = []

        async def _fake_gate(*, task_id, verb, reversibility, payload):
            self._gate_calls.append({"verb": verb, "rev": reversibility})
            # Return a gate_action so dispatch never runs (mirrors a
            # founder veto). Returning None would let dispatch proceed.
            return Action(status="cancelled", error="gated-by-test")

        async def _fake_dispatch(task):
            self._dispatch_calls.append(task)
            return Action(status="completed", result={})

        async def _noop(*args, **kwargs):
            return None

        _m._await_confirmation = _fake_gate
        _m._run_dispatch = _fake_dispatch
        _m._log_action_event = _noop
        _m._log_external_publish = _noop

    async def asyncTearDown(self):
        m = self._mod
        m._await_confirmation = self._orig_gate
        m._run_dispatch = self._orig_dispatch
        m._log_action_event = self._orig_audit
        m._log_external_publish = self._orig_publish
        os.environ.pop("KUTAI_CONFIRM_POLICY", None)

    async def test_default_off_preserves_legacy_behaviour(self):
        """No policy + no explicit flag → no auto-arm. Pin the safe default."""
        os.environ.pop("KUTAI_CONFIRM_POLICY", None)
        task = {
            "id": 1, "mission_id": 1,
            "payload": {"action": "notify_user", "message": "x"},
            "context": {},
        }
        # Need to also pass the safety-guard pre-check — notify_user isn't
        # shell-executing so it skips the guard. Run.
        res = await self._mod.run(task)
        self.assertEqual(self._gate_calls, [], "gate fired with policy=off")
        self.assertEqual(len(self._dispatch_calls), 1)
        self.assertEqual(res.status, "completed")

    async def test_irreversible_only_policy_arms_irreversible_verbs(self):
        os.environ["KUTAI_CONFIRM_POLICY"] = "irreversible_only"
        task = {
            "id": 2, "mission_id": 1,
            # notify_user is tagged irreversible in VERB_REVERSIBILITY.
            "payload": {"action": "notify_user", "message": "x"},
            "context": {},
        }
        res = await self._mod.run(task)
        self.assertEqual(len(self._gate_calls), 1, "auto-arm did not fire")
        self.assertEqual(self._gate_calls[0]["verb"], "notify_user")
        self.assertEqual(self._gate_calls[0]["rev"], "irreversible")
        # Dispatch was short-circuited because the fake gate returned an
        # action.
        self.assertEqual(self._dispatch_calls, [])
        self.assertEqual(res.status, "cancelled")

    async def test_irreversible_only_skips_reversible_verbs(self):
        os.environ["KUTAI_CONFIRM_POLICY"] = "irreversible_only"
        task = {
            "id": 3, "mission_id": 1,
            # workspace_snapshot is reversible — irreversible_only policy
            # must not gate it.
            "payload": {
                "action": "workspace_snapshot",
                "workspace_path": "/tmp/x",
            },
            "context": {},
        }
        res = await self._mod.run(task)
        self.assertEqual(self._gate_calls, [])
        self.assertEqual(len(self._dispatch_calls), 1)
        self.assertEqual(res.status, "completed")

    async def test_ctx_policy_overrides_env_off(self):
        os.environ.pop("KUTAI_CONFIRM_POLICY", None)
        task = {
            "id": 4, "mission_id": 1,
            "payload": {"action": "notify_user", "message": "x"},
            "context": {"confirm_policy": "irreversible_only"},
        }
        await self._mod.run(task)
        self.assertEqual(len(self._gate_calls), 1)

    async def test_partial_or_worse_policy_arms_partial_verbs(self):
        os.environ["KUTAI_CONFIRM_POLICY"] = "partial_or_worse"
        # Pick a verb tagged partial via reversibility table.
        from mr_roboto.reversibility import VERB_REVERSIBILITY
        partial_verbs = [
            v for v, r in VERB_REVERSIBILITY.items() if r == "partial"
        ]
        self.assertTrue(partial_verbs, "no partial verbs to test with")
        task = {
            "id": 5, "mission_id": 1,
            "payload": {"action": partial_verbs[0]},
            "context": {},
        }
        await self._mod.run(task)
        self.assertEqual(len(self._gate_calls), 1)
        self.assertEqual(self._gate_calls[0]["rev"], "partial")


if __name__ == "__main__":
    unittest.main()
