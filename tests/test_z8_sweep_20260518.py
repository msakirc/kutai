"""Host-path tests for the Z8 P0/P1 fixes from the 2026-05-18 sweep.

Z8 was the worst-audited zone:
  - P0  All 13 ops recipes (backup_verify / cve_scan / cost_monitor /
        dependency_hygiene / synthetic_check) were undiscoverable —
        flat one-level dir layout vs list_recipes()'s two-level
        <name>/<version>/recipe.yaml.
  - P0  alert_triage classified severity then stopped — hardcoded
        oncall_routed=False, never enqueued an oncall_agent task.
  - P1  /ask saved a ticket + acked but never enqueued support_tier1;
        tickets piled up status='open' forever.
  - P1  synthetic_check executor had no dispatch branch in mr_roboto.
  - P1  5 ops crons (backup_verify, dependency_hygiene, cve_scan,
        secret_scan, cost_pull) had dispatch branches but no
        cron_seed.INTERNAL_CADENCES rows.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path
from unittest.mock import patch, AsyncMock


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestZ8P0OpsRecipesDiscoverable(unittest.TestCase):
    """list_recipes() must surface every ops recipe."""

    EXPECTED_OPS_RECIPES = {
        "backup_verify_postgres",
        "backup_verify_sqlite",
        "cost_monitor_aws",
        "cost_monitor_stripe",
        "cost_monitor_vercel",
        "cve_scan_docker",
        "cve_scan_node",
        "cve_scan_python",
        "dependency_hygiene_node",
        "dependency_hygiene_python",
        "synthetic_check_k6",
        "synthetic_check_lighthouse",
    }

    def test_ops_recipes_use_two_level_layout(self):
        for name in self.EXPECTED_OPS_RECIPES:
            with self.subTest(recipe=name):
                p = REPO_ROOT / "recipes" / name / "v1" / "recipe.yaml"
                self.assertTrue(p.exists(), f"missing {p}")

    def test_list_recipes_returns_all_ops_recipes(self):
        from src.infra.recipes import list_recipes
        names = {r.name for r in list_recipes()}
        missing = self.EXPECTED_OPS_RECIPES - names
        self.assertFalse(
            missing,
            f"list_recipes() missing ops recipes: {sorted(missing)}",
        )


class TestZ8P0AlertTriageEnqueuesOncall(unittest.IsolatedAsyncioTestCase):
    """Critical/high severities must enqueue an oncall_agent task."""

    async def test_critical_severity_enqueues_oncall(self):
        from mr_roboto.executors.alert_triage import run as triage_run
        enqueued: list[dict] = []

        async def _fake_enqueue(spec, **kw):
            enqueued.append({"spec": spec, "kw": kw})
            return 999

        with patch("general_beckman.enqueue", _fake_enqueue), \
             patch(
                 "mr_roboto.executors.alert_triage.classify",
                 return_value="critical",
             ):
            task = {
                "id": 1, "mission_id": 1,
                "payload": {
                    "integration_id": "uptime_robot",
                    "event_id": "evt-1",
                    "payload": {"type": "down"},
                },
            }
            res = await triage_run(task)
            self.assertEqual(res["severity"], "critical")
            self.assertTrue(res["oncall_routed"])
            self.assertEqual(res["oncall_task_id"], 999)
            self.assertEqual(len(enqueued), 1)
            spec = enqueued[0]["spec"]
            self.assertEqual(spec["agent_type"], "oncall_agent")
            self.assertEqual(spec["context"]["severity"], "critical")
            self.assertEqual(
                spec["context"]["triage_source_task_id"], 1,
            )

    async def test_low_severity_does_not_enqueue(self):
        from mr_roboto.executors.alert_triage import run as triage_run

        async def _should_not_run(*a, **kw):
            self.fail("enqueue called for low severity")

        with patch("general_beckman.enqueue", _should_not_run), \
             patch(
                 "mr_roboto.executors.alert_triage.classify",
                 return_value="low",
             ):
            task = {
                "id": 2, "mission_id": 1,
                "payload": {
                    "integration_id": "uptime_robot",
                    "event_id": "evt-2",
                    "payload": {"type": "ok"},
                },
            }
            res = await triage_run(task)
            self.assertEqual(res["severity"], "low")
            self.assertFalse(res["oncall_routed"])
            self.assertIsNone(res["oncall_task_id"])

    async def test_enqueue_failure_does_not_break_triage(self):
        """A broken enqueue must not crash the triage executor."""
        from mr_roboto.executors.alert_triage import run as triage_run

        async def _broken(*a, **kw):
            raise RuntimeError("db lock contention")

        with patch("general_beckman.enqueue", _broken), \
             patch(
                 "mr_roboto.executors.alert_triage.classify",
                 return_value="high",
             ):
            task = {
                "id": 3, "mission_id": 1,
                "payload": {
                    "integration_id": "x", "event_id": "y",
                    "payload": {},
                },
            }
            res = await triage_run(task)
            self.assertEqual(res["severity"], "high")
            self.assertFalse(res["oncall_routed"])
            self.assertIsNone(res["oncall_task_id"])


class TestZ8P1AskEnqueuesSupportTier1(unittest.TestCase):
    """cmd_ask must enqueue a support_tier1 task — source-grep guard."""

    def test_cmd_ask_calls_enqueue_with_support_tier1(self):
        src = (REPO_ROOT / "src/app/telegram_bot.py").read_text(encoding="utf-8")
        # Find the cmd_ask body
        m = re.search(
            r"async def cmd_ask\b.*?(?=\n    async def |\nclass )",
            src, re.DOTALL,
        )
        self.assertIsNotNone(m, "cmd_ask not found")
        body = m.group(0)
        self.assertIn("support_tier1", body)
        self.assertIn("enqueue", body)


class TestZ8P1SyntheticCheckDispatch(unittest.TestCase):
    """mr_roboto must route action='synthetic_check' to the executor."""

    def test_dispatch_branch_present(self):
        src = (
            REPO_ROOT
            / "packages/mr_roboto/src/mr_roboto/__init__.py"
        ).read_text(encoding="utf-8")
        self.assertIn('action == "synthetic_check"', src)
        # And the import points at the real executor.
        self.assertIn(
            "from mr_roboto.executors.synthetic_check import run",
            src,
        )


class TestZ8P1OpsCronsSeeded(unittest.TestCase):
    """5 ops crons must be in INTERNAL_CADENCES with the right _executor."""

    EXPECTED = {
        "ops_backup_verify": "cron_backup_verify",
        "ops_dependency_hygiene": "cron_dep_hygiene",
        "ops_cve_scan": "cron_cve_scan",
        "ops_secret_scan": "cron_secret_scan",
        "ops_cost_pull": "cron_cost_pull",
    }

    def test_every_ops_cadence_seeded(self):
        from general_beckman.cron_seed import INTERNAL_CADENCES
        by_title = {c["title"]: c for c in INTERNAL_CADENCES}
        for title, expected_executor in self.EXPECTED.items():
            with self.subTest(cadence=title):
                self.assertIn(title, by_title, f"{title} not seeded")
                cadence = by_title[title]
                self.assertEqual(
                    cadence["payload"].get("_executor"),
                    expected_executor,
                )
                # Daily or weekly — guard against typo.
                interval = cadence.get("interval_seconds") or 0
                self.assertIn(
                    interval, (86400, 604800),
                    f"{title}: unexpected interval {interval}",
                )


if __name__ == "__main__":
    unittest.main()
