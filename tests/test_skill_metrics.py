"""Tests for skill A/B metrics tracking."""
import asyncio
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestSkillMetrics(unittest.TestCase):
    """Test skill metrics recording and summary."""

    def setUp(self):
        """Use a temp DB for testing."""
        os.environ["DB_PATH"] = ":memory:"
        # Reset the singleton connection so each test gets a fresh DB
        import src.infra.db as _db_mod
        _db_mod._db_connection = None

    def test_record_skill_metric(self):
        from src.infra.db import init_db, record_skill_metric, get_skill_metrics_summary
        async def _test():
            await init_db()
            await record_skill_metric(1, "currency_api_routing", True, 3, "assistant", 15.0)
            await record_skill_metric(2, "currency_api_routing", True, 2, "assistant", 10.0)
            await record_skill_metric(3, "currency_api_routing", False, 5, "assistant", 30.0)
            summary = await get_skill_metrics_summary()
            per_skill = summary["per_skill"]
            self.assertEqual(len(per_skill), 1)
            self.assertEqual(per_skill[0]["name"], "currency_api_routing")
            self.assertEqual(per_skill[0]["total"], 3)
            self.assertEqual(per_skill[0]["successes"], 2)
            self.assertAlmostEqual(per_skill[0]["success_rate"], 66.7, places=1)
        run_async(_test())

    def test_record_no_skill_baseline(self):
        from src.infra.db import init_db, record_no_skill_metric, get_skill_metrics_summary
        async def _test():
            await init_db()
            await record_no_skill_metric(10, True, 4, "executor", 20.0)
            await record_no_skill_metric(11, False, 6, "executor", 40.0)
            summary = await get_skill_metrics_summary()
            baseline = summary["overall"].get("baseline", {})
            self.assertEqual(baseline["total"], 2)
            self.assertEqual(baseline["successes"], 1)
            self.assertEqual(baseline["success_rate"], 50.0)
        run_async(_test())

    def test_ab_comparison(self):
        from src.infra.db import init_db, record_skill_metric, record_no_skill_metric, get_skill_metrics_summary
        async def _test():
            await init_db()
            # With skills: 3 successes out of 4
            for i in range(3):
                await record_skill_metric(i, "test_skill", True, 3, "assistant", 10.0)
            await record_skill_metric(3, "test_skill", False, 5, "assistant", 20.0)
            # Baseline: 1 success out of 4
            await record_no_skill_metric(10, True, 5, "assistant", 25.0)
            for i in range(3):
                await record_no_skill_metric(11+i, False, 6, "assistant", 30.0)

            summary = await get_skill_metrics_summary()
            with_skills = summary["overall"]["with_skills"]
            baseline = summary["overall"]["baseline"]
            self.assertEqual(with_skills["success_rate"], 75.0)
            self.assertEqual(baseline["success_rate"], 25.0)
            # Skills show 50% lift
            self.assertGreater(with_skills["success_rate"], baseline["success_rate"])
        run_async(_test())

    def test_empty_metrics(self):
        from src.infra.db import init_db, get_skill_metrics_summary
        async def _test():
            await init_db()
            summary = await get_skill_metrics_summary()
            self.assertEqual(summary["overall"], {})
            self.assertEqual(summary["per_skill"], [])
        run_async(_test())

    def test_avg_iterations_tracked(self):
        from src.infra.db import init_db, record_skill_metric, get_skill_metrics_summary
        async def _test():
            await init_db()
            await record_skill_metric(1, "s1", True, 2, "a", 5.0)
            await record_skill_metric(2, "s1", True, 4, "a", 15.0)
            summary = await get_skill_metrics_summary()
            self.assertAlmostEqual(summary["per_skill"][0]["avg_iterations"], 3.0, places=1)
        run_async(_test())


if __name__ == "__main__":
    unittest.main()
