# tests/test_quota_planner.py
import sys, os, unittest, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fatih_hoca.requirements import QuotaPlanner


class TestQuotaPlanner(unittest.TestCase):

    def _make_planner(self):
        return QuotaPlanner()

    def test_default_threshold_is_conservative(self):
        planner = self._make_planner()
        self.assertGreaterEqual(planner.expensive_threshold, 7)

    def test_healthy_quota_lowers_threshold(self):
        planner = self._make_planner()
        planner.update_paid_utilization("openai", 20.0, reset_in=3600)
        planner.update_paid_utilization("anthropic", 15.0, reset_in=3600)
        planner.set_max_upcoming_difficulty(4)
        planner.recalculate()
        self.assertLessEqual(planner.expensive_threshold, 4)

    def test_exhausted_quota_raises_threshold(self):
        planner = self._make_planner()
        planner.update_paid_utilization("openai", 90.0, reset_in=3600)
        planner.update_paid_utilization("anthropic", 85.0, reset_in=3600)
        planner.set_max_upcoming_difficulty(5)
        planner.recalculate()
        self.assertGreaterEqual(planner.expensive_threshold, 8)

    def test_hard_task_queued_reserves_capacity(self):
        planner = self._make_planner()
        planner.update_paid_utilization("openai", 40.0, reset_in=3600)
        planner.set_max_upcoming_difficulty(9)
        planner.recalculate()
        self.assertGreaterEqual(planner.expensive_threshold, 8)

    def test_quota_reset_soon_lowers_threshold(self):
        planner = self._make_planner()
        planner.update_paid_utilization("openai", 70.0, reset_in=120)
        planner.set_max_upcoming_difficulty(4)
        planner.recalculate()
        self.assertLessEqual(planner.expensive_threshold, 6)

    def test_recent_429s_raise_threshold(self):
        planner = self._make_planner()
        planner.update_paid_utilization("openai", 50.0, reset_in=3600)
        for _ in range(5):
            planner.record_429("openai")
        planner.set_max_upcoming_difficulty(4)
        planner.recalculate()
        self.assertGreaterEqual(planner.expensive_threshold, 7)

    def test_on_quota_restored_triggers_recalculate(self):
        planner = self._make_planner()
        planner.update_paid_utilization("openai", 90.0, reset_in=3600)
        planner.recalculate()
        high = planner.expensive_threshold

        planner.on_quota_restored("openai", new_remaining_pct=80.0)
        self.assertLessEqual(planner.expensive_threshold, high)

    def test_threshold_never_below_1(self):
        planner = self._make_planner()
        planner.update_paid_utilization("openai", 0.0, reset_in=3600)
        planner.set_max_upcoming_difficulty(1)
        planner.recalculate()
        self.assertGreaterEqual(planner.expensive_threshold, 1)

    def test_get_status(self):
        planner = self._make_planner()
        planner.update_paid_utilization("openai", 50.0, reset_in=300)
        status = planner.get_status()
        self.assertIn("expensive_threshold", status)
        self.assertIn("paid_utilization", status)
        self.assertEqual(status["paid_utilization"]["openai"], 50.0)


if __name__ == "__main__":
    unittest.main()
