# tests/test_rate_limiter.py
import sys, os, unittest, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.rate_limiter import RateLimitState, RateLimitManager
from src.models.header_parser import RateLimitSnapshot


class TestRateLimitStateHeaders(unittest.TestCase):

    def test_update_from_snapshot_sets_limits(self):
        state = RateLimitState(rpm_limit=30, tpm_limit=100000)
        snap = RateLimitSnapshot(
            rpm_limit=50, rpm_remaining=45,
            tpm_limit=200000, tpm_remaining=180000,
        )
        state.update_from_snapshot(snap)
        self.assertEqual(state.rpm_limit, 50)
        self.assertEqual(state.tpm_limit, 200000)
        self.assertEqual(state._original_rpm, 50)
        self.assertEqual(state._original_tpm, 200000)
        self.assertTrue(state._limits_discovered)

    def test_update_from_snapshot_lower_limit_accepted(self):
        state = RateLimitState(rpm_limit=100, tpm_limit=500000)
        snap = RateLimitSnapshot(rpm_limit=30, tpm_limit=100000)
        state.update_from_snapshot(snap)
        self.assertEqual(state.rpm_limit, 30)
        self.assertEqual(state.tpm_limit, 100000)

    def test_has_capacity_uses_header_remaining_when_fresh(self):
        state = RateLimitState(rpm_limit=30, tpm_limit=100000)
        snap = RateLimitSnapshot(rpm_remaining=5, tpm_remaining=50000)
        state.update_from_snapshot(snap)
        self.assertTrue(state.has_capacity(1000))

    def test_has_capacity_detects_exhausted_from_headers(self):
        state = RateLimitState(rpm_limit=30, tpm_limit=100000)
        snap = RateLimitSnapshot(rpm_remaining=0, tpm_remaining=50000)
        state.update_from_snapshot(snap)
        self.assertFalse(state.has_capacity(0))

    def test_daily_limit_exhaustion_blocks_capacity(self):
        state = RateLimitState(rpm_limit=30, tpm_limit=100000)
        snap = RateLimitSnapshot(
            rpm_remaining=10, tpm_remaining=50000,
            rpd_remaining=0, rpd_reset_at=time.time() + 3600,
        )
        state.update_from_snapshot(snap)
        self.assertFalse(state.has_capacity(0))

    def test_update_from_snapshot_restores_adaptive_reduction(self):
        """If headers show higher limit than our adapted limit, restore."""
        state = RateLimitState(rpm_limit=30, tpm_limit=100000)
        state.record_429()
        self.assertLess(state.rpm_limit, 30)

        snap = RateLimitSnapshot(rpm_limit=50, rpm_remaining=48)
        state.update_from_snapshot(snap)
        self.assertEqual(state.rpm_limit, 50)
        self.assertEqual(state._rate_limit_hits, 0)


class TestRateLimitManagerHeaders(unittest.TestCase):

    def _make_manager(self):
        mgr = RateLimitManager()
        mgr.register_model(
            litellm_name="gpt-4o", provider="openai",
            rpm=500, tpm=200000,
            provider_aggregate_rpm=500, provider_aggregate_tpm=2000000,
        )
        mgr.register_model(
            litellm_name="groq/llama-3.3-70b-versatile", provider="groq",
            rpm=30, tpm=131072,
            provider_aggregate_rpm=30, provider_aggregate_tpm=131072,
        )
        return mgr

    def test_update_from_headers_updates_model_state(self):
        mgr = self._make_manager()
        snap = RateLimitSnapshot(
            rpm_limit=600, rpm_remaining=590,
            tpm_limit=300000, tpm_remaining=290000,
        )
        mgr.update_from_headers("gpt-4o", "openai", snap)
        state = mgr.model_limits["gpt-4o"]
        self.assertEqual(state.rpm_limit, 600)
        self.assertEqual(state.tpm_limit, 300000)

    def test_update_from_headers_updates_provider_state(self):
        mgr = self._make_manager()
        snap = RateLimitSnapshot(
            rpm_limit=600, rpm_remaining=590,
            tpm_limit=3000000, tpm_remaining=2900000,
        )
        mgr.update_from_headers("gpt-4o", "openai", snap)
        prov = mgr.provider_limits["openai"]
        self.assertEqual(prov.rpm_limit, 600)
        self.assertEqual(prov.tpm_limit, 3000000)

    def test_update_from_headers_model_and_provider_differ(self):
        mgr = self._make_manager()
        snap1 = RateLimitSnapshot(rpm_limit=30, tpm_limit=100000)
        mgr.update_from_headers("groq/llama-3.3-70b-versatile", "groq", snap1)
        mgr.register_model(
            litellm_name="groq/llama-3.1-8b-instant", provider="groq",
            rpm=30, tpm=131072,
        )
        snap2 = RateLimitSnapshot(rpm_limit=30, rpm_remaining=28, tpm_remaining=125000)
        mgr.update_from_headers("groq/llama-3.1-8b-instant", "groq", snap2)
        self.assertIn("groq/llama-3.3-70b-versatile", mgr.model_limits)
        self.assertIn("groq/llama-3.1-8b-instant", mgr.model_limits)

    def test_get_status_includes_header_info(self):
        mgr = self._make_manager()
        snap = RateLimitSnapshot(rpm_remaining=490, tpm_remaining=195000)
        mgr.update_from_headers("gpt-4o", "openai", snap)
        status = mgr.get_status()
        model_status = status["models"]["gpt-4o"]
        self.assertIn("discovered", model_status)


if __name__ == "__main__":
    unittest.main()
