"""Comprehensive tests for shopping resilience modules:
circuit_breaker, rate_budget, error_recovery, fallback_chain.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite


# ---------------------------------------------------------------------------
# Async test helper
# ---------------------------------------------------------------------------

def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ═══════════════════════════════════════════════════════════════════════════
# 1. TestCircuitBreaker
# ═══════════════════════════════════════════════════════════════════════════

class TestCircuitBreaker(unittest.TestCase):
    """Test circuit breaker state transitions."""

    def _make_breaker(self, threshold=3, cooldown=1.0):
        from src.shopping.resilience.circuit_breaker import CircuitBreaker
        return CircuitBreaker(failure_threshold=threshold, cooldown_seconds=cooldown)

    def test_initial_state_closed(self):
        cb = self._make_breaker()
        self.assertEqual(cb.get_state("example.com"), "CLOSED")

    def test_is_allowed_closed(self):
        cb = self._make_breaker()
        self.assertTrue(cb.is_allowed("example.com"))

    def test_closed_to_open_after_failures(self):
        cb = self._make_breaker(threshold=3)
        for _ in range(3):
            cb.on_failure("example.com")
        self.assertEqual(cb.get_state("example.com"), "OPEN")

    def test_open_rejects_requests(self):
        cb = self._make_breaker(threshold=2, cooldown=60)
        cb.on_failure("example.com")
        cb.on_failure("example.com")
        self.assertFalse(cb.is_allowed("example.com"))

    def test_open_to_half_open_after_cooldown(self):
        cb = self._make_breaker(threshold=2, cooldown=0.01)
        cb.on_failure("example.com")
        cb.on_failure("example.com")
        self.assertEqual(cb.get_state("example.com"), "OPEN")
        time.sleep(0.02)
        # After cooldown, should transition to HALF_OPEN
        self.assertTrue(cb.is_allowed("example.com"))
        self.assertEqual(cb.get_state("example.com"), "HALF_OPEN")

    def test_half_open_to_closed_on_success(self):
        cb = self._make_breaker(threshold=2, cooldown=0.01)
        cb.on_failure("example.com")
        cb.on_failure("example.com")
        time.sleep(0.02)
        cb.is_allowed("example.com")  # Triggers HALF_OPEN
        cb.on_success("example.com")
        self.assertEqual(cb.get_state("example.com"), "CLOSED")

    def test_half_open_to_open_on_failure(self):
        cb = self._make_breaker(threshold=2, cooldown=0.01)
        cb.on_failure("example.com")
        cb.on_failure("example.com")
        time.sleep(0.02)
        cb.is_allowed("example.com")  # HALF_OPEN
        cb.on_failure("example.com")
        self.assertEqual(cb.get_state("example.com"), "OPEN")

    def test_success_resets_failure_count(self):
        cb = self._make_breaker(threshold=3)
        cb.on_failure("example.com")
        cb.on_failure("example.com")
        cb.on_success("example.com")
        self.assertEqual(cb._failure_counts["example.com"], 0)
        cb.on_failure("example.com")
        # Should still be CLOSED — only 1 failure after reset
        self.assertEqual(cb.get_state("example.com"), "CLOSED")

    def test_multiple_domains_isolated(self):
        cb = self._make_breaker(threshold=2)
        cb.on_failure("a.com")
        cb.on_failure("a.com")
        cb.on_failure("b.com")
        self.assertEqual(cb.get_state("a.com"), "OPEN")
        self.assertEqual(cb.get_state("b.com"), "CLOSED")

    def test_get_all_states(self):
        cb = self._make_breaker(threshold=2)
        cb.is_allowed("a.com")
        cb.is_allowed("b.com")
        states = cb.get_all_states()
        self.assertIn("a.com", states)
        self.assertIn("b.com", states)

    def test_global_functions(self):
        """Test the module-level async wrappers."""
        from src.shopping.resilience.circuit_breaker import (
            check_circuit, record_success, record_failure, get_circuit_status,
        )
        domain = "test_global_func.com"
        self.assertTrue(run_async(check_circuit(domain)))
        run_async(record_success(domain))
        status = get_circuit_status()
        self.assertIn(domain, status)


# ═══════════════════════════════════════════════════════════════════════════
# 2. TestRateBudget
# ═══════════════════════════════════════════════════════════════════════════

class TestRateBudget(unittest.TestCase):
    """Test rate budget tracking with temp DB."""

    def setUp(self):
        fd, self._tmpfile = tempfile.mkstemp(suffix=".db")
        os.close(fd)

        # Patch the module-level DB singleton and path
        import src.shopping.resilience.rate_budget as rb
        self._rb = rb
        self._original_db = rb._budget_db
        self._original_path = rb.BUDGET_DB_PATH
        rb._budget_db = None
        rb.BUDGET_DB_PATH = self._tmpfile

    def tearDown(self):
        if self._rb._budget_db:
            run_async(self._rb._budget_db.close())
        self._rb._budget_db = self._original_db
        self._rb.BUDGET_DB_PATH = self._original_path
        if os.path.exists(self._tmpfile):
            os.unlink(self._tmpfile)
        for suffix in ("-wal", "-shm"):
            p = self._tmpfile + suffix
            if os.path.exists(p):
                os.unlink(p)

    def test_init_and_get_remaining(self):
        run_async(self._rb.init_rate_budget_db())
        remaining = run_async(self._rb.get_remaining_budget("akakce"))
        self.assertEqual(remaining, 100)

    def test_consume_budget(self):
        run_async(self._rb.init_rate_budget_db())
        run_async(self._rb.consume_budget("akakce", 10))
        remaining = run_async(self._rb.get_remaining_budget("akakce"))
        self.assertEqual(remaining, 90)

    def test_budget_exhaustion(self):
        run_async(self._rb.init_rate_budget_db())
        run_async(self._rb.consume_budget("akakce", 100))
        remaining = run_async(self._rb.get_remaining_budget("akakce"))
        self.assertEqual(remaining, 0)

    def test_over_consume_stays_zero(self):
        run_async(self._rb.init_rate_budget_db())
        run_async(self._rb.consume_budget("akakce", 150))
        remaining = run_async(self._rb.get_remaining_budget("akakce"))
        self.assertEqual(remaining, 0)

    def test_unknown_domain_zero(self):
        run_async(self._rb.init_rate_budget_db())
        remaining = run_async(self._rb.get_remaining_budget("unknown_domain"))
        self.assertEqual(remaining, 0)

    def test_reset_daily_budgets(self):
        run_async(self._rb.init_rate_budget_db())
        run_async(self._rb.consume_budget("trendyol", 30))
        run_async(self._rb.reset_daily_budgets())
        remaining = run_async(self._rb.get_remaining_budget("trendyol"))
        self.assertEqual(remaining, 50)  # Full budget restored

    def test_budget_summary(self):
        run_async(self._rb.init_rate_budget_db())
        run_async(self._rb.consume_budget("akakce", 5))
        summary = run_async(self._rb.get_budget_summary())
        self.assertIn("akakce", summary)
        self.assertEqual(summary["akakce"]["used"], 5)
        self.assertEqual(summary["akakce"]["remaining"], 95)
        self.assertEqual(summary["akakce"]["total"], 100)

    def test_multiple_domains(self):
        run_async(self._rb.init_rate_budget_db())
        run_async(self._rb.consume_budget("akakce", 10))
        run_async(self._rb.consume_budget("trendyol", 5))
        r1 = run_async(self._rb.get_remaining_budget("akakce"))
        r2 = run_async(self._rb.get_remaining_budget("trendyol"))
        self.assertEqual(r1, 90)
        self.assertEqual(r2, 45)


# ═══════════════════════════════════════════════════════════════════════════
# 3. TestErrorRecovery
# ═══════════════════════════════════════════════════════════════════════════

class TestErrorRecovery(unittest.TestCase):
    """Test error classification and recovery recommendations."""

    def test_classify_transient(self):
        from src.shopping.resilience.scraper_failure_handler import classify_error
        self.assertEqual(classify_error(TimeoutError("Connection timed out")), "transient")

    def test_classify_rate_limit(self):
        from src.shopping.resilience.scraper_failure_handler import classify_error
        self.assertEqual(classify_error(Exception("429 Too Many Requests")), "rate_limit")

    def test_classify_blocked(self):
        from src.shopping.resilience.scraper_failure_handler import classify_error
        self.assertEqual(classify_error(Exception("403 Forbidden captcha")), "blocked")

    def test_classify_parse_error(self):
        from src.shopping.resilience.scraper_failure_handler import classify_error
        self.assertEqual(classify_error(ValueError("JSON decode error")), "parse_error")

    def test_classify_permanent(self):
        from src.shopping.resilience.scraper_failure_handler import classify_error
        self.assertEqual(classify_error(Exception("Something completely unknown")), "permanent")

    def test_handle_scraper_transient_first_try(self):
        from src.shopping.resilience.scraper_failure_handler import handle_scraper_error
        result = run_async(handle_scraper_error(
            "trendyol", TimeoutError("timeout"), {"retry_count": 0}
        ))
        self.assertEqual(result["action"], "retry")
        self.assertEqual(result["error_class"], "transient")
        self.assertGreater(result["delay"], 0)

    def test_handle_scraper_transient_exhausted(self):
        from src.shopping.resilience.scraper_failure_handler import handle_scraper_error
        result = run_async(handle_scraper_error(
            "trendyol", TimeoutError("timeout"), {"retry_count": 5}
        ))
        self.assertEqual(result["action"], "fallback")

    def test_handle_scraper_blocked(self):
        from src.shopping.resilience.scraper_failure_handler import handle_scraper_error
        result = run_async(handle_scraper_error(
            "trendyol", Exception("403 Forbidden"), {}
        ))
        self.assertEqual(result["action"], "fallback")
        self.assertEqual(result["error_class"], "blocked")

    def test_handle_scraper_rate_limit(self):
        from src.shopping.resilience.scraper_failure_handler import handle_scraper_error
        result = run_async(handle_scraper_error(
            "trendyol", Exception("429 rate limit"), {}
        ))
        self.assertEqual(result["action"], "skip")
        self.assertEqual(result["error_class"], "rate_limit")

    def test_handle_scraper_permanent(self):
        from src.shopping.resilience.scraper_failure_handler import handle_scraper_error
        result = run_async(handle_scraper_error(
            "trendyol", Exception("fatal unexpected error"), {}
        ))
        self.assertEqual(result["action"], "abort")

    def test_handle_llm_transient(self):
        from src.shopping.resilience.scraper_failure_handler import handle_llm_error
        result = run_async(handle_llm_error(
            TimeoutError("connection timeout"), {"retry_count": 0}
        ))
        self.assertEqual(result["action"], "retry")

    def test_handle_llm_token_limit(self):
        from src.shopping.resilience.scraper_failure_handler import handle_llm_error
        result = run_async(handle_llm_error(
            Exception("token limit exceeded"), {}
        ))
        self.assertEqual(result["action"], "retry")
        self.assertIn("truncate", result["reason"])

    def test_handle_llm_rate_limit(self):
        from src.shopping.resilience.scraper_failure_handler import handle_llm_error
        result = run_async(handle_llm_error(
            Exception("429 rate limit"), {"retry_count": 0}
        ))
        self.assertEqual(result["action"], "retry")
        self.assertGreater(result["delay"], 0)

    def test_handle_llm_unrecoverable(self):
        from src.shopping.resilience.scraper_failure_handler import handle_llm_error
        result = run_async(handle_llm_error(
            Exception("some fatal error"), {"retry_count": 10}
        ))
        self.assertEqual(result["action"], "abort")


# ═══════════════════════════════════════════════════════════════════════════
# 4. TestFallbackChain
# ═══════════════════════════════════════════════════════════════════════════

class TestFallbackChain(unittest.TestCase):
    """Test fallback chain execution."""

    def test_primary_succeeds(self):
        from src.shopping.resilience.fallback_chain import execute_with_fallback

        async def primary():
            return "primary_result"

        async def fallback():
            return "fallback_result"

        result = run_async(execute_with_fallback(primary, [fallback]))
        self.assertEqual(result, "primary_result")

    def test_primary_fails_fallback_succeeds(self):
        from src.shopping.resilience.fallback_chain import execute_with_fallback

        async def primary():
            raise RuntimeError("primary failed")

        async def fallback():
            return "fallback_result"

        result = run_async(execute_with_fallback(primary, [fallback]))
        self.assertEqual(result, "fallback_result")

    def test_all_fail_raises_runtime_error(self):
        from src.shopping.resilience.fallback_chain import execute_with_fallback

        async def fail1():
            raise RuntimeError("fail1")

        async def fail2():
            raise RuntimeError("fail2")

        with self.assertRaises(RuntimeError):
            run_async(execute_with_fallback(fail1, [fail2]))

    def test_sync_functions_work(self):
        from src.shopping.resilience.fallback_chain import execute_with_fallback

        def primary():
            return "sync_result"

        result = run_async(execute_with_fallback(primary, []))
        self.assertEqual(result, "sync_result")

    def test_primary_returns_none_tries_fallback(self):
        from src.shopping.resilience.fallback_chain import execute_with_fallback

        async def primary():
            return None

        async def fallback():
            return "fallback_result"

        result = run_async(execute_with_fallback(primary, [fallback]))
        self.assertEqual(result, "fallback_result")

    def test_args_forwarded(self):
        from src.shopping.resilience.fallback_chain import execute_with_fallback

        async def fn(x, y):
            return x + y

        result = run_async(execute_with_fallback(fn, [], 3, 4))
        self.assertEqual(result, 7)

    def test_kwargs_forwarded(self):
        from src.shopping.resilience.fallback_chain import execute_with_fallback

        async def fn(x, multiplier=1):
            return x * multiplier

        result = run_async(execute_with_fallback(fn, [], 5, multiplier=3))
        self.assertEqual(result, 15)

    def test_build_fallback_chain(self):
        from src.shopping.resilience.fallback_chain import build_fallback_chain
        chain = build_fallback_chain("akakce")
        # scraper + google_cse (Perplexica excluded — returns text, not products)
        self.assertTrue(len(chain) >= 2)

    def test_build_fallback_chain_default(self):
        from src.shopping.resilience.fallback_chain import build_fallback_chain
        chain = build_fallback_chain("default")
        # Default: google_cse only (no dedicated scraper, no Perplexica)
        self.assertTrue(len(chain) >= 1)


# ═══════════════════════════════════════════════════════════════════════════
# 5. TestGetProductWithFallback
# ═══════════════════════════════════════════════════════════════════════════

class TestGetProductWithFallback(unittest.TestCase):
    """Test get_product_with_fallback tiered search strategy."""

    def _mock_get_scraper(self, called_sources, results_for=None):
        """Return a tracking get_scraper that returns mock scrapers.
        results_for: dict mapping source name -> list of results."""
        results_for = results_for or {}

        def tracking(source):
            called_sources.append(source)
            mock_cls = MagicMock()
            mock_cls.return_value.search = AsyncMock(
                return_value=results_for.get(source, [])
            )
            return mock_cls

        return tracking

    def test_no_sources_tries_all_tiers(self):
        """Without explicit sources, all tiers (aggregators, major, specialty) are tried."""
        from src.shopping.resilience.fallback_chain import (
            get_product_with_fallback, _AGGREGATORS, _MAJOR_RETAILERS, _SPECIALTY_RETAILERS,
        )

        called: list[str] = []

        async def _run():
            with patch("src.shopping.scrapers.get_scraper", side_effect=self._mock_get_scraper(called)), \
                 patch("src.shopping.resilience.cache_fallback.get_stale_product", new_callable=AsyncMock, return_value=None):
                return await get_product_with_fallback("siemens s100")

        run_async(_run())

        all_expected = set(_AGGREGATORS + _MAJOR_RETAILERS + _SPECIALTY_RETAILERS)
        tried = set(called)
        self.assertTrue(
            all_expected.issubset(tried),
            f"Expected all tiers {all_expected} to be tried, "
            f"but only tried: {tried}",
        )

    def test_all_tiers_searched_even_if_first_has_results(self):
        """All tiers are searched for price comparison, not just the first success."""
        from src.shopping.resilience.fallback_chain import get_product_with_fallback

        called: list[str] = []
        fake_akakce = MagicMock()
        fake_trendyol = MagicMock()

        async def _run():
            with patch("src.shopping.scrapers.get_scraper",
                       side_effect=self._mock_get_scraper(called, {
                           "akakce": [fake_akakce],
                           "trendyol": [fake_trendyol],
                       })), \
                 patch("src.shopping.resilience.cache_fallback.get_stale_product", new_callable=AsyncMock, return_value=None):
                return await get_product_with_fallback("iphone 15")

        result = run_async(_run())

        # Both sources should be in results
        self.assertIn(fake_akakce, result)
        self.assertIn(fake_trendyol, result)
        # All tiers were searched
        self.assertIn("akakce", called)
        self.assertIn("trendyol", called)

    def test_explicit_sources_tried_first(self):
        """When explicit sources are given, they're tried before default tiers."""
        from src.shopping.resilience.fallback_chain import get_product_with_fallback

        called: list[str] = []
        fake_product = MagicMock()

        async def _run():
            with patch("src.shopping.scrapers.get_scraper",
                       side_effect=self._mock_get_scraper(called, {"migros": [fake_product]})), \
                 patch("src.shopping.resilience.cache_fallback.get_stale_product", new_callable=AsyncMock, return_value=None):
                return await get_product_with_fallback("süt", sources=["migros", "getir"])

        result = run_async(_run())

        self.assertEqual(result, [fake_product])
        # migros found results, so no other tiers needed
        self.assertNotIn("akakce", called)


# ═══════════════════════════════════════════════════════════════════════════
# 6. TestGetCommunityData
# ═══════════════════════════════════════════════════════════════════════════

class TestGetCommunityData(unittest.TestCase):
    """Test get_community_data searches all community sources."""

    def test_searches_all_community_sources(self):
        from src.shopping.resilience.fallback_chain import get_community_data, _COMMUNITY_SOURCES

        called: list[str] = []

        def tracking(source):
            called.append(source)
            mock_cls = MagicMock()
            mock_cls.return_value.search = AsyncMock(return_value=[])
            return mock_cls

        async def _run():
            with patch("src.shopping.scrapers.get_scraper", side_effect=tracking):
                return await get_community_data("siemens s100")

        run_async(_run())

        self.assertEqual(sorted(called), sorted(_COMMUNITY_SOURCES))

    def test_collects_all_results(self):
        """Community data should merge results from all sources, not first-wins."""
        from src.shopping.resilience.fallback_chain import get_community_data

        fake_a = MagicMock()
        fake_b = MagicMock()

        def tracking(source):
            mock_cls = MagicMock()
            if source == "technopat":
                mock_cls.return_value.search = AsyncMock(return_value=[fake_a])
            elif source == "sikayetvar":
                mock_cls.return_value.search = AsyncMock(return_value=[fake_b])
            else:
                mock_cls.return_value.search = AsyncMock(return_value=[])
            return mock_cls

        async def _run():
            with patch("src.shopping.scrapers.get_scraper", side_effect=tracking):
                return await get_community_data("siemens s100")

        result = run_async(_run())

        self.assertEqual(len(result), 2)
        self.assertIn(fake_a, result)
        self.assertIn(fake_b, result)


if __name__ == "__main__":
    unittest.main()
