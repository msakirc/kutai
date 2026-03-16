# tests/test_header_parser.py
import sys, os, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.header_parser import RateLimitSnapshot, parse_rate_limit_headers


class TestRateLimitSnapshot(unittest.TestCase):

    def test_openai_standard_headers(self):
        headers = {
            "x-ratelimit-limit-requests": "500",
            "x-ratelimit-limit-tokens": "200000",
            "x-ratelimit-remaining-requests": "490",
            "x-ratelimit-remaining-tokens": "195000",
            "x-ratelimit-reset-requests": "12ms",
            "x-ratelimit-reset-tokens": "6s",
        }
        snap = parse_rate_limit_headers("openai", headers)
        self.assertIsNotNone(snap)
        self.assertEqual(snap.rpm_limit, 500)
        self.assertEqual(snap.tpm_limit, 200000)
        self.assertEqual(snap.rpm_remaining, 490)
        self.assertEqual(snap.tpm_remaining, 195000)
        self.assertIsNotNone(snap.rpm_reset_at)
        self.assertIsNotNone(snap.tpm_reset_at)

    def test_groq_headers(self):
        """Groq uses same format as OpenAI."""
        headers = {
            "x-ratelimit-limit-requests": "30",
            "x-ratelimit-limit-tokens": "131072",
            "x-ratelimit-remaining-requests": "28",
            "x-ratelimit-remaining-tokens": "120000",
            "x-ratelimit-reset-requests": "2s",
            "x-ratelimit-reset-tokens": "1.5s",
        }
        snap = parse_rate_limit_headers("groq", headers)
        self.assertIsNotNone(snap)
        self.assertEqual(snap.rpm_limit, 30)
        self.assertEqual(snap.rpm_remaining, 28)

    def test_empty_headers_returns_none(self):
        snap = parse_rate_limit_headers("openai", {})
        self.assertIsNone(snap)

    def test_partial_headers_still_parse(self):
        headers = {"x-ratelimit-remaining-requests": "10"}
        snap = parse_rate_limit_headers("openai", headers)
        self.assertIsNotNone(snap)
        self.assertEqual(snap.rpm_remaining, 10)
        self.assertIsNone(snap.rpm_limit)

    def test_anthropic_headers(self):
        headers = {
            "anthropic-ratelimit-requests-limit": "50",
            "anthropic-ratelimit-requests-remaining": "45",
            "anthropic-ratelimit-requests-reset": "2026-01-27T12:00:30Z",
            "anthropic-ratelimit-tokens-limit": "80000",
            "anthropic-ratelimit-tokens-remaining": "72000",
            "anthropic-ratelimit-tokens-reset": "2026-01-27T12:00:30Z",
        }
        snap = parse_rate_limit_headers("anthropic", headers)
        self.assertIsNotNone(snap)
        self.assertEqual(snap.rpm_limit, 50)
        self.assertEqual(snap.rpm_remaining, 45)
        self.assertEqual(snap.tpm_limit, 80000)
        self.assertEqual(snap.tpm_remaining, 72000)
        self.assertIsNotNone(snap.rpm_reset_at)

    def test_cerebras_headers_daily_requests_minute_tokens(self):
        headers = {
            "x-ratelimit-limit-tokens-minute": "131072",
            "x-ratelimit-remaining-tokens-minute": "120000",
            "x-ratelimit-reset-tokens-minute": "45.5",
            "x-ratelimit-limit-requests-day": "1000",
            "x-ratelimit-remaining-requests-day": "950",
            "x-ratelimit-reset-requests-day": "33011.382867",
        }
        snap = parse_rate_limit_headers("cerebras", headers)
        self.assertIsNotNone(snap)
        self.assertEqual(snap.tpm_limit, 131072)
        self.assertEqual(snap.tpm_remaining, 120000)
        self.assertEqual(snap.rpd_limit, 1000)
        self.assertEqual(snap.rpd_remaining, 950)
        self.assertIsNone(snap.rpm_limit)

    def test_sambanova_headers(self):
        headers = {
            "x-ratelimit-limit-requests-minute": "20",
            "x-ratelimit-remaining-requests-minute": "18",
            "x-ratelimit-reset-requests-minute": "30",
            "x-ratelimit-limit-requests-day": "5000",
            "x-ratelimit-remaining-requests-day": "4900",
            "x-ratelimit-reset-requests-day": "43200",
        }
        snap = parse_rate_limit_headers("sambanova", headers)
        self.assertIsNotNone(snap)
        self.assertEqual(snap.rpm_limit, 20)
        self.assertEqual(snap.rpm_remaining, 18)
        self.assertEqual(snap.rpd_limit, 5000)
        self.assertEqual(snap.rpd_remaining, 4900)

    def test_gemini_with_daily_limits(self):
        headers = {
            "x-ratelimit-limit-requests": "15",
            "x-ratelimit-remaining-requests": "12",
            "x-ratelimit-reset-requests": "30s",
            "x-ratelimit-limit-tokens": "1000000",
            "x-ratelimit-remaining-tokens": "950000",
            "x-ratelimit-reset-tokens": "30s",
            "x-ratelimit-limit-requests-day": "1500",
            "x-ratelimit-remaining-requests-day": "1400",
            "x-ratelimit-reset-requests-day": "43200",
        }
        snap = parse_rate_limit_headers("gemini", headers)
        self.assertIsNotNone(snap)
        self.assertEqual(snap.rpm_limit, 15)
        self.assertEqual(snap.tpm_limit, 1000000)
        self.assertEqual(snap.rpd_limit, 1500)

    def test_llm_provider_prefix_stripped(self):
        """litellm proxy adds llm_provider- prefix."""
        headers = {
            "llm_provider-x-ratelimit-limit-requests": "100",
            "llm_provider-x-ratelimit-remaining-requests": "90",
        }
        snap = parse_rate_limit_headers("openai", headers)
        self.assertIsNotNone(snap)
        self.assertEqual(snap.rpm_limit, 100)
        self.assertEqual(snap.rpm_remaining, 90)

    def test_unknown_provider_uses_openai_style(self):
        headers = {
            "x-ratelimit-limit-requests": "60",
            "x-ratelimit-remaining-requests": "55",
        }
        snap = parse_rate_limit_headers("unknown_provider", headers)
        self.assertIsNotNone(snap)
        self.assertEqual(snap.rpm_limit, 60)


if __name__ == "__main__":
    unittest.main()
