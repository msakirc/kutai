"""
Tests for seed_skills: seeding, idempotency, and query matching.
"""
import asyncio
import os
import sys
import tempfile
import unittest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestSeedSkills(unittest.TestCase):
    """Test skill seeding against a real temp database."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        cls._tmp.close()
        cls.db_path = cls._tmp.name

        import src.app.config as config
        cls._orig_db_path = config.DB_PATH
        config.DB_PATH = cls.db_path

        import src.infra.db as db_mod
        db_mod.DB_PATH = cls.db_path
        db_mod._db_connection = None
        cls.db_mod = db_mod

        run_async(db_mod.init_db())

    @classmethod
    def tearDownClass(cls):
        run_async(cls.db_mod.close_db())
        import src.app.config as config
        config.DB_PATH = cls._orig_db_path
        try:
            os.unlink(cls.db_path)
        except OSError:
            pass

    def test_01_seed_adds_all_skills(self):
        """First seed should add all skills."""
        from src.memory.seed_skills import seed_skills, SEED_SKILLS
        added = run_async(seed_skills())
        self.assertEqual(added, len(SEED_SKILLS))
        self.assertEqual(added, 23)

    def test_02_seed_is_idempotent(self):
        """Second seed should add 0 new skills."""
        from src.memory.seed_skills import seed_skills
        added = run_async(seed_skills())
        self.assertEqual(added, 0)

    def test_03_list_skills_returns_all(self):
        """list_skills should return all 23 seeded skills."""
        from src.memory.skills import list_skills
        skills = run_async(list_skills())
        self.assertEqual(len(skills), 23)

    def test_currency_query_matches_currency_skill(self):
        """'dolar kuru' should match currency_api_routing via regex."""
        import re
        from src.memory.seed_skills import SEED_SKILLS
        currency_skill = next(s for s in SEED_SKILLS if s["name"] == "currency_api_routing")
        pattern = currency_skill["trigger_pattern"]
        self.assertTrue(re.search(pattern, "dolar kuru", re.IGNORECASE))
        self.assertTrue(re.search(pattern, "EUR/TRY exchange rate", re.IGNORECASE))

    def test_weather_query_matches_weather_skill(self):
        """'weather istanbul' should match weather_api_routing."""
        import re
        from src.memory.seed_skills import SEED_SKILLS
        skill = next(s for s in SEED_SKILLS if s["name"] == "weather_api_routing")
        pattern = skill["trigger_pattern"]
        self.assertTrue(re.search(pattern, "weather istanbul", re.IGNORECASE))
        self.assertTrue(re.search(pattern, "hava durumu ankara", re.IGNORECASE))

    def test_github_query_matches_github_skill(self):
        """'llama.cpp github' should match github_routing."""
        import re
        from src.memory.seed_skills import SEED_SKILLS
        skill = next(s for s in SEED_SKILLS if s["name"] == "github_routing")
        pattern = skill["trigger_pattern"]
        self.assertTrue(re.search(pattern, "llama.cpp github repo", re.IGNORECASE))

    def test_shopping_query_matches_shopping_skill(self):
        """'iPhone fiyat' should match shopping_turkish_sources."""
        import re
        from src.memory.seed_skills import SEED_SKILLS
        skill = next(s for s in SEED_SKILLS if s["name"] == "shopping_turkish_sources")
        pattern = skill["trigger_pattern"]
        self.assertTrue(re.search(pattern, "iPhone 15 fiyat", re.IGNORECASE))

    def test_unrelated_query_does_not_match(self):
        """A generic greeting should NOT match any skill trigger pattern."""
        import re
        from src.memory.seed_skills import SEED_SKILLS
        query = "merhaba nasilsin"
        for skill in SEED_SKILLS:
            pattern = skill["trigger_pattern"]
            self.assertIsNone(
                re.search(pattern, query, re.IGNORECASE),
                f"'{query}' should not match skill '{skill['name']}' pattern '{pattern}'",
            )

    def test_unrelated_recipe_query_does_not_match(self):
        """A cooking recipe query should NOT match any skill trigger pattern."""
        import re
        from src.memory.seed_skills import SEED_SKILLS
        query = "how to make baklava"
        for skill in SEED_SKILLS:
            pattern = skill["trigger_pattern"]
            self.assertIsNone(
                re.search(pattern, query, re.IGNORECASE),
                f"'{query}' should not match skill '{skill['name']}' pattern '{pattern}'",
            )

    def test_04_record_skill_outcome(self):
        """record_skill_outcome should increment success/failure counts."""
        from src.memory.skills import record_skill_outcome, list_skills
        # Record a success for currency skill
        run_async(record_skill_outcome("currency_api_routing", success=True))
        run_async(record_skill_outcome("currency_api_routing", success=True))
        run_async(record_skill_outcome("currency_api_routing", success=False))

        skills = run_async(list_skills())
        currency = next(s for s in skills if s["name"] == "currency_api_routing")
        self.assertEqual(currency["success_count"], 2)
        self.assertEqual(currency["failure_count"], 1)


if __name__ == "__main__":
    unittest.main()
