"""
Tests for seed_skills: seeding, idempotency, and query matching.
"""
import asyncio
import os
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

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
        # Patch out ChromaDB dedup and embedding so tests run in isolation
        with patch("src.memory.skills._find_duplicate_skill", new_callable=AsyncMock, return_value=None), \
             patch("src.memory.skills._embed_skill", new_callable=AsyncMock):
            added = run_async(seed_skills())
        self.assertEqual(added, len(SEED_SKILLS))
        self.assertEqual(added, 24)

    def test_02_seed_is_idempotent(self):
        """Second seed should add 0 new skills."""
        from src.memory.seed_skills import seed_skills
        with patch("src.memory.skills._find_duplicate_skill", new_callable=AsyncMock, return_value=None), \
             patch("src.memory.skills._embed_skill", new_callable=AsyncMock):
            added = run_async(seed_skills())
        self.assertEqual(added, 0)

    def test_03_list_skills_returns_all(self):
        """list_skills should return all 24 seeded skills."""
        from src.memory.skills import list_skills
        skills = run_async(list_skills())
        self.assertEqual(len(skills), 24)

    def test_seed_skills_have_description(self):
        """All seed skills should have a non-empty description field."""
        from src.memory.seed_skills import SEED_SKILLS
        for skill in SEED_SKILLS:
            self.assertTrue(
                skill.get("description"),
                f"Skill '{skill['name']}' missing description"
            )

    def test_seed_skills_have_strategy_summary(self):
        """All seed skills should have a non-empty strategy_summary field."""
        from src.memory.seed_skills import SEED_SKILLS
        for skill in SEED_SKILLS:
            self.assertTrue(
                skill.get("strategy_summary"),
                f"Skill '{skill['name']}' missing strategy_summary"
            )

    def test_seed_skills_have_tools_used_list(self):
        """All seed skills should have tools_used as a list."""
        from src.memory.seed_skills import SEED_SKILLS
        for skill in SEED_SKILLS:
            tools = skill.get("tools_used", [])
            self.assertIsInstance(
                tools, list,
                f"Skill '{skill['name']}' tools_used is not a list"
            )

    def test_seed_skills_no_old_fields(self):
        """Seed skills must not contain old-schema fields."""
        from src.memory.seed_skills import SEED_SKILLS
        old_fields = {"trigger_pattern", "tool_sequence", "examples",
                      "success_count", "failure_count"}
        for skill in SEED_SKILLS:
            for field in old_fields:
                self.assertNotIn(
                    field, skill,
                    f"Skill '{skill['name']}' still has old field '{field}'"
                )

    def test_currency_skill_exists(self):
        """currency_lookup skill should be present."""
        from src.memory.seed_skills import SEED_SKILLS
        names = {s["name"] for s in SEED_SKILLS}
        self.assertIn("currency_lookup", names)

    def test_weather_skill_exists(self):
        """weather_check skill should be present."""
        from src.memory.seed_skills import SEED_SKILLS
        names = {s["name"] for s in SEED_SKILLS}
        self.assertIn("weather_check", names)

    def test_github_skill_exists(self):
        """github_code_research skill should be present."""
        from src.memory.seed_skills import SEED_SKILLS
        names = {s["name"] for s in SEED_SKILLS}
        self.assertIn("github_code_research", names)

    def test_shopping_skill_exists(self):
        """turkish_product_shopping skill should be present."""
        from src.memory.seed_skills import SEED_SKILLS
        names = {s["name"] for s in SEED_SKILLS}
        self.assertIn("turkish_product_shopping", names)

    def test_04_record_skill_outcome(self):
        """record_skill_outcome should increment injection counts on the correct columns."""
        from src.memory.skills import record_skill_outcome, list_skills
        # success=True -> record_injection_success -> increments injection_success only
        # success=False -> record_injection -> increments injection_count only
        run_async(record_skill_outcome("currency_lookup", success=True))
        run_async(record_skill_outcome("currency_lookup", success=True))
        run_async(record_skill_outcome("currency_lookup", success=False))

        skills = run_async(list_skills())
        currency = next(s for s in skills if s["name"] == "currency_lookup")
        # 1 failure -> injection_count=1; 2 successes -> injection_success=2
        self.assertEqual(currency["injection_count"], 1)
        self.assertEqual(currency["injection_success"], 2)


if __name__ == "__main__":
    unittest.main()
