"""Tests for the skills v2 schema, migration, and DB helpers."""
import asyncio
import json
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


class TestSkillsV2Schema(unittest.TestCase):
    """Tests for the new skills schema columns and constraints."""

    def setUp(self):
        os.environ["DB_PATH"] = ":memory:"
        import src.infra.db as _db_mod
        _db_mod._db_connection = None

    def test_skills_table_has_new_columns(self):
        """skills table must have name, description, skill_type, strategies, injection_count, injection_success."""
        from src.infra.db import init_db, get_db

        async def _test():
            await init_db()
            db = await get_db()
            cursor = await db.execute("PRAGMA table_info(skills)")
            columns = {row[1] for row in await cursor.fetchall()}
            required = {"name", "description", "skill_type", "strategies",
                        "injection_count", "injection_success", "created_at", "updated_at"}
            for col in required:
                self.assertIn(col, columns, f"Missing column: {col}")

        run_async(_test())

    def test_skills_table_does_not_have_old_columns(self):
        """skills table must NOT have old columns: trigger_pattern, tool_sequence, success_count, failure_count."""
        from src.infra.db import init_db, get_db

        async def _test():
            await init_db()
            db = await get_db()
            cursor = await db.execute("PRAGMA table_info(skills)")
            columns = {row[1] for row in await cursor.fetchall()}
            forbidden = {"trigger_pattern", "tool_sequence", "success_count", "failure_count"}
            for col in forbidden:
                self.assertNotIn(col, columns, f"Old column still present: {col}")

        run_async(_test())

    def test_insert_skill_with_json_strategies(self):
        """Can insert a skill with a JSON strategies column."""
        from src.infra.db import init_db, get_db

        async def _test():
            await init_db()
            db = await get_db()
            strategies = json.dumps([
                {
                    "summary": "Use web_search then summarize",
                    "tool_template": "web_search({query})",
                    "tools_used": ["web_search"],
                }
            ])
            await db.execute(
                "INSERT INTO skills (name, description, skill_type, strategies) VALUES (?, ?, ?, ?)",
                ("test_skill", "A test skill", "seed", strategies),
            )
            await db.commit()

            cursor = await db.execute("SELECT strategies FROM skills WHERE name = 'test_skill'")
            row = await cursor.fetchone()
            self.assertIsNotNone(row)
            parsed = json.loads(row[0])
            self.assertEqual(len(parsed), 1)
            self.assertEqual(parsed[0]["tools_used"], ["web_search"])

        run_async(_test())


class TestSkillsV2Helpers(unittest.TestCase):
    """Tests for the DB helper functions: upsert_skill, get_all_skills, etc."""

    def setUp(self):
        os.environ["DB_PATH"] = ":memory:"
        import src.infra.db as _db_mod
        _db_mod._db_connection = None

    def test_upsert_skill_insert(self):
        """upsert_skill inserts a new skill and returns a row id."""
        from src.infra.db import init_db, upsert_skill

        async def _test():
            await init_db()
            row_id = await upsert_skill(
                name="my_skill",
                description="Does something useful",
                skill_type="seed",
                strategies=[{"summary": "step1", "tool_template": "", "tools_used": []}],
            )
            self.assertIsNotNone(row_id)
            self.assertGreater(row_id, 0)

        run_async(_test())

    def test_upsert_skill_update(self):
        """upsert_skill updates description when name already exists."""
        from src.infra.db import init_db, upsert_skill, get_skill_by_name

        async def _test():
            await init_db()
            await upsert_skill("dup_skill", "Original desc", "auto", [])
            await upsert_skill("dup_skill", "Updated desc", "auto", [{"summary": "new"}])
            skill = await get_skill_by_name("dup_skill")
            self.assertIsNotNone(skill)
            self.assertEqual(skill["description"], "Updated desc")
            parsed = json.loads(skill["strategies"])
            self.assertEqual(parsed[0]["summary"], "new")

        run_async(_test())

    def test_get_all_skills(self):
        """get_all_skills returns all inserted skills."""
        from src.infra.db import init_db, upsert_skill, get_all_skills

        async def _test():
            await init_db()
            await upsert_skill("skill_a", "Skill A", "seed", [])
            await upsert_skill("skill_b", "Skill B", "auto", [])
            skills = await get_all_skills()
            names = {s["name"] for s in skills}
            self.assertIn("skill_a", names)
            self.assertIn("skill_b", names)

        run_async(_test())

    def test_get_skill_by_name_found(self):
        """get_skill_by_name returns the skill dict when found."""
        from src.infra.db import init_db, upsert_skill, get_skill_by_name

        async def _test():
            await init_db()
            await upsert_skill("named_skill", "Desc", "seed", [])
            result = await get_skill_by_name("named_skill")
            self.assertIsNotNone(result)
            self.assertEqual(result["name"], "named_skill")
            self.assertEqual(result["description"], "Desc")
            self.assertEqual(result["skill_type"], "seed")

        run_async(_test())

    def test_get_skill_by_name_not_found(self):
        """get_skill_by_name returns None for missing skills."""
        from src.infra.db import init_db, get_skill_by_name

        async def _test():
            await init_db()
            result = await get_skill_by_name("nonexistent_skill")
            self.assertIsNone(result)

        run_async(_test())

    def test_injection_count_tracking(self):
        """increment_skill_injection increments injection_count."""
        from src.infra.db import init_db, upsert_skill, increment_skill_injection, get_skill_by_name

        async def _test():
            await init_db()
            await upsert_skill("tracked_skill", "Desc", "auto", [])
            await increment_skill_injection("tracked_skill")
            await increment_skill_injection("tracked_skill")
            skill = await get_skill_by_name("tracked_skill")
            self.assertEqual(skill["injection_count"], 2)
            self.assertEqual(skill["injection_success"], 0)

        run_async(_test())

    def test_injection_success_tracking(self):
        """increment_skill_success increments injection_success."""
        from src.infra.db import init_db, upsert_skill, increment_skill_injection, increment_skill_success, get_skill_by_name

        async def _test():
            await init_db()
            await upsert_skill("success_skill", "Desc", "auto", [])
            await increment_skill_injection("success_skill")
            await increment_skill_injection("success_skill")
            await increment_skill_injection("success_skill")
            await increment_skill_success("success_skill")
            await increment_skill_success("success_skill")
            skill = await get_skill_by_name("success_skill")
            self.assertEqual(skill["injection_count"], 3)
            self.assertEqual(skill["injection_success"], 2)

        run_async(_test())

    def test_injection_count_defaults_to_zero(self):
        """Freshly inserted skills have injection_count and injection_success of 0."""
        from src.infra.db import init_db, upsert_skill, get_skill_by_name

        async def _test():
            await init_db()
            await upsert_skill("fresh_skill", "Desc", "seed", [])
            skill = await get_skill_by_name("fresh_skill")
            self.assertEqual(skill["injection_count"], 0)
            self.assertEqual(skill["injection_success"], 0)

        run_async(_test())


class TestSkillsMigration(unittest.TestCase):
    """Tests for migration from old skills table to new schema."""

    def setUp(self):
        os.environ["DB_PATH"] = ":memory:"
        import src.infra.db as _db_mod
        _db_mod._db_connection = None

    def test_migration_renames_old_table(self):
        """After migration, skills_old_backup exists only if old skills table existed."""
        from src.infra.db import init_db, get_db

        async def _test():
            db = await get_db()
            # Pre-create old skills table to simulate migration scenario
            await db.execute("""
                CREATE TABLE IF NOT EXISTS skills (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT NOT NULL,
                    trigger_pattern TEXT DEFAULT '',
                    tool_sequence TEXT DEFAULT '',
                    examples TEXT DEFAULT '',
                    success_count INTEGER DEFAULT 0,
                    failure_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await db.execute(
                "INSERT INTO skills (name, description, tool_sequence) VALUES (?, ?, ?)",
                ("seed:weather", "Get weather data", "web_search")
            )
            await db.execute(
                "INSERT INTO skills (name, description, tool_sequence) VALUES (?, ?, ?)",
                ("auto:some_task", "Auto-captured skill", "some_tool")
            )
            await db.commit()

            await init_db()

            # Old table should be backed up
            cursor = await db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='skills_old_backup'"
            )
            backup_exists = await cursor.fetchone() is not None
            self.assertTrue(backup_exists, "skills_old_backup should exist after migration")

            # New skills table should have new columns
            cursor2 = await db.execute("PRAGMA table_info(skills)")
            columns = {row[1] for row in await cursor2.fetchall()}
            self.assertIn("strategies", columns)
            self.assertIn("injection_count", columns)
            self.assertNotIn("trigger_pattern", columns)

        run_async(_test())

    def test_seed_skills_migrated(self):
        """Seed skills (not 'auto:' prefix) are migrated to new table."""
        from src.infra.db import init_db, get_db, get_skill_by_name

        async def _test():
            db = await get_db()
            await db.execute("""
                CREATE TABLE IF NOT EXISTS skills (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT NOT NULL,
                    trigger_pattern TEXT DEFAULT '',
                    tool_sequence TEXT DEFAULT '',
                    examples TEXT DEFAULT '',
                    success_count INTEGER DEFAULT 0,
                    failure_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await db.execute(
                "INSERT INTO skills (name, description, tool_sequence) VALUES (?, ?, ?)",
                ("seed:weather", "Get weather data", "web_search")
            )
            await db.commit()

            await init_db()

            skill = await get_skill_by_name("seed:weather")
            self.assertIsNotNone(skill, "Seed skill should be migrated to new table")
            self.assertEqual(skill["name"], "seed:weather")
            strategies = json.loads(skill["strategies"])
            self.assertEqual(len(strategies), 1)
            self.assertEqual(strategies[0]["summary"], "web_search")

        run_async(_test())

    def test_auto_skills_not_migrated(self):
        """Auto-captured skills ('auto:' prefix) are NOT migrated."""
        from src.infra.db import init_db, get_db, get_all_skills

        async def _test():
            db = await get_db()
            await db.execute("""
                CREATE TABLE IF NOT EXISTS skills (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT NOT NULL,
                    trigger_pattern TEXT DEFAULT '',
                    tool_sequence TEXT DEFAULT '',
                    examples TEXT DEFAULT '',
                    success_count INTEGER DEFAULT 0,
                    failure_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await db.execute(
                "INSERT INTO skills (name, description) VALUES (?, ?)",
                ("auto:some_task", "Auto-captured skill")
            )
            await db.commit()

            await init_db()

            skills = await get_all_skills()
            names = {s["name"] for s in skills}
            self.assertNotIn("auto:some_task", names, "Auto skills should NOT be migrated")

        run_async(_test())


class TestSkillsV2Core(unittest.TestCase):
    """Tests for the rewritten skills.py core module."""

    def setUp(self):
        os.environ["DB_PATH"] = ":memory:"
        import src.infra.db as _db_mod
        _db_mod._db_connection = None

    def test_add_skill_creates_new(self):
        """add_skill creates a new skill when no duplicate exists."""
        from src.infra.db import init_db, get_skill_by_name
        from unittest.mock import AsyncMock, patch

        async def _test():
            await init_db()
            with patch("src.memory.skills._find_duplicate_skill", new_callable=AsyncMock, return_value=None), \
                 patch("src.memory.skills._embed_skill", new_callable=AsyncMock):
                from src.memory.skills import add_skill
                result = await add_skill(
                    name="test_new_skill",
                    description="A brand new skill",
                    strategy_summary="Do X then Y",
                    tools_used=["web_search"],
                )
                self.assertEqual(result, "test_new_skill")

            skill = await get_skill_by_name("test_new_skill")
            self.assertIsNotNone(skill)
            strategies = json.loads(skill["strategies"])
            self.assertEqual(len(strategies), 1)
            self.assertEqual(strategies[0]["summary"], "Do X then Y")
            self.assertEqual(strategies[0]["tools_used"], ["web_search"])

        run_async(_test())

    def test_add_skill_merges_duplicate(self):
        """add_skill merges strategy into existing skill when duplicate found."""
        from src.infra.db import init_db, upsert_skill, get_skill_by_name
        from unittest.mock import AsyncMock, patch

        async def _test():
            await init_db()
            # Pre-create existing skill
            await upsert_skill(
                "existing_skill", "Does something", "auto",
                [{"summary": "Original approach", "tools_used": ["tool_a"],
                  "tool_template": "", "injection_count": 0, "injection_success": 0}],
            )
            existing = await get_skill_by_name("existing_skill")

            with patch("src.memory.skills._find_duplicate_skill", new_callable=AsyncMock, return_value=existing):
                from src.memory.skills import add_skill
                result = await add_skill(
                    name="new_name_ignored",
                    description="Does something similar",
                    strategy_summary="New approach",
                    tools_used=["tool_b"],
                )
                self.assertEqual(result, "existing_skill")

            skill = await get_skill_by_name("existing_skill")
            strategies = json.loads(skill["strategies"])
            self.assertEqual(len(strategies), 2)
            self.assertEqual(strategies[1]["summary"], "New approach")

        run_async(_test())

    def test_format_skill_verbose(self):
        """format_skill_verbose includes key markdown elements."""
        from src.memory.skills import format_skill_verbose

        skill = {
            "name": "weather_lookup",
            "description": "Look up weather for a city",
            "strategies": json.dumps([{
                "summary": "Use weather API",
                "tool_template": "web_search({city} weather)",
                "tools_used": ["web_search"],
                "injection_count": 10,
                "injection_success": 8,
            }]),
            "injection_count": 10,
            "injection_success": 8,
        }
        output = format_skill_verbose(skill)
        self.assertIn("## Skill: weather_lookup", output)
        self.assertIn("**Situation:**", output)
        self.assertIn("**Strategy:** Use weather API", output)
        self.assertIn("**Track record:**", output)
        self.assertIn("80%", output)

    def test_format_skill_compact(self):
        """format_skill_compact returns a single line with expected format."""
        from src.memory.skills import format_skill_compact

        skill = {
            "name": "price_check",
            "description": "Check product prices",
            "strategies": json.dumps([{
                "summary": "Scrape price sites",
                "tools_used": ["web_search", "scraper"],
                "injection_count": 0,
                "injection_success": 0,
            }]),
            "injection_count": 5,
            "injection_success": 4,
        }
        output = format_skill_compact(skill)
        self.assertTrue(output.startswith("- price_check:"))
        self.assertIn("web_search", output)
        self.assertIn("scraper", output)
        self.assertIn("80%", output)

    def test_select_injection_depth_trusted(self):
        """High-confidence skill gets verbose depth with 1 skill."""
        from src.memory.skills import select_injection_depth

        skills = [{
            "name": "trusted_skill",
            "injection_count": 10,
            "injection_success": 9,
            "strategies": "[]",
        }]
        depth, selected = select_injection_depth(skills, context_budget=4096)
        self.assertEqual(depth, "verbose")
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["name"], "trusted_skill")

    def test_select_injection_depth_uncertain(self):
        """Low-confidence skill gets compact depth with multiple skills."""
        from src.memory.skills import select_injection_depth

        skills = [
            {"name": "s1", "injection_count": 2, "injection_success": 1, "strategies": "[]"},
            {"name": "s2", "injection_count": 3, "injection_success": 1, "strategies": "[]"},
            {"name": "s3", "injection_count": 1, "injection_success": 0, "strategies": "[]"},
            {"name": "s4", "injection_count": 0, "injection_success": 0, "strategies": "[]"},
        ]
        depth, selected = select_injection_depth(skills, context_budget=8192)
        self.assertEqual(depth, "compact")
        self.assertEqual(len(selected), 3)  # max 3 for budget >= 4096

    def test_select_injection_depth_small_context(self):
        """Small context budget limits to 1 skill in compact mode."""
        from src.memory.skills import select_injection_depth

        skills = [
            {"name": "s1", "injection_count": 2, "injection_success": 1, "strategies": "[]"},
            {"name": "s2", "injection_count": 3, "injection_success": 1, "strategies": "[]"},
        ]
        depth, selected = select_injection_depth(skills, context_budget=1024)
        self.assertEqual(depth, "compact")
        self.assertEqual(len(selected), 1)

    def test_get_tools_to_inject_high_confidence(self):
        """Tools returned for high-confidence skills."""
        from src.memory.skills import get_tools_to_inject

        skills = [{
            "name": "confident_skill",
            "injection_count": 10,
            "injection_success": 8,
            "strategies": json.dumps([{
                "summary": "Do stuff",
                "tools_used": ["web_search", "calculator"],
                "injection_count": 10,
                "injection_success": 8,
            }]),
        }]
        tools = get_tools_to_inject(skills)
        self.assertIn("web_search", tools)
        self.assertIn("calculator", tools)

    def test_get_tools_to_inject_low_confidence(self):
        """No tools returned for low-confidence skills."""
        from src.memory.skills import get_tools_to_inject

        skills = [{
            "name": "new_skill",
            "injection_count": 2,
            "injection_success": 1,
            "strategies": json.dumps([{
                "summary": "Do stuff",
                "tools_used": ["web_search"],
                "injection_count": 0,
                "injection_success": 0,
            }]),
        }]
        tools = get_tools_to_inject(skills)
        self.assertEqual(tools, [])


class TestSkillsV2Helpers(unittest.TestCase):
    """Tests for pure helper functions."""

    def test_injection_success_rate_no_injections(self):
        """Zero injections returns 0.5."""
        from src.memory.skills import _injection_success_rate
        self.assertEqual(_injection_success_rate({"injection_count": 0, "injection_success": 0}), 0.5)

    def test_injection_success_rate_capped(self):
        """Below MIN_INJECTIONS_FOR_CONFIDENCE, rate capped at 0.5."""
        from src.memory.skills import _injection_success_rate
        # 3 out of 4 = 0.75, but capped to 0.5
        rate = _injection_success_rate({"injection_count": 4, "injection_success": 3})
        self.assertEqual(rate, 0.5)

    def test_injection_success_rate_uncapped(self):
        """At or above MIN_INJECTIONS_FOR_CONFIDENCE, real rate returned."""
        from src.memory.skills import _injection_success_rate
        rate = _injection_success_rate({"injection_count": 10, "injection_success": 8})
        self.assertAlmostEqual(rate, 0.8)

    def test_best_strategy_proven(self):
        """Proven strategy with best rate is returned."""
        from src.memory.skills import _best_strategy
        skill = {
            "strategies": json.dumps([
                {"summary": "A", "injection_count": 10, "injection_success": 5},
                {"summary": "B", "injection_count": 10, "injection_success": 9},
            ])
        }
        best = _best_strategy(skill)
        self.assertEqual(best["summary"], "B")

    def test_best_strategy_unproven_newest(self):
        """Unproven strategies: newest (last) is returned."""
        from src.memory.skills import _best_strategy
        skill = {
            "strategies": json.dumps([
                {"summary": "Old", "injection_count": 0, "injection_success": 0},
                {"summary": "New", "injection_count": 2, "injection_success": 1},
            ])
        }
        best = _best_strategy(skill)
        self.assertEqual(best["summary"], "New")

    def test_prune_strategies_keeps_unproven(self):
        """Pruning never drops unproven strategies."""
        from src.memory.skills import _prune_strategies

        strategies = [
            {"summary": f"proven_{i}", "injection_count": 10, "injection_success": i}
            for i in range(4)
        ] + [
            {"summary": "unproven_a", "injection_count": 0, "injection_success": 0},
            {"summary": "unproven_b", "injection_count": 3, "injection_success": 2},
        ]
        pruned = _prune_strategies(strategies)
        self.assertLessEqual(len(pruned), 5)
        summaries = [s["summary"] for s in pruned]
        self.assertIn("unproven_a", summaries)
        self.assertIn("unproven_b", summaries)


class TestSeedSkillsNewFormat(unittest.TestCase):
    """Tests that SEED_SKILLS uses the new execution recipe format."""

    def test_seed_skills_have_new_format(self):
        from src.memory.seed_skills import SEED_SKILLS
        for skill in SEED_SKILLS:
            self.assertIn("name", skill)
            self.assertIn("description", skill)
            self.assertIn("strategy_summary", skill,
                          f"{skill['name']} missing strategy_summary")
            self.assertNotIn("trigger_pattern", skill,
                             f"{skill['name']} still has trigger_pattern")
            self.assertNotIn("tool_sequence", skill,
                             f"{skill['name']} still has tool_sequence")
            self.assertNotIn("examples", skill,
                             f"{skill['name']} still has examples")

    def test_seed_skills_count(self):
        from src.memory.seed_skills import SEED_SKILLS
        self.assertEqual(len(SEED_SKILLS), 24, "Expected 24 seed skills")

    def test_seed_skills_names_are_clean(self):
        from src.memory.seed_skills import SEED_SKILLS
        old_names = {
            "currency_api_routing", "weather_api_routing", "time_api_routing",
            "wikipedia_routing", "play_store_routing", "github_routing",
            "shopping_turkish_sources", "shopping_review_sources",
            "sports_web_search", "pdf_processing", "coding_error_search",
            "translation_routing", "news_routing", "network_tools_routing",
            "i2p_competitor_research", "pharmacy_on_duty", "earthquake_data",
            "fuel_price_routing", "gold_price_routing", "map_directions_routing",
            "prayer_times_routing", "travel_ticket_routing",
            "epey_spec_comparison", "turkish_holidays_routing",
        }
        names = {s["name"] for s in SEED_SKILLS}
        overlap = names & old_names
        self.assertEqual(overlap, set(), f"Old-format names still present: {overlap}")

    def test_seed_skills_tools_used_is_list(self):
        from src.memory.seed_skills import SEED_SKILLS
        for skill in SEED_SKILLS:
            tools = skill.get("tools_used", [])
            self.assertIsInstance(tools, list,
                                  f"{skill['name']} tools_used is not a list")


if __name__ == "__main__":
    unittest.main()
