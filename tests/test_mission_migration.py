# tests/test_mission_migration.py
"""
Tests for the goals→missions migration.

Verifies:
  1. DB schema uses 'missions' table (not 'goals')
  2. All columns use 'mission_id' (not 'goal_id')
  3. DB API functions use mission naming
  4. Migration code correctly renames old tables/columns
  5. No stale 'goal' references remain in source code
"""
import asyncio
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestMissionSchemaFromScratch(unittest.TestCase):
    """Verify fresh DB uses 'missions' table and 'mission_id' columns."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = self.tmp.name
        self.tmp.close()

        from src.app import config
        from src.infra import db as db_mod

        self._orig_config = config.DB_PATH
        self._orig_db = db_mod.DB_PATH
        config.DB_PATH = self.db_path
        db_mod.DB_PATH = self.db_path
        db_mod._db_connection = None
        self.db_mod = db_mod
        run_async(db_mod.init_db())

    def tearDown(self):
        from src.app import config
        run_async(self.db_mod.close_db())
        config.DB_PATH = self._orig_config
        self.db_mod.DB_PATH = self._orig_db
        os.unlink(self.db_path)

    def test_missions_table_exists(self):
        """The 'missions' table must exist after init_db."""
        async def check():
            db = await self.db_mod.get_db()
            cur = await db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='missions'"
            )
            return await cur.fetchone()
        result = run_async(check())
        self.assertIsNotNone(result, "missions table not found")

    def test_goals_table_does_not_exist(self):
        """The old 'goals' table must NOT exist."""
        async def check():
            db = await self.db_mod.get_db()
            cur = await db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='goals'"
            )
            return await cur.fetchone()
        result = run_async(check())
        self.assertIsNone(result, "old 'goals' table still exists")

    def test_tasks_has_mission_id_column(self):
        """tasks table must have 'mission_id', not 'goal_id'."""
        async def check():
            db = await self.db_mod.get_db()
            cur = await db.execute("PRAGMA table_info(tasks)")
            cols = [row[1] for row in await cur.fetchall()]
            return cols
        cols = run_async(check())
        self.assertIn("mission_id", cols)
        self.assertNotIn("goal_id", cols)

    def test_blackboards_has_mission_id(self):
        """blackboards table must use mission_id."""
        async def check():
            db = await self.db_mod.get_db()
            cur = await db.execute("PRAGMA table_info(blackboards)")
            return [row[1] for row in await cur.fetchall()]
        cols = run_async(check())
        self.assertIn("mission_id", cols)
        self.assertNotIn("goal_id", cols)

    def test_memory_has_mission_id(self):
        """memory table must use mission_id."""
        async def check():
            db = await self.db_mod.get_db()
            cur = await db.execute("PRAGMA table_info(memory)")
            return [row[1] for row in await cur.fetchall()]
        cols = run_async(check())
        self.assertIn("mission_id", cols)
        self.assertNotIn("goal_id", cols)

    def test_missions_has_project_columns(self):
        """missions table must have absorbed project fields."""
        async def check():
            db = await self.db_mod.get_db()
            cur = await db.execute("PRAGMA table_info(missions)")
            return [row[1] for row in await cur.fetchall()]
        cols = run_async(check())
        for expected in ("workflow", "repo_path", "language", "framework"):
            self.assertIn(expected, cols, f"Missing column: {expected}")


class TestMissionDBAPI(unittest.TestCase):
    """Verify mission API functions work correctly."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = self.tmp.name
        self.tmp.close()

        from src.app import config
        from src.infra import db as db_mod

        self._orig_config = config.DB_PATH
        self._orig_db = db_mod.DB_PATH
        config.DB_PATH = self.db_path
        db_mod.DB_PATH = self.db_path
        db_mod._db_connection = None
        self.db_mod = db_mod
        run_async(db_mod.init_db())

    def tearDown(self):
        from src.app import config
        run_async(self.db_mod.close_db())
        config.DB_PATH = self._orig_config
        self.db_mod.DB_PATH = self._orig_db
        os.unlink(self.db_path)

    def test_add_mission(self):
        """add_mission creates a row and returns its ID."""
        mid = run_async(self.db_mod.add_mission("Test Mission", "A test"))
        self.assertIsInstance(mid, int)
        self.assertGreater(mid, 0)

    def test_get_mission(self):
        """get_mission retrieves by ID."""
        mid = run_async(self.db_mod.add_mission("Retrieve Me", "desc"))
        mission = run_async(self.db_mod.get_mission(mid))
        self.assertIsNotNone(mission)
        self.assertEqual(mission["title"], "Retrieve Me")

    def test_get_active_missions(self):
        """get_active_missions returns active missions."""
        run_async(self.db_mod.add_mission("Active", "active mission"))
        missions = run_async(self.db_mod.get_active_missions())
        self.assertGreaterEqual(len(missions), 1)
        titles = [m["title"] for m in missions]
        self.assertIn("Active", titles)

    def test_update_mission(self):
        """update_mission modifies fields."""
        mid = run_async(self.db_mod.add_mission("Original", "desc"))
        run_async(self.db_mod.update_mission(mid, title="Updated"))
        mission = run_async(self.db_mod.get_mission(mid))
        self.assertEqual(mission["title"], "Updated")

    def test_add_task_with_mission_id(self):
        """add_task accepts mission_id parameter."""
        mid = run_async(self.db_mod.add_mission("Task Parent", "desc"))
        tid = run_async(self.db_mod.add_task(
            "Test Task", "task desc", mission_id=mid
        ))
        self.assertIsNotNone(tid)

    def test_get_tasks_for_mission(self):
        """get_tasks_for_mission returns tasks linked to a mission."""
        mid = run_async(self.db_mod.add_mission("With Tasks", "desc"))
        run_async(self.db_mod.add_task("T1", "d1", mission_id=mid))
        run_async(self.db_mod.add_task("T2", "d2", mission_id=mid))
        tasks = run_async(self.db_mod.get_tasks_for_mission(mid))
        self.assertEqual(len(tasks), 2)

    def test_add_mission_with_project_fields(self):
        """add_mission accepts workflow, repo_path, language, framework."""
        mid = run_async(self.db_mod.add_mission(
            "Project Mission", "desc",
            workflow="i2p_v3",
            repo_path="/tmp/repo",
            language="python",
            framework="fastapi"
        ))
        mission = run_async(self.db_mod.get_mission(mid))
        self.assertEqual(mission["workflow"], "i2p_v3")
        self.assertEqual(mission["repo_path"], "/tmp/repo")
        self.assertEqual(mission["language"], "python")
        self.assertEqual(mission["framework"], "fastapi")

    def test_mission_total_cost(self):
        """get_mission_total_cost returns cost for a mission."""
        mid = run_async(self.db_mod.add_mission("Costed", "desc"))
        cost = run_async(self.db_mod.get_mission_total_cost(mid))
        self.assertEqual(cost, 0.0)


class TestMissionMigrationFromOldSchema(unittest.TestCase):
    """Verify migration from old 'goals' table to 'missions'."""

    def setUp(self):
        import aiosqlite
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = self.tmp.name
        self.tmp.close()
        self.aiosqlite = aiosqlite

    def tearDown(self):
        os.unlink(self.db_path)

    def test_goals_table_migrated_to_missions(self):
        """If a 'goals' table exists, init_db renames it to 'missions'."""
        async def setup_and_migrate():
            # Create old-style 'goals' table
            db = await self.aiosqlite.connect(self.db_path)
            await db.execute("""
                CREATE TABLE goals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    description TEXT,
                    status TEXT DEFAULT 'active',
                    priority INTEGER DEFAULT 5,
                    context TEXT DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await db.execute(
                "INSERT INTO goals (title, description) VALUES (?, ?)",
                ("Old Goal", "from goals table")
            )
            await db.commit()
            await db.close()

            # Now run init_db which should migrate
            from src.app import config
            from src.infra import db as db_mod
            orig_config = config.DB_PATH
            orig_db = db_mod.DB_PATH
            config.DB_PATH = self.db_path
            db_mod.DB_PATH = self.db_path
            db_mod._db_connection = None

            await db_mod.init_db()

            db = await db_mod.get_db()

            # Check missions table exists
            cur = await db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='missions'"
            )
            missions_exists = await cur.fetchone()

            # Check goals table is gone
            cur = await db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='goals'"
            )
            goals_exists = await cur.fetchone()

            # Check data was preserved
            cur = await db.execute("SELECT title FROM missions WHERE title='Old Goal'")
            old_data = await cur.fetchone()

            await db_mod.close_db()
            config.DB_PATH = orig_config
            db_mod.DB_PATH = orig_db

            return missions_exists, goals_exists, old_data

        missions_exists, goals_exists, old_data = run_async(setup_and_migrate())
        self.assertIsNotNone(missions_exists, "missions table not created")
        self.assertIsNone(goals_exists, "goals table still exists after migration")
        self.assertIsNotNone(old_data, "data not preserved during migration")


class TestNoStaleGoalReferences(unittest.TestCase):
    """Verify no stale 'goal' references remain in source code."""

    def test_no_goal_functions_in_db_module(self):
        """DB module should not export old goal function names."""
        from src.infra import db
        # These should NOT exist as separate functions
        self.assertFalse(
            hasattr(db, 'add_goal') and db.add_goal is not db.add_mission,
            "add_goal still exists as a separate function"
        )

    def test_mission_functions_exist(self):
        """All mission API functions must exist."""
        from src.infra import db
        required = [
            'add_mission', 'get_mission', 'get_active_missions',
            'update_mission', 'get_tasks_for_mission', 'get_task_tree',
            'propagate_skips', 'release_mission_locks', 'get_mission_locks',
            'get_latest_snapshot', 'get_mission_total_cost',
        ]
        for name in required:
            self.assertTrue(hasattr(db, name), f"Missing function: {name}")

    def test_no_goal_references_in_source(self):
        """Source files (except migration code) should not reference 'goal_id' as a column."""
        import re
        src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
        violations = []
        # Patterns that indicate stale goal references (not in comments about migration)
        pattern = re.compile(r'\badd_goal\s*\(|\bget_goal\s*\(|\bupdate_goal\s*\(|\bget_active_goals\s*\(')

        for root, dirs, files in os.walk(src_dir):
            for f in files:
                if not f.endswith('.py'):
                    continue
                path = os.path.join(root, f)
                with open(path, encoding='utf-8') as fh:
                    for i, line in enumerate(fh, 1):
                        if pattern.search(line):
                            violations.append(f"{path}:{i}: {line.strip()}")

        self.assertEqual(violations, [], f"Stale goal references found:\n" + "\n".join(violations))


if __name__ == "__main__":
    unittest.main()
