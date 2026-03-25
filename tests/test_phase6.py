# tests/test_phase6.py
"""
Tests for Phase 6: Workspace & Isolation

  6.1 Per-mission workspace directories
  6.2 File locking mechanism
  6.3 Branch-per-mission git workflow
  6.4 Workspace snapshots (file hashing)
  6.5 Multi-project support
  6.6 Orchestrator integration
"""
import asyncio
import json
import os
import re
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import aiosqlite
    HAS_AIOSQLITE = True
except ImportError:
    HAS_AIOSQLITE = False


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _patch_db_path(db_mod, db_path):
    import config
    config.DB_PATH = db_path
    db_mod.DB_PATH = db_path


class _DBTestBase(unittest.TestCase):
    def setUp(self):
        if not HAS_AIOSQLITE:
            self.skipTest("aiosqlite not installed")
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = self.tmp.name
        self.tmp.close()

        import config
        import db as db_mod
        self._orig_config_path = config.DB_PATH
        self._orig_db_path = db_mod.DB_PATH
        self.db_mod = db_mod

        _patch_db_path(db_mod, self.db_path)
        db_mod._db_connection = None
        run_async(db_mod.init_db())

    def tearDown(self):
        run_async(self.db_mod.close_db())
        import config
        config.DB_PATH = self._orig_config_path
        self.db_mod.DB_PATH = self._orig_db_path
        for suffix in ("", "-wal", "-shm"):
            try:
                os.unlink(self.db_path + suffix)
            except OSError:
                pass


# ─── 6.1 Per-Mission Workspace Directories ──────────────────────────────────

class TestMissionWorkspace(unittest.TestCase):
    """Test per-mission workspace directory creation and management."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        # Patch WORKSPACE_DIR
        import tools.workspace as ws_mod
        self._orig_ws = ws_mod.WORKSPACE_DIR
        ws_mod.WORKSPACE_DIR = self.tmp_dir

    def tearDown(self):
        import tools.workspace as ws_mod
        ws_mod.WORKSPACE_DIR = self._orig_ws
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_get_mission_workspace_creates_dir(self):
        from tools.workspace import get_mission_workspace
        path = get_mission_workspace(42)
        self.assertTrue(os.path.isdir(path))
        self.assertTrue(path.endswith("mission_42"))

    def test_get_mission_workspace_idempotent(self):
        from tools.workspace import get_mission_workspace
        p1 = get_mission_workspace(7)
        p2 = get_mission_workspace(7)
        self.assertEqual(p1, p2)

    def test_get_mission_workspace_relative(self):
        from tools.workspace import get_mission_workspace_relative
        self.assertEqual(get_mission_workspace_relative(5), "mission_5")

    def test_list_mission_workspaces_empty(self):
        from tools.workspace import list_mission_workspaces
        self.assertEqual(list_mission_workspaces(), [])

    def test_list_mission_workspaces_with_missions(self):
        from tools.workspace import get_mission_workspace, list_mission_workspaces
        get_mission_workspace(1)
        get_mission_workspace(3)
        # Create a file in mission_1
        with open(os.path.join(self.tmp_dir, "mission_1", "test.py"), "w") as f:
            f.write("# test")
        result = list_mission_workspaces()
        self.assertEqual(len(result), 2)
        ids = {w["mission_id"] for w in result}
        self.assertEqual(ids, {1, 3})
        # mission_1 should have 1 file
        g1 = [w for w in result if w["mission_id"] == 1][0]
        self.assertEqual(g1["file_count"], 1)

    def test_cleanup_mission_workspace(self):
        from tools.workspace import get_mission_workspace, cleanup_mission_workspace
        get_mission_workspace(10)
        self.assertTrue(os.path.isdir(
            os.path.join(self.tmp_dir, "mission_10")
        ))
        result = cleanup_mission_workspace(10)
        self.assertTrue(result)
        self.assertFalse(os.path.isdir(
            os.path.join(self.tmp_dir, "mission_10")
        ))

    def test_cleanup_nonexistent_workspace(self):
        from tools.workspace import cleanup_mission_workspace
        self.assertFalse(cleanup_mission_workspace(999))


# ─── 6.2 File Locking ────────────────────────────────────────────────────

class TestFileLocking(_DBTestBase):
    """Test advisory file locking mechanism."""

    def test_acquire_lock(self):
        result = run_async(
            self.db_mod.acquire_file_lock("src/main.py", mission_id=1, task_id=10)
        )
        self.assertTrue(result)

    def test_acquire_lock_twice_fails(self):
        run_async(
            self.db_mod.acquire_file_lock("src/main.py", mission_id=1, task_id=10)
        )
        result = run_async(
            self.db_mod.acquire_file_lock("src/main.py", mission_id=2, task_id=20)
        )
        self.assertFalse(result)

    def test_release_lock(self):
        run_async(
            self.db_mod.acquire_file_lock("src/main.py", mission_id=1, task_id=10)
        )
        run_async(self.db_mod.release_file_lock("src/main.py"))
        # Should be acquirable again
        result = run_async(
            self.db_mod.acquire_file_lock("src/main.py", mission_id=2)
        )
        self.assertTrue(result)

    def test_release_task_locks(self):
        run_async(
            self.db_mod.acquire_file_lock("a.py", mission_id=1, task_id=5)
        )
        run_async(
            self.db_mod.acquire_file_lock("b.py", mission_id=1, task_id=5)
        )
        run_async(self.db_mod.release_task_locks(5))
        # Both should be available now
        self.assertTrue(
            run_async(self.db_mod.acquire_file_lock("a.py"))
        )
        self.assertTrue(
            run_async(self.db_mod.acquire_file_lock("b.py"))
        )

    def test_release_mission_locks(self):
        run_async(
            self.db_mod.acquire_file_lock("x.py", mission_id=3, task_id=10)
        )
        run_async(
            self.db_mod.acquire_file_lock("y.py", mission_id=3, task_id=11)
        )
        run_async(self.db_mod.release_mission_locks(3))
        self.assertTrue(
            run_async(self.db_mod.acquire_file_lock("x.py"))
        )

    def test_get_file_lock(self):
        run_async(
            self.db_mod.acquire_file_lock(
                "test.py", mission_id=1, task_id=5, agent_type="coder"
            )
        )
        lock = run_async(self.db_mod.get_file_lock("test.py"))
        self.assertIsNotNone(lock)
        self.assertEqual(lock["filepath"], "test.py")
        self.assertEqual(lock["mission_id"], 1)
        self.assertEqual(lock["task_id"], 5)
        self.assertEqual(lock["agent_type"], "coder")

    def test_get_file_lock_not_found(self):
        lock = run_async(self.db_mod.get_file_lock("nonexistent.py"))
        self.assertIsNone(lock)

    def test_get_mission_locks(self):
        run_async(
            self.db_mod.acquire_file_lock("a.py", mission_id=2, task_id=1)
        )
        run_async(
            self.db_mod.acquire_file_lock("b.py", mission_id=2, task_id=2)
        )
        run_async(
            self.db_mod.acquire_file_lock("c.py", mission_id=9, task_id=3)
        )
        locks = run_async(self.db_mod.get_mission_locks(2))
        self.assertEqual(len(locks), 2)


# ─── 6.3 Branch-per-Mission Git Workflow ─────────────────────────────────────

class TestBranchPerMission(unittest.TestCase):
    """Test branch name generation (pure function)."""

    def test_slugify(self):
        from tools.git_ops import _slugify
        self.assertEqual(_slugify("Hello World!"), "hello-world")
        self.assertEqual(_slugify("Build REST API"), "build-rest-api")

    def test_slugify_truncates(self):
        from tools.git_ops import _slugify
        result = _slugify("A" * 100, max_len=20)
        self.assertEqual(len(result), 20)

    def test_slugify_special_chars(self):
        from tools.git_ops import _slugify
        result = _slugify("Fix #123: bug_fix (urgent)")
        self.assertNotIn("#", result)
        self.assertNotIn("(", result)


# ─── 6.4 Workspace Snapshots ─────────────────────────────────────────────

class TestWorkspaceSnapshots(unittest.TestCase):
    """Test file hashing and snapshot comparison."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_compute_hashes_empty_dir(self):
        from tools.workspace import compute_workspace_hashes
        hashes = compute_workspace_hashes(self.tmp_dir)
        self.assertEqual(hashes, {})

    def test_compute_hashes_with_files(self):
        from tools.workspace import compute_workspace_hashes
        with open(os.path.join(self.tmp_dir, "a.py"), "w") as f:
            f.write("print('hello')")
        with open(os.path.join(self.tmp_dir, "b.txt"), "w") as f:
            f.write("some text")
        hashes = compute_workspace_hashes(self.tmp_dir)
        self.assertIn("a.py", hashes)
        self.assertIn("b.txt", hashes)
        self.assertEqual(len(hashes), 2)

    def test_compute_hashes_skips_pycache(self):
        from tools.workspace import compute_workspace_hashes
        pycache = os.path.join(self.tmp_dir, "__pycache__")
        os.makedirs(pycache)
        with open(os.path.join(pycache, "mod.pyc"), "wb") as f:
            f.write(b"bytecode")
        with open(os.path.join(self.tmp_dir, "main.py"), "w") as f:
            f.write("code")
        hashes = compute_workspace_hashes(self.tmp_dir)
        self.assertIn("main.py", hashes)
        self.assertNotIn(os.path.join("__pycache__", "mod.pyc"), hashes)

    def test_diff_snapshots_added(self):
        from tools.workspace import diff_snapshots
        before = {"a.py": "abc123"}
        after = {"a.py": "abc123", "b.py": "def456"}
        diff = diff_snapshots(before, after)
        self.assertEqual(diff["added"], ["b.py"])
        self.assertEqual(diff["modified"], [])
        self.assertEqual(diff["deleted"], [])

    def test_diff_snapshots_modified(self):
        from tools.workspace import diff_snapshots
        before = {"a.py": "abc123"}
        after = {"a.py": "xyz789"}
        diff = diff_snapshots(before, after)
        self.assertEqual(diff["modified"], ["a.py"])

    def test_diff_snapshots_deleted(self):
        from tools.workspace import diff_snapshots
        before = {"a.py": "abc123", "b.py": "def456"}
        after = {"a.py": "abc123"}
        diff = diff_snapshots(before, after)
        self.assertEqual(diff["deleted"], ["b.py"])

    def test_diff_snapshots_combined(self):
        from tools.workspace import diff_snapshots
        before = {"keep.py": "aaa", "changed.py": "bbb", "gone.py": "ccc"}
        after = {"keep.py": "aaa", "changed.py": "ddd", "new.py": "eee"}
        diff = diff_snapshots(before, after)
        self.assertEqual(diff["added"], ["new.py"])
        self.assertEqual(diff["modified"], ["changed.py"])
        self.assertEqual(diff["deleted"], ["gone.py"])


class TestSnapshotDB(_DBTestBase):
    """Test workspace snapshot DB operations."""

    def test_save_snapshot(self):
        sid = run_async(self.db_mod.save_workspace_snapshot(
            mission_id=1,
            file_hashes={"a.py": "abc123", "b.py": "def456"},
            task_id=10,
            branch_name="mission/1-test",
            commit_sha="abc123def456",
        ))
        self.assertIsNotNone(sid)
        self.assertGreater(sid, 0)

    def test_get_latest_snapshot(self):
        run_async(self.db_mod.save_workspace_snapshot(
            mission_id=1,
            file_hashes={"a.py": "111"},
            task_id=5,
        ))
        run_async(self.db_mod.save_workspace_snapshot(
            mission_id=1,
            file_hashes={"a.py": "222", "b.py": "333"},
            task_id=6,
        ))
        snap = run_async(self.db_mod.get_latest_snapshot(1))
        self.assertIsNotNone(snap)
        self.assertEqual(snap["task_id"], 6)
        self.assertIn("b.py", snap["file_hashes"])

    def test_get_latest_snapshot_no_match(self):
        snap = run_async(self.db_mod.get_latest_snapshot(999))
        self.assertIsNone(snap)

    def test_get_snapshot_by_id(self):
        sid = run_async(self.db_mod.save_workspace_snapshot(
            mission_id=2,
            file_hashes={"x.py": "aaa"},
        ))
        snap = run_async(self.db_mod.get_snapshot(sid))
        self.assertIsNotNone(snap)
        self.assertEqual(snap["mission_id"], 2)
        self.assertIn("x.py", snap["file_hashes"])


# ─── 6.5 Multi-Project Support ───────────────────────────────────────────

class TestMultiProject(unittest.TestCase):
    """Test multi-project configuration."""

    def setUp(self):
        self.tmp_file = tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        )
        json.dump([
            {"name": "MyApp", "path": "/home/user/myapp", "language": "python"},
            {"name": "Frontend", "path": "/home/user/frontend", "language": "typescript"},
        ], self.tmp_file)
        self.tmp_file.close()

        import config
        self._orig_path = config.PROJECTS_CONFIG_PATH
        config.PROJECTS_CONFIG_PATH = self.tmp_file.name

    def tearDown(self):
        import config
        config.PROJECTS_CONFIG_PATH = self._orig_path
        os.unlink(self.tmp_file.name)

    def test_load_projects(self):
        from tools.workspace import load_projects_config
        projects = load_projects_config()
        self.assertEqual(len(projects), 2)
        self.assertEqual(projects[0]["name"], "MyApp")

    def test_get_project_found(self):
        from tools.workspace import get_project
        p = get_project("myapp")
        self.assertIsNotNone(p)
        self.assertEqual(p["language"], "python")

    def test_get_project_not_found(self):
        from tools.workspace import get_project
        p = get_project("nonexistent")
        self.assertIsNone(p)

    def test_get_project_case_insensitive(self):
        from tools.workspace import get_project
        p = get_project("FRONTEND")
        self.assertIsNotNone(p)
        self.assertEqual(p["language"], "typescript")


class TestMultiProjectMissing(unittest.TestCase):
    """Test when projects.json doesn't exist."""

    def setUp(self):
        import config
        self._orig_path = config.PROJECTS_CONFIG_PATH
        config.PROJECTS_CONFIG_PATH = "/nonexistent/path/projects.json"

    def tearDown(self):
        import config
        config.PROJECTS_CONFIG_PATH = self._orig_path

    def test_load_missing_config(self):
        from tools.workspace import load_projects_config
        self.assertEqual(load_projects_config(), [])


# ─── 6.6 Config Constants ────────────────────────────────────────────────

class TestPhase6Config(unittest.TestCase):
    """Verify Phase 6 config constants."""

    def test_max_concurrent_missions(self):
        from config import MAX_CONCURRENT_MISSIONS
        self.assertIsInstance(MAX_CONCURRENT_MISSIONS, int)
        self.assertGreater(MAX_CONCURRENT_MISSIONS, 0)

    def test_projects_config_path(self):
        from config import PROJECTS_CONFIG_PATH
        self.assertIsInstance(PROJECTS_CONFIG_PATH, str)
        self.assertTrue(PROJECTS_CONFIG_PATH.endswith("projects.json"))


# ─── 6.7 Orchestrator Integration ────────────────────────────────────────

class TestOrchestratorIntegration(unittest.TestCase):
    """Verify orchestrator has workspace isolation wiring."""

    def setUp(self):
        src_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "orchestrator.py",
        )
        with open(src_path, encoding="utf-8") as f:
            self.src = f.read()

    def test_imports_workspace_functions(self):
        self.assertIn("get_mission_workspace", self.src)
        self.assertIn("compute_workspace_hashes", self.src)
        self.assertIn("save_workspace_snapshot", self.src)

    def test_imports_git_branch_functions(self):
        self.assertIn("create_mission_branch", self.src)
        self.assertIn("get_current_branch", self.src)
        self.assertIn("get_commit_sha", self.src)

    def test_plan_mission_creates_workspace(self):
        self.assertIn("get_mission_workspace(mission_id)", self.src)
        self.assertIn("create_mission_branch", self.src)

    def test_process_task_snapshots(self):
        self.assertIn("compute_workspace_hashes", self.src)
        self.assertIn("save_workspace_snapshot", self.src)

    def test_process_task_releases_locks(self):
        self.assertIn("release_task_locks", self.src)

    def test_mission_completion_releases_locks(self):
        self.assertIn("release_mission_locks", self.src)

    def test_workspace_snapshot_for_coder_tasks(self):
        # The snapshot should only happen for coder/pipeline/fixer agents
        self.assertIn(
            '"coder", "pipeline", "implementer", "fixer"', self.src
        )


# ─── 6.8 DB Schema ───────────────────────────────────────────────────────

class TestPhase6Schema(_DBTestBase):
    """Verify Phase 6 tables exist."""

    def test_file_locks_table(self):
        async def check():
            db = await self.db_mod.get_db()
            cursor = await db.execute("PRAGMA table_info(file_locks)")
            cols = [row[1] for row in await cursor.fetchall()]
            return cols

        cols = run_async(check())
        self.assertIn("filepath", cols)
        self.assertIn("mission_id", cols)
        self.assertIn("task_id", cols)
        self.assertIn("agent_type", cols)

    def test_workspace_snapshots_table(self):
        async def check():
            db = await self.db_mod.get_db()
            cursor = await db.execute("PRAGMA table_info(workspace_snapshots)")
            cols = [row[1] for row in await cursor.fetchall()]
            return cols

        cols = run_async(check())
        self.assertIn("mission_id", cols)
        self.assertIn("file_hashes", cols)
        self.assertIn("branch_name", cols)
        self.assertIn("commit_sha", cols)


if __name__ == "__main__":
    unittest.main()
