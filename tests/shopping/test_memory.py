"""Comprehensive tests for shopping memory modules:
user_profile, price_watch, session, purchase_history.

Each test class creates a temp SQLite DB, initialises it, and tears it down.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
import unittest
from unittest.mock import patch

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


# ---------------------------------------------------------------------------
# Base class that sets up a temp DB and patches get_memory_db
# ---------------------------------------------------------------------------

class MemoryTestBase(unittest.TestCase):
    """Base class providing temp SQLite DB for memory module tests."""

    _db: aiosqlite.Connection | None = None
    _tmpfile: str = ""

    def setUp(self):
        fd, self._tmpfile = tempfile.mkstemp(suffix=".db")
        os.close(fd)

        async def _setup_db():
            self._db = await aiosqlite.connect(self._tmpfile)
            self._db.row_factory = aiosqlite.Row
            await self._db.execute("PRAGMA journal_mode=WAL")
            return self._db

        run_async(_setup_db())

    def tearDown(self):
        if self._db:
            run_async(self._db.close())
            self._db = None
        if os.path.exists(self._tmpfile):
            os.unlink(self._tmpfile)
        # Clean up WAL/SHM files
        for suffix in ("-wal", "-shm"):
            p = self._tmpfile + suffix
            if os.path.exists(p):
                os.unlink(p)

    def _mock_get_db(self):
        """Return an AsyncMock that returns our test DB connection."""
        async def _get_db():
            return self._db
        return _get_db


# ═══════════════════════════════════════════════════════════════════════════
# 1. TestUserProfile
# ═══════════════════════════════════════════════════════════════════════════

class TestUserProfile(MemoryTestBase):
    """Test user profile CRUD."""

    def _patch_target(self):
        return "src.shopping.memory.user_profile.get_memory_db"

    def _init_db(self):
        from src.shopping.memory.user_profile import init_user_profile_db
        with patch(self._patch_target(), new=self._mock_get_db()):
            run_async(init_user_profile_db())

    def setUp(self):
        super().setUp()
        self._init_db()

    def test_create_and_get_profile(self):
        from src.shopping.memory.user_profile import get_user_profile
        with patch(self._patch_target(), new=self._mock_get_db()):
            profile = run_async(get_user_profile(1001))
        self.assertEqual(profile["user_id"], 1001)
        self.assertEqual(profile["owned_items"], [])
        self.assertEqual(profile["preferences"], {})

    def test_update_dietary_restrictions(self):
        from src.shopping.memory.user_profile import update_user_profile, get_user_profile
        with patch(self._patch_target(), new=self._mock_get_db()):
            run_async(update_user_profile(1001, dietary_restrictions=["vegan", "gluten-free"]))
            profile = run_async(get_user_profile(1001))
        self.assertEqual(profile["dietary_restrictions"], ["vegan", "gluten-free"])

    def test_update_location(self):
        from src.shopping.memory.user_profile import update_user_profile, get_user_profile
        with patch(self._patch_target(), new=self._mock_get_db()):
            run_async(update_user_profile(1001, location="Istanbul"))
            profile = run_async(get_user_profile(1001))
        self.assertEqual(profile["location"], "Istanbul")

    def test_add_owned_item(self):
        from src.shopping.memory.user_profile import add_owned_item, get_user_profile
        item = {"name": "iPhone 15", "category": "phone"}
        with patch(self._patch_target(), new=self._mock_get_db()):
            run_async(add_owned_item(1001, item))
            profile = run_async(get_user_profile(1001))
        self.assertEqual(len(profile["owned_items"]), 1)
        self.assertEqual(profile["owned_items"][0]["name"], "iPhone 15")

    def test_remove_owned_item(self):
        from src.shopping.memory.user_profile import add_owned_item, remove_owned_item, get_user_profile
        with patch(self._patch_target(), new=self._mock_get_db()):
            run_async(add_owned_item(1001, {"name": "iPhone 15", "category": "phone"}))
            run_async(add_owned_item(1001, {"name": "MacBook Pro", "category": "laptop"}))
            run_async(remove_owned_item(1001, "iPhone"))
            profile = run_async(get_user_profile(1001))
        self.assertEqual(len(profile["owned_items"]), 1)
        self.assertEqual(profile["owned_items"][0]["name"], "MacBook Pro")

    def test_set_preference(self):
        from src.shopping.memory.user_profile import set_preference, get_user_profile
        with patch(self._patch_target(), new=self._mock_get_db()):
            run_async(set_preference(1001, "preferred_brand", "Samsung"))
            profile = run_async(get_user_profile(1001))
        self.assertEqual(profile["preferences"]["preferred_brand"], "Samsung")

    def test_set_inferred_preference(self):
        from src.shopping.memory.user_profile import set_preference, get_user_profile
        with patch(self._patch_target(), new=self._mock_get_db()):
            run_async(set_preference(1001, "price_sensitivity", "high", inferred=True))
            profile = run_async(get_user_profile(1001))
        self.assertEqual(profile["inferred_preferences"]["price_sensitivity"], "high")

    def test_upsert_preference(self):
        from src.shopping.memory.user_profile import set_preference, get_user_profile
        with patch(self._patch_target(), new=self._mock_get_db()):
            run_async(set_preference(1001, "brand", "Apple"))
            run_async(set_preference(1001, "brand", "Samsung"))
            profile = run_async(get_user_profile(1001))
        self.assertEqual(profile["preferences"]["brand"], "Samsung")

    def test_record_behavior(self):
        from src.shopping.memory.user_profile import record_behavior, get_user_profile
        with patch(self._patch_target(), new=self._mock_get_db()):
            run_async(record_behavior(1001, "always_picks_cheapest"))
            profile = run_async(get_user_profile(1001))
        self.assertIn("always_picks_cheapest", profile["behaviors"])

    def test_clear_user_data(self):
        from src.shopping.memory.user_profile import (
            add_owned_item, set_preference, record_behavior, clear_user_data, get_user_profile,
        )
        with patch(self._patch_target(), new=self._mock_get_db()):
            run_async(add_owned_item(1001, {"name": "Test"}))
            run_async(set_preference(1001, "key", "val"))
            run_async(record_behavior(1001, "test"))
            run_async(clear_user_data(1001))
            profile = run_async(get_user_profile(1001))
        self.assertEqual(profile["owned_items"], [])
        self.assertEqual(profile["preferences"], {})
        self.assertEqual(profile["behaviors"], [])

    def test_multiple_users_isolated(self):
        from src.shopping.memory.user_profile import add_owned_item, get_user_profile
        with patch(self._patch_target(), new=self._mock_get_db()):
            run_async(add_owned_item(1001, {"name": "Item A"}))
            run_async(add_owned_item(2002, {"name": "Item B"}))
            p1 = run_async(get_user_profile(1001))
            p2 = run_async(get_user_profile(2002))
        self.assertEqual(len(p1["owned_items"]), 1)
        self.assertEqual(p1["owned_items"][0]["name"], "Item A")
        self.assertEqual(len(p2["owned_items"]), 1)
        self.assertEqual(p2["owned_items"][0]["name"], "Item B")


# ═══════════════════════════════════════════════════════════════════════════
# 2. TestPriceWatch
# ═══════════════════════════════════════════════════════════════════════════

class TestPriceWatch(MemoryTestBase):
    """Test price watch memory."""

    def _patch_target(self):
        return "src.shopping.memory.price_watch.get_memory_db"

    def _init_db(self):
        from src.shopping.memory.price_watch import init_price_watch_db
        with patch(self._patch_target(), new=self._mock_get_db()):
            run_async(init_price_watch_db())

    def setUp(self):
        super().setUp()
        self._init_db()

    def test_add_watch(self):
        from src.shopping.memory.price_watch import add_price_watch, get_active_watches
        with patch(self._patch_target(), new=self._mock_get_db()):
            watch_id = run_async(add_price_watch(1001, "iPhone 15", 40000, target_price=35000))
            watches = run_async(get_active_watches(1001))
        self.assertIsInstance(watch_id, int)
        self.assertEqual(len(watches), 1)
        self.assertEqual(watches[0]["product_name"], "iPhone 15")

    def test_update_price(self):
        from src.shopping.memory.price_watch import add_price_watch, update_watch_price, get_active_watches
        with patch(self._patch_target(), new=self._mock_get_db()):
            wid = run_async(add_price_watch(1001, "Test", 1000))
            run_async(update_watch_price(wid, 900, "trendyol"))
            watches = run_async(get_active_watches(1001))
        self.assertEqual(watches[0]["current_price"], 900)
        self.assertEqual(watches[0]["historical_low"], 900)

    def test_update_price_tracks_historical_low(self):
        from src.shopping.memory.price_watch import add_price_watch, update_watch_price, get_active_watches
        with patch(self._patch_target(), new=self._mock_get_db()):
            wid = run_async(add_price_watch(1001, "Test", 1000))
            run_async(update_watch_price(wid, 800, "a"))
            run_async(update_watch_price(wid, 900, "b"))
            watches = run_async(get_active_watches(1001))
        self.assertEqual(watches[0]["historical_low"], 800)

    def test_trigger_watch(self):
        from src.shopping.memory.price_watch import add_price_watch, trigger_watch, get_active_watches
        with patch(self._patch_target(), new=self._mock_get_db()):
            wid = run_async(add_price_watch(1001, "Test", 1000))
            run_async(trigger_watch(wid))
            watches = run_async(get_active_watches(1001))
        self.assertEqual(len(watches), 0)  # triggered watches are not active

    def test_expire_old_watches(self):
        from src.shopping.memory.price_watch import add_price_watch, expire_old_watches, get_active_watches
        with patch(self._patch_target(), new=self._mock_get_db()):
            wid = run_async(add_price_watch(1001, "Old", 1000))
            # Manually backdate the watch
            run_async(self._db.execute(
                "UPDATE price_watches SET created_at = ? WHERE id = ?",
                (time.time() - 100 * 86400, wid),
            ))
            run_async(self._db.commit())
            run_async(expire_old_watches(days=90))
            watches = run_async(get_active_watches(1001))
        self.assertEqual(len(watches), 0)

    def test_remove_watch(self):
        from src.shopping.memory.price_watch import add_price_watch, remove_watch, get_active_watches
        with patch(self._patch_target(), new=self._mock_get_db()):
            wid = run_async(add_price_watch(1001, "Test", 1000))
            run_async(remove_watch(wid))
            watches = run_async(get_active_watches(1001))
        self.assertEqual(len(watches), 0)

    def test_get_all_active_watches(self):
        from src.shopping.memory.price_watch import add_price_watch, get_all_active_watches
        with patch(self._patch_target(), new=self._mock_get_db()):
            run_async(add_price_watch(1001, "A", 100))
            run_async(add_price_watch(2002, "B", 200))
            all_watches = run_async(get_all_active_watches())
        self.assertEqual(len(all_watches), 2)


# ═══════════════════════════════════════════════════════════════════════════
# 3. TestSession
# ═══════════════════════════════════════════════════════════════════════════

class TestSession(MemoryTestBase):
    """Test shopping session memory."""

    def _patch_target(self):
        return "src.shopping.memory.session.get_memory_db"

    def _init_db(self):
        from src.shopping.memory.session import init_session_db
        with patch(self._patch_target(), new=self._mock_get_db()):
            run_async(init_session_db())

    def setUp(self):
        super().setUp()
        self._init_db()

    def test_create_session(self):
        from src.shopping.memory.session import create_session, get_session
        with patch(self._patch_target(), new=self._mock_get_db()):
            sid = run_async(create_session(1001, "laptop search"))
            session = run_async(get_session(sid))
        self.assertIsNotNone(sid)
        self.assertEqual(session["topic"], "laptop search")
        self.assertEqual(session["status"], "active")

    def test_get_nonexistent_session(self):
        from src.shopping.memory.session import get_session
        with patch(self._patch_target(), new=self._mock_get_db()):
            session = run_async(get_session("nonexistent-id"))
        self.assertEqual(session, {})

    def test_add_products(self):
        from src.shopping.memory.session import create_session, add_session_product, get_session
        with patch(self._patch_target(), new=self._mock_get_db()):
            sid = run_async(create_session(1001, "phone"))
            run_async(add_session_product(sid, {"name": "iPhone 15", "price": 40000}))
            run_async(add_session_product(sid, {"name": "Samsung S24", "price": 35000}))
            session = run_async(get_session(sid))
        self.assertEqual(len(session["products"]), 2)

    def test_add_questions(self):
        from src.shopping.memory.session import create_session, add_session_question, get_session
        with patch(self._patch_target(), new=self._mock_get_db()):
            sid = run_async(create_session(1001, "laptop"))
            run_async(add_session_question(sid, "RAM ne kadar?", "16 GB"))
            session = run_async(get_session(sid))
        self.assertEqual(len(session["questions"]), 1)
        self.assertEqual(session["questions"][0]["question"], "RAM ne kadar?")
        self.assertEqual(session["questions"][0]["answer"], "16 GB")

    def test_update_session(self):
        from src.shopping.memory.session import create_session, update_session, get_session
        with patch(self._patch_target(), new=self._mock_get_db()):
            sid = run_async(create_session(1001, "test"))
            run_async(update_session(sid, status="completed", summary="Found a good laptop"))
            session = run_async(get_session(sid))
        self.assertEqual(session["status"], "completed")
        self.assertEqual(session["summary"], "Found a good laptop")

    def test_recent_session_lookup(self):
        from src.shopping.memory.session import create_session, get_recent_session
        with patch(self._patch_target(), new=self._mock_get_db()):
            run_async(create_session(1001, "laptop search"))
            recent = run_async(get_recent_session(1001, "laptop search"))
        self.assertIsNotNone(recent)
        self.assertEqual(recent["topic"], "laptop search")

    def test_recent_session_expired(self):
        from src.shopping.memory.session import create_session, get_recent_session
        with patch(self._patch_target(), new=self._mock_get_db()):
            sid = run_async(create_session(1001, "old"))
            # Backdate
            run_async(self._db.execute(
                "UPDATE shopping_sessions SET updated_at = ? WHERE session_id = ?",
                (time.time() - 25 * 3600, sid),
            ))
            run_async(self._db.commit())
            recent = run_async(get_recent_session(1001, hours=24))
        self.assertIsNone(recent)

    def test_clear_session(self):
        from src.shopping.memory.session import create_session, add_session_product, clear_session, get_session
        with patch(self._patch_target(), new=self._mock_get_db()):
            sid = run_async(create_session(1001, "test"))
            run_async(add_session_product(sid, {"name": "X"}))
            run_async(clear_session(sid))
            session = run_async(get_session(sid))
        self.assertEqual(session, {})


# ═══════════════════════════════════════════════════════════════════════════
# 4. TestPurchaseHistory
# ═══════════════════════════════════════════════════════════════════════════

class TestPurchaseHistory(MemoryTestBase):
    """Test purchase history memory."""

    def _patch_target(self):
        return "src.shopping.memory.purchase_history.get_memory_db"

    def _init_db(self):
        from src.shopping.memory.purchase_history import init_purchase_history_db
        with patch(self._patch_target(), new=self._mock_get_db()):
            run_async(init_purchase_history_db())

    def setUp(self):
        super().setUp()
        self._init_db()

    def test_log_purchase(self):
        from src.shopping.memory.purchase_history import log_purchase, get_purchase_history
        with patch(self._patch_target(), new=self._mock_get_db()):
            pid = run_async(log_purchase(1001, "iPhone 15 Pro", 45000, "trendyol", "phone"))
            history = run_async(get_purchase_history(1001))
        self.assertIsInstance(pid, int)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["product_name"], "iPhone 15 Pro")

    def test_purchase_history_order(self):
        from src.shopping.memory.purchase_history import log_purchase, get_purchase_history
        with patch(self._patch_target(), new=self._mock_get_db()):
            run_async(log_purchase(1001, "First", 100))
            run_async(log_purchase(1001, "Second", 200))
            history = run_async(get_purchase_history(1001))
        # Most recent first
        self.assertEqual(history[0]["product_name"], "Second")

    def test_has_purchased_exact(self):
        from src.shopping.memory.purchase_history import log_purchase, has_purchased
        with patch(self._patch_target(), new=self._mock_get_db()):
            run_async(log_purchase(1001, "Samsung Galaxy S24", 35000))
            result = run_async(has_purchased(1001, "Samsung Galaxy S24"))
        self.assertTrue(result)

    def test_has_purchased_fuzzy(self):
        from src.shopping.memory.purchase_history import log_purchase, has_purchased
        with patch(self._patch_target(), new=self._mock_get_db()):
            run_async(log_purchase(1001, "Samsung Galaxy S24 Ultra 256GB", 35000))
            result = run_async(has_purchased(1001, "Galaxy S24"))
        self.assertTrue(result)

    def test_has_not_purchased(self):
        from src.shopping.memory.purchase_history import has_purchased
        with patch(self._patch_target(), new=self._mock_get_db()):
            result = run_async(has_purchased(1001, "iPhone"))
        self.assertFalse(result)

    def test_get_recent_purchases(self):
        from src.shopping.memory.purchase_history import log_purchase, get_recent_purchases
        with patch(self._patch_target(), new=self._mock_get_db()):
            run_async(log_purchase(1001, "Recent", 100))
            # Backdate an old purchase
            run_async(self._db.execute(
                "INSERT INTO purchases (user_id, product_name, price, purchased_at) VALUES (?, ?, ?, ?)",
                (1001, "Old", 50, time.time() - 120 * 86400),
            ))
            run_async(self._db.commit())
            recent = run_async(get_recent_purchases(1001, days=90))
        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0]["product_name"], "Recent")

    def test_complementary_suggestions(self):
        from src.shopping.memory.purchase_history import log_purchase, get_complementary_suggestions
        with patch(self._patch_target(), new=self._mock_get_db()):
            run_async(log_purchase(1001, "Laptop X", 10000, category="laptop"))
            suggestions = run_async(get_complementary_suggestions(1001))
        self.assertTrue(len(suggestions) > 0)
        suggestion_names = [s["suggestion"] for s in suggestions]
        self.assertTrue(any("bag" in s.lower() or "mouse" in s.lower() for s in suggestion_names))

    def test_history_limit(self):
        from src.shopping.memory.purchase_history import log_purchase, get_purchase_history
        with patch(self._patch_target(), new=self._mock_get_db()):
            for i in range(10):
                run_async(log_purchase(1001, f"Item {i}", i * 100))
            history = run_async(get_purchase_history(1001, limit=5))
        self.assertEqual(len(history), 5)


if __name__ == "__main__":
    unittest.main()
