"""User profile memory — owned items, preferences, constraints, and behavior patterns."""

import json
import time

import aiosqlite

from src.shopping.memory._db import get_memory_db
from src.infra.logging_config import get_logger

logger = get_logger("shopping.memory.user_profile")

# ─── Schema ──────────────────────────────────────────────────────────────────


async def init_user_profile_db() -> None:
    """Create user-profile tables if they don't exist."""
    db = await get_memory_db()

    await db.execute("""
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id     INTEGER PRIMARY KEY,
            dietary_restrictions TEXT NOT NULL DEFAULT '[]',
            location    TEXT,
            created_at  REAL NOT NULL,
            updated_at  REAL NOT NULL
        )
    """)

    await db.execute("""
        CREATE TABLE IF NOT EXISTS owned_items (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            item_json   TEXT    NOT NULL,
            added_at    REAL   NOT NULL,
            FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
        )
    """)
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_owned_items_user ON owned_items(user_id)"
    )

    await db.execute("""
        CREATE TABLE IF NOT EXISTS preferences (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            key         TEXT    NOT NULL,
            value       TEXT    NOT NULL,
            inferred    INTEGER NOT NULL DEFAULT 0,
            set_at      REAL    NOT NULL,
            UNIQUE(user_id, key),
            FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
        )
    """)
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_prefs_user ON preferences(user_id)"
    )

    await db.execute("""
        CREATE TABLE IF NOT EXISTS behaviors (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            behavior    TEXT    NOT NULL,
            observed_at REAL   NOT NULL,
            FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
        )
    """)
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_behaviors_user ON behaviors(user_id)"
    )

    await db.commit()
    logger.info("User-profile tables initialised")


# ─── Helpers ─────────────────────────────────────────────────────────────────


async def _ensure_profile(db: aiosqlite.Connection, user_id: int) -> None:
    """Insert a skeleton profile row if one doesn't exist yet."""
    now = time.time()
    await db.execute(
        """
        INSERT OR IGNORE INTO user_profiles (user_id, created_at, updated_at)
        VALUES (?, ?, ?)
        """,
        (user_id, now, now),
    )


# ─── Public API ──────────────────────────────────────────────────────────────


async def get_user_profile(user_id: int) -> dict:
    """Return the full user profile as a dict.

    Keys: owned_items, preferences, constraints, dietary_restrictions,
    location, inferred_preferences, behaviors.
    """
    db = await get_memory_db()
    await _ensure_profile(db, user_id)
    await db.commit()

    # Core profile row
    cur = await db.execute(
        "SELECT dietary_restrictions, location FROM user_profiles WHERE user_id = ?",
        (user_id,),
    )
    row = await cur.fetchone()
    dietary = json.loads(row["dietary_restrictions"]) if row else []
    location = row["location"] if row else None

    # Owned items
    cur = await db.execute(
        "SELECT item_json, added_at FROM owned_items WHERE user_id = ? ORDER BY added_at",
        (user_id,),
    )
    owned_items = [
        {**json.loads(r["item_json"]), "_added_at": r["added_at"]}
        for r in await cur.fetchall()
    ]

    # Preferences (split stated vs inferred)
    cur = await db.execute(
        "SELECT key, value, inferred FROM preferences WHERE user_id = ?",
        (user_id,),
    )
    preferences: dict[str, str] = {}
    inferred_preferences: dict[str, str] = {}
    for r in await cur.fetchall():
        if r["inferred"]:
            inferred_preferences[r["key"]] = r["value"]
        else:
            preferences[r["key"]] = r["value"]

    # Behaviors
    cur = await db.execute(
        "SELECT DISTINCT behavior FROM behaviors WHERE user_id = ? ORDER BY observed_at DESC",
        (user_id,),
    )
    behaviors = [r["behavior"] for r in await cur.fetchall()]

    return {
        "user_id": user_id,
        "owned_items": owned_items,
        "preferences": preferences,
        "constraints": {},  # reserved for future budget / shipping constraints
        "dietary_restrictions": dietary,
        "location": location,
        "inferred_preferences": inferred_preferences,
        "behaviors": behaviors,
    }


async def update_user_profile(user_id: int, **fields) -> None:
    """Update specific top-level profile fields.

    Supported fields: dietary_restrictions (list), location (str).
    """
    db = await get_memory_db()
    await _ensure_profile(db, user_id)

    sets: list[str] = []
    params: list = []

    if "dietary_restrictions" in fields:
        sets.append("dietary_restrictions = ?")
        params.append(json.dumps(fields["dietary_restrictions"], ensure_ascii=False))
    if "location" in fields:
        sets.append("location = ?")
        params.append(fields["location"])

    if not sets:
        await db.commit()
        return

    sets.append("updated_at = ?")
    params.append(time.time())
    params.append(user_id)

    await db.execute(
        f"UPDATE user_profiles SET {', '.join(sets)} WHERE user_id = ?",
        params,
    )
    await db.commit()

    # Phase C: Re-embed user profile into vector store
    try:
        from src.shopping.intelligence.vector_bridge import embed_user_shopping_profile
        await embed_user_shopping_profile(user_id)
    except Exception as e:
        logger.debug("Profile embedding skipped: %s", e)


async def add_owned_item(user_id: int, item: dict) -> None:
    """Add an owned item (PC specs, appliance, etc.) to the user's list."""
    db = await get_memory_db()
    await _ensure_profile(db, user_id)
    await db.execute(
        "INSERT INTO owned_items (user_id, item_json, added_at) VALUES (?, ?, ?)",
        (user_id, json.dumps(item, ensure_ascii=False), time.time()),
    )
    await db.commit()

    # Phase C: Re-embed profile to include new owned item
    try:
        from src.shopping.intelligence.vector_bridge import embed_user_shopping_profile
        await embed_user_shopping_profile(user_id)
    except Exception as e:
        logger.debug("Owned item embedding skipped: %s", e)


async def remove_owned_item(user_id: int, item_name: str) -> None:
    """Remove an owned item by name (case-insensitive partial match on the 'name' key)."""
    db = await get_memory_db()
    cur = await db.execute(
        "SELECT id, item_json FROM owned_items WHERE user_id = ?",
        (user_id,),
    )
    for row in await cur.fetchall():
        item = json.loads(row["item_json"])
        stored_name = item.get("name", "")
        if item_name.lower() in stored_name.lower():
            await db.execute("DELETE FROM owned_items WHERE id = ?", (row["id"],))
    await db.commit()


async def set_preference(user_id: int, key: str, value: str, inferred: bool = False) -> None:
    """Store a stated or inferred preference (upserts on user_id + key)."""
    db = await get_memory_db()
    await _ensure_profile(db, user_id)
    await db.execute(
        """
        INSERT INTO preferences (user_id, key, value, inferred, set_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(user_id, key) DO UPDATE SET value = excluded.value,
                                                inferred = excluded.inferred,
                                                set_at = excluded.set_at
        """,
        (user_id, key, value, int(inferred), time.time()),
    )
    await db.commit()

    # Phase C: Re-embed profile preferences into vector store
    try:
        from src.shopping.intelligence.vector_bridge import embed_user_shopping_profile
        await embed_user_shopping_profile(user_id)
    except Exception as e:
        logger.debug("Preference embedding skipped: %s", e)


async def record_behavior(user_id: int, behavior: str) -> None:
    """Track a purchase-adjacent behavior pattern.

    Examples: always_picks_cheapest, always_checks_reviews, prefers_known_brands.
    """
    db = await get_memory_db()
    await _ensure_profile(db, user_id)
    await db.execute(
        "INSERT INTO behaviors (user_id, behavior, observed_at) VALUES (?, ?, ?)",
        (user_id, behavior, time.time()),
    )
    await db.commit()


async def clear_user_data(user_id: int) -> None:
    """Delete all user data ('forget everything about me')."""
    db = await get_memory_db()
    await db.execute("DELETE FROM behaviors WHERE user_id = ?", (user_id,))
    await db.execute("DELETE FROM preferences WHERE user_id = ?", (user_id,))
    await db.execute("DELETE FROM owned_items WHERE user_id = ?", (user_id,))
    await db.execute("DELETE FROM user_profiles WHERE user_id = ?", (user_id,))
    await db.commit()
    logger.info("Cleared all profile data for user %s", user_id)
