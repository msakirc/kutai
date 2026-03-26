"""Purchase history memory — log purchases and derive suggestions."""

import json
import time

import aiosqlite

from src.shopping.memory._db import get_memory_db
from src.infra.logging_config import get_logger

logger = get_logger("shopping.memory.purchase_history")

# ─── Complementary product mapping ──────────────────────────────────────────
# Maps product categories to likely complementary categories / items.
_COMPLEMENTARY_MAP: dict[str, list[str]] = {
    "laptop":       ["laptop bag", "mouse", "keyboard", "monitor", "USB hub", "laptop stand"],
    "phone":        ["phone case", "screen protector", "charger", "earbuds", "power bank"],
    "camera":       ["memory card", "camera bag", "tripod", "lens filter", "extra battery"],
    "headphones":   ["ear tips", "headphone stand", "DAC", "carrying case"],
    "monitor":      ["monitor arm", "HDMI cable", "DisplayPort cable", "webcam"],
    "keyboard":     ["wrist rest", "keycaps", "desk mat"],
    "mouse":        ["mouse pad", "wrist rest"],
    "printer":      ["ink cartridge", "paper", "USB cable"],
    "tablet":       ["tablet case", "stylus", "screen protector", "keyboard cover"],
    "gaming console": ["extra controller", "headset", "charging dock", "HDMI cable"],
    "coffee maker":  ["coffee grinder", "coffee beans", "water filter", "descaler"],
    "blender":      ["extra blades", "to-go cups", "recipe book"],
}

# ─── Schema ──────────────────────────────────────────────────────────────────


async def init_purchase_history_db() -> None:
    """Create purchase-history tables if they don't exist."""
    db = await get_memory_db()

    await db.execute("""
        CREATE TABLE IF NOT EXISTS purchases (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id       INTEGER NOT NULL,
            product_name  TEXT    NOT NULL,
            price         REAL,
            store         TEXT,
            category      TEXT,
            purchased_at  REAL    NOT NULL
        )
    """)
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_purchases_user ON purchases(user_id)"
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_purchases_user_time ON purchases(user_id, purchased_at)"
    )

    await db.commit()
    logger.info("Purchase-history tables initialised")


# ─── Public API ──────────────────────────────────────────────────────────────


async def log_purchase(
    user_id: int,
    product_name: str,
    price: float = None,
    store: str = None,
    category: str = None,
) -> int:
    """Log a purchase and return its record id."""
    db = await get_memory_db()
    cur = await db.execute(
        """
        INSERT INTO purchases (user_id, product_name, price, store, category, purchased_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (user_id, product_name, price, store, category, time.time()),
    )
    purchase_id = cur.lastrowid
    await db.commit()
    logger.info("Logged purchase #%d for user %s: %s", purchase_id, user_id, product_name)
    return purchase_id


async def get_purchase_history(user_id: int, limit: int = 50) -> list[dict]:
    """Return purchase history for a user, most recent first."""
    db = await get_memory_db()
    cur = await db.execute(
        """
        SELECT id, product_name, price, store, category, purchased_at
        FROM purchases
        WHERE user_id = ?
        ORDER BY purchased_at DESC
        LIMIT ?
        """,
        (user_id, limit),
    )
    return [dict(r) for r in await cur.fetchall()]


async def get_recent_purchases(user_id: int, days: int = 90) -> list[dict]:
    """Return purchases from the last *days* days, most recent first."""
    db = await get_memory_db()
    cutoff = time.time() - days * 86400
    cur = await db.execute(
        """
        SELECT id, product_name, price, store, category, purchased_at
        FROM purchases
        WHERE user_id = ? AND purchased_at > ?
        ORDER BY purchased_at DESC
        """,
        (user_id, cutoff),
    )
    return [dict(r) for r in await cur.fetchall()]


async def has_purchased(user_id: int, product_name: str) -> bool:
    """Check if a user has purchased a product (case-insensitive fuzzy match).

    Uses SQL LIKE with the product name as a substring.
    """
    db = await get_memory_db()
    pattern = f"%{product_name}%"
    cur = await db.execute(
        """
        SELECT 1 FROM purchases
        WHERE user_id = ? AND product_name LIKE ? COLLATE NOCASE
        LIMIT 1
        """,
        (user_id, pattern),
    )
    return (await cur.fetchone()) is not None


async def get_complementary_suggestions(user_id: int) -> list[dict]:
    """Suggest complementary products based on recent purchases.

    Returns a list of dicts with 'suggestion', 'based_on', and 'category' keys.
    """
    recent = await get_recent_purchases(user_id, days=90)
    if not recent:
        return []

    suggestions: list[dict] = []
    seen_suggestions: set[str] = set()

    for purchase in recent:
        name_lower = purchase["product_name"].lower()
        cat_lower = (purchase["category"] or "").lower()

        # Try matching against the complementary map
        for key, complements in _COMPLEMENTARY_MAP.items():
            if key in name_lower or key in cat_lower:
                for suggestion in complements:
                    if suggestion not in seen_suggestions:
                        seen_suggestions.add(suggestion)
                        suggestions.append({
                            "suggestion": suggestion,
                            "based_on": purchase["product_name"],
                            "category": key,
                        })

    return suggestions
