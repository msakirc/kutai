"""Z1 Tier 3 — add ``missions.legacy_pre_per_screen_plans`` column.

Idempotent. Mirrors the prior Z1 migration scripts. Backfills existing
missions to ``1`` because they predate the per-screen plan + HTML
prototype reshape of phase 5 (steps ``5.1 generate_per_screen_plans`` and
``5.2 generate_html_prototypes``).

Usage:
    python scripts/z1_migrate_legacy_per_screen_plans_flag.py [DB_PATH]

DB_PATH defaults to ``$KUTAI_DB`` then ``C:/Users/sakir/ai/kutai/kutai.db``.
"""
from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path


COLUMN = "legacy_pre_per_screen_plans"


def main() -> int:
    db_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.environ.get("KUTAI_DB") or r"C:\Users\sakir\ai\kutai\kutai.db"
    )
    if not Path(db_path).exists():
        print(f"DB not found: {db_path}", file=sys.stderr)
        return 1
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(missions)")
        cols = {row[1] for row in cur.fetchall()}
        if COLUMN in cols:
            print(f"{COLUMN} already present — skipping ALTER")
        else:
            cur.execute(
                f"ALTER TABLE missions ADD COLUMN {COLUMN} INTEGER DEFAULT 0"
            )
            cur.execute(f"UPDATE missions SET {COLUMN} = 1")
            conn.commit()
            print(
                f"added {COLUMN}; backfilled {cur.rowcount} existing missions to 1"
            )
        cur.execute(f"SELECT COUNT(*) FROM missions WHERE {COLUMN} = 1")
        legacy = cur.fetchone()[0]
        cur.execute(f"SELECT COUNT(*) FROM missions WHERE {COLUMN} = 0")
        modern = cur.fetchone()[0]
        print(f"missions: legacy_per_screen_plans={legacy} modern={modern}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
