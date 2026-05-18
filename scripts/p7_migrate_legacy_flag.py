"""P7 — add `missions.legacy_pre_p7` column and backfill.

Idempotent. Safe to run multiple times. The same migration is also wired into
``src/infra/db.py`` so a fresh DB or normal startup applies it automatically;
this script exists for ops-driven runs against a paused DB.

Usage:
    python scripts/p7_migrate_legacy_flag.py [DB_PATH]

DB_PATH defaults to ``$KUTAI_DB`` then ``C:/Users/sakir/ai/kutai/kutai.db``.
"""
from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path


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
        if "legacy_pre_p7" in cols:
            print("legacy_pre_p7 already present — skipping ALTER")
        else:
            cur.execute(
                "ALTER TABLE missions ADD COLUMN legacy_pre_p7 INTEGER DEFAULT 0"
            )
            cur.execute("UPDATE missions SET legacy_pre_p7 = 1")
            conn.commit()
            print(
                f"added legacy_pre_p7; backfilled {cur.rowcount} existing missions to 1"
            )
        cur.execute(
            "SELECT COUNT(*) FROM missions WHERE legacy_pre_p7 = 1"
        )
        legacy = cur.fetchone()[0]
        cur.execute(
            "SELECT COUNT(*) FROM missions WHERE legacy_pre_p7 = 0"
        )
        modern = cur.fetchone()[0]
        print(f"missions: legacy={legacy} modern={modern}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
