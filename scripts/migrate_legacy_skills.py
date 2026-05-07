"""One-shot: migrate 1 legacy auto skill to seed, delete the other 32.

Triage rationale:
  - Workflow-step value is now served by workflow_exemplars (commit 8b22ab5).
  - A legacy auto skill stays only if it carries real tools AND a cross-agent
    capability description — `auto:deal_analyst:[3.1]` is the sole entry that
    qualifies (real tools: shopping_compare, shopping_reviews).
  - Everything else: empty filtered tools, step-bound descriptions, or test
    pollution. Drop.
"""
from __future__ import annotations

import asyncio
import io
import json
import sqlite3
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DB_PATH = r"C:\Users\sakir\ai\kutai\kutai.db"
CHROMA_PATH = r"data/chroma"

KEEP_LEGACY = "auto:deal_analyst:[3.1] analyze_value_and_risks"
NEW_NAME = "product_deal_analysis"
NEW_DESC = (
    "Evaluating product deals, discounts, and user reviews to determine "
    "value, risks, and timing for purchase decisions in Turkish e-commerce"
)
NEW_STRATEGY = {
    "summary": (
        "Use shopping_compare to aggregate retailer prices, then "
        "shopping_reviews/shopping_fetch_reviews for sentiment and "
        "complaints. Synthesize: value (price vs alternatives), risks "
        "(complaint patterns, brand reputation), timing (deal "
        "freshness, inventory signals)."
    ),
    "tool_template": "",
    "tools_used": ["shopping_compare", "shopping_reviews"],
    "avg_iterations": 3,
    "source_grade": "great",
    "source_task_id": 0,
    "injection_count": 0,
    "injection_success": 0,
}


async def main() -> None:
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row

    # Snapshot before
    print("auto skills before:", db.execute(
        "SELECT COUNT(*) FROM skills WHERE skill_type='auto'"
    ).fetchone()[0])

    # Get list of names to drop (everything except the keeper)
    drop_names = [
        r["name"]
        for r in db.execute(
            "SELECT name FROM skills WHERE skill_type='auto' AND name<>?",
            (KEEP_LEGACY,),
        )
    ]
    print(f"will drop {len(drop_names)} skills")
    print(f"will migrate {KEEP_LEGACY} -> {NEW_NAME}")

    # 1) Delete the 32
    cur = db.cursor()
    cur.executemany(
        "DELETE FROM skills WHERE name=?", [(n,) for n in drop_names]
    )

    # 2) Migrate the keeper: rename + retype + replace strategies with the
    #    cleaned single strategy. Inject counts reset (was 376/218 from
    #    polluted era; new data starts fresh with seed semantics).
    cur.execute(
        "UPDATE skills SET name=?, description=?, skill_type='seed', "
        "strategies=?, injection_count=0, injection_success=0, "
        "updated_at=strftime('%Y-%m-%d %H:%M:%S','now','localtime') "
        "WHERE name=?",
        (
            NEW_NAME,
            NEW_DESC,
            json.dumps([NEW_STRATEGY]),
            KEEP_LEGACY,
        ),
    )
    db.commit()

    print()
    print("auto skills after:", db.execute(
        "SELECT COUNT(*) FROM skills WHERE skill_type='auto'"
    ).fetchone()[0])
    print("seed skills after:", db.execute(
        "SELECT COUNT(*) FROM skills WHERE skill_type='seed'"
    ).fetchone()[0])
    db.close()

    # 3) Vector cleanup: drop embeddings of all 32 dropped names; re-embed
    #    the migrated skill under its NEW name with the NEW description.
    import chromadb

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    col = client.get_collection("semantic")
    print(f"\nchroma semantic count before: {col.count()}")

    purged = 0
    for n in drop_names:
        r = col.get(where={"skill_name": n})
        ids = r.get("ids", [])
        if ids:
            col.delete(ids=ids)
            purged += len(ids)
    # Old keeper name embedding goes too
    r = col.get(where={"skill_name": KEEP_LEGACY})
    if r.get("ids"):
        col.delete(ids=r["ids"])
        purged += len(r["ids"])
    print(f"  purged {purged} stale embeddings")

    # Re-embed the migrated skill via the live skills module so its
    # embedding lives under the right name + uses the persisted embed fn.
    from src.memory.skills import _embed_skill

    await _embed_skill(NEW_NAME, NEW_DESC)
    print(f"  re-embedded {NEW_NAME}")

    # New count
    client2 = chromadb.PersistentClient(path=CHROMA_PATH)
    col2 = client2.get_collection("semantic")
    print(f"chroma semantic count after: {col2.count()}")


asyncio.run(main())
