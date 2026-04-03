"""
One-time migration: wipe garbage auto-captured skills, keep seed skills.
Run: python -m scripts.migrate_skills_v2
"""
import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def main():
    from src.infra.db import get_db, init_db
    await init_db()
    db = await get_db()

    # 1. Count current state
    cursor = await db.execute("SELECT COUNT(*) FROM skills WHERE name LIKE 'auto:%'")
    auto_count = (await cursor.fetchone())[0]
    cursor = await db.execute("SELECT COUNT(*) FROM skills WHERE name NOT LIKE 'auto:%'")
    seed_count = (await cursor.fetchone())[0]
    print(f"Current state: {auto_count} auto-captured, {seed_count} seed/manual")

    # 2. Show non-i2p auto skills
    cursor = await db.execute(
        "SELECT name, description FROM skills WHERE name LIKE 'auto:%' LIMIT 20"
    )
    auto_skills = await cursor.fetchall()
    if auto_skills:
        print(f"\nAuto skills ({len(auto_skills)}):")
        for row in auto_skills:
            print(f"  {row[0]}: {row[1][:80] if row[1] else 'no desc'}")

    # 3. Delete all auto-captured
    await db.execute("DELETE FROM skills WHERE name LIKE 'auto:%'")
    await db.commit()
    print(f"\nDeleted {auto_count} auto-captured skills")

    # 4. Rebuild ChromaDB
    try:
        from src.memory.vector_store import get_collection, embed_and_store
        collection = await get_collection("semantic")
        results = collection.get(where={"type": "skill"})
        if results and results["ids"]:
            collection.delete(ids=results["ids"])
            print(f"Cleaned {len(results['ids'])} skill embeddings from ChromaDB")

        cursor = await db.execute("SELECT name, description FROM skills")
        remaining = await cursor.fetchall()
        for row in remaining:
            await embed_and_store(
                text=row[1],
                metadata={"type": "skill", "skill_name": row[0]},
                collection="semantic",
                doc_id=f"skill:{row[0]}",
            )
        print(f"Re-embedded {len(remaining)} seed skills in ChromaDB")
    except Exception as e:
        print(f"ChromaDB cleanup failed (non-critical): {e}")

    print("\nMigration complete!")


if __name__ == "__main__":
    asyncio.run(main())
