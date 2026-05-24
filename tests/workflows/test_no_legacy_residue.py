"""End-state lock: after legacy removal, the i2p_v3 workflow JSON and the
missions schema that db.py builds must carry zero `legacy_pre_` residue
(the `_LEGACY_DROP_COLS` migration list in db.py is allowed)."""
from __future__ import annotations
import io, os, re, asyncio, tempfile

WF = "src/workflows/i2p/i2p_v3.json"

def test_workflow_has_no_legacy_pre_gates():
    text = io.open(WF, encoding="utf-8").read()
    hits = re.findall(r"legacy_pre_\w+", text)
    assert not hits, f"workflow still references: {sorted(set(hits))}"

def test_fresh_missions_schema_has_no_legacy_columns():
    # Build a FRESH schema in an isolated tmp DB -- never touch the live DB.
    os.environ["DB_PATH"] = tempfile.mktemp(suffix=".db")
    import importlib, src.infra.db as _db
    importlib.reload(_db)
    async def _check():
        await _db.init_db()
        conn = await _db.get_db()
        cur = await conn.execute("PRAGMA table_info(missions)")
        cols = [r[1] for r in await cur.fetchall()]
        await cur.close()
        return [c for c in cols if c.startswith("legacy_pre_")]
    leftover = asyncio.run(_check())
    assert not leftover, f"fresh missions schema still has: {leftover}"
