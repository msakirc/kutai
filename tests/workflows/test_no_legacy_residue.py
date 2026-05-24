"""End-state lock: after legacy removal, the i2p_v3 workflow JSON and the
missions schema that db.py builds must carry zero `legacy_pre_` residue
(the `_LEGACY_DROP_COLS` migration list in db.py is allowed)."""
from __future__ import annotations
import io, re

WF = "src/workflows/i2p/i2p_v3.json"
DB_SRC = "src/infra/db.py"


def test_workflow_has_no_legacy_pre_gates():
    text = io.open(WF, encoding="utf-8").read()
    hits = re.findall(r"legacy_pre_\w+", text)
    assert not hits, f"workflow still references: {sorted(set(hits))}"


def test_fresh_missions_schema_has_no_legacy_columns():
    """Parse the CREATE TABLE missions statement from db.py source and assert
    no legacy_pre_ column is defined.  Fully synchronous -- no DB open, no
    asyncio, no Windows teardown hang."""
    src = io.open(DB_SRC, encoding="utf-8").read()

    # Extract the CREATE TABLE missions block (from CREATE TABLE missions up to
    # the next empty line or closing paren that ends the statement).
    match = re.search(
        r"CREATE TABLE IF NOT EXISTS missions\s*\((.+?)\)",
        src,
        re.DOTALL | re.IGNORECASE,
    )
    assert match, "Could not find CREATE TABLE missions statement in db.py"

    cols_block = match.group(1)
    legacy_cols = re.findall(r"legacy_pre_\w+", cols_block)
    assert not legacy_cols, (
        f"CREATE TABLE missions in db.py still defines columns: {legacy_cols}"
    )
