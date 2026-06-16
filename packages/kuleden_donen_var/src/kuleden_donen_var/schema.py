"""kdv_state schema, owned by kuleden_donen_var. SINGLE source of truth for the
KDV persistence table's DDL.

Phase B §4 relocation: the DDL was moved here from packages/db/src/dabidabi
(init_db) so the engine no longer owns a domain table it knows nothing about.
``create_kdv_schema`` is registered with the engine via
``dabidabi.register_schema`` and runs inside init_db's registration loop.
``CREATE ... IF NOT EXISTS`` keeps the overlap safe while the table still lives
in core.db (no file-split yet).

Registration is a side effect of importing this module; kuleden_donen_var's
package __init__ imports it so that any process importing the package (or run.py
/ the cold-init CLIs, which import it explicitly) registers the schema before
init_db() runs.
"""
import dabidabi

KDV_DDL = [
    # One row per (scope, scope_key). scope in {"model","provider","breaker",
    # "outcomes","meta"}. snapshot_json holds the dict from snapshot_state();
    # last_persisted is unix epoch (loader drops rows older than stale_hours).
    """
        CREATE TABLE IF NOT EXISTS kdv_state (
            scope TEXT NOT NULL,
            scope_key TEXT NOT NULL,
            snapshot_json TEXT NOT NULL,
            last_persisted REAL NOT NULL,
            PRIMARY KEY (scope, scope_key)
        )
    """,
]


async def create_kdv_schema(db) -> None:
    """Async executor (engine registration path). db = aiosqlite connection."""
    for sql in KDV_DDL:
        await db.execute(sql)


dabidabi.register_schema("kuleden_donen_var_kdv", create_kdv_schema)
