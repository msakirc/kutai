# load_manager.py
"""
GPU load mode manager. Three modes:
  full    — use all available GPU/RAM (default)
  shared  — cap at ~50% VRAM; prefer smaller models or cloud
  minimal — zero local GPU; all inference offloaded to cloud

Mode is persisted in the DB and applied to gpu_scheduler on change.
"""

import asyncio
from src.infra.logging_config import get_logger
from src.infra.db import get_db

logger = get_logger("infra.load_manager")

LOAD_MODES = ("full", "shared", "minimal")
_current_mode: str = "full"


async def _init_table():
    async with get_db() as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS load_mode (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                mode TEXT NOT NULL DEFAULT 'full',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.execute(
            "INSERT OR IGNORE INTO load_mode (id, mode) VALUES (1, 'full')"
        )
        await db.commit()


async def get_load_mode() -> str:
    """Return the current load mode ('full', 'shared', or 'minimal')."""
    global _current_mode
    try:
        await _init_table()
        async with get_db() as db:
            cur = await db.execute("SELECT mode FROM load_mode WHERE id = 1")
            row = await cur.fetchone()
        _current_mode = row["mode"] if row else "full"
    except Exception as e:
        logger.warning("could not read load_mode from DB", error=str(e))
    return _current_mode


async def set_load_mode(mode: str) -> str:
    """Set the load mode. Returns confirmation string."""
    global _current_mode
    if mode not in LOAD_MODES:
        return f"Unknown mode '{mode}'. Choose: {', '.join(LOAD_MODES)}"
    prev = _current_mode
    _current_mode = mode
    try:
        await _init_table()
        async with get_db() as db:
            await db.execute(
                "UPDATE load_mode SET mode = ?, updated_at = CURRENT_TIMESTAMP WHERE id = 1",
                (mode,)
            )
            await db.commit()
    except Exception as e:
        logger.error("failed to persist load_mode", error=str(e))

    logger.info("load mode changed", prev=prev, new=mode)

    # Notify runtime state
    try:
        from src.infra.runtime_state import runtime_state
        runtime_state["load_mode"] = mode
    except Exception:
        pass

    descriptions = {
        "full":    "Full GPU — all local capacity available",
        "shared":  "Shared GPU — 50% VRAM cap, prefer cloud for heavy tasks",
        "minimal": "Minimal GPU — local inference disabled, cloud only",
    }
    return f"Load mode set to *{mode}*: {descriptions[mode]}"


def is_local_inference_allowed() -> bool:
    """Returns False when in 'minimal' mode."""
    return _current_mode != "minimal"


def get_vram_budget_fraction() -> float:
    """Fraction of VRAM to use: 1.0 (full), 0.5 (shared), 0.0 (minimal)."""
    return {"full": 1.0, "shared": 0.5, "minimal": 0.0}.get(_current_mode, 1.0)
