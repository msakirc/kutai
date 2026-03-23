# load_manager.py
"""
GPU load mode manager. Four modes:
  full    — use all available GPU/RAM (default)
  heavy   — cap at ~90% VRAM; slight headroom for OS/desktop
  shared  — cap at ~50% VRAM; prefer smaller models or cloud
  minimal — zero local GPU; all inference offloaded to cloud

Mode is persisted in the DB and applied to gpu_scheduler on change.
Auto-detection can dynamically switch modes based on external GPU usage.
"""

import asyncio
import os
import time
from src.infra.logging_config import get_logger
from src.infra.db import get_db

logger = get_logger("infra.load_manager")

LOAD_MODES = ("full", "heavy", "shared", "minimal")

VRAM_BUDGETS: dict[str, float] = {
    "full": 1.0,
    "heavy": 0.9,
    "shared": 0.5,
    "minimal": 0.0,
}

DESCRIPTIONS: dict[str, str] = {
    "full":    "Full GPU — all local capacity available",
    "heavy":   "Heavy GPU — 90% VRAM cap, slight headroom for OS/desktop",
    "shared":  "Shared GPU — 50% VRAM cap, prefer cloud for heavy tasks",
    "minimal": "Minimal GPU — local inference disabled, cloud only",
}

# Ordered from most restrictive to least — index used for comparisons
MODE_ORDER = ("minimal", "shared", "heavy", "full")

_current_mode: str = "full"
_auto_managed: bool = True  # start with auto-management enabled


async def _init_table():
    db = await get_db()
    await db.execute("""
        CREATE TABLE IF NOT EXISTS load_mode (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            mode TEXT NOT NULL DEFAULT 'full',
            auto_managed INTEGER NOT NULL DEFAULT 1,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    await db.execute(
        "INSERT OR IGNORE INTO load_mode (id, mode, auto_managed) VALUES (1, 'full', 1)"
    )
    await db.commit()


async def get_load_mode() -> str:
    """Return the current load mode."""
    global _current_mode, _auto_managed
    try:
        await _init_table()
        db = await get_db()
        cur = await db.execute("SELECT mode, auto_managed FROM load_mode WHERE id = 1")
        row = await cur.fetchone()
        if row:
            _current_mode = row["mode"] if row["mode"] in LOAD_MODES else "full"
            _auto_managed = bool(row["auto_managed"])
    except Exception as e:
        logger.warning("could not read load_mode from DB", error=str(e))
    return _current_mode


async def set_load_mode(mode: str, source: str = "user") -> str:
    """Set the load mode. source='user' disables auto-management, source='auto' preserves it."""
    global _current_mode, _auto_managed
    if mode not in LOAD_MODES:
        return f"Unknown mode '{mode}'. Choose: {', '.join(LOAD_MODES)}"

    prev = _current_mode
    _current_mode = mode

    if source == "user":
        _auto_managed = False  # user took manual control

    try:
        await _init_table()
        db = await get_db()
        await db.execute(
            "UPDATE load_mode SET mode = ?, auto_managed = ?, updated_at = CURRENT_TIMESTAMP WHERE id = 1",
            (mode, int(_auto_managed))
        )
        await db.commit()
    except Exception as e:
        logger.error("failed to persist load_mode", error=str(e))

    logger.info("load mode changed", prev=prev, new=mode, source=source)

    # Notify runtime state
    try:
        from src.infra.runtime_state import runtime_state
        runtime_state["load_mode"] = mode
    except Exception:
        pass

    return f"Load mode set to *{mode}*: {DESCRIPTIONS[mode]}"


async def enable_auto_management():
    """Re-enable auto-management (e.g. from /load auto)."""
    global _auto_managed
    _auto_managed = True
    try:
        await _init_table()
        db = await get_db()
        await db.execute(
            "UPDATE load_mode SET auto_managed = 1, updated_at = CURRENT_TIMESTAMP WHERE id = 1"
        )
        await db.commit()
    except Exception as e:
        logger.error("failed to enable auto_managed", error=str(e))
    logger.info("auto-management enabled")


def is_local_inference_allowed() -> bool:
    """Returns False when in 'minimal' mode."""
    return _current_mode != "minimal"


def is_auto_managed() -> bool:
    """Whether mode is being auto-managed (vs manually set by user)."""
    return _auto_managed


def get_vram_budget_fraction() -> float:
    """Fraction of VRAM to use: 1.0 (full), 0.9 (heavy), 0.5 (shared), 0.0 (minimal)."""
    return VRAM_BUDGETS.get(_current_mode, 1.0)


def suggest_mode_for_external_usage(external_vram_fraction: float) -> str:
    """Map external VRAM fraction to the appropriate load mode.

    The more VRAM external processes use, the more we should back off:
      external < 10%  → full   (negligible external usage)
      external < 30%  → heavy  (light external usage, keep 90%)
      external < 60%  → shared (moderate, give them room)
      external >= 60% → minimal (heavy external load like gaming)
    """
    if external_vram_fraction < 0.10:
        return "full"
    elif external_vram_fraction < 0.30:
        return "heavy"
    elif external_vram_fraction < 0.60:
        return "shared"
    else:
        return "minimal"


def _mode_index(mode: str) -> int:
    """Return mode severity index (higher = less restrictive)."""
    try:
        return MODE_ORDER.index(mode)
    except ValueError:
        return 3  # default to full


# ─── Auto-detect loop ───────────────────────────────────────────

_DETECT_INTERVAL = int(os.getenv("GPU_DETECT_INTERVAL", "30"))
_UPGRADE_DELAY = int(os.getenv("GPU_UPGRADE_DELAY", "300"))  # 5 minutes


async def run_gpu_autodetect_loop(notify_fn=None):
    """Background loop: check external GPU usage every 30s, auto-switch mode.

    - Downgrade immediately when external usage increases
    - Upgrade only after sustained decrease for UPGRADE_DELAY seconds
    - Respects manual overrides (stops auto-managing when user sets mode)
    """
    from src.models.gpu_monitor import get_gpu_monitor

    monitor = get_gpu_monitor()
    upgrade_candidate: str | None = None
    upgrade_stable_since: float = 0.0

    logger.info("GPU auto-detect loop started",
                interval=_DETECT_INTERVAL, upgrade_delay=_UPGRADE_DELAY)

    while True:
        try:
            await asyncio.sleep(_DETECT_INTERVAL)

            if not _auto_managed:
                # User has manual control — skip detection
                upgrade_candidate = None
                continue

            ext = monitor.detect_external_gpu_usage()
            suggested = suggest_mode_for_external_usage(ext.external_vram_fraction)
            current = _current_mode
            now = time.time()

            current_idx = _mode_index(current)
            suggested_idx = _mode_index(suggested)

            if suggested_idx < current_idx:
                # ── Downgrade: act immediately ──
                await set_load_mode(suggested, source="auto")
                upgrade_candidate = None
                upgrade_stable_since = 0.0
                msg = (
                    f"🔻 *GPU auto-detect*: external usage at {ext.external_vram_fraction:.0%} "
                    f"({ext.external_vram_mb}MB, {ext.external_process_count} processes)\n"
                    f"Switched *{current}* → *{suggested}*\n"
                    f"Use `/load full` to override or `/load auto` to re-enable."
                )
                logger.info("auto-downgrade", prev=current, new=suggested,
                            ext_frac=ext.external_vram_fraction,
                            ext_mb=ext.external_vram_mb)
                if notify_fn:
                    try:
                        await notify_fn(msg)
                    except Exception:
                        pass

            elif suggested_idx > current_idx:
                # ── Upgrade: wait for sustained decrease ──
                if upgrade_candidate != suggested:
                    upgrade_candidate = suggested
                    upgrade_stable_since = now
                elif now - upgrade_stable_since >= _UPGRADE_DELAY:
                    # Sustained for long enough — upgrade
                    await set_load_mode(suggested, source="auto")
                    msg = (
                        f"🔺 *GPU auto-detect*: external usage dropped to {ext.external_vram_fraction:.0%} "
                        f"({ext.external_vram_mb}MB)\n"
                        f"Upgraded *{current}* → *{suggested}*"
                    )
                    logger.info("auto-upgrade", prev=current, new=suggested,
                                ext_frac=ext.external_vram_fraction)
                    upgrade_candidate = None
                    upgrade_stable_since = 0.0
                    if notify_fn:
                        try:
                            await notify_fn(msg)
                        except Exception:
                            pass
            else:
                # Same mode — reset upgrade tracking
                upgrade_candidate = None
                upgrade_stable_since = 0.0

        except asyncio.CancelledError:
            logger.info("GPU auto-detect loop cancelled")
            break
        except Exception as e:
            logger.debug("GPU auto-detect error", error=str(e))
