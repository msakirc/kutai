# periodic_checks.py — timestamp-gated background jobs the pump fires.
#
# Extracted from the orchestrator (P5, 2026-06-07). Each check is a
# self-gating, failure-swallowing job: the pump calls ``run_due()`` once per
# tick and every individual check decides whether it is due. A check must
# NEVER raise out of run_due — a periodic-job failure must not disturb the
# dispatch pump.
#
# The orchestrator imports ZERO from yalayut/founder_actions at module load —
# the mechanical executors own those imports; the enqueue-a-mechanical-task
# checks here pull them lazily. The cron_seed cadence rows are the
# restart-survivable backstop; these in-process checks give a finer cadence
# and fire promptly after boot.
import time

from src.infra.logging_config import get_logger

logger = get_logger("core.orchestrator")


class PeriodicChecks:
    """Owns the orchestrator's timestamp-gated background jobs + their state."""

    _YALAYUT_DISCOVERY_INTERVAL_S: float = 86400.0   # 24h
    _SOURCE_SCOUT_INTERVAL_S: float = 86400.0        # 24h
    _FOUNDER_SWEEP_INTERVAL_S: float = 60.0          # ~once per minute

    def __init__(self) -> None:
        # 0.0 → first pump tick after boot fires the discovery/scout checks
        # immediately, then they self-gate to 24h.
        self._last_yalayut_discovery: float = 0.0
        self._last_source_scout: float = 0.0
        # Founder sweep starts gated (boot already runs startup_recovery).
        self._last_founder_sweep: float = time.time()

    async def run_due(self) -> None:
        """Run every check that is due. Order is immaterial; each self-gates.

        Each check is individually guarded so one failing job never blocks
        the others or the pump.
        """
        for _check in (
            self.check_founder_sweep,
            self.check_mcp_idle_sweep,
            self.check_yalayut_discovery,
            self.check_source_scout,
        ):
            try:
                await _check()
            except Exception as e:  # pragma: no cover - defensive
                logger.debug("periodic check %s skipped: %s", _check.__name__, e)

    # ─── Founder mission-unblock sweep (Z6 T1E) ──────────────────────────
    async def check_founder_sweep(self) -> None:
        """Backstop sweep that unblocks missions whose founder_actions resolved.

        Founder may resolve actions via the Yaşar Usta bot or external tooling
        without the per-resolve hook firing in this process — this sweep is the
        backstop. One indexed SELECT + zero-to-N UPDATEs; cheap.
        """
        now = time.time()
        if now - self._last_founder_sweep < self._FOUNDER_SWEEP_INTERVAL_S:
            return
        self._last_founder_sweep = now
        try:
            import src.founder_actions as _fa
            n = await _fa.sweep_unblock_all()
            if n > 0:
                logger.info("z6 lifecycle sweep: unblocked %d mission(s)", n)
        except Exception as _e:
            logger.debug(f"z6 sweep skipped: {_e}")

    # ─── MCP idle sweep (Yalayut Phase 3) ────────────────────────────────
    async def check_mcp_idle_sweep(self) -> None:
        """Shut down idle MCP servers (lazy-start companion). Never starts one.

        Runs every tick (cheap select); only kills servers idle past their
        idle_timeout_s.
        """
        try:
            from yalayut.mcp_manager import get_manager
            killed = await get_manager().sweep_idle()
            if killed:
                logger.info("mcp idle sweep", killed=killed)
        except Exception:
            # Sweep failures must never disturb the pump.
            pass

    # ─── Yalayut Phase 4 periodic checks ─────────────────────────────────
    async def check_yalayut_discovery(self) -> None:
        """Enqueue a yalayut daily-discovery mechanical task when due."""
        now = time.time()
        if now - self._last_yalayut_discovery < self._YALAYUT_DISCOVERY_INTERVAL_S:
            return
        self._last_yalayut_discovery = now
        try:
            import general_beckman
            await general_beckman.enqueue(
                {
                    "agent_type": "mechanical",
                    "title": "Yalayut daily discovery",
                    "context": {
                        "executor": "mechanical",
                        "payload": {"action": "yalayut_discovery",
                                    "mode": "daily"},
                    },
                },
                lane="oneshot",
            )
            logger.info("enqueued yalayut daily discovery task")
        except Exception as e:
            logger.warning("yalayut discovery enqueue failed: %s", e)

    async def check_source_scout(self) -> None:
        """Enqueue a yalayut source-scout mechanical task when due."""
        now = time.time()
        if now - self._last_source_scout < self._SOURCE_SCOUT_INTERVAL_S:
            return
        self._last_source_scout = now
        try:
            import general_beckman
            await general_beckman.enqueue(
                {
                    "agent_type": "mechanical",
                    "title": "Yalayut source scout",
                    "context": {
                        "executor": "mechanical",
                        "payload": {"action": "source_scout"},
                    },
                },
                lane="oneshot",
            )
            logger.info("enqueued yalayut source-scout task")
        except Exception as e:
            logger.warning("source-scout enqueue failed: %s", e)
