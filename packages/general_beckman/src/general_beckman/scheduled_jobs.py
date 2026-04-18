"""Scheduled / proactive jobs that don't fit the task-orchestration main loop.

Todos reminders (every 2h), price watch re-scrape (daily), API discovery,
daily digest. Orchestrator calls tick_* methods on its main loop heartbeat.

Phase 2b target: each tick becomes a scheduled CronEvent that task master
accepts as intake, producing mechanical tasks that go through the dispatch
loop like anything else.
"""

import asyncio
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

from src.infra.logging_config import get_logger
from src.infra.times import utc_now, db_now, to_db, from_db
from src.infra.db import (
    get_due_scheduled_tasks, update_scheduled_task, add_task,
    get_active_missions, get_daily_stats,
)

logger = get_logger("app.scheduled_jobs")

_VALID_SUGGESTION_AGENTS = {"researcher", "shopping_advisor", "assistant", "coder"}

_BENCHMARK_FRESHNESS_HOURS = 24
_benchmark_refresh_in_flight = False


def _benchmark_cache_dir() -> Path:
    """Return the benchmark cache dir. Monkeypatched in tests."""
    return Path(".benchmark_cache")


def _benchmark_refresh_impl() -> tuple[int, int]:
    """Sync refresh via BenchmarkFetcher. Returns (before_count, after_count).
    Monkeypatched in tests.
    """
    from src.models.benchmark.benchmark_fetcher import BenchmarkFetcher

    fetcher = BenchmarkFetcher()
    before = len(fetcher.fetch_all_bulk())
    fetcher.refresh_cache()
    after = len(fetcher.fetch_all_bulk())
    return before, after


def _benchmark_cache_is_fresh() -> bool:
    cache_dir = _benchmark_cache_dir()
    if not cache_dir.exists():
        return False
    bulks = list(cache_dir.glob("_bulk_*.json"))
    if not bulks:
        return False
    newest = max(b.stat().st_mtime for b in bulks)
    age_hours = (time.time() - newest) / 3600
    return age_hours < _BENCHMARK_FRESHNESS_HOURS


def _parse_todo_suggestions(raw: str, todo_count: int) -> list[dict]:
    """Parse LLM response into per-todo suggestions.

    Returns a list of length todo_count. Each element is:
      {"suggestion": str | None, "agent": str}

    Lenient parser: handles N. or N) prefixes, optional [agent] tags,
    markdown bold around tags, extra whitespace.
    """
    results = [{"suggestion": None, "agent": "researcher"} for _ in range(todo_count)]
    if not raw or not raw.strip():
        return results

    # Build a map: line_number → parsed content
    parsed_lines: dict[int, tuple[str, str]] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        # Match: optional whitespace, number, . or ), rest
        m = re.match(r"(\d+)\s*[.)]\s*(.+)", line)
        if not m:
            continue
        idx = int(m.group(1)) - 1  # 0-based
        text = m.group(2).strip()

        # Skip "no suggestion" variants
        if re.match(r"(?i)no\s+suggestion|n/a|none|-$", text):
            continue
        if len(text) < 6:
            continue

        # Extract [agent_type] — handle optional markdown bold: **[agent]**
        text = re.sub(r"^\*{1,2}\[", "[", text)
        text = re.sub(r"\]\*{1,2}", "]", text)
        agent_m = re.match(r"\[(\w+)\]\s*(.+)", text)
        if agent_m and agent_m.group(1).lower() in _VALID_SUGGESTION_AGENTS:
            agent = agent_m.group(1).lower()
            suggestion = agent_m.group(2).strip()
        else:
            agent = "researcher"
            suggestion = text

        if 0 <= idx < todo_count:
            parsed_lines[idx] = (suggestion, agent)

    for idx, (suggestion, agent) in parsed_lines.items():
        results[idx] = {"suggestion": suggestion, "agent": agent}

    return results


class ScheduledJobs:
    """Container for all proactive / cron-driven jobs.

    Constructed once by Orchestrator and called via tick_* coroutines on
    the main loop heartbeat.
    """

    def __init__(self, telegram=None, **deps):
        self.telegram = telegram
        self._last_todo_run = 0.0
        self._last_api_discovery_run = 0.0
        self._last_digest_date = None
        self._last_price_watch_date = None
        # Store extra deps the ticks need (e.g. orchestrator ref, config)
        for k, v in deps.items():
            setattr(self, k, v)

    # ─── Public tick entry points ──────────────────────────────────────────

    async def tick_todos(self):
        """Called by orchestrator check_scheduled_tasks for todo_reminder entries."""
        await self._start_todo_suggestions()

    async def tick_api_discovery(self):
        """Called by orchestrator check_scheduled_tasks every cycle."""
        await self._check_api_discovery()

    async def tick_digest(self):
        """Called by orchestrator run_loop at briefing hour."""
        await self.daily_digest()

    async def tick_price_watches(self):
        """Called by orchestrator check_scheduled_tasks for price_watch_check entries."""
        from src.app.price_watch_checker import check_price_watches
        summary = await check_price_watches(self.telegram)
        logger.info(f"[Scheduler] Price watch check complete: {summary}")

    async def tick_benchmark_refresh(self):
        """Refresh benchmark cache when older than 24h. Fires on heartbeat."""
        global _benchmark_refresh_in_flight

        if _benchmark_refresh_in_flight:
            logger.debug("benchmark refresh already in flight — noop")
            return

        if _benchmark_cache_is_fresh():
            logger.debug("benchmark cache fresh — skip refresh")
            return

        _benchmark_refresh_in_flight = True
        try:
            before, after = await asyncio.to_thread(_benchmark_refresh_impl)
            delta = after - before
            logger.info(
                "benchmark refresh: matched %d→%d (%+d)",
                before, after, delta,
            )
        except Exception as exc:
            logger.warning("benchmark refresh failed: %s", exc, exc_info=True)
        finally:
            _benchmark_refresh_in_flight = False

    # ─── check_scheduled_tasks (full port from Orchestrator) ──────────────

    async def check_scheduled_tasks(self):
        """Check for due scheduled tasks and create task instances.

        Runs every 60s alongside the main loop.
        """
        try:
            # ── API discovery (8:30am daily, catch-up if missed) ──
            try:
                await self._check_api_discovery()
            except Exception as exc:
                logger.debug("API discovery check failed: %s", exc)

            due = await get_due_scheduled_tasks()
            if not due:
                return

            for sched in due:
                sched_id = sched["id"]
                title = sched["title"]
                logger.info(
                    f"[Scheduler] Triggering scheduled task #{sched_id}: "
                    f"'{title}'"
                )
                import json as _json
                sched_ctx = (
                    _json.loads(sched.get("context", "{}"))
                    if isinstance(sched.get("context"), str)
                    else sched.get("context", {})
                )

                # Special handling: todo reminders create suggestion tasks first
                if sched_ctx.get("type") == "todo_reminder":
                    try:
                        await self._start_todo_suggestions()
                    except Exception as e:
                        logger.error(f"[Scheduler] Todo suggestion creation failed: {e}")
                        # Fallback: send reminder without suggestions
                        try:
                            from src.app.reminders import send_todo_reminder
                            if self.telegram:
                                await send_todo_reminder(self.telegram)
                        except Exception:
                            pass
                    # Update last_run / next_run and skip task creation
                    now = utc_now()
                    next_run = self._compute_next_run(
                        sched.get("cron_expression", "0 * * * *"), now
                    )
                    await update_scheduled_task(
                        sched_id,
                        last_run=to_db(now),
                        next_run=to_db(next_run) if next_run else None,
                    )
                    continue

                # Special handling: one-shot reminders — send directly, then disable
                if sched_ctx.get("one_shot"):
                    reminder_text = sched_ctx.get("reminder_text", title)
                    try:
                        if self.telegram:
                            chat_id = sched_ctx.get("chat_id")
                            if chat_id:
                                await self.telegram.app.bot.send_message(
                                    chat_id=int(chat_id),
                                    text=f"⏰ *Hatırlatma*\n\n{reminder_text}",
                                    parse_mode="Markdown",
                                )
                            else:
                                await self.telegram.send_notification(
                                    f"⏰ *Hatırlatma*\n\n{reminder_text}"
                                )
                        logger.info(f"[Scheduler] One-shot reminder sent: {title}")
                    except Exception as e:
                        logger.error(f"[Scheduler] Reminder send failed: {e}")
                    # Disable — no next_run for one-shot
                    await update_scheduled_task(
                        sched_id,
                        last_run=db_now(),
                        next_run=None,
                        enabled=False,
                    )
                    continue

                # Special handling: price watch checker
                if sched_ctx.get("type") == "price_watch_check":
                    try:
                        from src.app.price_watch_checker import check_price_watches
                        summary = await check_price_watches(self.telegram)
                        logger.info(
                            f"[Scheduler] Price watch check complete: {summary}"
                        )
                    except Exception as e:
                        logger.error(f"[Scheduler] Price watch check failed: {e}")
                    now = utc_now()
                    next_run = self._compute_next_run(
                        sched.get("cron_expression", "0 * * * *"), now
                    )
                    await update_scheduled_task(
                        sched_id,
                        last_run=to_db(now),
                        next_run=to_db(next_run) if next_run else None,
                    )
                    continue

                task_id = await add_task(
                    title=title,
                    description=sched.get("description", ""),
                    agent_type=sched.get("agent_type", "executor"),
                    tier=sched.get("tier", "cheap"),
                    mission_id=sched_ctx.get("mission_id"),
                    context=sched_ctx,
                )
                if task_id:
                    logger.info(
                        f"[Scheduler] Created task #{task_id} from "
                        f"schedule #{sched_id}"
                    )

                # Update last_run and compute next_run
                now = utc_now()
                next_run = self._compute_next_run(
                    sched.get("cron_expression", "0 * * * *"), now
                )
                await update_scheduled_task(
                    sched_id,
                    last_run=to_db(now),
                    next_run=to_db(next_run) if next_run else None,
                )

        except Exception as e:
            logger.error(f"[Scheduler] Error checking schedules: {e}")

    # ─── Private helpers (verbatim port from Orchestrator) ────────────────

    async def _check_api_discovery(self):
        """Run API discovery daily at 8:30am, with catch-up if missed."""
        now = utc_now()

        # Check DB for last discovery time (persists across restarts)
        last_discovery = None
        try:
            from src.infra.db import get_db
            db = await get_db()
            cur = await db.execute(
                "SELECT MAX(timestamp) FROM smart_search_log WHERE source = 'discovery'"
            )
            row = await cur.fetchone()
            if row and row[0]:
                last_discovery = from_db(row[0])
        except Exception:
            pass

        # Already ran today? Skip.
        if last_discovery and (now - last_discovery).total_seconds() < 86400:
            return

        from src.infra.times import turkey_now
        tr_now = turkey_now()
        in_window = tr_now.hour == 8 and 25 <= tr_now.minute <= 35
        overdue = (
            last_discovery is None
            or (now - last_discovery).total_seconds() > 36 * 3600
        )

        if not in_window and not overdue:
            return

        logger.info("Starting API discovery run")
        try:
            from src.tools.free_apis import discover_new_apis, build_keyword_index, seed_category_patterns
            from src.infra.db import log_smart_search

            new_count = await discover_new_apis("all")
            await build_keyword_index()
            await seed_category_patterns()

            # Record discovery run in DB so we don't re-run on restart
            await log_smart_search("discovery", layer=0, source="discovery", success=True, response_ms=0)

            if new_count > 0:
                logger.info("API discovery complete: %d new APIs", new_count)
                if hasattr(self, "_morning_brief_extras"):
                    self._morning_brief_extras.append(
                        f"Discovered {new_count} new APIs/MCP tools."
                    )
                if new_count >= 5 and self.telegram:
                    await self.telegram.send_notification(
                        f"API discovery: {new_count} new APIs added to registry."
                    )
            else:
                logger.info("API discovery complete: no new APIs found")
        except Exception as exc:
            logger.warning("API discovery failed: %s", exc)

    async def _start_todo_suggestions(self):
        """Generate AI suggestions for pending todos that don't have one yet.

        Only queries LLM for todos where suggestion IS NULL and suggestion_at IS NULL
        (never attempted). Todos with suggestion_at set but suggestion NULL were
        previously attempted and failed — skip them.
        """
        from src.infra.db import get_todos, update_todo
        from src.app.reminders import send_todo_reminder

        todos = await get_todos(status="pending")
        if not todos:
            return

        # Filter to todos that need suggestions (never attempted)
        need_suggestions = [
            t for t in todos
            if t.get("suggestion") is None and t.get("suggestion_at") is None
        ]

        if need_suggestions:
            try:
                await self._generate_suggestions(need_suggestions)
            except Exception as e:
                logger.warning(f"[Todo] Suggestion generation failed: {e}")

        # Always send the reminder (suggestions are read from DB by reminders.py)
        if self.telegram:
            await send_todo_reminder(self.telegram)

    async def _generate_suggestions(self, todos: list[dict]):
        """Call LLM to generate suggestions for given todos, persist results."""
        from src.infra.db import update_todo

        todo_lines = "\n".join(
            f"{i+1}. {t['title']}"
            + (f" (priority: {t.get('priority', 'normal')})" if t.get("priority") != "normal" else "")
            + (f" (notes: {t['description'][:80]})" if t.get("description") else "")
            for i, t in enumerate(todos[:10])
        )
        prompt = (
            f"The user has {len(todos)} pending todo item(s):\n\n"
            f"{todo_lines}\n\n"
            f"For each item, suggest ONE concrete, actionable way an AI assistant could help "
            f"(e.g. search, compare prices, book, remind, draft a message). "
            f"Be creative — even mundane tasks like 'buy milk' could mean price comparison or online ordering. "
            f"If you genuinely cannot help with an item, write 'no suggestion'.\n\n"
            f"Also pick the best agent type for each suggestion:\n"
            f"  researcher — web search, information gathering, fact-checking\n"
            f"  shopping_advisor — product search, price comparison, deal finding\n"
            f"  assistant — drafting messages, reminders, general help\n"
            f"  coder — writing code, scripts, technical tasks\n\n"
            f"Reply ONLY with a numbered list. Format: NUMBER. [agent_type] suggestion text\n"
            f"Example: 1. [researcher] Search for nearby tire shops and compare prices.\n"
            f"No preamble, no extra commentary."
        )

        now_str = db_now()

        try:
            from src.core.llm_dispatcher import get_dispatcher, CallCategory

            dispatcher = get_dispatcher()
            response = await asyncio.wait_for(
                dispatcher.request(
                    category=CallCategory.OVERHEAD,
                    task="assistant",
                    difficulty=2,
                    messages=[{"role": "user", "content": prompt}],
                    estimated_input_tokens=400,
                    estimated_output_tokens=150,
                    prefer_speed=True,
                    priority=2,
                ),
                timeout=45,
            )
            raw = (response.get("content") or "").strip()
            logger.info(f"[Todo] Suggestion LLM response ({len(raw)} chars)")

            parsed = _parse_todo_suggestions(raw, len(todos[:10]))

            for i, todo in enumerate(todos[:10]):
                entry = parsed[i]
                if entry["suggestion"]:
                    await update_todo(
                        todo["id"],
                        suggestion=entry["suggestion"],
                        suggestion_agent=entry["agent"],
                        suggestion_at=now_str,
                    )
                else:
                    # Mark as attempted-but-failed so we don't retry
                    await update_todo(todo["id"], suggestion_at=now_str)

            generated = sum(1 for p in parsed if p["suggestion"])
            logger.info(f"[Todo] Generated {generated}/{len(todos[:10])} suggestions")

        except asyncio.TimeoutError:
            logger.warning("[Todo] Suggestion LLM call timed out — marking todos as attempted")
            for todo in todos[:10]:
                await update_todo(todo["id"], suggestion_at=now_str)
        except Exception as exc:
            logger.warning(f"[Todo] Suggestion LLM call failed: {exc} — marking todos as attempted")
            for todo in todos[:10]:
                await update_todo(todo["id"], suggestion_at=now_str)

    async def daily_digest(self):
        """Phase 14.1: Enhanced morning briefing with overnight results and system health."""
        stats = await get_daily_stats()
        missions = await get_active_missions()

        missions_text = "\n".join(f"  - {g['title']}" for g in missions[:5]) or "  None"

        # Gather additional intelligence
        pending_approvals = 0
        try:
            from src.infra.db import get_db
            db = await get_db()
            cursor = await db.execute(
                "SELECT COUNT(*) FROM tasks WHERE status = 'awaiting_approval'"
            )
            row = await cursor.fetchone()
            pending_approvals = row[0] if row else 0
        except Exception:
            pass

        # System health indicators
        health_lines = []
        try:
            from src.infra.load_manager import get_load_mode
            mode = await get_load_mode()
            health_lines.append(f"  GPU: {mode} mode")
        except Exception:
            pass
        try:
            from src.infra.metrics import get_all_counters
            counters = get_all_counters()
            queue = int(counters.get("queue_depth", 0))
            health_lines.append(f"  Queue: {queue} tasks")
        except Exception:
            pass

        health_text = "\n".join(health_lines) if health_lines else "  All systems normal"
        approval_line = f"\n*Pending approvals:* {pending_approvals}" if pending_approvals else ""

        await self.telegram.send_notification(
            f"*Morning Briefing*\n\n"
            f"*Tasks (last 24h):*\n"
            f"  Completed: {stats['completed']}\n"
            f"  Pending: {stats['pending']}\n"
            f"  Processing: {stats['processing']}\n"
            f"  Failed: {stats['failed']}\n"
            f"  Cost: ${stats['today_cost']:.4f}\n\n"
            f"*Active missions:*\n{missions_text}"
            f"{approval_line}\n\n"
            f"*System health:*\n{health_text}"
        )

    @staticmethod
    def _compute_next_run(
        cron_expr: str, after: datetime
    ) -> datetime | None:
        """Simple cron parser supporting: minute hour day month weekday.

        Examples: "0 * * * *" (hourly), "30 9 * * *" (daily 9:30),
                  "0 0 * * 1" (Monday midnight).
        Returns the next datetime after *after*, or None on parse failure.
        """
        try:
            parts = cron_expr.strip().split()
            if len(parts) != 5:
                return None

            minute, hour, day, month, weekday = parts

            # Simple: advance by fixed intervals for common patterns
            if minute != "*" and hour == "*":
                # Every hour at minute M
                m = int(minute)
                candidate = after.replace(
                    minute=m, second=0, microsecond=0
                )
                if candidate <= after:
                    candidate += timedelta(hours=1)
                return candidate

            if minute != "*" and hour != "*":
                m = int(minute)
                # Handle comma-separated hours (e.g., "9,11,13,15,17,19,21")
                if "," in hour:
                    hours = sorted(int(h) for h in hour.split(","))
                    # Find next hour that's still in the future today
                    for h in hours:
                        candidate = after.replace(
                            hour=h, minute=m, second=0, microsecond=0
                        )
                        if candidate > after:
                            return candidate
                    # All hours passed today — first hour tomorrow
                    candidate = after.replace(
                        hour=hours[0], minute=m, second=0, microsecond=0
                    )
                    return candidate + timedelta(days=1)
                # Single hour: daily at H:M
                h = int(hour)
                candidate = after.replace(
                    hour=h, minute=m, second=0, microsecond=0
                )
                if candidate <= after:
                    candidate += timedelta(days=1)
                return candidate

            # Fallback: every hour from now
            return after + timedelta(hours=1)
        except Exception:
            return None
