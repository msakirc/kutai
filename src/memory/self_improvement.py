# memory/self_improvement.py
"""
Phase 13.4 — Self-Improvement Proposals

Analyzes accumulated feedback, failure patterns, and model performance
to generate actionable improvement proposals. Can run on-demand (/improve)
or as a weekly scheduled task.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta

from src.infra.logging_config import get_logger
from src.infra.times import utc_now
from src.infra.db import get_db

logger = get_logger("memory.self_improvement")


async def analyze_and_propose() -> list[dict]:
    """
    Analyze recent system performance and generate improvement proposals.

    Returns a list of proposal dicts:
      [{"category": str, "title": str, "detail": str, "priority": int, "action": str}]
    """
    proposals: list[dict] = []
    db = await get_db()
    cutoff = (utc_now() - timedelta(days=7)).isoformat()

    # ── 1. Agent failure rate analysis ──
    try:
        cursor = await db.execute("""
            SELECT agent_type, COUNT(*) as total,
                   SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                   SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed
            FROM tasks
            WHERE created_at > ? AND agent_type IS NOT NULL
            GROUP BY agent_type
            HAVING total >= 5
        """, (cutoff,))
        rows = await cursor.fetchall()
        for row in rows:
            agent, total, completed, failed = row
            if total > 0 and failed / total > 0.3:
                rate = failed / total * 100
                proposals.append({
                    "category": "agent_reliability",
                    "title": f"High failure rate for '{agent}' agent ({rate:.0f}%)",
                    "detail": (
                        f"Agent '{agent}' failed {failed}/{total} tasks in the last 7 days. "
                        f"Review system prompt, reduce complexity per task, or "
                        f"consider splitting into sub-agents."
                    ),
                    "priority": 8 if rate > 50 else 5,
                    "action": f"review_prompt:{agent}",
                })
    except Exception as e:
        logger.debug(f"Agent failure analysis failed: {e}")

    # ── 2. Negative feedback clustering ──
    try:
        cursor = await db.execute("""
            SELECT agent_type, COUNT(*) as bad_count, GROUP_CONCAT(reason, ' | ')
            FROM task_feedback
            WHERE score < 0 AND created_at > ?
            GROUP BY agent_type
            HAVING bad_count >= 3
        """, (cutoff,))
        rows = await cursor.fetchall()
        for row in rows:
            agent, count, reasons = row
            reasons_str = (reasons or "no reasons given")[:300]
            proposals.append({
                "category": "quality",
                "title": f"Repeated negative feedback for '{agent}' ({count} complaints)",
                "detail": (
                    f"Users gave negative feedback {count} times for '{agent}' agent. "
                    f"Common complaints: {reasons_str}"
                ),
                "priority": 7,
                "action": f"improve_quality:{agent}",
            })
    except Exception:
        pass

    # ── 3. Model performance degradation ──
    try:
        cursor = await db.execute("""
            SELECT model, COUNT(*) as calls,
                   AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as success_rate,
                   AVG(cost) as avg_cost
            FROM model_stats
            WHERE recorded_at > ?
            GROUP BY model
            HAVING calls >= 10
        """, (cutoff,))
        rows = await cursor.fetchall()
        for row in rows:
            model, calls, success_rate, avg_cost = row
            if success_rate < 0.6:
                proposals.append({
                    "category": "model_health",
                    "title": f"Model '{model}' success rate dropped to {success_rate:.0%}",
                    "detail": (
                        f"Model '{model}' had {success_rate:.0%} success rate over "
                        f"{calls} calls (avg cost ${avg_cost:.4f}). Consider demoting "
                        f"or replacing with alternative."
                    ),
                    "priority": 6,
                    "action": f"demote_model:{model}",
                })
    except Exception:
        pass

    # ── 4. Cost anomalies ──
    try:
        cursor = await db.execute("""
            SELECT SUM(cost) as weekly_cost, COUNT(*) as task_count
            FROM tasks WHERE completed_at > ? AND cost > 0
        """, (cutoff,))
        row = await cursor.fetchone()
        if row and row[0]:
            weekly_cost, task_count = row
            avg_per_task = weekly_cost / max(task_count, 1)
            if weekly_cost > 20.0:
                proposals.append({
                    "category": "cost",
                    "title": f"Weekly cost ${weekly_cost:.2f} ({task_count} tasks)",
                    "detail": (
                        f"Spent ${weekly_cost:.2f} on {task_count} tasks this week "
                        f"(${avg_per_task:.4f}/task avg). Review if cloud models are "
                        f"overused for simple tasks."
                    ),
                    "priority": 4,
                    "action": "review_routing",
                })
    except Exception:
        pass

    # ── 5. Prompt version candidates ready for promotion ──
    try:
        cursor = await db.execute("""
            SELECT agent_type, version, task_count, avg_quality
            FROM prompt_versions
            WHERE is_active = 0 AND task_count >= 10 AND avg_quality > 7.0
        """)
        rows = await cursor.fetchall()
        for row in rows:
            agent, version, count, quality = row
            proposals.append({
                "category": "prompt_version",
                "title": f"Prompt v{version} for '{agent}' ready for promotion",
                "detail": (
                    f"Candidate prompt v{version} for agent '{agent}' has "
                    f"{count} tasks with avg quality {quality:.1f}. "
                    f"Consider promoting to active."
                ),
                "priority": 3,
                "action": f"promote_prompt:{agent}:{version}",
            })
    except Exception:
        pass

    # ── 6. Underutilized skills ──
    try:
        cursor = await db.execute("""
            SELECT name, injection_success, injection_count
            FROM skills
            WHERE injection_count >= 5
            AND CAST(injection_success AS REAL) / injection_count < 0.5
        """)
        rows = await cursor.fetchall()
        for row in rows:
            name, success, count = row
            rate = (success / count * 100) if count > 0 else 0
            proposals.append({
                "category": "skills",
                "title": f"Skill '{name}' has low success rate ({rate:.0f}%)",
                "detail": (
                    f"Skill '{name}' succeeded {success}/{count} injections ({rate:.0f}%). "
                    f"Review or remove to avoid misleading agents."
                ),
                "priority": 2,
                "action": f"review_skill:{name}",
            })
    except Exception:
        pass

    # Sort by priority (highest first)
    proposals.sort(key=lambda p: -p["priority"])
    return proposals


def format_proposals_for_telegram(proposals: list[dict]) -> str:
    """Format proposals for Telegram notification."""
    if not proposals:
        return "No improvement proposals at this time. System is healthy."

    lines = ["*Weekly Self-Improvement Report*\n"]
    priority_emoji = {
        8: "🔴", 7: "🟠", 6: "🟡", 5: "🟡",
        4: "🔵", 3: "🔵", 2: "⚪", 1: "⚪",
    }
    for i, p in enumerate(proposals[:10], 1):
        emoji = priority_emoji.get(p["priority"], "⚪")
        lines.append(f"{emoji} *{i}. {p['title']}*")
        lines.append(f"   {p['detail'][:150]}")
        lines.append("")

    lines.append(f"_Total: {len(proposals)} proposals_")
    return "\n".join(lines)


async def format_proposals_for_file(proposals: list[dict]) -> str:
    """Format proposals as markdown for saving to workspace."""
    lines = [
        f"# Self-Improvement Report — {utc_now().strftime('%Y-%m-%d')}",
        "",
    ]
    for i, p in enumerate(proposals, 1):
        lines.append(f"## {i}. [{p['category']}] {p['title']}")
        lines.append(f"**Priority:** {p['priority']}/10")
        lines.append(f"**Action:** `{p['action']}`")
        lines.append(f"\n{p['detail']}\n")

    return "\n".join(lines)
