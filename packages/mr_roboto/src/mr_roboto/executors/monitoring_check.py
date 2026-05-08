"""Monitoring check executor.

Cron-seeded mechanical task: runs URL uptime checks and GitHub repo polls,
then enqueues ``notify_user`` sub-tasks for each alert via General Beckman.
No direct Telegram sends — all notifications flow through mr_roboto notify_user.
"""
from __future__ import annotations

import os
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.monitoring_check")


async def run(task: dict[str, Any]) -> dict[str, Any]:
    """Execute one monitoring cycle.

    Reads MONITOR_URLS and MONITOR_GITHUB_REPOS from env.
    For each alert, enqueues a notify_user mechanical sub-task.
    Always returns a result dict — errors are caught and logged.
    """
    from src.infra.monitoring import check_url_uptime, check_github_repo, _url_statuses
    from src.infra.db import add_task
    from general_beckman.apply import _mechanical_context

    task_id = task.get("id")
    alerts: list[str] = []

    # --- URL uptime checks ---
    monitor_urls = [u.strip() for u in os.getenv("MONITOR_URLS", "").split(",") if u.strip()]
    for url in monitor_urls:
        try:
            is_up, detail = await check_url_uptime(url)
            was_up = _url_statuses.get(url, True)
            _url_statuses[url] = is_up

            if not is_up and was_up:
                alerts.append(f"\U0001f534 *DOWN*: `{url}` — {detail}")
                logger.warning("URL down", url=url, detail=detail)
            elif is_up and not was_up:
                alerts.append(f"\U0001f7e2 *RECOVERED*: `{url}`")
                logger.info("URL recovered", url=url)
        except Exception as exc:
            alerts.append(f"\U0001f534 *ERROR* checking `{url}`: {str(exc)[:80]}")
            logger.warning("URL check raised", url=url, error=str(exc))

    # --- GitHub repo polls ---
    monitor_repos = [r.strip() for r in os.getenv("MONITOR_GITHUB_REPOS", "").split(",") if r.strip()]
    for repo in monitor_repos:
        try:
            new_items = await check_github_repo(repo)
            for item in new_items:
                itype = item["type"]
                alerts.append(
                    f"\U0001f4cb New {itype} in `{repo}` #{item['number']}: "
                    f"{item['title'][:60]}\n{item['url']}"
                )
        except Exception as exc:
            logger.debug("GitHub check raised", repo=repo, error=str(exc))

    # --- Enqueue notify_user sub-tasks for each alert ---
    enqueued = 0
    for alert_msg in alerts:
        try:
            msg = "\U0001f50d *Monitoring Alert*\n\n" + alert_msg
            await add_task(
                title="Notify: monitoring alert",
                description=alert_msg[:120],
                agent_type="mechanical",
                parent_task_id=task_id,
                context=_mechanical_context("notify_user", message=msg),
                depends_on=[],
            )
            enqueued += 1
        except Exception as exc:
            logger.warning("Failed to enqueue notify_user sub-task", error=str(exc))

    logger.info(
        "monitoring_check complete",
        urls_checked=len(monitor_urls),
        repos_checked=len(monitor_repos),
        alerts=len(alerts),
        enqueued=enqueued,
    )
    return {
        "urls_checked": len(monitor_urls),
        "repos_checked": len(monitor_repos),
        "alerts": len(alerts),
        "enqueued": enqueued,
    }
