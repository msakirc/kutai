# infra/monitoring.py
"""
Phase 14.2 — Proactive Monitoring

Background checks: configured URLs (uptime), GitHub repos (new issues/PRs).
Notifies via Telegram with actionable suggestions.
Configure URLs via MONITOR_URLS env var (comma-separated).
Configure GitHub repos via MONITOR_GITHUB_REPOS env var (comma-separated, owner/repo format).
"""
from __future__ import annotations
import asyncio
import os
import time
from typing import Optional

from .logging_config import get_logger
from .runtime_state import runtime_state

logger = get_logger("infra.monitoring")

# State tracking to avoid re-alerting on same issue
_last_check: dict[str, float] = {}
_known_issue_ids: set[str] = set()
_url_statuses: dict[str, bool] = {}

CHECK_INTERVAL_SECS = int(os.getenv("MONITOR_INTERVAL", "300"))  # 5 min default


async def check_url_uptime(url: str) -> tuple[bool, str]:
    """Check if a URL is reachable. Returns (is_up, detail)."""
    try:
        import aiohttp
        async with aiohttp.ClientSession() as s:
            async with s.get(url, timeout=aiohttp.ClientTimeout(total=10)) as r:
                up = r.status < 500
                return up, f"HTTP {r.status}"
    except asyncio.TimeoutError:
        return False, "timeout"
    except Exception as exc:
        return False, str(exc)[:100]


async def check_github_repo(repo: str) -> list[dict]:
    """
    Check a GitHub repo for new open issues/PRs.
    repo format: owner/repo
    Returns list of new items since last check.
    """
    try:
        import aiohttp
        token = os.getenv("GITHUB_TOKEN", "")
        headers = {"Accept": "application/vnd.github.v3+json"}
        if token:
            headers["Authorization"] = f"token {token}"

        new_items = []
        url = f"https://api.github.com/repos/{repo}/issues?state=open&per_page=5&sort=created"
        async with aiohttp.ClientSession() as s:
            async with s.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as r:
                if r.status == 200:
                    issues = await r.json()
                    for issue in issues:
                        issue_id = str(issue.get("id", ""))
                        if issue_id and issue_id not in _known_issue_ids:
                            _known_issue_ids.add(issue_id)
                            new_items.append({
                                "repo": repo,
                                "number": issue.get("number"),
                                "title": issue.get("title", ""),
                                "url": issue.get("html_url", ""),
                                "type": "PR" if issue.get("pull_request") else "Issue",
                            })
        return new_items
    except Exception as exc:
        logger.debug(f"GitHub check failed for {repo}: {exc}")
        return []


async def run_monitoring_cycle() -> None:
    """Run one monitoring cycle. Called periodically."""
    alerts = []

    # Check configured URLs
    monitor_urls = [u.strip() for u in os.getenv("MONITOR_URLS", "").split(",") if u.strip()]
    for url in monitor_urls:
        is_up, detail = await check_url_uptime(url)
        was_up = _url_statuses.get(url, True)
        _url_statuses[url] = is_up

        if not is_up and was_up:
            alerts.append(f"🔴 *DOWN*: `{url}` — {detail}")
            logger.warning(f"URL down: {url} ({detail})")
        elif is_up and not was_up:
            alerts.append(f"🟢 *RECOVERED*: `{url}`")
            logger.info(f"URL recovered: {url}")

    # Check GitHub repos
    monitor_repos = [r.strip() for r in os.getenv("MONITOR_GITHUB_REPOS", "").split(",") if r.strip()]
    for repo in monitor_repos:
        new_items = await check_github_repo(repo)
        for item in new_items:
            itype = item["type"]
            alerts.append(
                f"📋 New {itype} in `{repo}` #{item['number']}: "
                f"{item['title'][:60]}\n{item['url']}"
            )

    # Send alerts
    if alerts:
        await _send_monitoring_alerts(alerts)


async def _send_monitoring_alerts(alerts: list[str]) -> None:
    """Send monitoring alerts via Telegram."""
    try:
        from ..app.config import TELEGRAM_BOT_TOKEN, TELEGRAM_ADMIN_CHAT_ID
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_ADMIN_CHAT_ID:
            return
        import aiohttp
        msg = "🔍 *Monitoring Alert*\n\n" + "\n\n".join(alerts)
        if len(msg) > 4000:
            msg = msg[:4000] + "\n..."
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        async with aiohttp.ClientSession() as s:
            await s.post(url, json={
                "chat_id": TELEGRAM_ADMIN_CHAT_ID,
                "text": msg,
                "parse_mode": "Markdown",
            }, timeout=aiohttp.ClientTimeout(total=5))
    except Exception as exc:
        logger.debug(f"Monitoring alert send failed: {exc}")


async def run_monitoring_loop() -> None:
    """Background loop that runs monitoring checks periodically."""
    logger.info(f"Monitoring loop started (interval: {CHECK_INTERVAL_SECS}s)")
    while True:
        try:
            await run_monitoring_cycle()
        except Exception as exc:
            logger.debug(f"Monitoring cycle error: {exc}")
        await asyncio.sleep(CHECK_INTERVAL_SECS)
