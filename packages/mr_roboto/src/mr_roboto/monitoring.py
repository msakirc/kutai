# mr_roboto/monitoring.py (formerly src/infra/monitoring.py — Phase 14.2)
"""
Proactive Monitoring

Background checks: configured URLs (uptime), GitHub repos (new issues/PRs).
Pure helper functions — invoked by the mr_roboto monitoring_check executor
(cron-seeded mechanical task). No background loop lives here any more.
Configure URLs via MONITOR_URLS env var (comma-separated).
Configure GitHub repos via MONITOR_GITHUB_REPOS env var (comma-separated, owner/repo format).
"""
from __future__ import annotations
import asyncio
import os

from yazbunu import get_logger

logger = get_logger("mr_roboto.monitoring")

# State tracking to avoid re-alerting on the same URL being down across runs.
# Shared dict mutated by the monitoring_check executor on each cycle.
_known_issue_ids: set[str] = set()
_url_statuses: dict[str, bool] = {}


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
