"""GitHub repository and code search via REST API."""

import asyncio
import os

import aiohttp

from src.infra.logging_config import get_logger

logger = get_logger("tools.github_search")

_GITHUB_API = "https://api.github.com"


async def github_search_repos(query: str, count: int = 10, sort: str = "stars") -> list[dict]:
    """Search GitHub repositories."""
    headers = {"Accept": "application/vnd.github.v3+json"}
    token = os.getenv("GITHUB_TOKEN", "")
    if token:
        headers["Authorization"] = f"token {token}"

    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{_GITHUB_API}/search/repositories",
            params={"q": query, "sort": sort, "per_page": count},
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            if resp.status != 200:
                return [{"error": f"GitHub API: HTTP {resp.status}"}]
            data = await resp.json()

    return [
        {
            "name": r.get("full_name", ""),
            "description": (r.get("description") or "")[:200],
            "stars": r.get("stargazers_count", 0),
            "forks": r.get("forks_count", 0),
            "language": r.get("language", ""),
            "updated": r.get("updated_at", "")[:10],
            "url": r.get("html_url", ""),
            "topics": r.get("topics", [])[:5],
        }
        for r in data.get("items", [])[:count]
    ]


async def github_search_code(query: str, count: int = 10) -> list[dict]:
    """Search GitHub code (requires auth token for best results)."""
    headers = {"Accept": "application/vnd.github.v3+json"}
    token = os.getenv("GITHUB_TOKEN", "")
    if token:
        headers["Authorization"] = f"token {token}"
    else:
        return [{"error": "GITHUB_TOKEN not set — code search requires authentication"}]

    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{_GITHUB_API}/search/code",
            params={"q": query, "per_page": count},
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            if resp.status != 200:
                return [{"error": f"GitHub API: HTTP {resp.status}"}]
            data = await resp.json()

    return [
        {
            "name": r.get("name", ""),
            "path": r.get("path", ""),
            "repo": r.get("repository", {}).get("full_name", ""),
            "url": r.get("html_url", ""),
        }
        for r in data.get("items", [])[:count]
    ]


async def github_repo_readme(owner_repo: str) -> str:
    """Fetch a repo's README content."""
    headers = {"Accept": "application/vnd.github.v3.raw"}
    token = os.getenv("GITHUB_TOKEN", "")
    if token:
        headers["Authorization"] = f"token {token}"

    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{_GITHUB_API}/repos/{owner_repo}/readme",
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            if resp.status != 200:
                return f"Error: HTTP {resp.status}"
            text = await resp.text()
            if len(text) > 5000:
                text = text[:5000] + "\n...(truncated)"
            return text
