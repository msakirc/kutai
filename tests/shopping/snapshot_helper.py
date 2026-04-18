"""Snapshot helper for review-scraper development.

Fetches a URL once via the tiered scraper, persists HTML/JSON to
``tests/shopping/fixtures/review_snapshots/``, and serves cached content
on subsequent calls. Lets parser iteration happen offline without
re-hitting live sites.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
from pathlib import Path
from typing import Any

SNAPSHOT_DIR = Path(__file__).parent / "fixtures" / "review_snapshots"
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


def _slug(url: str, label: str | None = None) -> str:
    """Produce a stable filename from a URL + optional label."""
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
    base = label or re.sub(r"[^a-zA-Z0-9]+", "_", url)[:80].strip("_")
    return f"{base}__{h}"


def snapshot_path(url: str, label: str | None = None, *, ext: str = "html") -> Path:
    return SNAPSHOT_DIR / f"{_slug(url, label)}.{ext}"


async def fetch_snapshot(
    url: str,
    *,
    label: str | None = None,
    tier: str = "TLS",
    refresh: bool = False,
    timeout: float = 30.0,
    ext: str = "html",
) -> tuple[str, int, dict]:
    """Fetch *url* once and cache; return (text, status, headers).

    On subsequent calls, returns the cached snapshot unless ``refresh=True``.
    Tier choices: HTTP, TLS, STEALTH, BROWSER.
    """
    path = snapshot_path(url, label, ext=ext)
    meta_path = path.with_suffix(path.suffix + ".meta.json")

    if path.exists() and not refresh:
        text = path.read_text(encoding="utf-8")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        return text, meta.get("status", 200), meta.get("headers", {})

    from src.tools.scraper import scrape_url, ScrapeTier

    tier_map = {
        "HTTP": ScrapeTier.HTTP,
        "TLS": ScrapeTier.TLS,
        "STEALTH": ScrapeTier.STEALTH,
        "BROWSER": ScrapeTier.BROWSER,
    }
    result = await scrape_url(url, max_tier=tier_map[tier], timeout=timeout)

    path.write_text(result.html, encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {"url": url, "status": result.status, "headers": result.headers, "tier_used": getattr(result, "tier_used", tier)},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return result.html, result.status, result.headers


def load_snapshot(url: str, label: str | None = None, *, ext: str = "html") -> str | None:
    path = snapshot_path(url, label, ext=ext)
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


async def snapshot_many(urls: list[tuple[str, str]], *, tier: str = "TLS", refresh: bool = False) -> dict[str, tuple[str, int]]:
    """Snapshot a list of (url, label) pairs sequentially. Returns label -> (text, status)."""
    out: dict[str, tuple[str, int]] = {}
    for url, label in urls:
        try:
            text, status, _ = await fetch_snapshot(url, label=label, tier=tier, refresh=refresh)
            out[label] = (text, status)
            print(f"  [{status}] {label}  ({len(text)} bytes)")
        except Exception as exc:
            print(f"  [ERR] {label}  {exc}")
            out[label] = ("", 0)
        await asyncio.sleep(0.5)
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python -m tests.shopping.snapshot_helper <url> [label] [tier]")
        sys.exit(1)
    url = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else None
    tier = sys.argv[3] if len(sys.argv) > 3 else "TLS"
    text, status, _ = asyncio.run(fetch_snapshot(url, label=label, tier=tier, refresh=True))
    print(f"status={status} bytes={len(text)} -> {snapshot_path(url, label)}")
