"""Z7 T5 B2 — /changelog, /changelog.rss, and /changelog/latest.json route handlers.

Self-hosted public changelog page served by the existing FastAPI webhook listener.
Static-rendered + cached (simple module-level TTL cache; good enough for
low-traffic self-hosted deployment without external infra).

Same render+cache pattern as status_page.py (B3).

Routes:
  GET /changelog              — HTML version history, anchor links per release
  GET /changelog.rss          — RSS 2.0 feed of published changelog entries
  GET /changelog/latest.json  — In-app banner data (most-recent published entry)

All routes render from the changelog_entries table (published=1 only).
Cache TTL: 60 seconds (configurable via CHANGELOG_CACHE_TTL_SECONDS env var).
"""
from __future__ import annotations

import html
import json
import os
import time
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("app.changelog_page")

_CACHE_TTL: float = float(os.getenv("CHANGELOG_CACHE_TTL_SECONDS", "60"))

# Module-level cache: {html: str, rss: str, latest: dict|None, fetched_at: float}
_cache: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

async def _fetch_published_entries(limit: int = 50) -> list[dict]:
    """Return published changelog entries, newest first."""
    try:
        from src.infra.db import get_db
        db = await get_db()
        async with db.execute(
            "SELECT entry_id, product_id, version, released_at, title, body_md, "
            "kind_breakdown_json, shipped_features_json, related_mission_ids_json, "
            "external_url "
            "FROM changelog_entries "
            "WHERE published=1 "
            "ORDER BY released_at DESC, entry_id DESC "
            "LIMIT ?",
            (limit,),
        ) as cur:
            rows = await cur.fetchall()

        result = []
        for r in rows:
            try:
                kac = json.loads(r[6] or "{}")
            except Exception:
                kac = {}
            try:
                features = json.loads(r[7] or "[]")
            except Exception:
                features = []
            result.append({
                "entry_id": r[0],
                "product_id": r[1] or "",
                "version": r[2] or "",
                "released_at": r[3] or "",
                "title": r[4] or "",
                "body_md": r[5] or "",
                "kind_breakdown": kac,
                "shipped_features": features,
                "external_url": r[9],
            })
        return result
    except Exception as exc:
        logger.warning("changelog_page: _fetch_published_entries failed", error=str(exc))
        return []


# ---------------------------------------------------------------------------
# HTML renderer
# ---------------------------------------------------------------------------

_KAC_SECTION_ORDER = ["added", "changed", "fixed", "deprecated", "removed"]
_KAC_SECTION_EMOJI = {
    "added": "Added",
    "changed": "Changed",
    "fixed": "Fixed",
    "deprecated": "Deprecated",
    "removed": "Removed",
}


def _render_kac_section(kac: dict) -> str:
    parts: list[str] = []
    for bucket in _KAC_SECTION_ORDER:
        items = kac.get(bucket, [])
        if not items:
            continue
        label = _KAC_SECTION_EMOJI.get(bucket, bucket.capitalize())
        parts.append(f'<h4 class="kac-bucket kac-{html.escape(bucket)}">{label}</h4>')
        parts.append('<ul>')
        for item in items:
            parts.append(f'<li>{html.escape(str(item))}</li>')
        parts.append('</ul>')
    return "".join(parts)


def _render_html(entries: list[dict]) -> str:
    if not entries:
        changelog_body = '<p class="empty">No releases published yet.</p>'
    else:
        cards: list[str] = []
        for entry in entries:
            version = html.escape(entry["version"])
            title = html.escape(entry["title"])
            released = html.escape((entry["released_at"] or "")[:10])
            anchor = f"v{version.replace('.', '-')}"
            ext_link = ""
            if entry.get("external_url"):
                url = html.escape(entry["external_url"])
                ext_link = f' &middot; <a href="{url}" rel="noopener noreferrer">Release notes</a>'

            kac_html = _render_kac_section(entry["kind_breakdown"])
            if not kac_html:
                # Fall back to body_md as pre-formatted text
                body_escaped = html.escape(entry["body_md"] or "")
                kac_html = f'<pre class="body-md">{body_escaped}</pre>'

            cards.append(
                f'<div class="release" id="{html.escape(anchor)}">'
                f'<div class="release-header">'
                f'<a class="version-tag" href="#{html.escape(anchor)}">v{version}</a>'
                f'<span class="release-title">{title}</span>'
                f'<span class="release-date">{released}{ext_link}</span>'
                f'</div>'
                f'<div class="release-body">{kac_html}</div>'
                f'</div>'
            )
        changelog_body = "".join(cards)

    now_str = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Changelog</title>
<link rel="alternate" type="application/rss+xml" title="Changelog RSS" href="/changelog.rss">
<style>
body{{font-family:system-ui,sans-serif;margin:0;padding:0;background:#f7f7f7;color:#222}}
.container{{max-width:840px;margin:0 auto;padding:24px 16px}}
h1{{margin-bottom:4px}}
.subtitle{{color:#666;font-size:14px;margin-bottom:32px}}
.release{{background:#fff;border-radius:8px;padding:20px 24px;margin-bottom:24px;box-shadow:0 1px 3px rgba(0,0,0,.1)}}
.release-header{{display:flex;align-items:baseline;gap:12px;margin-bottom:12px;flex-wrap:wrap}}
.version-tag{{font-size:18px;font-weight:700;color:#2b6cb0;text-decoration:none}}
.version-tag:hover{{text-decoration:underline}}
.release-title{{font-size:15px;font-weight:500;color:#444}}
.release-date{{margin-left:auto;font-size:12px;color:#888}}
h4.kac-bucket{{font-size:13px;font-weight:600;text-transform:uppercase;letter-spacing:.05em;margin:12px 0 4px;color:#555}}
.kac-added{{color:#276749}}
.kac-changed{{color:#2b6cb0}}
.kac-fixed{{color:#2c7a7b}}
.kac-deprecated{{color:#975a16}}
.kac-removed{{color:#9b2c2c}}
ul{{margin:0 0 8px;padding-left:20px;font-size:14px;line-height:1.6}}
pre.body-md{{font-size:13px;white-space:pre-wrap;background:#f4f4f4;padding:12px;border-radius:4px}}
.empty{{color:#999;font-style:italic}}
footer{{color:#aaa;font-size:12px;text-align:center;margin-top:40px}}
</style>
</head>
<body>
<div class="container">
<h1>Changelog</h1>
<p class="subtitle">All notable changes &middot; Last updated: {now_str} &middot; <a href="/changelog.rss">RSS feed</a></p>
{changelog_body}
<footer>Powered by KutAI &middot; <a href="/changelog.rss">Subscribe via RSS</a></footer>
</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# RSS renderer
# ---------------------------------------------------------------------------

def _render_rss(entries: list[dict]) -> str:
    items: list[str] = []
    for entry in entries[:20]:
        version = entry["version"]
        title_text = f"v{version}: {entry['title']}"
        pub_date = entry.get("released_at") or ""
        body = html.escape(entry.get("body_md") or "")
        link = f"/changelog#v{version.replace('.', '-')}"
        guid = f"changelog-entry-{entry['entry_id']}"
        ext_url = entry.get("external_url") or link
        items.append(
            f"  <item>\n"
            f"    <title>{html.escape(title_text)}</title>\n"
            f"    <link>{html.escape(ext_url)}</link>\n"
            f"    <pubDate>{pub_date}</pubDate>\n"
            f"    <description>{body}</description>\n"
            f"    <guid isPermaLink='false'>{guid}</guid>\n"
            f"  </item>"
        )

    items_str = "\n".join(items)
    now_str = time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime())
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Changelog</title>
    <link>/changelog</link>
    <description>Product changelog — all notable releases</description>
    <language>en</language>
    <lastBuildDate>{now_str}</lastBuildDate>
    <atom:link href="/changelog.rss" rel="self" type="application/rss+xml" xmlns:atom="http://www.w3.org/2005/Atom"/>
{items_str}
  </channel>
</rss>"""


# ---------------------------------------------------------------------------
# Latest-entry dict (for in-app banner)
# ---------------------------------------------------------------------------

def _render_latest(entries: list[dict]) -> dict | None:
    """Return a dict for the in-app banner (most-recent published entry)."""
    if not entries:
        return None
    latest = entries[0]
    return {
        "entry_id": latest["entry_id"],
        "version": latest["version"],
        "title": latest["title"],
        "released_at": latest["released_at"],
        "external_url": latest.get("external_url"),
        "shipped_features": latest["shipped_features"],
    }


# ---------------------------------------------------------------------------
# Cached fetch-and-render
# ---------------------------------------------------------------------------

async def _get_cached() -> tuple[str, str, dict | None]:
    """Return (html, rss_xml, latest_dict), refreshing if TTL expired."""
    now = time.monotonic()
    if _cache.get("html") and (now - _cache.get("fetched_at", 0)) < _CACHE_TTL:
        return _cache["html"], _cache["rss"], _cache["latest"]

    entries = await _fetch_published_entries(limit=50)

    rendered_html = _render_html(entries)
    rendered_rss = _render_rss(entries)
    latest_data = _render_latest(entries)

    _cache["html"] = rendered_html
    _cache["rss"] = rendered_rss
    _cache["latest"] = latest_data
    _cache["fetched_at"] = now
    return rendered_html, rendered_rss, latest_data


def invalidate_cache() -> None:
    """Force next request to re-render (call after changelog/publish)."""
    _cache.clear()


# ---------------------------------------------------------------------------
# Route handler functions (imported by webhook_listener.py)
# ---------------------------------------------------------------------------

async def changelog_html_handler() -> str:
    html_content, _, _ = await _get_cached()
    return html_content


async def changelog_rss_handler() -> str:
    _, rss_content, _ = await _get_cached()
    return rss_content


async def changelog_latest_json_handler() -> dict | None:
    _, _, latest = await _get_cached()
    return latest
