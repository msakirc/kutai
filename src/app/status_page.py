"""Z7 T3D — B3: /status and /status.rss route handlers.

Self-hosted status page served by the existing FastAPI webhook listener.
Static-rendered + cached (simple module-level TTL cache; good enough for
low-traffic self-hosted deployment without external infra).

Routes:
  GET /status            — HTML status page (current up/down, active incidents,
                           90-day uptime stats)
  GET /status.rss        — RSS 2.0 feed of recent status_updates

Both routes render from the incidents + status_updates tables.
Cache TTL: 60 seconds (configurable via STATUS_CACHE_TTL_SECONDS env var).
"""
from __future__ import annotations

import html
import json
import os
import time
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("app.status_page")

_CACHE_TTL: float = float(os.getenv("STATUS_CACHE_TTL_SECONDS", "60"))

# Module-level cache: (rendered_html, rss_xml, fetched_at)
_cache: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

async def _fetch_active_incidents(product_id: str | None = None) -> list[dict]:
    """Return open incidents (resolved_at IS NULL), newest first."""
    try:
        from src.infra.db import get_db
        db = await get_db()
        if product_id:
            sql = (
                "SELECT incident_id, product_id, opened_at, severity, "
                "affected_components_json, customer_impact_summary, current_status_md "
                "FROM incidents WHERE resolved_at IS NULL AND product_id = ? "
                "ORDER BY opened_at DESC LIMIT 20"
            )
            params = (product_id,)
        else:
            sql = (
                "SELECT incident_id, product_id, opened_at, severity, "
                "affected_components_json, customer_impact_summary, current_status_md "
                "FROM incidents WHERE resolved_at IS NULL "
                "ORDER BY opened_at DESC LIMIT 20"
            )
            params = ()
        async with db.execute(sql, params) as cur:
            rows = await cur.fetchall()
        result = []
        for r in rows:
            result.append({
                "incident_id": r[0],
                "product_id": r[1],
                "opened_at": r[2] or "",
                "severity": r[3] or "minor",
                "affected_components": _safe_json_list(r[4]),
                "customer_impact_summary": r[5] or "",
                "current_status_md": r[6] or "",
            })
        return result
    except Exception as exc:
        logger.warning("status_page: _fetch_active_incidents failed", error=str(exc))
        return []


async def _fetch_recent_updates(limit: int = 20) -> list[dict]:
    """Return most recent status_updates across all incidents."""
    try:
        from src.infra.db import get_db
        db = await get_db()
        async with db.execute(
            "SELECT su.update_id, su.product_id, su.incident_id, su.posted_at, "
            "su.body_md, su.status_kind "
            "FROM status_updates su "
            "ORDER BY su.posted_at DESC LIMIT ?",
            (limit,),
        ) as cur:
            rows = await cur.fetchall()
        return [
            {
                "update_id": r[0],
                "product_id": r[1],
                "incident_id": r[2],
                "posted_at": r[3] or "",
                "body_md": r[4] or "",
                "status_kind": r[5] or "investigating",
            }
            for r in rows
        ]
    except Exception as exc:
        logger.warning("status_page: _fetch_recent_updates failed", error=str(exc))
        return []


async def _fetch_uptime_stats(days: int = 90) -> dict:
    """Return per-component uptime percentage over the last *days* days.

    Simple implementation: counts days with at least one active incident
    per component name from affected_components_json.  Returns a dict:
      {"component_name": uptime_pct_float}
    """
    try:
        from src.infra.db import get_db
        db = await get_db()
        async with db.execute(
            "SELECT affected_components_json, opened_at, resolved_at "
            "FROM incidents "
            f"WHERE opened_at >= datetime('now', '-{days} days') "
            "ORDER BY opened_at ASC"
        ) as cur:
            rows = await cur.fetchall()
    except Exception as exc:
        logger.warning("status_page: _fetch_uptime_stats failed", error=str(exc))
        return {}

    # Count incident-days per component.
    component_outage_days: dict[str, set] = {}
    for r in rows:
        components = _safe_json_list(r[0])
        opened = (r[1] or "")[:10]  # YYYY-MM-DD
        resolved = (r[2] or "")[:10] or _today_str()
        if not opened:
            continue
        # Expand date range to day strings.
        try:
            from datetime import datetime, timedelta
            d0 = datetime.strptime(opened, "%Y-%m-%d")
            d1 = datetime.strptime(resolved, "%Y-%m-%d")
            day_count = max((d1 - d0).days, 1)
            for delta in range(day_count):
                day_str = (d0 + timedelta(days=delta)).strftime("%Y-%m-%d")
                for comp in components:
                    component_outage_days.setdefault(comp, set()).add(day_str)
        except Exception:
            continue

    result = {}
    for comp, outage_days_set in component_outage_days.items():
        pct = max(0.0, 100.0 - (len(outage_days_set) / days) * 100.0)
        result[comp] = round(pct, 2)
    return result


def _today_str() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


def _safe_json_list(raw: str | None) -> list:
    if not raw:
        return []
    try:
        val = json.loads(raw)
        return val if isinstance(val, list) else []
    except Exception:
        return []


# ---------------------------------------------------------------------------
# HTML renderer
# ---------------------------------------------------------------------------

_SEVERITY_BADGE = {
    "critical": '<span style="background:#e53e3e;color:#fff;padding:2px 8px;border-radius:4px;font-size:12px">CRITICAL</span>',
    "major": '<span style="background:#dd6b20;color:#fff;padding:2px 8px;border-radius:4px;font-size:12px">MAJOR</span>',
    "minor": '<span style="background:#d69e2e;color:#fff;padding:2px 8px;border-radius:4px;font-size:12px">MINOR</span>',
}

_STATUS_KIND_LABEL = {
    "investigating": "Investigating",
    "identified": "Identified",
    "monitoring": "Monitoring",
    "resolved": "Resolved",
}


def _render_html(
    active_incidents: list[dict],
    uptime_stats: dict,
    recent_updates: list[dict],
) -> str:
    incident_section = ""
    if active_incidents:
        cards = []
        for inc in active_incidents:
            badge = _SEVERITY_BADGE.get(inc["severity"], "")
            comps = ", ".join(html.escape(c) for c in inc["affected_components"]) or "General"
            summary = html.escape(inc["customer_impact_summary"] or "We are investigating an issue.")
            current_md = html.escape(inc["current_status_md"] or "")
            opened = html.escape(inc["opened_at"][:16] or "")
            cards.append(
                f'<div class="incident-card">'
                f'<div class="incident-header">'
                f'<strong>Incident #{inc["incident_id"]}</strong> {badge}'
                f'<span class="inc-time">{opened} UTC</span>'
                f'</div>'
                f'<div class="affected">Affected: <em>{comps}</em></div>'
                f'<p>{summary}</p>'
                f'<p class="current-status">{current_md}</p>'
                f'</div>'
            )
        incident_section = (
            '<section class="incidents">'
            '<h2>Active Incidents</h2>'
            + "".join(cards)
            + "</section>"
        )
    else:
        incident_section = (
            '<section class="all-ok">'
            '<div class="ok-banner">All systems operational</div>'
            "</section>"
        )

    uptime_rows = ""
    if uptime_stats:
        for comp, pct in sorted(uptime_stats.items()):
            bar_class = "green" if pct >= 99.5 else ("yellow" if pct >= 95.0 else "red")
            uptime_rows += (
                f'<tr><td>{html.escape(comp)}</td>'
                f'<td><div class="bar {bar_class}" style="width:{pct}%"></div>'
                f'<span>{pct:.2f}%</span></td></tr>'
            )
        uptime_section = (
            '<section class="uptime">'
            "<h2>90-Day Uptime</h2>"
            '<table><thead><tr><th>Component</th><th>Uptime</th></tr></thead>'
            f"<tbody>{uptime_rows}</tbody></table>"
            "</section>"
        )
    else:
        uptime_section = ""

    now_str = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Status</title>
<link rel="alternate" type="application/rss+xml" title="Status RSS" href="/status.rss">
<style>
body{{font-family:system-ui,sans-serif;margin:0;padding:0;background:#f7f7f7;color:#222}}
.container{{max-width:800px;margin:0 auto;padding:24px 16px}}
h1{{margin-bottom:4px}}
.subtitle{{color:#666;font-size:14px;margin-bottom:24px}}
.all-ok .ok-banner{{background:#276749;color:#fff;padding:16px 24px;border-radius:8px;font-size:18px;font-weight:600}}
.incidents h2{{color:#9b2c2c}}
.incident-card{{background:#fff;border-left:4px solid #e53e3e;border-radius:6px;padding:16px 20px;margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,.1)}}
.incident-header{{display:flex;align-items:center;gap:8px;margin-bottom:8px}}
.inc-time{{margin-left:auto;font-size:12px;color:#666}}
.affected{{font-size:13px;color:#444;margin-bottom:8px}}
.current-status{{font-size:14px;color:#555;border-top:1px solid #eee;margin-top:8px;padding-top:8px}}
.uptime table{{width:100%;border-collapse:collapse;background:#fff;border-radius:6px;overflow:hidden}}
.uptime th,.uptime td{{padding:10px 14px;text-align:left;border-bottom:1px solid #eee;font-size:14px}}
.bar{{height:12px;border-radius:3px;display:inline-block;vertical-align:middle;margin-right:8px}}
.bar.green{{background:#48bb78}}.bar.yellow{{background:#ecc94b}}.bar.red{{background:#fc8181}}
footer{{color:#aaa;font-size:12px;text-align:center;margin-top:40px}}
</style>
</head>
<body>
<div class="container">
<h1>System Status</h1>
<p class="subtitle">Last updated: {now_str} &middot; <a href="/status.rss">RSS feed</a></p>
{incident_section}
{uptime_section}
<footer>Powered by KutAI &middot; Updates via <a href="/status.rss">RSS</a></footer>
</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# RSS renderer
# ---------------------------------------------------------------------------

def _render_rss(recent_updates: list[dict]) -> str:
    items = []
    for u in recent_updates[:20]:
        title_text = (
            f"[{_STATUS_KIND_LABEL.get(u['status_kind'], u['status_kind'])}] "
            f"Incident #{u['incident_id']}"
        )
        pub_date = u.get("posted_at") or ""
        body = html.escape(u.get("body_md") or "")
        link = f"/status#{u.get('incident_id', '')}"
        items.append(
            f"  <item>\n"
            f"    <title>{html.escape(title_text)}</title>\n"
            f"    <link>{link}</link>\n"
            f"    <pubDate>{pub_date}</pubDate>\n"
            f"    <description>{body}</description>\n"
            f"    <guid isPermaLink='false'>status-update-{u.get('update_id', '')}</guid>\n"
            f"  </item>"
        )

    items_str = "\n".join(items)
    now_str = time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime())
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>System Status</title>
    <link>/status</link>
    <description>Status updates for system incidents</description>
    <language>en</language>
    <lastBuildDate>{now_str}</lastBuildDate>
    <atom:link href="/status.rss" rel="self" type="application/rss+xml" xmlns:atom="http://www.w3.org/2005/Atom"/>
{items_str}
  </channel>
</rss>"""


# ---------------------------------------------------------------------------
# Cached fetch-and-render
# ---------------------------------------------------------------------------

async def _get_cached() -> tuple[str, str]:
    """Return (html, rss_xml), refreshing if TTL expired."""
    now = time.monotonic()
    if _cache.get("html") and (now - _cache.get("fetched_at", 0)) < _CACHE_TTL:
        return _cache["html"], _cache["rss"]

    active_incidents = await _fetch_active_incidents()
    uptime_stats = await _fetch_uptime_stats(days=90)
    recent_updates = await _fetch_recent_updates(limit=20)

    rendered_html = _render_html(active_incidents, uptime_stats, recent_updates)
    rendered_rss = _render_rss(recent_updates)

    _cache["html"] = rendered_html
    _cache["rss"] = rendered_rss
    _cache["fetched_at"] = now
    return rendered_html, rendered_rss


def invalidate_cache() -> None:
    """Force next request to re-render (call after publish_status)."""
    _cache.clear()


# ---------------------------------------------------------------------------
# Route handler functions (imported by webhook_listener.py)
# ---------------------------------------------------------------------------

async def status_html_handler() -> str:
    html_content, _ = await _get_cached()
    return html_content


async def status_rss_handler() -> str:
    _, rss_content = await _get_cached()
    return rss_content
