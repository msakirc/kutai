"""Z7 T6 A11 — mention_polls/<source> mechanical executor.

Thin adapters for each mention source.  All network calls are mockable —
tests patch the internal fetcher functions directly.

Sources
-------
- ``hn``       : Algolia HN Search API — free, no key required.
- ``reddit``   : PRAW (Python Reddit API Wrapper) — free, rate-limited,
                 requires REDDIT_CLIENT_ID + REDDIT_CLIENT_SECRET in .env.
                 Searches specified subreddits for product_name mentions.
- ``google``   : Google Alerts RSS — free, no key, URL-encoded RSS feed.
- ``twitter``  : X API v2 — PAID, OFF by default behind
                 MENTION_TWITTER_ENABLED=1 consent flag.
- ``discord``  : Discord bot — founder OAuth-authorised, reads specified
                 channel IDs for product mentions.

Signal scoring (shared across all sources)
------------------------------------------
  score = clamp(
      (crm_match * 4) +
      (log10(max(1, followers)) / log10(1_000_000)) * 3 +
      (keyword_density * 3)
  , 0, 10)

  crm_match: 1 if author handle/username is in CRM relationships table,
             else 0 (founder-known contact → higher signal).
  followers: author_followers field (source-specific; 0 when unavailable).
  keyword_density: fraction of product keywords found in text (0.0-1.0).

Score tiers
-----------
  <4   → silent log (no founder action)
  4-7  → add to daily digest batch
  >=7  → immediate founder_action

Negative cluster trigger
------------------------
  If >=3 neg-sentiment mentions with score>=4 arrive within 1h for a
  product, a ``crisis_comms_draft`` founder_action is surfaced (B6).

Public API
----------
  run(payload) -> dict
      mr_roboto executor entry point.
      payload keys:
        source   (str)  — 'hn' | 'reddit' | 'google' | 'twitter' | 'discord'
        product_id (str)
        product_name (str) — search keyword
        config (dict, optional) — source-specific (subreddits, channel_ids, etc.)

  poll_source(source, product_id, product_name, config) -> dict
      {"ingested": int, "immediate": int, "digest": int, "silent": int,
       "crisis_triggered": bool}
"""
from __future__ import annotations

import math
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote_plus

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.mention_polls")

SUPPORTED_SOURCES = frozenset({"hn", "reddit", "google", "twitter", "discord"})

# Default product keywords weighting density computation.
_DEFAULT_KEYWORDS: list[str] = []


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _keyword_density(text: str, keywords: list[str]) -> float:
    """Fraction of keywords found in text (0.0-1.0)."""
    if not keywords:
        return 0.5  # no keyword list → neutral density
    hits = sum(1 for kw in keywords if kw.lower() in text.lower())
    return hits / len(keywords)


def _score_mention(
    *,
    text: str,
    author_followers: int,
    crm_match: bool,
    keywords: list[str],
) -> int:
    """Compute 0-10 signal score."""
    crm_pts = 4.0 if crm_match else 0.0
    followers_pts = 0.0
    if author_followers > 0:
        followers_pts = (math.log10(max(1, author_followers)) / math.log10(1_000_000)) * 3.0
    keyword_pts = _keyword_density(text, keywords) * 3.0
    raw = crm_pts + followers_pts + keyword_pts
    return min(10, max(0, round(raw)))


async def _check_crm_match(product_id: str, author: str) -> bool:
    """Return True if *author* handle matches a CRM contact for *product_id*."""
    try:
        from src.app.crm import get_contact_by_handle
        contact = await get_contact_by_handle(product_id, author)
        return contact is not None
    except Exception:
        return False


def _canonical_url(url: str | None) -> str | None:
    """Strip tracking params and fragments for cross-source dedup."""
    if not url:
        return None
    # Strip fragment and common tracking params
    url = url.split("#")[0]
    url = re.sub(r"[?&](utm_[^&]+|ref=[^&]+|source=[^&]+)", "", url)
    return url.rstrip("?&") or None


async def _cross_source_dedup(
    db: Any,
    canonical_url: str | None,
    product_id: str,
    window_hours: int = 24,
) -> bool:
    """Return True if canonical_url seen in window_hours for product (skip dedup if no URL)."""
    if not canonical_url:
        return False
    cutoff = (
        datetime.now(timezone.utc) - timedelta(hours=window_hours)
    ).strftime("%Y-%m-%d %H:%M:%S")
    cur = await db.execute(
        "SELECT 1 FROM mentions "
        "WHERE canonical_url = ? AND product_id = ? AND seen_at >= ?",
        (canonical_url, product_id, cutoff),
    )
    row = await cur.fetchone()
    return row is not None


async def _ingest_mention(
    *,
    product_id: str,
    source: str,
    source_id: str,
    url: str | None,
    author: str,
    author_followers: int,
    text: str,
    sentiment: str,
    keywords: list[str],
) -> dict[str, Any]:
    """Ingest one mention row; return {"ingested": bool, "score": int, "tier": str}.

    Skips on:
    - UNIQUE(source, source_id) conflict (INSERT OR IGNORE)
    - cross-source dedup on canonical_url within 24h
    """
    from src.infra.db import get_db

    db = await get_db()

    canonical = _canonical_url(url)
    if await _cross_source_dedup(db, canonical, product_id):
        return {"ingested": False, "score": 0, "tier": "dedup_skip"}

    crm_match = await _check_crm_match(product_id, author)
    score = _score_mention(
        text=text,
        author_followers=author_followers,
        crm_match=crm_match,
        keywords=keywords,
    )
    tier = "silent" if score < 4 else ("immediate" if score >= 7 else "digest")

    try:
        cur = await db.execute(
            "INSERT OR IGNORE INTO mentions "
            "(product_id, source, source_id, url, canonical_url, author, "
            " author_followers, text, sentiment, signal_score) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                product_id, source, source_id, url, canonical,
                author, author_followers, text, sentiment, score,
            ),
        )
        await db.commit()
        # INSERT OR IGNORE: rowcount is 0 when the row was ignored (duplicate key)
        if cur.rowcount == 0:
            return {"ingested": False, "score": score, "tier": "dedup_skip"}
    except Exception as exc:
        logger.error("mention_polls: DB insert failed", error=str(exc))
        return {"ingested": False, "score": score, "tier": tier}

    if tier == "immediate":
        await _surface_founder_action(
            product_id=product_id,
            source=source,
            author=author,
            text=text[:500],
            score=score,
            url=url,
        )
        # Mark the mention as acted-on now that a founder_action has been
        # surfaced for it — this is the one place a mention is acted upon,
        # so it owns the acted_on write.
        try:
            await db.execute(
                "UPDATE mentions SET acted_on = 1 "
                "WHERE product_id = ? AND source = ? AND source_id = ?",
                (product_id, source, source_id),
            )
            await db.commit()
        except Exception as exc:
            logger.warning(
                "mention_polls: failed to set acted_on",
                product_id=product_id,
                source=source,
                source_id=source_id,
                error=str(exc),
            )

    return {"ingested": True, "score": score, "tier": tier}


async def _surface_founder_action(
    *,
    product_id: str,
    source: str,
    author: str,
    text: str,
    score: int,
    url: str | None,
) -> None:
    """Surface an immediate founder_action for high-signal mentions."""
    try:
        from packages.general_beckman.src.general_beckman import enqueue  # type: ignore
    except ImportError:
        try:
            from general_beckman import enqueue  # type: ignore
        except ImportError:
            logger.debug("mention_polls: general_beckman not available; skipping enqueue")
            return

    await enqueue(
        {
            "description": (
                f"[mention monitor] score={score} from {source}: "
                f"@{author} — {text[:200]}"
            ),
            "agent_type": "mechanical",
            "context": {
                "payload": {
                    "action": "notify_user",
                    "message": (
                        f"High-signal mention (score={score}) from {source}:\n"
                        f"@{author}: {text[:300]}"
                        + (f"\n{url}" if url else "")
                    ),
                }
            },
        }
    )


async def _check_crisis_threshold(product_id: str) -> bool:
    """Return True if >=3 neg-sentiment mentions with score>=4 arrived in the last 1h."""
    try:
        from src.infra.db import get_db
        db = await get_db()
        cutoff = (
            datetime.now(timezone.utc) - timedelta(hours=1)
        ).strftime("%Y-%m-%d %H:%M:%S")
        cur = await db.execute(
            "SELECT COUNT(*) FROM mentions "
            "WHERE product_id = ? AND sentiment = 'neg' "
            "AND signal_score >= 4 AND seen_at >= ?",
            (product_id, cutoff),
        )
        row = await cur.fetchone()
        return (row[0] if row else 0) >= 3
    except Exception:
        return False


async def _trigger_crisis_action(product_id: str) -> None:
    """Surface crisis_comms_draft founder_action for negative mention cluster."""
    try:
        from packages.general_beckman.src.general_beckman import enqueue  # type: ignore
    except ImportError:
        try:
            from general_beckman import enqueue  # type: ignore
        except ImportError:
            logger.debug("mention_polls: general_beckman not available for crisis trigger")
            return

    await enqueue(
        {
            "description": (
                f"[crisis_comms_draft] {product_id}: >=3 negative mentions in 1h"
            ),
            "agent_type": "mechanical",
            "context": {
                "payload": {
                    "action": "notify_user",
                    "message": (
                        f"CRISIS ALERT ({product_id}): 3+ negative mentions in the last hour. "
                        "Consider reviewing with /mention_monitor status. "
                        "Draft a crisis comms response? (B6)"
                    ),
                }
            },
        }
    )


# ---------------------------------------------------------------------------
# Source fetchers (mocked in tests)
# ---------------------------------------------------------------------------

async def _fetch_hn(product_name: str, config: dict) -> list[dict]:
    """Algolia HN Search API — free, no key needed."""
    import json
    try:
        from vecihi import fetch_json as vecihi_fetch  # type: ignore
    except ImportError:
        logger.debug("mention_polls.hn: vecihi not available")
        return []

    query = quote_plus(product_name)
    url = f"https://hn.algolia.com/api/v1/search_by_date?query={query}&tags=(story,comment)&hitsPerPage=30"
    try:
        raw = await vecihi_fetch(url)
        data = raw if isinstance(raw, dict) else json.loads(raw)
        hits = data.get("hits") or []
        results = []
        for h in hits:
            results.append({
                "source_id": str(h.get("objectID") or ""),
                "url": h.get("url") or f"https://news.ycombinator.com/item?id={h.get('objectID')}",
                "author": h.get("author") or "",
                "author_followers": 0,  # HN has no follower count
                "text": (h.get("comment_text") or h.get("story_text") or h.get("title") or ""),
            })
        return results
    except Exception as exc:
        logger.warning("mention_polls.hn: fetch failed", error=str(exc))
        return []


async def _fetch_reddit(product_name: str, config: dict) -> list[dict]:
    """PRAW search — free, requires REDDIT_CLIENT_ID + REDDIT_CLIENT_SECRET."""
    subreddits = config.get("subreddits") or ["all"]
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    if not client_id or not client_secret:
        logger.debug("mention_polls.reddit: REDDIT_CLIENT_ID/SECRET not set; skipping")
        return []
    try:
        import praw  # type: ignore
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent="kutai_mention_monitor/1.0",
        )
        results = []
        for sub_name in subreddits:
            sub = reddit.subreddit(sub_name)
            for submission in sub.search(product_name, sort="new", time_filter="day", limit=25):
                results.append({
                    "source_id": submission.id,
                    "url": f"https://www.reddit.com{submission.permalink}",
                    "author": str(submission.author) if submission.author else "",
                    "author_followers": 0,
                    "text": (submission.title + "\n" + (submission.selftext or ""))[:2000],
                })
        return results
    except ImportError:
        logger.debug("mention_polls.reddit: praw not installed; skipping")
        return []
    except Exception as exc:
        logger.warning("mention_polls.reddit: fetch failed", error=str(exc))
        return []


async def _fetch_google(product_name: str, config: dict) -> list[dict]:
    """Google Alerts RSS — free, no key needed."""
    try:
        from vecihi import fetch_rss as vecihi_rss  # type: ignore
    except ImportError:
        logger.debug("mention_polls.google: vecihi not available")
        return []

    query = quote_plus(f'"{product_name}"')
    # Google Alerts RSS: standard undocumented but stable endpoint
    url = f"https://www.google.com/alerts/feeds/0/{query}"
    try:
        entries = await vecihi_rss(url)
        results = []
        for e in entries[:30]:
            results.append({
                "source_id": e.get("id") or e.get("link") or "",
                "url": e.get("link") or "",
                "author": e.get("author") or "",
                "author_followers": 0,
                "text": (e.get("summary") or e.get("title") or "")[:2000],
            })
        return results
    except Exception as exc:
        logger.warning("mention_polls.google: fetch failed", error=str(exc))
        return []


async def _fetch_twitter(product_name: str, config: dict) -> list[dict]:
    """X API v2 — PAID. Off by default; gated behind MENTION_TWITTER_ENABLED=1."""
    if os.getenv("MENTION_TWITTER_ENABLED", "").strip() != "1":
        logger.debug("mention_polls.twitter: MENTION_TWITTER_ENABLED not set; skipping")
        return []
    bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
    if not bearer_token:
        logger.debug("mention_polls.twitter: TWITTER_BEARER_TOKEN not set; skipping")
        return []
    try:
        from vecihi import fetch_json as vecihi_fetch  # type: ignore
        import json

        query = quote_plus(f"{product_name} -is:retweet lang:en")
        url = (
            f"https://api.twitter.com/2/tweets/search/recent"
            f"?query={query}&max_results=25"
            f"&tweet.fields=author_id,created_at,public_metrics,text"
            f"&expansions=author_id"
            f"&user.fields=public_metrics,username"
        )
        raw = await vecihi_fetch(url, headers={"Authorization": f"Bearer {bearer_token}"})
        data = raw if isinstance(raw, dict) else json.loads(raw)
        tweets = data.get("data") or []
        users = {u["id"]: u for u in (data.get("includes") or {}).get("users") or []}
        results = []
        for t in tweets:
            author_id = t.get("author_id") or ""
            user = users.get(author_id) or {}
            pm = user.get("public_metrics") or {}
            results.append({
                "source_id": t.get("id") or "",
                "url": f"https://twitter.com/i/web/status/{t.get('id')}",
                "author": user.get("username") or author_id,
                "author_followers": pm.get("followers_count") or 0,
                "text": t.get("text") or "",
            })
        return results
    except ImportError:
        logger.debug("mention_polls.twitter: vecihi not available")
        return []
    except Exception as exc:
        logger.warning("mention_polls.twitter: fetch failed", error=str(exc))
        return []


async def _fetch_discord(product_name: str, config: dict) -> list[dict]:
    """Discord bot — founder OAuth-authorised. Reads specified channel_ids."""
    bot_token = os.getenv("DISCORD_BOT_TOKEN")
    channel_ids: list[str] = config.get("channel_ids") or []
    if not bot_token or not channel_ids:
        logger.debug("mention_polls.discord: DISCORD_BOT_TOKEN or channel_ids not set; skipping")
        return []
    try:
        from vecihi import fetch_json as vecihi_fetch  # type: ignore
        import json

        results = []
        for channel_id in channel_ids:
            url = f"https://discord.com/api/v10/channels/{channel_id}/messages?limit=50"
            raw = await vecihi_fetch(
                url, headers={"Authorization": f"Bot {bot_token}"}
            )
            messages = raw if isinstance(raw, list) else json.loads(raw)
            for msg in messages:
                content = msg.get("content") or ""
                if product_name.lower() not in content.lower():
                    continue
                author = (msg.get("author") or {}).get("username") or ""
                results.append({
                    "source_id": f"{channel_id}:{msg.get('id') or ''}",
                    "url": (
                        f"https://discord.com/channels/"
                        f"{(msg.get('guild_id') or 'unknown')}/{channel_id}/{msg.get('id') or ''}"
                    ),
                    "author": author,
                    "author_followers": 0,
                    "text": content[:2000],
                })
        return results
    except ImportError:
        logger.debug("mention_polls.discord: vecihi not available")
        return []
    except Exception as exc:
        logger.warning("mention_polls.discord: fetch failed", error=str(exc))
        return []


# ---------------------------------------------------------------------------
# Sentiment classifier (lightweight; no LLM)
# ---------------------------------------------------------------------------

_NEG_PATTERNS = re.compile(
    r"\b(broken|terrible|worst|scam|fraud|hate|useless|crash|awful|"
    r"never works|doesn't work|waste|refund|disappointed|frustrated|"
    r"cancel|buggy|horrible)\b",
    re.IGNORECASE,
)
_POS_PATTERNS = re.compile(
    r"\b(great|excellent|amazing|love|awesome|fantastic|perfect|"
    r"brilliant|helpful|recommend|best|outstanding|impressive)\b",
    re.IGNORECASE,
)


def _classify_sentiment(text: str) -> str:
    """Simple keyword-based sentiment classifier. Returns 'pos'|'neg'|'neu'."""
    neg = len(_NEG_PATTERNS.findall(text))
    pos = len(_POS_PATTERNS.findall(text))
    if neg > pos:
        return "neg"
    if pos > neg:
        return "pos"
    return "neu"


# ---------------------------------------------------------------------------
# Core poll logic
# ---------------------------------------------------------------------------

_FETCHERS = {
    "hn": _fetch_hn,
    "reddit": _fetch_reddit,
    "google": _fetch_google,
    "twitter": _fetch_twitter,
    "discord": _fetch_discord,
}


async def poll_source(
    source: str,
    product_id: str,
    product_name: str,
    config: dict | None = None,
) -> dict:
    """Fetch + ingest mentions from *source* for *product_id*.

    Returns:
        {"ingested": int, "immediate": int, "digest": int, "silent": int,
         "skipped": int, "crisis_triggered": bool}
    """
    if source not in SUPPORTED_SOURCES:
        return {
            "error": f"unsupported source: {source}",
            "ingested": 0, "skipped": 0,
            "immediate": 0, "digest": 0, "silent": 0,
            "crisis_triggered": False,
        }

    cfg = config or {}
    keywords = cfg.get("keywords") or [product_name]
    fetcher = _FETCHERS[source]

    raw_items = await fetcher(product_name, cfg)

    counts = {"ingested": 0, "immediate": 0, "digest": 0, "silent": 0, "skipped": 0}
    for item in raw_items:
        if not item.get("source_id"):
            continue
        text = item.get("text") or ""
        if not text.strip():
            continue
        sentiment = _classify_sentiment(text)
        result = await _ingest_mention(
            product_id=product_id,
            source=source,
            source_id=item["source_id"],
            url=item.get("url"),
            author=item.get("author") or "",
            author_followers=int(item.get("author_followers") or 0),
            text=text,
            sentiment=sentiment,
            keywords=keywords,
        )
        if not result["ingested"]:
            counts["skipped"] += 1
            continue
        counts["ingested"] += 1
        tier = result.get("tier") or "silent"
        if tier in counts:
            counts[tier] += 1

    crisis_triggered = False
    if counts["ingested"] > 0:
        crisis_triggered = await _check_crisis_threshold(product_id)
        if crisis_triggered:
            await _trigger_crisis_action(product_id)

    return {**counts, "crisis_triggered": crisis_triggered}


# ---------------------------------------------------------------------------
# mr_roboto executor entry point
# ---------------------------------------------------------------------------

async def run(payload: dict) -> dict:
    """Entry point: ``action == "mention_polls/<source>"`` dispatch.

    payload keys:
      source       (str)  — hn | reddit | google | twitter | discord
      product_id   (str)
      product_name (str)
      config       (dict, optional) — source-specific config
    """
    source = str(payload.get("source") or "")
    product_id = str(payload.get("product_id") or "")
    product_name = str(payload.get("product_name") or "")
    config = payload.get("config") or {}

    if not source:
        return {"status": "failed", "error": "mention_polls: missing 'source'"}
    if not product_id:
        return {"status": "failed", "error": "mention_polls: missing 'product_id'"}
    if not product_name:
        return {"status": "failed", "error": "mention_polls: missing 'product_name'"}

    result = await poll_source(
        source=source,
        product_id=product_id,
        product_name=product_name,
        config=config,
    )
    return {"status": "ok", **result}
