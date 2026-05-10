"""Z1 Tier 6B (P5) — web-grounded prior art search.

Module exposing :func:`find_prior_art` that escalates through public APIs
to assemble a structured "graveyard" report for an idea. Output schema is
documented in ``docs/i2p-evolution/01-pre-code-plan-v3.md`` lines 1159+.

Sources tier (escalating, stops on first sufficient response):

1. ``hn_algolia`` — HN search (https://hn.algolia.com/api/v1/search), no auth.
2. ``wikipedia`` — Wikipedia REST (https://en.wikipedia.org/w/api.php), no auth.
3. ``wayback`` — Wayback Machine availability + cdx.
4. ``product_hunt`` — public RSS feed (https://www.producthunt.com/feed).

Fallback: if any source returns 429 OR ≥3 sources return 0 results within
30s, fall back to the local ``prior_art_cache`` SQLite table. Cache hits
are annotated in ``search_summary.cache_hit_for_sources``.

URL resolution sweep: every candidate URL is HEAD-checked inside this
module. Unreachable URLs are flagged with ``status: "dead"``.

Q4 lock — this module backs the LLM-callable ``find_prior_art`` tool; the
mechanical post-hook ``prior_art_min_coverage`` (mr_roboto) gates the
artifact afterwards.
"""
from __future__ import annotations

import asyncio
import datetime as _dt
import hashlib
import json
import logging
import re
import sqlite3
from typing import Any

try:
    import aiohttp  # type: ignore
except Exception:  # pragma: no cover — aiohttp is a hard dep but defend
    aiohttp = None  # type: ignore[assignment]


logger = logging.getLogger("vecihi.prior_art")

SCHEMA_VERSION = "1"
DEFAULT_TTL_HOURS = 168  # 7 days

HN_ALGOLIA = "https://hn.algolia.com/api/v1/search"
WIKI_API = "https://en.wikipedia.org/w/api.php"
WAYBACK_AVAIL = "https://archive.org/wayback/available"
WAYBACK_CDX = "https://web.archive.org/cdx/search/cdx"
PRODUCT_HUNT_FEED = "https://www.producthunt.com/feed"

_SOURCE_TIMEOUT = 10.0  # per-source budget
_FALLBACK_TIMEOUT = 30.0  # total budget before falling back to cache
_DEFAULT_K = 10


# ---------------------------------------------------------------------------
# Cache (SQLite)
# ---------------------------------------------------------------------------
def _cache_db_path() -> str:
    """Resolve the prior_art_cache DB path.

    Honours ``KUTAI_DB_PATH`` (same env var src/infra/db.py honours), else
    falls back to ``./kutai.db`` relative to CWD. The table is created
    lazily by :func:`_ensure_cache_table`.
    """
    import os
    return os.environ.get("KUTAI_DB_PATH") or os.environ.get("DB_PATH") or "kutai.db"


def _ensure_cache_table(con: sqlite3.Connection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS prior_art_cache (
            domain_keywords_hash TEXT PRIMARY KEY,
            results_json TEXT NOT NULL,
            fetched_at TEXT NOT NULL,
            ttl_hours INTEGER NOT NULL DEFAULT 168
        )
        """
    )
    con.commit()


def _hash_keywords(domain_keywords: list[str]) -> str:
    norm = "|".join(sorted(k.strip().lower() for k in domain_keywords if k))
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def _read_cache(cache_key: str, db_path: str | None = None) -> dict[str, Any] | None:
    path = db_path or _cache_db_path()
    try:
        con = sqlite3.connect(path)
        try:
            _ensure_cache_table(con)
            cur = con.execute(
                "SELECT results_json, fetched_at, ttl_hours FROM "
                "prior_art_cache WHERE domain_keywords_hash = ?",
                (cache_key,),
            )
            row = cur.fetchone()
        finally:
            con.close()
    except Exception as e:  # pragma: no cover — DB blip
        logger.warning("prior_art cache read failed: %s", e)
        return None
    if not row:
        return None
    results_json, fetched_at, ttl_hours = row
    try:
        fetched = _dt.datetime.fromisoformat(fetched_at)
    except Exception:
        return None
    age = _dt.datetime.utcnow() - fetched
    if age.total_seconds() > int(ttl_hours) * 3600:
        return None
    try:
        return json.loads(results_json)
    except Exception:
        return None


def _write_cache(
    cache_key: str,
    report: dict[str, Any],
    ttl_hours: int = DEFAULT_TTL_HOURS,
    db_path: str | None = None,
) -> None:
    path = db_path or _cache_db_path()
    try:
        con = sqlite3.connect(path)
        try:
            _ensure_cache_table(con)
            con.execute(
                "INSERT OR REPLACE INTO prior_art_cache "
                "(domain_keywords_hash, results_json, fetched_at, ttl_hours) "
                "VALUES (?, ?, ?, ?)",
                (
                    cache_key,
                    json.dumps(report, ensure_ascii=False),
                    _dt.datetime.utcnow().isoformat(),
                    int(ttl_hours),
                ),
            )
            con.commit()
        finally:
            con.close()
    except Exception as e:  # pragma: no cover
        logger.warning("prior_art cache write failed: %s", e)


# ---------------------------------------------------------------------------
# Query construction
# ---------------------------------------------------------------------------
def _build_queries(idea_summary: str, domain_keywords: list[str]) -> list[str]:
    """Build a small set of probe queries.

    Strategy:
    - Seed: the idea summary truncated to 6-8 tokens.
    - Plus: each domain keyword paired with ``"saas"``, ``"tool"``,
      ``"startup"`` — three short phrases that match "graveyard"-style
      shutdown discussions on HN.
    """
    base = re.sub(r"\s+", " ", (idea_summary or "")).strip()
    tokens = base.split(" ")
    seed = " ".join(tokens[:8]) if tokens else ""
    out: list[str] = []
    if seed:
        out.append(seed)
    suffixes = ["saas", "tool", "startup"]
    for kw in (domain_keywords or [])[:3]:
        for suf in suffixes:
            phrase = f"{kw.strip()} {suf}".strip()
            if phrase and phrase not in out:
                out.append(phrase)
        if len(out) >= 5:
            break
    return out[:5]


# ---------------------------------------------------------------------------
# Source fetchers
# ---------------------------------------------------------------------------
async def _fetch_json(
    session: "aiohttp.ClientSession",
    url: str,
    params: dict[str, Any] | None = None,
    timeout: float = _SOURCE_TIMEOUT,
) -> tuple[int, Any]:
    """Return (status, parsed_json or text). Raises only on network error."""
    async with session.get(url, params=params, timeout=timeout) as r:
        status = r.status
        try:
            data = await r.json(content_type=None)
        except Exception:
            data = await r.text()
        return status, data


async def _query_hn(
    session: "aiohttp.ClientSession", queries: list[str]
) -> tuple[list[dict[str, Any]], int | None]:
    """Return (hits, status_429_seen). One probe per query, max 3 queries."""
    hits: list[dict[str, Any]] = []
    rate_limited = None
    for q in queries[:3]:
        try:
            status, data = await _fetch_json(
                session, HN_ALGOLIA,
                params={"query": q, "tags": "story", "hitsPerPage": 20},
            )
            if status == 429:
                rate_limited = 429
                break
            if isinstance(data, dict):
                hits.extend(data.get("hits") or [])
        except Exception as e:
            logger.debug("hn query failed: %s", e)
    return hits, rate_limited


async def _query_wikipedia(
    session: "aiohttp.ClientSession", queries: list[str]
) -> tuple[list[dict[str, Any]], int | None]:
    hits: list[dict[str, Any]] = []
    rate_limited = None
    for q in queries[:3]:
        try:
            status, data = await _fetch_json(
                session, WIKI_API,
                params={
                    "action": "query", "list": "search", "format": "json",
                    "srsearch": q, "srlimit": 10,
                },
            )
            if status == 429:
                rate_limited = 429
                break
            if isinstance(data, dict):
                results = (data.get("query") or {}).get("search") or []
                hits.extend(results)
        except Exception as e:
            logger.debug("wiki query failed: %s", e)
    return hits, rate_limited


async def _query_wayback(
    session: "aiohttp.ClientSession", url: str
) -> dict[str, Any] | None:
    """Return Wayback availability info for ``url`` or None."""
    try:
        status, data = await _fetch_json(
            session, WAYBACK_AVAIL, params={"url": url},
        )
        if status == 429 or not isinstance(data, dict):
            return None
        snap = (
            data.get("archived_snapshots", {})
            .get("closest", {})
        )
        if not snap:
            return None
        return {
            "wayback_first_capture": snap.get("timestamp"),
            "wayback_url": snap.get("url"),
            "wayback_available": bool(snap.get("available")),
        }
    except Exception as e:
        logger.debug("wayback query failed: %s", e)
        return None


async def _query_product_hunt(
    session: "aiohttp.ClientSession", query: str
) -> tuple[list[dict[str, Any]], int | None]:
    """Public RSS feed scrape. Title-only filter."""
    try:
        async with session.get(
            PRODUCT_HUNT_FEED, timeout=_SOURCE_TIMEOUT,
        ) as r:
            if r.status == 429:
                return [], 429
            text = await r.text()
    except Exception as e:
        logger.debug("product_hunt query failed: %s", e)
        return [], None
    items: list[dict[str, Any]] = []
    # Token-based match: any 4+ char word from query present in title.
    qtokens = {t.lower() for t in re.findall(r"[a-zA-Z]{4,}", query or "")}
    item_blocks = re.findall(
        r"<item>(.*?)</item>", text, flags=re.DOTALL
    )
    for blk in item_blocks[:50]:
        title_m = re.search(r"<title><!\[CDATA\[(.*?)\]\]></title>", blk, re.DOTALL) or \
            re.search(r"<title>(.*?)</title>", blk, re.DOTALL)
        link_m = re.search(r"<link>(.*?)</link>", blk, re.DOTALL)
        if not title_m or not link_m:
            continue
        title = title_m.group(1).strip()
        link = link_m.group(1).strip()
        title_tokens = {t.lower() for t in re.findall(r"[a-zA-Z]{4,}", title)}
        if not qtokens or (qtokens & title_tokens):
            items.append({"title": title, "url": link, "source": "product_hunt"})
    return items, None


# ---------------------------------------------------------------------------
# Normalization & classification
# ---------------------------------------------------------------------------
def _normalize_hn(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for h in hits:
        title = h.get("title") or h.get("story_title") or ""
        if not title:
            continue
        url = h.get("url") or h.get("story_url") or ""
        out.append({
            "name": title.strip(),
            "url": url,
            "thesis_summary": title.strip(),
            "sources": [
                f"https://news.ycombinator.com/item?id={h.get('objectID', '')}"
            ],
            "_source_kind": "hn_algolia",
            "founded_year": None,
            "status": "unknown",
            "evidence_refs": ["agent_inference"],
        })
    return out


def _normalize_wiki(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for h in hits:
        title = (h.get("title") or "").strip()
        if not title:
            continue
        slug = title.replace(" ", "_")
        url = f"https://en.wikipedia.org/wiki/{slug}"
        snippet = re.sub(r"<[^>]+>", "", h.get("snippet") or "")
        out.append({
            "name": title,
            "url": url,
            "thesis_summary": snippet[:200],
            "sources": [url],
            "_source_kind": "wikipedia",
            "founded_year": None,
            "status": "unknown",
            "evidence_refs": ["agent_inference"],
        })
    return out


def _normalize_ph(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for h in hits:
        title = (h.get("title") or "").strip()
        if not title:
            continue
        out.append({
            "name": title,
            "url": h.get("url", ""),
            "thesis_summary": title,
            "sources": [h.get("url", "")] if h.get("url") else [],
            "_source_kind": "product_hunt",
            "founded_year": None,
            "status": "unknown",
            "evidence_refs": ["agent_inference"],
        })
    return out


def _dedup_by_name(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for it in items:
        key = (it.get("name") or "").strip().lower()
        if not key:
            continue
        if key in seen:
            # Merge sources
            srcs = list(seen[key].get("sources") or [])
            for s in (it.get("sources") or []):
                if s and s not in srcs:
                    srcs.append(s)
            seen[key]["sources"] = srcs
        else:
            seen[key] = it
    return list(seen.values())


def _split_by_relevance(
    candidates: list[dict[str, Any]], idea_summary: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Naive split: items whose name shares a token with idea_summary
    are 'attempted', else 'adjacent'."""
    tokens = {t.lower() for t in re.findall(r"[a-zA-Z]{4,}", idea_summary or "")}
    attempted: list[dict[str, Any]] = []
    adjacent: list[dict[str, Any]] = []
    for c in candidates:
        name_tokens = {t.lower() for t in re.findall(r"[a-zA-Z]{4,}", c.get("name", ""))}
        if tokens & name_tokens:
            attempted.append(c)
        else:
            adjacent.append(c)
    # If everything matched as adjacent, promote up to 5 to attempted
    if not attempted and adjacent:
        attempted = adjacent[:5]
        adjacent = adjacent[5:]
    return attempted, adjacent


def _extract_lessons(attempted: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Surface a key_lesson per dead/dormant solution we have evidence for.

    Naive but honest: one lesson per dead/dormant entry, citing its source.
    LLM caller can re-write these — this module just guarantees structure.
    """
    lessons: list[dict[str, Any]] = []
    for i, sol in enumerate(attempted):
        if sol.get("status") in ("dead", "dormant"):
            lessons.append({
                "lesson_id": f"L-{i+1:03d}",
                "lesson": (
                    f"{sol.get('name')} ({sol.get('status')}) — "
                    f"investigate failure mode before competing on the same surface."
                ),
                "evidence_refs": list(sol.get("sources") or []) or ["agent_inference"],
                "applies_to_us": "Document a concrete differentiator before phase 2.",
            })
    return lessons


def _classify_verdict(
    attempted: list[dict[str, Any]],
    total_inspected: int,
    queries: list[str],
) -> str:
    graveyard = sum(1 for s in attempted if s.get("status") in ("dead", "dormant"))
    if graveyard >= 2:
        return "graveyard_well_populated"
    if attempted:
        return "graveyard_thin"
    if len(queries) >= 3 and total_inspected >= 20:
        return "blue_ocean_validated"
    return "blue_ocean_suspicious"


# ---------------------------------------------------------------------------
# URL resolution sweep
# ---------------------------------------------------------------------------
async def _head_ok(
    session: "aiohttp.ClientSession", url: str, timeout: float = 6.0,
) -> bool:
    if not url or not url.startswith(("http://", "https://")):
        return False
    try:
        async with session.head(
            url, timeout=timeout, allow_redirects=True,
        ) as r:
            if 200 <= r.status < 400:
                return True
            # Some sites 405 HEAD; fall through to GET probe with byte limit.
            if r.status in (403, 405):
                pass
            else:
                return False
    except Exception:
        pass
    try:
        async with session.get(url, timeout=timeout) as r:
            return 200 <= r.status < 400
    except Exception:
        return False


async def _resolve_urls(
    session: "aiohttp.ClientSession", items: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Validate each item's URL is reachable; mark dead ones."""
    if not items:
        return items
    coros = [_head_ok(session, it.get("url", "")) for it in items]
    try:
        results = await asyncio.gather(*coros, return_exceptions=True)
    except Exception:
        results = [False] * len(items)
    for it, ok in zip(items, results):
        if isinstance(ok, BaseException):
            ok = False
        if not ok:
            it["status"] = "dead"
    return items


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _annotate_cache_hit(
    cached: dict[str, Any], sources: list[str]
) -> dict[str, Any]:
    out = json.loads(json.dumps(cached))  # deep copy
    summary = out.setdefault("search_summary", {})
    existing = list(summary.get("cache_hit_for_sources") or [])
    for s in sources:
        if s not in existing:
            existing.append(s)
    summary["cache_hit_for_sources"] = existing
    return out


async def find_prior_art(
    idea_summary: str,
    domain_keywords: list[str] | None = None,
    k: int = _DEFAULT_K,
    ambition_tier: str = "private_beta",
    *,
    db_path: str | None = None,
    session: "aiohttp.ClientSession | None" = None,
    ttl_hours: int = DEFAULT_TTL_HOURS,
) -> dict[str, Any]:
    """Run a web-grounded prior-art search.

    See module docstring for sources, fallback rules, and schema.

    Parameters
    ----------
    idea_summary:
        One-paragraph summary of the idea (used for query construction +
        relevance split).
    domain_keywords:
        Cluster of 1-5 domain words ("price tracker", "trendyol",
        "saas reseller"). Hashed to form the cache key.
    k:
        Top-k attempted_solutions to return.
    ambition_tier:
        ``"private_beta"`` / ``"public_launch"`` / ``"revenue_product"``.
        Higher tiers add Wayback validation + Product Hunt probes.
    db_path:
        Override SQLite path (tests).
    session:
        Optional aiohttp session (tests inject mocks); otherwise a session
        is created internally.
    ttl_hours:
        Cache TTL for fresh writes (default 168 = 7 days).
    """
    if aiohttp is None:  # pragma: no cover — defended at top of module
        raise RuntimeError("aiohttp required for vecihi.prior_art")

    domain_keywords = list(domain_keywords or [])
    queries = _build_queries(idea_summary, domain_keywords)
    cache_key = _hash_keywords(domain_keywords or [idea_summary[:64]])

    # Cache fast-path
    cached = _read_cache(cache_key, db_path=db_path)
    # We always try fresh first; fall back to cache only on rate-limit /
    # empty-results trigger. Pre-load cached for that branch.

    candidates: list[dict[str, Any]] = []
    sources_used: list[str] = []
    rate_limit_hits: list[dict[str, str]] = []
    total_inspected = 0
    empty_count = 0

    own_session = False
    if session is None:
        session = aiohttp.ClientSession()
        own_session = True

    started = asyncio.get_event_loop().time()
    try:
        # Tier 1: HN
        try:
            hn_hits, hn_rate = await asyncio.wait_for(
                _query_hn(session, queries), timeout=_SOURCE_TIMEOUT + 1,
            )
            if hn_rate:
                rate_limit_hits.append({"source": "hn_algolia", "err": "429"})
            if hn_hits:
                sources_used.append("hn_algolia")
                norm = _normalize_hn(hn_hits)
                candidates.extend(norm)
                total_inspected += len(hn_hits)
            else:
                empty_count += 1
                sources_used.append("hn_algolia")
        except asyncio.TimeoutError:
            rate_limit_hits.append({"source": "hn_algolia", "err": "timeout"})
            empty_count += 1

        # Tier 2: Wikipedia
        try:
            wiki_hits, wiki_rate = await asyncio.wait_for(
                _query_wikipedia(session, queries), timeout=_SOURCE_TIMEOUT + 1,
            )
            if wiki_rate:
                rate_limit_hits.append({"source": "wikipedia", "err": "429"})
            if wiki_hits:
                sources_used.append("wikipedia")
                candidates.extend(_normalize_wiki(wiki_hits))
                total_inspected += len(wiki_hits)
            else:
                empty_count += 1
                sources_used.append("wikipedia")
        except asyncio.TimeoutError:
            rate_limit_hits.append({"source": "wikipedia", "err": "timeout"})
            empty_count += 1

        # Tier 3: Wayback (per top-k candidate, private_beta+)
        if (ambition_tier in ("private_beta", "public_launch", "revenue_product")
                and candidates):
            sources_used.append("wayback")
            top = candidates[:k]
            wb_coros = [
                _query_wayback(session, c["url"]) for c in top if c.get("url")
            ]
            if wb_coros:
                try:
                    wb_results = await asyncio.wait_for(
                        asyncio.gather(*wb_coros, return_exceptions=True),
                        timeout=_SOURCE_TIMEOUT + 4,
                    )
                except asyncio.TimeoutError:
                    wb_results = [None] * len(wb_coros)
                # Apply the wayback fields back to the candidates that had urls
                idx = 0
                for c in top:
                    if not c.get("url"):
                        continue
                    res = wb_results[idx] if idx < len(wb_results) else None
                    idx += 1
                    if isinstance(res, dict):
                        c.update({k: v for k, v in res.items()
                                 if k.startswith("wayback_")})
                        # If it's archived but unreachable, mark dormant.
                        if res.get("wayback_first_capture") and not c.get("status"):
                            c["status"] = "dormant"

        # Tier 4: Product Hunt (public_launch+)
        if ambition_tier in ("public_launch", "revenue_product") and queries:
            try:
                ph_hits, ph_rate = await asyncio.wait_for(
                    _query_product_hunt(session, queries[0]),
                    timeout=_SOURCE_TIMEOUT + 1,
                )
                if ph_rate:
                    rate_limit_hits.append({"source": "product_hunt", "err": "429"})
                if ph_hits:
                    sources_used.append("product_hunt")
                    candidates.extend(_normalize_ph(ph_hits))
                    total_inspected += len(ph_hits)
            except asyncio.TimeoutError:
                rate_limit_hits.append({"source": "product_hunt", "err": "timeout"})

        # Fallback rule: any source 429 OR ≥3 empty within 30s
        elapsed = asyncio.get_event_loop().time() - started
        should_fallback = bool(rate_limit_hits) or (
            empty_count >= 3 and elapsed <= _FALLBACK_TIMEOUT
        )
        if should_fallback and cached:
            return _annotate_cache_hit(
                cached,
                sources=cached.get("search_summary", {}).get(
                    "sources_used", []
                ) or sources_used,
            )

        # Dedup + URL resolution sweep (broken URLs marked dead)
        candidates = _dedup_by_name(candidates)
        candidates = await _resolve_urls(session, candidates)
    finally:
        if own_session:
            try:
                await session.close()
            except Exception:
                pass

    attempted, adjacent = _split_by_relevance(candidates, idea_summary)
    lessons = _extract_lessons(attempted)
    graveyard_count = sum(
        1 for s in attempted if s.get("status") in ("dead", "dormant")
    )
    verdict = _classify_verdict(attempted, total_inspected, queries)

    # Strip private fields
    def _clean(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{k: v for k, v in it.items() if not k.startswith("_")}
                for it in items]

    report = {
        "_schema_version": SCHEMA_VERSION,
        "search_summary": {
            "queries_run": queries[:5],
            "sources_used": sources_used,
            "rate_limit_hits": rate_limit_hits,
            "total_results_inspected": total_inspected,
            "cache_hit_for_sources": [],
        },
        "attempted_solutions": _clean(attempted[:k]),
        "adjacent_failures": _clean(adjacent[:5]),
        "key_lessons": lessons,
        "graveyard_count": graveyard_count,
        "verdict": verdict,
    }

    # Best-effort cache write (skip in pure-cache fallback)
    try:
        _write_cache(cache_key, report, ttl_hours=ttl_hours, db_path=db_path)
    except Exception:  # pragma: no cover
        pass

    return report


__all__ = [
    "find_prior_art",
    "SCHEMA_VERSION",
    "DEFAULT_TTL_HOURS",
]
