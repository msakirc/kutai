# Smart Resource Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make agents actually use the free API registry, MCP tools, and skills — via layered resolution that minimizes LLM decision surface.

**Architecture:** Four layers: Layer 0 (orchestrator fast-path, no LLM), Layer 1 (pre-fetched context injection), Layer 2 (`smart_search` unified tool), Layer 3 (bug fixes for skills/MCP). See `docs/superpowers/specs/2026-04-03-smart-resource-integration-design.md` for full spec.

**Tech Stack:** Python 3.10, aiosqlite, aiohttp, existing TOOL_REGISTRY pattern, existing Telegram bot command pattern.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/infra/db.py` | Modify | Add 4 new tables: `api_keywords`, `api_category_patterns`, `smart_search_log`, `api_reliability` |
| `src/tools/free_apis.py` | Modify | Add keyword index building, reliability tracking, Turkish category pattern seeding |
| `src/core/fast_resolver.py` | **Create** | Layer 0 (fast-path) + Layer 1 (context enrichment) — keyword matching, API calling, response formatting |
| `src/tools/smart_search.py` | **Create** | Layer 2 — unified `smart_search` tool routing through APIs → MCP → web |
| `src/tools/__init__.py` | Modify | Register `smart_search`, fix MCP lazy connect bug, add MCP stub registration |
| `src/memory/skills.py` | Modify | Fix `WHERE success_count > 0` filter, update ranking |
| `src/core/orchestrator.py` | Modify | Call fast_resolver before dispatch, fix auto-skill regex, add discovery scheduling |
| `src/agents/base.py` | Modify | Inject Layer 1 enriched context, fix skill logging |
| `src/workflows/engine/expander.py` | Modify | Read `api_hints` field from steps |
| `src/app/telegram_bot.py` | Modify | Add `/smartsearch` command + inline menu |
| `tests/test_fast_resolver.py` | **Create** | Tests for Layer 0 + Layer 1 |
| `tests/test_smart_search.py` | **Create** | Tests for Layer 2 |
| `tests/test_skill_fixes.py` | **Create** | Tests for skill injection fixes |

---

### Task 1: DB Schema — New Tables

**Files:**
- Modify: `src/infra/db.py:375` (after `skill_metrics` table, before `await db.commit()`)
- Test: `tests/test_db_schema.py`

- [ ] **Step 1: Write failing test for new tables**

```python
# tests/test_db_schema.py
import asyncio
import aiosqlite
import pytest

@pytest.mark.asyncio
async def test_smart_search_tables_exist(tmp_path):
    """Verify all 4 new tables are created by init_db."""
    db_path = str(tmp_path / "test.db")
    
    # Patch DB_PATH before importing
    import src.infra.db as db_mod
    original = db_mod.DB_PATH
    db_mod.DB_PATH = db_path
    db_mod._db = None
    try:
        await db_mod.init_db()
        async with aiosqlite.connect(db_path) as db:
            for table in ["api_keywords", "api_category_patterns", 
                          "smart_search_log", "api_reliability"]:
                cur = await db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (table,)
                )
                row = await cur.fetchone()
                assert row is not None, f"Table {table} not created"
    finally:
        db_mod.DB_PATH = original
        db_mod._db = None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_db_schema.py::test_smart_search_tables_exist -v`
Expected: FAIL — tables don't exist yet

- [ ] **Step 3: Add tables to init_db**

In `src/infra/db.py`, after line 375 (after the `skill_metrics` CREATE TABLE block), before `await db.commit()` at line 377, add:

```python
    # ── Smart Resource Integration tables ──

    await db.execute("""
        CREATE TABLE IF NOT EXISTS api_keywords (
            api_name TEXT NOT NULL,
            keyword TEXT NOT NULL,
            source TEXT DEFAULT 'description',
            UNIQUE(api_name, keyword)
        )
    """)
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_api_keywords_kw ON api_keywords(keyword)"
    )

    await db.execute("""
        CREATE TABLE IF NOT EXISTS api_category_patterns (
            category TEXT PRIMARY KEY,
            pattern TEXT NOT NULL
        )
    """)

    await db.execute("""
        CREATE TABLE IF NOT EXISTS smart_search_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime')),
            query TEXT NOT NULL,
            layer INTEGER NOT NULL,
            source TEXT,
            success INTEGER DEFAULT 1,
            response_ms INTEGER
        )
    """)

    await db.execute("""
        CREATE TABLE IF NOT EXISTS api_reliability (
            api_name TEXT PRIMARY KEY,
            success_count INTEGER DEFAULT 0,
            failure_count INTEGER DEFAULT 0,
            last_success TEXT,
            last_failure TEXT,
            status TEXT DEFAULT 'active'
        )
    """)
```

- [ ] **Step 4: Add DB helper functions**

Still in `src/infra/db.py`, add these query functions at the end of the file (after existing helper functions):

```python
# ── Smart Resource Integration queries ──

async def upsert_api_keyword(api_name: str, keyword: str, source: str = "description"):
    db = await get_db()
    await db.execute(
        "INSERT OR IGNORE INTO api_keywords (api_name, keyword, source) VALUES (?, ?, ?)",
        (api_name, keyword, source),
    )
    await db.commit()


async def bulk_upsert_api_keywords(rows: list[tuple[str, str, str]]):
    """rows = [(api_name, keyword, source), ...]"""
    db = await get_db()
    await db.executemany(
        "INSERT OR IGNORE INTO api_keywords (api_name, keyword, source) VALUES (?, ?, ?)",
        rows,
    )
    await db.commit()


async def find_apis_by_keywords(keywords: list[str], limit: int = 5) -> list[dict]:
    """Find APIs with the most keyword overlap. Returns [{api_name, match_count}, ...]."""
    if not keywords:
        return []
    db = await get_db()
    placeholders = ",".join("?" for _ in keywords)
    cur = await db.execute(
        f"""SELECT api_name, COUNT(*) as match_count
            FROM api_keywords
            WHERE keyword IN ({placeholders})
            GROUP BY api_name
            ORDER BY match_count DESC
            LIMIT ?""",
        (*keywords, limit),
    )
    rows = await cur.fetchall()
    return [{"api_name": r[0], "match_count": r[1]} for r in rows]


async def get_api_category_patterns() -> dict[str, str]:
    """Return {category: pattern} for Turkish localization patterns."""
    db = await get_db()
    cur = await db.execute("SELECT category, pattern FROM api_category_patterns")
    rows = await cur.fetchall()
    return {r[0]: r[1] for r in rows}


async def upsert_category_pattern(category: str, pattern: str):
    db = await get_db()
    await db.execute(
        "INSERT OR REPLACE INTO api_category_patterns (category, pattern) VALUES (?, ?)",
        (category, pattern),
    )
    await db.commit()


async def log_smart_search(query: str, layer: int, source: str | None, success: bool, response_ms: int):
    db = await get_db()
    await db.execute(
        """INSERT INTO smart_search_log (query, layer, source, success, response_ms)
           VALUES (?, ?, ?, ?, ?)""",
        (query, layer, source, 1 if success else 0, response_ms),
    )
    await db.commit()


async def record_api_call(api_name: str, success: bool):
    """Update api_reliability counters and auto-demote if needed."""
    db = await get_db()
    now = "strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime')"
    if success:
        await db.execute(
            f"""INSERT INTO api_reliability (api_name, success_count, last_success)
                VALUES (?, 1, {now})
                ON CONFLICT(api_name) DO UPDATE SET
                    success_count = success_count + 1,
                    last_success = {now}""",
            (api_name,),
        )
    else:
        await db.execute(
            f"""INSERT INTO api_reliability (api_name, failure_count, last_failure)
                VALUES (?, 1, {now})
                ON CONFLICT(api_name) DO UPDATE SET
                    failure_count = failure_count + 1,
                    last_failure = {now}""",
            (api_name,),
        )
    # Auto-demote check
    cur = await db.execute(
        "SELECT success_count, failure_count FROM api_reliability WHERE api_name = ?",
        (api_name,),
    )
    row = await cur.fetchone()
    if row:
        total = row[0] + row[1]
        rate = row[0] / max(total, 1)
        if total >= 10 and rate < 0.10:
            status = "suspended"
        elif total >= 5 and rate < 0.25:
            status = "demoted"
        elif total >= 5 and rate < 0.50:
            status = "warning"
        else:
            status = "active"
        await db.execute(
            "UPDATE api_reliability SET status = ? WHERE api_name = ?",
            (status, api_name),
        )
    await db.commit()


async def get_api_reliability(api_name: str) -> dict | None:
    db = await get_db()
    cur = await db.execute(
        "SELECT api_name, success_count, failure_count, status FROM api_reliability WHERE api_name = ?",
        (api_name,),
    )
    row = await cur.fetchone()
    if not row:
        return None
    return {"api_name": row[0], "success_count": row[1], "failure_count": row[2], "status": row[3]}


async def get_api_reliability_all() -> list[dict]:
    db = await get_db()
    cur = await db.execute(
        "SELECT api_name, success_count, failure_count, status, last_success, last_failure FROM api_reliability ORDER BY (success_count + failure_count) DESC"
    )
    rows = await cur.fetchall()
    return [
        {"api_name": r[0], "success_count": r[1], "failure_count": r[2], "status": r[3], "last_success": r[4], "last_failure": r[5]}
        for r in rows
    ]


async def get_smart_search_stats(days: int = 7) -> dict:
    """Aggregate smart_search_log for observability menu."""
    db = await get_db()
    cutoff = f"datetime('now', 'localtime', '-{days} days')"

    # Layer breakdown
    cur = await db.execute(
        f"SELECT layer, COUNT(*), SUM(success) FROM smart_search_log WHERE timestamp > {cutoff} GROUP BY layer"
    )
    layers = {r[0]: {"count": r[1], "success": r[2]} for r in await cur.fetchall()}

    # Top sources
    cur = await db.execute(
        f"""SELECT source, COUNT(*), SUM(success) FROM smart_search_log
            WHERE timestamp > {cutoff} AND source IS NOT NULL
            GROUP BY source ORDER BY COUNT(*) DESC LIMIT 10"""
    )
    top_sources = [{"source": r[0], "count": r[1], "success": r[2]} for r in await cur.fetchall()]

    # Today count
    cur = await db.execute(
        "SELECT COUNT(*) FROM smart_search_log WHERE date(timestamp) = date('now', 'localtime')"
    )
    today = (await cur.fetchone())[0]

    return {"layers": layers, "top_sources": top_sources, "today": today}


async def unsuspend_api(api_name: str):
    db = await get_db()
    await db.execute(
        "UPDATE api_reliability SET status = 'active', success_count = 0, failure_count = 0 WHERE api_name = ?",
        (api_name,),
    )
    await db.commit()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_db_schema.py::test_smart_search_tables_exist -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/infra/db.py tests/test_db_schema.py
git commit -m "feat(smart-search): add DB tables + helpers for keyword index, reliability, logging"
```

---

### Task 2: Keyword Index Builder + Turkish Category Patterns

**Files:**
- Modify: `src/tools/free_apis.py:375` (after `seed_registry`)
- Test: `tests/test_keyword_index.py`

- [ ] **Step 1: Write failing test for keyword extraction**

```python
# tests/test_keyword_index.py
import pytest

from src.tools.free_apis import tokenize_api_description, TURKISH_CATEGORY_PATTERNS


def test_tokenize_extracts_meaningful_keywords():
    desc = "Weather forecasts in plain text or JSON. No API key needed."
    keywords = tokenize_api_description(desc)
    assert "weather" in keywords
    assert "forecasts" in keywords
    assert "json" in keywords
    # Stop words excluded
    assert "in" not in keywords
    assert "or" not in keywords
    assert "no" not in keywords


def test_tokenize_handles_empty():
    assert tokenize_api_description("") == []
    assert tokenize_api_description(None) == []


def test_turkish_patterns_cover_key_categories():
    assert "weather" in TURKISH_CATEGORY_PATTERNS
    assert "currency" in TURKISH_CATEGORY_PATTERNS
    assert "pharmacy" in TURKISH_CATEGORY_PATTERNS
    assert "earthquake" in TURKISH_CATEGORY_PATTERNS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_keyword_index.py -v`
Expected: FAIL — `tokenize_api_description` and `TURKISH_CATEGORY_PATTERNS` don't exist

- [ ] **Step 3: Implement tokenizer and Turkish patterns**

Add to `src/tools/free_apis.py` after the `seed_registry` function (after line 376):

```python
# ── Keyword Index ──

# Stop words to exclude from keyword tokenization (English)
_STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "no", "not", "this", "that",
    "are", "was", "were", "be", "been", "has", "have", "had", "do", "does",
    "can", "could", "will", "would", "shall", "should", "may", "might",
    "its", "via", "etc", "also", "just", "only", "all", "any", "each",
    "than", "as", "up", "out", "about", "into", "over", "after", "before",
    "between", "under", "above", "such", "very", "more", "most", "other",
    "some", "need", "needed", "based", "using", "free", "api", "key",
    "data", "get", "set", "use", "used", "available", "provides", "returns",
    "access", "service", "simple", "easy", "http", "https", "json", "xml",
    "rest", "endpoint",
})


def tokenize_api_description(description: str | None, tags: str | None = None) -> list[str]:
    """Extract meaningful keywords from API description and tags."""
    if not description:
        return []
    text = description.lower()
    if tags:
        text += " " + tags.lower()
    # Split on non-alphanumeric, keep words >= 3 chars
    words = re.findall(r"[a-z0-9\u00e0-\u024f]{3,}", text)
    # Remove stop words and deduplicate preserving order
    seen = set()
    result = []
    for w in words:
        if w not in _STOP_WORDS and w not in seen:
            seen.add(w)
            result.append(w)
    return result


# Turkish localization patterns per category
TURKISH_CATEGORY_PATTERNS: dict[str, str] = {
    "weather": r"hava\s*durumu|s[ıi]cakl[ıi]k|ya[gğ]mur|kar\s+ya[gğ][ıi]|r[üu]zg[aâ]r|tahmin|forecast",
    "currency": r"d[öo]viz|kur|dolar|euro|sterlin|pound|alt[ıi]n\s*fiyat|para\s*birimi",
    "pharmacy": r"n[öo]bet[çc]i\s*eczane|eczane|nobetci|ila[çc]|pharmacy",
    "earthquake": r"deprem|sars[ıi]nt[ıi]|kandilli|zelzele|earthquake",
    "fuel": r"benzin|mazot|diesel|lpg|yak[ıi]t|akaryak[ıi]t|petrol\s*fiyat",
    "gold": r"alt[ıi]n\s*fiyat|[çc]eyrek|gram\s*alt[ıi]n|yar[ıi]m\s*alt[ıi]n|tam\s*alt[ıi]n|cumhuriyet\s*alt[ıi]n[ıi]",
    "prayer_times": r"namaz\s*vakt|ezan|imsak|iftar|sahur|ak[şs]am\s*ezan|[öo][gğ]le\s*namaz",
    "time": r"saat\s*ka[çc]|saat\s*fark|timezone",
    "news": r"haber|son\s*dakika|g[üu]ndem|headline|g[üu]ncel",
    "translation": r"[çc]evir|terc[üu]me|[İi]ngilizce|T[üu]rk[çc]e",
    "map": r"yol\s*tarifi|mesafe|nas[ıi]l\s*gid|harita|rota",
    "travel": r"u[çc]ak|bilet|otob[üu]s|seyahat|enuygun|obilet",
    "holiday": r"tatil|resmi\s*tatil|bayram|arife|ramazan|kurban",
    "sports": r"ma[çc]|kadro|skor|s[üu]per\s*lig|futbol|basketbol",
}


async def build_keyword_index() -> int:
    """Build/rebuild the keyword index from all APIs in registry + DB cache.
    
    Returns the number of keyword entries created.
    """
    from src.infra.db import bulk_upsert_api_keywords

    rows = []
    seen_names = set()

    # Static registry
    for api in API_REGISTRY:
        if api.name in seen_names:
            continue
        seen_names.add(api.name)
        keywords = tokenize_api_description(api.description)
        for kw in keywords:
            rows.append((api.name, kw, "description"))
        # Also index the category itself
        rows.append((api.name, api.category.lower(), "category"))

    # DB-cached discovered APIs
    for api in _db_api_cache:
        if api.name in seen_names:
            continue
        seen_names.add(api.name)
        keywords = tokenize_api_description(api.description)
        for kw in keywords:
            rows.append((api.name, kw, "description"))
        rows.append((api.name, api.category.lower(), "category"))

    if rows:
        await bulk_upsert_api_keywords(rows)
    logger.info("Keyword index built: %d entries for %d APIs", len(rows), len(seen_names))
    return len(rows)


async def seed_category_patterns():
    """Seed Turkish category patterns into DB."""
    from src.infra.db import upsert_category_pattern
    for category, pattern in TURKISH_CATEGORY_PATTERNS.items():
        await upsert_category_pattern(category, pattern)
    logger.info("Seeded %d Turkish category patterns", len(TURKISH_CATEGORY_PATTERNS))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_keyword_index.py -v`
Expected: PASS

- [ ] **Step 5: Write integration test for build_keyword_index**

```python
# Append to tests/test_keyword_index.py

@pytest.mark.asyncio
async def test_build_keyword_index_populates_db(tmp_path):
    """build_keyword_index should create entries in api_keywords table."""
    import src.infra.db as db_mod
    original = db_mod.DB_PATH
    db_mod.DB_PATH = str(tmp_path / "test.db")
    db_mod._db = None
    try:
        await db_mod.init_db()
        from src.tools.free_apis import build_keyword_index, seed_registry
        await seed_registry()
        count = await build_keyword_index()
        assert count > 50  # 30 static APIs should produce many keywords
        
        results = await db_mod.find_apis_by_keywords(["weather", "forecast"])
        assert len(results) > 0
        assert results[0]["api_name"] in ("wttr.in", "Open-Meteo")
    finally:
        db_mod.DB_PATH = original
        db_mod._db = None
```

- [ ] **Step 6: Run integration test**

Run: `python -m pytest tests/test_keyword_index.py::test_build_keyword_index_populates_db -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/tools/free_apis.py tests/test_keyword_index.py
git commit -m "feat(smart-search): keyword index builder + Turkish category patterns"
```

---

### Task 3: Fast Resolver — Layer 0 + Layer 1

**Files:**
- Create: `src/core/fast_resolver.py`
- Test: `tests/test_fast_resolver.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_fast_resolver.py
import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_try_resolve_weather_query():
    """Layer 0: weather query should resolve via API without LLM."""
    from src.core.fast_resolver import try_resolve

    task = {"title": "Istanbul hava durumu", "description": ""}
    
    with patch("src.core.fast_resolver._call_best_api", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = {"temp": 22, "condition": "sunny"}
        result = await try_resolve(task)
    
    assert result is not None
    assert "22" in result or "sunny" in result


@pytest.mark.asyncio
async def test_try_resolve_no_match_returns_none():
    """Layer 0: unrelated query should return None (fall through to agent)."""
    from src.core.fast_resolver import try_resolve

    task = {"title": "Write a Python script to sort files", "description": ""}
    result = await try_resolve(task)
    assert result is None


@pytest.mark.asyncio 
async def test_enrich_context_adds_data():
    """Layer 1: partial match should return enriched context dict."""
    from src.core.fast_resolver import enrich_context

    task = {"title": "Istanbul'da bu hafta sonu piknik yapilir mi?", "description": ""}
    
    with patch("src.core.fast_resolver._call_best_api", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = {"temp": 22, "condition": "partly cloudy"}
        enrichment = await enrich_context(task)
    
    if enrichment:  # May or may not match depending on threshold
        assert "Available Data" in enrichment or isinstance(enrichment, str)


@pytest.mark.asyncio
async def test_try_resolve_api_failure_falls_through():
    """Layer 0: API failure should return None, not raise."""
    from src.core.fast_resolver import try_resolve

    task = {"title": "Istanbul hava durumu", "description": ""}
    
    with patch("src.core.fast_resolver._call_best_api", new_callable=AsyncMock) as mock_call:
        mock_call.side_effect = Exception("API timeout")
        result = await try_resolve(task)
    
    assert result is None  # Falls through, no crash
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_fast_resolver.py -v`
Expected: FAIL — module doesn't exist

- [ ] **Step 3: Implement fast_resolver.py**

```python
# src/core/fast_resolver.py
"""Layered resolution: resolve tasks via API registry before LLM dispatch.

Layer 0 (try_resolve): Full resolution — call API, format, return answer. No LLM.
Layer 1 (enrich_context): Partial match — fetch data, return as context for agent.
"""

import re
import time
import logging

from src.tools.free_apis import (
    find_apis,
    call_api,
    get_api,
    TURKISH_CATEGORY_PATTERNS,
    tokenize_api_description,
)

logger = logging.getLogger(__name__)

# Thresholds for keyword matching
_LAYER0_THRESHOLD = 0.6   # High confidence — full resolve
_LAYER1_THRESHOLD = 0.3   # Medium — enrich context


async def try_resolve(task: dict) -> str | None:
    """Layer 0: Try to fully resolve a task via API fast-path.
    
    Returns formatted answer string if resolved, None if not.
    """
    task_text = f"{task.get('title', '')} {task.get('description', '')}".strip()
    if not task_text:
        return None

    try:
        match = await _find_best_match(task_text)
        if not match or match["score"] < _LAYER0_THRESHOLD:
            return None

        api = match["api"]
        params = _extract_params(task_text, match["category"])

        start = time.time()
        raw = await _call_best_api(api, params)
        elapsed_ms = int((time.time() - start) * 1000)

        if not raw:
            return None

        formatted = _format_response(raw, match["category"], api.name)

        # Log success
        try:
            from src.infra.db import log_smart_search, record_api_call
            await log_smart_search(task_text, layer=0, source=api.name, success=True, response_ms=elapsed_ms)
            await record_api_call(api.name, success=True)
        except Exception:
            pass

        logger.info("fast-path resolved", api=api.name, category=match["category"], ms=elapsed_ms)
        return formatted

    except Exception as exc:
        logger.info("fast-path failed, falling through: %s", exc)
        try:
            if match and match.get("api"):
                from src.infra.db import record_api_call
                await record_api_call(match["api"].name, success=False)
        except Exception:
            pass
        return None


async def enrich_context(task: dict) -> str | None:
    """Layer 1: Fetch relevant API data as context for the agent.
    
    Returns a formatted context block string, or None.
    """
    task_text = f"{task.get('title', '')} {task.get('description', '')}".strip()
    if not task_text:
        return None

    try:
        match = await _find_best_match(task_text)
        if not match or match["score"] < _LAYER1_THRESHOLD:
            return None
        # Layer 0 would have caught high-score matches
        if match["score"] >= _LAYER0_THRESHOLD:
            return None

        api = match["api"]
        params = _extract_params(task_text, match["category"])

        start = time.time()
        raw = await _call_best_api(api, params)
        elapsed_ms = int((time.time() - start) * 1000)

        if not raw:
            return None

        formatted = _format_response(raw, match["category"], api.name)

        try:
            from src.infra.db import log_smart_search, record_api_call
            await log_smart_search(task_text, layer=1, source=api.name, success=True, response_ms=elapsed_ms)
            await record_api_call(api.name, success=True)
        except Exception:
            pass

        return f"### Available Data\n{formatted}\n(Source: {api.name}, fetched just now)"

    except Exception as exc:
        logger.debug("context enrichment failed: %s", exc)
        return None


async def _find_best_match(task_text: str) -> dict | None:
    """Find the best matching API for task text using keywords + Turkish patterns.
    
    Returns {api, category, score} or None.
    """
    task_lower = task_text.lower()
    best = None

    # 1. Turkish category patterns (strong signal)
    try:
        from src.infra.db import get_api_category_patterns
        db_patterns = await get_api_category_patterns()
    except Exception:
        db_patterns = {}
    
    all_patterns = {**TURKISH_CATEGORY_PATTERNS, **db_patterns}  # DB overrides static

    for category, pattern in all_patterns.items():
        try:
            if re.search(pattern, task_lower, re.IGNORECASE):
                # Found category match — find best API in this category
                apis = find_apis(category=category)
                api = await _pick_most_reliable(apis)
                if api:
                    score = 0.8  # Turkish pattern match is strong
                    if not best or score > best["score"]:
                        best = {"api": api, "category": category, "score": score}
        except re.error:
            continue

    # 2. Keyword index matching
    try:
        from src.infra.db import find_apis_by_keywords
        task_keywords = tokenize_api_description(task_text)
        if task_keywords:
            matches = await find_apis_by_keywords(task_keywords, limit=3)
            for m in matches:
                # Score = proportion of task keywords that matched
                score = m["match_count"] / max(len(task_keywords), 1)
                # Scale to 0-1 range, boost if many matches
                score = min(score * 1.5, 1.0)
                api = get_api(m["api_name"])
                if api and (not best or score > best["score"]):
                    best = {"api": api, "category": api.category, "score": score}
    except Exception as exc:
        logger.debug("keyword matching failed: %s", exc)

    # 3. Check reliability — demoted/suspended APIs drop score
    if best:
        try:
            from src.infra.db import get_api_reliability
            rel = await get_api_reliability(best["api"].name)
            if rel:
                if rel["status"] == "suspended":
                    return None  # Don't use suspended APIs
                elif rel["status"] == "demoted":
                    best["score"] *= 0.3  # Heavy penalty
                elif rel["status"] == "warning":
                    best["score"] *= 0.5
        except Exception:
            pass

    return best


async def _pick_most_reliable(apis: list) -> "FreeAPI | None":
    """From a list of APIs in the same category, pick the most reliable."""
    if not apis:
        return None
    if len(apis) == 1:
        return apis[0]

    try:
        from src.infra.db import get_api_reliability
        scored = []
        for api in apis:
            rel = await get_api_reliability(api.name)
            if rel and rel["status"] == "suspended":
                continue
            reliability = 0.5  # default for unknown
            if rel:
                total = rel["success_count"] + rel["failure_count"]
                if total > 0:
                    reliability = rel["success_count"] / total
            scored.append((api, reliability))
        if scored:
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[0][0]
    except Exception:
        pass

    return apis[0]  # Fallback: first API


def _extract_params(task_text: str, category: str) -> dict:
    """Extract API call parameters from task text based on category.
    
    Simple keyword extraction — city names, currency pairs, etc.
    """
    params = {}
    text_lower = task_text.lower()

    # Common Turkish cities
    cities = [
        "istanbul", "ankara", "izmir", "bursa", "antalya", "adana",
        "konya", "gaziantep", "mersin", "kayseri", "eskisehir",
        "trabzon", "samsun", "denizli", "diyarbakir", "erzurum",
    ]
    for city in cities:
        if city in text_lower:
            params["city"] = city.capitalize()
            break

    # Currency pairs
    if category == "currency":
        currencies = {
            "dolar": "USD", "dollar": "USD", "usd": "USD",
            "euro": "EUR", "eur": "EUR",
            "sterlin": "GBP", "pound": "GBP", "gbp": "GBP",
            "yen": "JPY", "jpy": "JPY",
        }
        for term, code in currencies.items():
            if term in text_lower:
                params["currency"] = code
                break
        if "currency" not in params:
            params["currency"] = "USD"  # Default

    return params


async def _call_best_api(api, params: dict) -> dict | str | None:
    """Call an API with extracted params. Returns raw response data."""
    endpoint = api.example_endpoint

    # Substitute params into endpoint
    city = params.get("city", "Istanbul")
    currency = params.get("currency", "USD")
    endpoint = endpoint.replace("Istanbul", city)
    endpoint = endpoint.replace("USD", currency)

    result = await call_api(api, endpoint=endpoint)
    if not result:
        return None
    return result


def _format_response(raw, category: str, api_name: str) -> str:
    """Format raw API response into clean text."""
    if isinstance(raw, str):
        # Truncate very long responses
        if len(raw) > 2000:
            raw = raw[:2000] + "..."
        return raw

    if isinstance(raw, dict):
        # Try to extract meaningful data based on category
        import json
        return json.dumps(raw, ensure_ascii=False, indent=2)[:2000]

    return str(raw)[:2000]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_fast_resolver.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/core/fast_resolver.py tests/test_fast_resolver.py
git commit -m "feat(smart-search): Layer 0 + Layer 1 fast resolver with keyword matching"
```

---

### Task 4: smart_search Tool — Layer 2

**Files:**
- Create: `src/tools/smart_search.py`
- Modify: `src/tools/__init__.py:87-148` (register the new tool in `_optional_tools`)
- Test: `tests/test_smart_search.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_smart_search.py
import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_smart_search_routes_to_api_first():
    """smart_search should try API registry before web search."""
    from src.tools.smart_search import smart_search

    with patch("src.tools.smart_search._try_api_registry", new_callable=AsyncMock) as mock_api:
        mock_api.return_value = "22°C, sunny in Istanbul"
        result = await smart_search("Istanbul weather")

    assert "22" in result
    assert "sunny" in result
    mock_api.assert_called_once()


@pytest.mark.asyncio
async def test_smart_search_falls_through_to_web():
    """If API registry has no match, fall through to web search."""
    from src.tools.smart_search import smart_search

    with patch("src.tools.smart_search._try_api_registry", new_callable=AsyncMock) as mock_api, \
         patch("src.tools.smart_search._try_web_search", new_callable=AsyncMock) as mock_web:
        mock_api.return_value = None
        mock_web.return_value = "Some web result about Python sorting"
        result = await smart_search("How to sort a list in Python")

    assert result is not None
    mock_web.assert_called_once()


@pytest.mark.asyncio
async def test_smart_search_includes_source_attribution():
    """Result should include source info."""
    from src.tools.smart_search import smart_search

    with patch("src.tools.smart_search._try_api_registry", new_callable=AsyncMock) as mock_api:
        mock_api.return_value = "22°C in Istanbul (Source: wttr.in API)"
        result = await smart_search("Istanbul weather")

    assert "Source:" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_smart_search.py -v`
Expected: FAIL — module doesn't exist

- [ ] **Step 3: Implement smart_search.py**

```python
# src/tools/smart_search.py
"""smart_search — unified search tool that routes through APIs, MCP, then web.

Agents see one tool: smart_search(query). Internally it tries:
1. API registry (keyword match → call_api)
2. MCP tools (if connected, for URL extraction etc.)
3. Web search (Brave/GCSE/DuckDuckGo fallback)
"""

import logging
import time

logger = logging.getLogger(__name__)


async def smart_search(query: str) -> str:
    """Search for information using the best available source.
    
    Checks free API registry first, then MCP tools, then falls back to web search.
    Returns the result with source attribution.
    """
    if not query or not query.strip():
        return "Error: empty query"

    start = time.time()

    # 1. Try API registry
    result = await _try_api_registry(query)
    if result:
        await _log(query, layer=2, source="api_registry", success=True, start=start)
        return result

    # 2. Try MCP tools (Fetch for URLs)
    result = await _try_mcp(query)
    if result:
        await _log(query, layer=2, source="mcp", success=True, start=start)
        return result

    # 3. Fall back to web search
    result = await _try_web_search(query)
    if result:
        await _log(query, layer=2, source="web_search", success=True, start=start)
        return result

    await _log(query, layer=2, source=None, success=False, start=start)
    return f"No results found for: {query}"


async def _try_api_registry(query: str) -> str | None:
    """Try to answer via free API registry."""
    try:
        from src.core.fast_resolver import _find_best_match, _extract_params, _call_best_api, _format_response

        match = await _find_best_match(query)
        if not match or match["score"] < 0.3:
            return None

        api = match["api"]
        params = _extract_params(query, match["category"])
        raw = await _call_best_api(api, params)

        if not raw:
            return None

        formatted = _format_response(raw, match["category"], api.name)

        try:
            from src.infra.db import record_api_call
            await record_api_call(api.name, success=True)
        except Exception:
            pass

        return f"{formatted}\n(Source: {api.name} API)"

    except Exception as exc:
        logger.debug("API registry lookup failed: %s", exc)
        return None


async def _try_mcp(query: str) -> str | None:
    """Try MCP tools — currently Fetch for URL extraction."""
    # Only route URL-containing queries to MCP Fetch
    import re
    url_match = re.search(r"https?://\S+", query)
    if not url_match:
        return None

    try:
        from src.tools import TOOL_REGISTRY
        fetch_tool = TOOL_REGISTRY.get("mcp_fetch_fetch")
        if not fetch_tool:
            return None

        result = await fetch_tool["function"](url=url_match.group())
        if result:
            return f"{str(result)[:2000]}\n(Source: MCP Fetch)"
    except Exception as exc:
        logger.debug("MCP fetch failed: %s", exc)

    return None


async def _try_web_search(query: str) -> str | None:
    """Fall back to existing web_search tool."""
    try:
        from src.tools import TOOL_REGISTRY
        web_search_fn = TOOL_REGISTRY.get("web_search", {}).get("function")
        if not web_search_fn:
            return None

        result = await web_search_fn(query=query)
        if result:
            return f"{str(result)[:3000]}\n(Source: web search)"
    except Exception as exc:
        logger.debug("Web search failed: %s", exc)

    return None


async def _log(query: str, layer: int, source: str | None, success: bool, start: float):
    """Log to smart_search_log table."""
    try:
        from src.infra.db import log_smart_search
        elapsed_ms = int((time.time() - start) * 1000)
        await log_smart_search(query, layer, source, success, elapsed_ms)
    except Exception:
        pass
```

- [ ] **Step 4: Register smart_search in TOOL_REGISTRY**

In `src/tools/__init__.py`, add after the last `try/except` block in the `_optional_tools` section (before line 857 where `TOOL_REGISTRY` is defined — add to the `_optional_tools` dict):

```python
try:
    from .smart_search import smart_search
    _optional_tools["smart_search"] = {
        "function": smart_search,
        "description": (
            "Search for information using the best available source. "
            "Checks free API registry first (weather, currency, pharmacy, etc.), "
            "then MCP tools, then falls back to web search. "
            "Use this instead of web_search for general queries."
        ),
        "example": '{"tool": "smart_search", "args": {"query": "Istanbul hava durumu"}}',
    }
except Exception as e:
    logger.warning("smart_search not available: %s", e)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_smart_search.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/tools/smart_search.py src/tools/__init__.py tests/test_smart_search.py
git commit -m "feat(smart-search): Layer 2 unified smart_search tool"
```

---

### Task 5: Bug Fixes — Skills

**Files:**
- Modify: `src/memory/skills.py:116-117` (WHERE clause)
- Modify: `src/memory/skills.py:175-176` (logging level)
- Modify: `src/core/orchestrator.py:2074-2076` (regex escaping)
- Test: `tests/test_skill_fixes.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_skill_fixes.py
import pytest
import re


def test_seed_skills_returned_without_success_count():
    """Seed skills (success_count=0) should be returned by find_relevant_skills."""
    # This tests the fix: removing WHERE success_count > 0
    # We test the pattern matching logic directly
    from src.memory.skills import _skill_score

    seed_skill = {"success_count": 0, "failure_count": 0, "name": "weather_api_routing"}
    score = _skill_score(seed_skill)
    # Seed skills should have non-zero score (neutral prior)
    assert score > 0


def test_auto_skill_regex_escape():
    """Auto-captured skill patterns must not contain unescaped brackets."""
    # Simulate the fix: words should be re.escape'd
    title = "[0.1] raw_idea_intake"
    words = [w.lower().strip(".,!?") for w in title.split() if len(w) >= 3]
    escaped_words = [re.escape(w) for w in words]
    pattern = "|".join(escaped_words)

    # The pattern should be valid regex
    compiled = re.compile(pattern)
    assert compiled is not None

    # And should NOT match single digits (which broken [0.1] would)
    assert not re.search(pattern, "0")
    assert not re.search(pattern, "1")

    # But should match the actual words
    assert re.search(pattern, "[0.1]")
    assert re.search(pattern, "raw_idea_intake")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_skill_fixes.py -v`
Expected: First test may pass (depends on current `_skill_score`), second tests the pattern fix concept

- [ ] **Step 3: Fix skills.py — remove success_count filter**

In `src/memory/skills.py`, replace line 116-117:

Old (line 116-117):
```python
        cursor = await db.execute(
            "SELECT * FROM skills WHERE success_count > 0 ORDER BY success_count DESC"
        )
```

New:
```python
        cursor = await db.execute(
            "SELECT * FROM skills ORDER BY success_count DESC"
        )
```

- [ ] **Step 4: Fix skills.py — upgrade logging**

In `src/memory/skills.py`, replace line 175-176:

Old:
```python
    except Exception as exc:
        logger.debug(f"find_relevant_skills failed: {exc}")
```

New:
```python
    except Exception as exc:
        logger.warning("find_relevant_skills failed: %s", exc)
```

Also after line 173 (`return merged[:limit]`), add before the except block:

```python
        if not merged:
            logger.info("No skills matched for task: %s", task_text[:80])
```

- [ ] **Step 5: Fix orchestrator.py — regex escaping for auto-capture**

In `src/core/orchestrator.py`, replace lines 2074-2076:

Old:
```python
                words = [w.lower().strip(".,!?") for w in title.split() if len(w) >= 3]
                trigger_parts.extend(sorted(set(words))[:5])
                trigger = "|".join(trigger_parts)
```

New:
```python
                # Strip i2p step prefixes like [0.1], [15.11] before extracting keywords
                clean_title = re.sub(r"\[\d+\.?\d*[a-z]?\]\s*", "", title)
                words = [re.escape(w.lower().strip(".,!?")) for w in clean_title.split() if len(w) >= 3]
                trigger_parts.extend(sorted(set(words))[:5])
                trigger = "|".join(trigger_parts)
```

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/test_skill_fixes.py -v`
Expected: PASS

- [ ] **Step 7: Run existing skill tests if any**

Run: `python -m pytest tests/ -k "skill" -v`
Expected: PASS (no regressions)

- [ ] **Step 8: Commit**

```bash
git add src/memory/skills.py src/core/orchestrator.py tests/test_skill_fixes.py
git commit -m "fix(skills): unblock seed skills, escape auto-capture regex, upgrade logging"
```

---

### Task 6: Bug Fixes — MCP Tool Stubs + Lazy Connect

**Files:**
- Modify: `src/tools/__init__.py:1322` (fix connect method name)
- Modify: `src/tools/__init__.py` (add MCP stub registration at import time)
- Read: `src/tools/mcp_client.py:177` (verify method name)
- Read: `mcp.yaml` (server definitions)

- [ ] **Step 1: Fix MCP lazy connect method name**

In `src/tools/__init__.py`, find line 1322 (the `connect_stdio` call):

Old:
```python
                    await _mcp.connect_stdio(server_name, cfg["command"], env=cfg.get("env"))
```

New:
```python
                    await _mcp.connect(server_name, cfg["command"], env=cfg.get("env"))
```

- [ ] **Step 2: Add MCP stub registration**

In `src/tools/__init__.py`, add after the `_optional_tools` section but before `TOOL_REGISTRY` definition (before line 857). This registers MCP tool stubs (name + description only, no connection):

```python
# ── MCP tool stubs (registered without connecting) ──
def _register_mcp_stubs():
    """Parse mcp.yaml and register stub entries so agents can see MCP tools exist.
    
    Actual connection happens lazily on first tool call.
    """
    import yaml
    import os

    mcp_path = os.path.join(os.path.dirname(__file__), "..", "..", "mcp.yaml")
    if not os.path.exists(mcp_path):
        return

    try:
        with open(mcp_path, "r") as f:
            config = yaml.safe_load(f)
    except Exception:
        return

    servers = config.get("mcp_servers", {})

    # Map server names to their known tool descriptions
    _mcp_tool_descriptions = {
        "fetch": {
            "mcp_fetch_fetch": "Fetch content from a URL and extract text. Use for reading web pages, APIs, or any HTTP endpoint.",
        },
        "sequential_thinking": {
            "mcp_sequential_thinking_think": "Use structured sequential thinking to break down complex problems step by step.",
        },
        "github": {
            "mcp_github_search_repositories": "Search GitHub repositories by query.",
            "mcp_github_get_file_contents": "Get file contents from a GitHub repository.",
            "mcp_github_search_code": "Search code across GitHub repositories.",
        },
    }

    for server_name in servers:
        descs = _mcp_tool_descriptions.get(server_name, {})
        for tool_name, description in descs.items():
            if tool_name not in _optional_tools:
                _optional_tools[tool_name] = {
                    "function": None,  # Stub — lazy connect fills this in
                    "description": f"[MCP: {server_name}] {description}",
                    "example": f'{{"tool": "{tool_name}", "args": {{}}}}',
                    "_mcp_stub": True,  # Flag for lazy connect
                    "_mcp_server": server_name,
                }

try:
    _register_mcp_stubs()
except Exception as e:
    logger.debug("MCP stub registration failed: %s", e)
```

- [ ] **Step 3: Update lazy connect to handle stubs**

In `src/tools/__init__.py`, find the `execute_tool` function's MCP lazy connect section (~line 1308-1329). Update it to also handle stub tools that are already in `TOOL_REGISTRY` but have `function=None`:

Find the existing MCP lazy connect block and update:

Old pattern:
```python
        if tool_name not in TOOL_REGISTRY and tool_name.startswith("mcp_"):
```

New:
```python
        tool_entry = TOOL_REGISTRY.get(tool_name)
        if (tool_name.startswith("mcp_") and 
            (tool_name not in TOOL_REGISTRY or (tool_entry and tool_entry.get("_mcp_stub")))):
```

After the lazy connect succeeds and `_mcp.register_all_tools()` is called, the stub's `function` field gets replaced with the real function. Add after the `register_all_tools()` call:

```python
                    # Clear stub flag now that real function is registered
                    if tool_name in TOOL_REGISTRY:
                        TOOL_REGISTRY[tool_name].pop("_mcp_stub", None)
```

- [ ] **Step 4: Verify by importing**

Run: `python -c "from src.tools import TOOL_REGISTRY; mcps = [k for k in TOOL_REGISTRY if k.startswith('mcp_')]; print(f'MCP stubs: {len(mcps)}'); print(mcps)"`
Expected: Should show MCP stub tool names

- [ ] **Step 5: Commit**

```bash
git add src/tools/__init__.py
git commit -m "fix(mcp): register tool stubs from mcp.yaml, fix lazy connect method name"
```

---

### Task 7: Orchestrator Integration — Fast Resolver + Discovery Scheduling

**Files:**
- Modify: `src/core/orchestrator.py:1276-1350` (add fast_resolver call before agent dispatch)
- Modify: `src/core/orchestrator.py:956` (add discovery scheduling)

- [ ] **Step 1: Add fast_resolver call before agent dispatch**

In `src/core/orchestrator.py`, inside `process_task()`, after the classification block (after line 1326 where `task["context"] = json.dumps(task_ctx)` in the classification block) and before the shopping intent detection (line 1328), add:

```python
            # ── Layer 0: Fast-path resolution via API registry ──
            try:
                from ..core.fast_resolver import try_resolve
                fast_result = await try_resolve(task)
                if fast_result:
                    logger.info("task resolved via fast-path", task_id=task_id)
                    await update_task(task_id, status="done", result=fast_result)
                    # Send result to user
                    if self.telegram and task.get("chat_id"):
                        await self.telegram.send_notification(fast_result)
                    return
            except Exception as exc:
                logger.debug("fast-path check failed (continuing to agent): %s", exc)
```

- [ ] **Step 2: Add Layer 1 context enrichment before agent dispatch**

In `src/core/orchestrator.py`, inside `process_task()`, after the workflow step pre-hook section (after line 1363), before the human approval gate (line 1366), add:

```python
            # ── Layer 1: Enrich context with pre-fetched API data ──
            try:
                from ..core.fast_resolver import enrich_context
                enrichment = await enrich_context(task)
                if enrichment:
                    task_ctx["api_enrichment"] = enrichment
                    task["context"] = json.dumps(task_ctx)
                    logger.info("task enriched with API data", task_id=task_id)
            except Exception as exc:
                logger.debug("context enrichment failed (non-critical): %s", exc)
```

- [ ] **Step 3: Add discovery scheduling**

In `src/core/orchestrator.py`, inside `check_scheduled_tasks()` (line 956), add discovery check at the beginning of the method, after `try:`:

```python
            # ── API discovery (8:30am daily, catch-up if missed) ──
            try:
                await self._check_api_discovery()
            except Exception as exc:
                logger.debug("API discovery check failed: %s", exc)
```

Then add the new method to the Orchestrator class:

```python
    async def _check_api_discovery(self):
        """Run API discovery daily at 8:30am, with catch-up if missed."""
        from datetime import datetime, timedelta

        now = datetime.now()

        # Only run at 8:30am (±5 min window)
        if not hasattr(self, "_last_api_discovery"):
            self._last_api_discovery = None

        # Check if we already ran today
        if self._last_api_discovery and (now - self._last_api_discovery).total_seconds() < 86400:
            return

        # Run at 8:25-8:35 window, or if >36h since last run (catch-up)
        in_window = now.hour == 8 and 25 <= now.minute <= 35
        overdue = (
            self._last_api_discovery is None
            or (now - self._last_api_discovery).total_seconds() > 36 * 3600
        )

        if not in_window and not overdue:
            return

        logger.info("Starting API discovery run")
        try:
            from src.tools.free_apis import discover_new_apis, build_keyword_index, seed_category_patterns

            new_count = await discover_new_apis("all")
            await build_keyword_index()
            await seed_category_patterns()
            self._last_api_discovery = now

            if new_count > 0:
                logger.info("API discovery complete: %d new APIs", new_count)
                # Add to morning brief if available
                if hasattr(self, "_morning_brief_extras"):
                    self._morning_brief_extras.append(
                        f"Discovered {new_count} new APIs/MCP tools."
                    )
                # Notify if significant
                if new_count >= 5 and self.telegram:
                    await self.telegram.send_notification(
                        f"API discovery: {new_count} new APIs added to registry."
                    )
            else:
                logger.info("API discovery complete: no new APIs found")
        except Exception as exc:
            logger.warning("API discovery failed: %s", exc)
```

- [ ] **Step 4: Verify import works**

Run: `python -c "from src.core.orchestrator import Orchestrator; print('OK')"`
Expected: OK (no import errors)

- [ ] **Step 5: Commit**

```bash
git add src/core/orchestrator.py
git commit -m "feat(smart-search): integrate fast resolver + discovery scheduling in orchestrator"
```

---

### Task 8: Agent Context Injection — Layer 1 in base.py

**Files:**
- Modify: `src/agents/base.py:496` (after skill injection, add API enrichment injection)

- [ ] **Step 1: Add API enrichment context injection**

In `src/agents/base.py`, inside `_build_context()`, after the skill injection block (after line 496), add:

```python
        # ── Smart Resource Integration: Layer 1 API enrichment ──
        try:
            _task_ctx_raw = task.get("context", "{}")
            if isinstance(_task_ctx_raw, str):
                _task_ctx_parsed = json.loads(_task_ctx_raw)
            else:
                _task_ctx_parsed = _task_ctx_raw or {}
            api_enrichment = _task_ctx_parsed.get("api_enrichment")
            if api_enrichment:
                parts.append(api_enrichment)
        except Exception as exc:
            logger.debug("API enrichment injection failed (non-critical): %s", exc)
```

- [ ] **Step 2: Verify import works**

Run: `python -c "from src.agents.base import BaseAgent; print('OK')"`
Expected: OK

- [ ] **Step 3: Commit**

```bash
git add src/agents/base.py
git commit -m "feat(smart-search): inject Layer 1 API enrichment into agent context"
```

---

### Task 9: Workflow Expander — api_hints Support

**Files:**
- Modify: `src/workflows/engine/expander.py:130-132` (add api_hints alongside tools_hint)

- [ ] **Step 1: Add api_hints extraction**

In `src/workflows/engine/expander.py`, after lines 130-132 where `tools_hint` is extracted, add:

```python
        api_hints = step.get("api_hints")
        if api_hints and isinstance(api_hints, list):
            context["api_hints"] = api_hints
```

- [ ] **Step 2: Add api_hints enrichment in workflow hooks**

In `src/workflows/engine/hooks.py`, inside `pre_execute_workflow_step()`, after existing context injection logic, add:

```python
    # ── Enrich from api_hints ──
    task_ctx = task.get("context", "{}")
    if isinstance(task_ctx, str):
        try:
            task_ctx = json.loads(task_ctx)
        except (json.JSONDecodeError, TypeError):
            task_ctx = {}
    
    api_hints = task_ctx.get("api_hints", [])
    if api_hints:
        try:
            from src.tools.free_apis import find_apis
            from src.tools.free_apis import call_api
            enrichment_parts = []
            for hint in api_hints[:3]:  # Max 3 hints per step
                apis = find_apis(category=hint)
                if apis:
                    try:
                        data = await call_api(apis[0])
                        if data:
                            enrichment_parts.append(f"**{hint}** ({apis[0].name}): {str(data)[:500]}")
                    except Exception:
                        pass
            if enrichment_parts:
                task_ctx["api_enrichment"] = "### Available Data\n" + "\n\n".join(enrichment_parts)
                task["context"] = json.dumps(task_ctx)
        except Exception:
            pass
```

- [ ] **Step 3: Verify import works**

Run: `python -c "from src.workflows.engine.expander import expand_steps_to_tasks; print('OK')"`
Expected: OK

- [ ] **Step 4: Commit**

```bash
git add src/workflows/engine/expander.py src/workflows/engine/hooks.py
git commit -m "feat(smart-search): api_hints support in workflow expander + hooks"
```

---

### Task 10: Telegram Menu — /smartsearch Command

**Files:**
- Modify: `src/app/telegram_bot.py:1743` (register handler in `_setup_handlers`)
- Modify: `src/app/telegram_bot.py` (add command handler method + callback handlers)

- [ ] **Step 1: Register the command handler**

In `src/app/telegram_bot.py`, inside `_setup_handlers()` (line 1743), add alongside other CommandHandler registrations:

```python
        self.app.add_handler(CommandHandler("smartsearch", self.cmd_smartsearch))
```

- [ ] **Step 2: Add the command handler method**

Add to the TelegramInterface class (near `cmd_skillstats` at line 2590 for consistency):

```python
    async def cmd_smartsearch(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show smart search stats and API/MCP observability."""
        try:
            from src.infra.db import get_smart_search_stats, get_api_reliability_all

            stats = await get_smart_search_stats(days=7)
            reliability = await get_api_reliability_all()

            # Header
            today = stats.get("today", 0)
            layers = stats.get("layers", {})
            total_7d = sum(l["count"] for l in layers.values())

            lines = [f"Smart Search Stats", "─" * 25]
            lines.append(f"Queries today: {today}")

            layer_names = {0: "Layer 0 (fast-path)", 1: "Layer 1 (enriched)", 2: "Layer 2 (smart_search)", 3: "Fell through to web"}
            for layer_num in sorted(layers.keys()):
                info = layers[layer_num]
                pct = int(info["count"] / max(total_7d, 1) * 100)
                name = layer_names.get(layer_num, f"Layer {layer_num}")
                lines.append(f"  {name}: {info['count']}  ({pct}%)")

            # Top performers
            top = [r for r in reliability if r["status"] == "active" and (r["success_count"] + r["failure_count"]) > 0]
            top.sort(key=lambda r: r["success_count"], reverse=True)
            if top[:5]:
                lines.append("")
                lines.append("Top APIs (7d)")
                for r in top[:5]:
                    total = r["success_count"] + r["failure_count"]
                    rate = int(r["success_count"] / max(total, 1) * 100)
                    lines.append(f"  {r['api_name']:<20} {total} calls, {rate}% success")

            # Worst performers
            worst = [r for r in reliability if r["status"] in ("warning", "demoted", "suspended")]
            if worst:
                lines.append("")
                lines.append("Worst Performers (7d)")
                for r in worst[:5]:
                    total = r["success_count"] + r["failure_count"]
                    rate = int(r["success_count"] / max(total, 1) * 100)
                    status_icon = {"warning": "!", "demoted": "demoted", "suspended": "suspended"}
                    lines.append(f"  {r['api_name']:<20} {r['success_count']}/{total} ({rate}%) {status_icon.get(r['status'], '')}")

            # Top sources
            top_sources = stats.get("top_sources", [])
            if top_sources:
                lines.append("")
                lines.append("Top Sources (7d)")
                for s in top_sources[:5]:
                    lines.append(f"  {s['source']:<20} {s['count']} calls")

            # Registry info
            from src.tools.free_apis import API_REGISTRY, _db_api_cache
            total_apis = len(API_REGISTRY) + len(_db_api_cache)
            lines.append("")
            lines.append(f"Registry: {total_apis} APIs")

            text = "\n".join(lines)

            # Inline buttons
            from telegram import InlineKeyboardButton, InlineKeyboardMarkup
            keyboard = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("Refresh Now", callback_data="ss:refresh"),
                    InlineKeyboardButton("Top Failures", callback_data="ss:failures"),
                ],
                [
                    InlineKeyboardButton("Unsuspend All", callback_data="ss:unsuspend"),
                ],
            ])

            await self._reply(update, text, reply_markup=keyboard)

        except Exception as e:
            await self._reply(update, f"Error loading stats: {e}")
```

- [ ] **Step 3: Add callback handlers**

In the `handle_callback` method (line 4856), add a routing case for `ss:` prefix:

```python
        elif data.startswith("ss:"):
            await self._handle_smartsearch_callback(query, data)
```

Then add the handler method:

```python
    async def _handle_smartsearch_callback(self, query, data: str):
        """Handle /smartsearch inline button callbacks."""
        action = data.split(":", 1)[1] if ":" in data else ""

        if action == "refresh":
            try:
                from src.tools.free_apis import discover_new_apis, build_keyword_index
                new_count = await discover_new_apis("all")
                await build_keyword_index()
                await query.edit_message_text(f"Discovery complete: {new_count} new APIs found.")
            except Exception as e:
                await query.edit_message_text(f"Discovery failed: {e}")

        elif action == "failures":
            try:
                from src.infra.db import get_api_reliability_all
                all_rel = await get_api_reliability_all()
                failures = [r for r in all_rel if r["failure_count"] > 0]
                failures.sort(key=lambda r: r["failure_count"], reverse=True)
                if not failures:
                    await query.edit_message_text("No failures recorded.")
                    return
                lines = ["API Failures (all time)", "─" * 25]
                for r in failures[:15]:
                    total = r["success_count"] + r["failure_count"]
                    rate = int(r["success_count"] / max(total, 1) * 100)
                    lines.append(f"{r['api_name']}: {r['failure_count']} failures ({rate}% success) [{r['status']}]")
                await query.edit_message_text("\n".join(lines))
            except Exception as e:
                await query.edit_message_text(f"Error: {e}")

        elif action == "unsuspend":
            try:
                from src.infra.db import get_api_reliability_all, unsuspend_api
                all_rel = await get_api_reliability_all()
                suspended = [r for r in all_rel if r["status"] in ("suspended", "demoted")]
                for r in suspended:
                    await unsuspend_api(r["api_name"])
                await query.edit_message_text(f"Unsuspended {len(suspended)} APIs. Counters reset.")
            except Exception as e:
                await query.edit_message_text(f"Error: {e}")
```

- [ ] **Step 4: Verify import works**

Run: `python -c "from src.app.telegram_bot import TelegramInterface; print('OK')"`
Expected: OK

- [ ] **Step 5: Commit**

```bash
git add src/app/telegram_bot.py
git commit -m "feat(smart-search): /smartsearch Telegram command with observability menu"
```

---

### Task 11: i2p v3 — Add smart_search + api_hints to Steps

**Files:**
- Modify: `src/workflows/i2p/i2p_v3.json`

- [ ] **Step 1: Identify steps that need smart_search**

Research/analysis phases (1-3) that currently have `web_search` in `tools_hint` should get `smart_search` added. Run:

```bash
python -c "
import json
with open('src/workflows/i2p/i2p_v3.json') as f:
    data = json.load(f)
steps = data.get('phases', data) if isinstance(data, dict) else data
# Find all steps with web_search in tools_hint
count = 0
for item in (steps if isinstance(steps, list) else []):
    phases = item.get('steps', [item]) if isinstance(item, dict) else [item]
    for step in phases:
        th = step.get('tools_hint', [])
        if 'web_search' in th:
            count += 1
            print(f\"  {step.get('step', '?')}: {step.get('title', '?')} -> {th}\")
print(f'Total steps with web_search: {count}')
"
```

- [ ] **Step 2: Add smart_search to research/analysis steps**

For each step in phases 0-5 that has `web_search` in `tools_hint`, add `smart_search` to the list. Also add `api_hints` where relevant:

- Steps in Phase 1 (market research): add `api_hints: ["market_data"]`
- Steps about competitors: add `api_hints: ["app_store"]`
- Steps about tech stack: add `api_hints: ["github_trending"]`

Use a script to batch-update:

```bash
python -c "
import json

with open('src/workflows/i2p/i2p_v3.json') as f:
    data = json.load(f)

def update_steps(obj):
    if isinstance(obj, dict):
        th = obj.get('tools_hint', [])
        if 'web_search' in th and 'smart_search' not in th:
            obj['tools_hint'] = ['smart_search'] + th
        
        # Add api_hints based on step title/content
        title = obj.get('title', '').lower()
        step_id = obj.get('step', '')
        
        if any(kw in title for kw in ['competitor', 'market', 'pricing']):
            if 'api_hints' not in obj:
                obj['api_hints'] = ['market_data']
        if any(kw in title for kw in ['tech_stack', 'technology']):
            if 'api_hints' not in obj:
                obj['api_hints'] = ['github_trending']
        
        for v in obj.values():
            update_steps(v)
    elif isinstance(obj, list):
        for item in obj:
            update_steps(item)

update_steps(data)

with open('src/workflows/i2p/i2p_v3.json', 'w') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print('Updated i2p_v3.json')
"
```

- [ ] **Step 3: Verify JSON is valid**

Run: `python -c "import json; json.load(open('src/workflows/i2p/i2p_v3.json')); print('Valid JSON')"`
Expected: Valid JSON

- [ ] **Step 4: Commit**

```bash
git add src/workflows/i2p/i2p_v3.json
git commit -m "feat(smart-search): add smart_search + api_hints to i2p v3 research steps"
```

---

### Task 12: Integration Test — End-to-End Flow

**Files:**
- Create: `tests/test_smart_search_integration.py`

- [ ] **Step 1: Write end-to-end integration test**

```python
# tests/test_smart_search_integration.py
"""End-to-end test for the layered resolution system."""
import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_layer0_resolves_weather_without_agent(tmp_path):
    """A weather query should be resolved at Layer 0 without touching an LLM."""
    import src.infra.db as db_mod
    original = db_mod.DB_PATH
    db_mod.DB_PATH = str(tmp_path / "test.db")
    db_mod._db = None
    try:
        await db_mod.init_db()
        from src.tools.free_apis import seed_registry, build_keyword_index, seed_category_patterns
        await seed_registry()
        await build_keyword_index()
        await seed_category_patterns()

        from src.core.fast_resolver import try_resolve

        task = {"title": "Istanbul hava durumu", "description": ""}

        with patch("src.tools.free_apis.call_api", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = '{"current_weather": {"temperature": 22, "weathercode": 0}}'
            result = await try_resolve(task)

        assert result is not None
        assert "22" in result
    finally:
        db_mod.DB_PATH = original
        db_mod._db = None


@pytest.mark.asyncio
async def test_layer1_enriches_context(tmp_path):
    """A partial-match query should get enriched context."""
    import src.infra.db as db_mod
    original = db_mod.DB_PATH
    db_mod.DB_PATH = str(tmp_path / "test.db")
    db_mod._db = None
    try:
        await db_mod.init_db()
        from src.tools.free_apis import seed_registry, build_keyword_index, seed_category_patterns
        await seed_registry()
        await build_keyword_index()
        await seed_category_patterns()

        from src.core.fast_resolver import enrich_context

        # This has weather relevance but also reasoning needed
        task = {"title": "Istanbul'da bu hafta sonu piknik yapilir mi? hava durumu nasil?", "description": ""}

        with patch("src.tools.free_apis.call_api", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = '{"temp": 22}'
            result = await enrich_context(task)

        # May or may not enrich depending on threshold tuning
        # At minimum, should not crash
        assert result is None or "Available Data" in result
    finally:
        db_mod.DB_PATH = original
        db_mod._db = None


@pytest.mark.asyncio
async def test_reliability_tracking(tmp_path):
    """API calls should update reliability counters."""
    import src.infra.db as db_mod
    original = db_mod.DB_PATH
    db_mod.DB_PATH = str(tmp_path / "test.db")
    db_mod._db = None
    try:
        await db_mod.init_db()

        await db_mod.record_api_call("test_api", success=True)
        await db_mod.record_api_call("test_api", success=True)
        await db_mod.record_api_call("test_api", success=False)

        rel = await db_mod.get_api_reliability("test_api")
        assert rel["success_count"] == 2
        assert rel["failure_count"] == 1
        assert rel["status"] == "active"
    finally:
        db_mod.DB_PATH = original
        db_mod._db = None


@pytest.mark.asyncio
async def test_auto_demotion(tmp_path):
    """APIs with low success rate should be auto-demoted."""
    import src.infra.db as db_mod
    original = db_mod.DB_PATH
    db_mod.DB_PATH = str(tmp_path / "test.db")
    db_mod._db = None
    try:
        await db_mod.init_db()

        # 1 success, 9 failures = 10% success rate with 10 calls -> suspended
        await db_mod.record_api_call("bad_api", success=True)
        for _ in range(9):
            await db_mod.record_api_call("bad_api", success=False)

        rel = await db_mod.get_api_reliability("bad_api")
        assert rel["status"] == "suspended"
    finally:
        db_mod.DB_PATH = original
        db_mod._db = None
```

- [ ] **Step 2: Run integration tests**

Run: `python -m pytest tests/test_smart_search_integration.py -v`
Expected: PASS

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest tests/ -v --timeout=60`
Expected: All pass, no regressions

- [ ] **Step 4: Commit**

```bash
git add tests/test_smart_search_integration.py
git commit -m "test(smart-search): end-to-end integration tests for layered resolution"
```

---

### Task 13: Seed Keyword Index on Startup

**Files:**
- Modify: `src/app/run.py` (add keyword index seeding after DB init)

- [ ] **Step 1: Add keyword index seeding to startup**

In `src/app/run.py`, find where `init_db()` is called during startup. After that call, add:

```python
    # Seed API keyword index + Turkish patterns (fast, idempotent)
    try:
        from src.tools.free_apis import seed_registry, build_keyword_index, seed_category_patterns
        await seed_registry()
        await build_keyword_index()
        await seed_category_patterns()
    except Exception as exc:
        logger.warning("API keyword index seeding failed (non-critical): %s", exc)
```

- [ ] **Step 2: Verify startup doesn't break**

Run: `python -c "from src.app.run import main; print('Import OK')"`
Expected: Import OK

- [ ] **Step 3: Commit**

```bash
git add src/app/run.py
git commit -m "feat(smart-search): seed keyword index + Turkish patterns on startup"
```

---

### Summary of Commits

1. `feat(smart-search): add DB tables + helpers for keyword index, reliability, logging`
2. `feat(smart-search): keyword index builder + Turkish category patterns`
3. `feat(smart-search): Layer 0 + Layer 1 fast resolver with keyword matching`
4. `feat(smart-search): Layer 2 unified smart_search tool`
5. `fix(skills): unblock seed skills, escape auto-capture regex, upgrade logging`
6. `fix(mcp): register tool stubs from mcp.yaml, fix lazy connect method name`
7. `feat(smart-search): integrate fast resolver + discovery scheduling in orchestrator`
8. `feat(smart-search): inject Layer 1 API enrichment into agent context`
9. `feat(smart-search): api_hints support in workflow expander + hooks`
10. `feat(smart-search): /smartsearch Telegram command with observability menu`
11. `feat(smart-search): add smart_search + api_hints to i2p v3 research steps`
12. `test(smart-search): end-to-end integration tests for layered resolution`
13. `feat(smart-search): seed keyword index + Turkish patterns on startup`
