# Smart Resource Integration — Design Spec

**Date**: 2026-04-03
**Status**: Draft
**Scope**: Make agents (especially i2p v3) actually use the API registry, skills, and MCP tools that already exist but are currently dead weight.

## Problem

Logs from the last 10 days show:
- **750 tool calls** total — 66% are just `web_search` + `read_file`
- **Free APIs**: Zero usage. Registry exists (30 static + 4 discovery sources) but agents never see it.
- **MCP tools**: Zero usage. 3 bugs in a chain prevent connection.
- **Skills**: Write-only. 30+ captured, zero injected back. 3 bugs.
- **API tools** (`api_call`, `api_lookup`, `discover_apis`): Registered but no small LLM has ever called them.

Core constraint: limited VRAM means small LLMs. Giving them menus of options leads to hallucination. The system must pre-resolve what it can and minimize LLM decision surface.

## Architecture: Layered Resolution

Four layers, each progressively involving more LLM reasoning:

```
Prerequisite: Bug fixes (Layer 3)
  Skills injection works, MCP tools visible, auto-skill regex fixed

Task arrives
  │
  ▼
Layer 0: Fast-path (no LLM)
  │ API keyword match → call API → format → return result
  │ If resolved: done ──→ return to user
  │ If partial match:
  ▼
Layer 1: Context enrichment (LLM gets pre-fetched data)
  │ Fetch relevant API data, inject as facts in agent context
  │ Agent dispatched with real data
  │ If no match:
  ▼
Layer 2: Agent with smart_search tool
  │ Agent calls smart_search(query) → routes internally
  │ APIs → MCP → web search fallback
```

### Layer 0 — Orchestrator Fast-Path

**New module**: `src/core/fast_resolver.py`

**Entry point**: Called from `orchestrator.py` before agent dispatch.

```python
result = await fast_resolver.try_resolve(task)
if result:
    return result  # done, no agent needed
```

**Matching strategy — keyword index, not hardcoded patterns**:

1. At discovery time, each API's `description` and `tags` are tokenized into keywords. Stored in `api_keywords` table.
2. Incoming task text is tokenized.
3. Keyword overlap score determines match. No per-API regex maintenance.
4. Thin Turkish localization layer: one category-level regex pattern table (`api_category_patterns`) for terms that don't appear in English API descriptions (`doviz`, `nobetci`, `hava durumu`, etc.). ~15-20 rows total.

**Auto-growth**: When `discover_new_apis()` pulls in new APIs, their descriptions are auto-tokenized into the keyword index. A new API about "air quality index for cities" becomes findable when someone asks about "hava kalitesi" without any manual pattern work.

**Parameter extraction**: After matching category, simple regex/keyword extraction pulls query params (city name, currency pair, etc.) from task text. No LLM.

**Response formatting**: Per-category `format_response(raw_data)` functions turn raw JSON into clean text.

**Escape hatch**: API call fails → fall through to Layer 1/agent dispatch. Never a hard failure.

### Layer 1 — Pre-fetched Context Injection

**When**: Task didn't fully match Layer 0 but APIs are relevant. Example: "Istanbul'da bu hafta sonu piknik yapilir mi?" — needs weather + reasoning.

**Where**: Same `fast_resolver.py`, method `enrich_context(task)`.

**How**: 
- Keyword index finds relevant APIs (score above medium threshold, below full-resolve threshold)
- Fetch data from top 1-2 matching APIs
- Inject as factual block in task context via `_build_context()`:
  ```
  ### Available Data
  Weather Istanbul (wttr.in, fetched just now): 22C, partly cloudy,
  Saturday 28C sunny, Sunday 19C rain expected
  ```
- Agent reasons over real data instead of hallucinating

**Two thresholds**:
- Score >= high → Layer 0 (full resolve)
- Score >= medium → Layer 1 (enrich context)
- Score < medium → skip

### Layer 2 — `smart_search` Tool

**What agents see**: One tool, one parameter.
```
smart_search(query: str) -> str
```

**Internal routing order**:
1. API registry — keyword index match → `call_api()`. If good response, return it.
2. MCP tools — if a connected MCP server handles the query type (e.g., Fetch for URL content extraction), route there.
3. Web search — fallback to existing `web_search` (Brave/GCSE/DuckDuckGo).

Note: Skills are injected separately into agent context (Layer 3 fix), not routed through `smart_search`. Skills provide execution strategy hints, not search results.

**Returns**: Best result with attribution line ("Source: wttr.in API" / "Source: web search via Brave").

**Coexistence**: Existing tools (`web_search`, `api_call`, `api_lookup`, `discover_apis`) stay untouched. `smart_search` is additive. New tool registered in `TOOL_REGISTRY`.

**i2p integration**: Steps that had `web_search` in `tools_hint` get `smart_search` added (or replacing, depending on step).

### Layer 3 — Bug Fixes

#### Skills (3 fixes)

1. **`skills.py:117`** — Remove `WHERE success_count > 0` filter. Replace with composite ranking: `(success_count / (success_count + failure_count + 1)) * 0.7 + seed_bonus * 0.3`. Seed skills get baseline trust.

2. **`orchestrator.py:~2074`** — `re.escape()` all words before joining trigger patterns. Strip bracket-heavy i2p prefixes (`[0.1]`) before keyword extraction. Auto-captured patterns become actually matchable.

3. **`base.py:496`** — Upgrade logging: INFO when zero skills match, WARNING on exceptions. Failures become visible.

#### MCP (3 fixes)

4. **MCP tool stubs at import time** — Parse `mcp.yaml`, register tool names and descriptions as stubs in `TOOL_REGISTRY` without connecting. Agents can see MCP tools exist. Respects `auto_connect: false` / no-auto-connect-on-startup rule.

5. **`tools/__init__.py:~1322`** — Fix `_mcp.connect_stdio()` → `_mcp.connect()`. Lazy connection works on first actual tool request.

6. **Chicken-and-egg solved** — Stubs (#4) + working lazy connect (#5) = agent sees stub → requests tool → lazy connect fires → MCP server starts → tool executes.

## Ranking: Relevance First, Reliability Breaks Ties

```
final_score = relevance_score * 0.7 + reliability_score * 0.3

relevance_score = keyword_overlap + category_boost + turkish_pattern_boost
reliability_score = success_count / (success_count + failure_count + 1)
```

Highly relevant but slightly flaky API beats irrelevant but reliable one. Between two weather APIs that both match, the more reliable one wins.

### Auto-Demotion

| Threshold | Condition (min 5 calls) | Effect |
|-----------|------------------------|--------|
| Warning | <50% success | Ranking score -50% |
| Demoted | <25% success | Excluded from Layer 0/1, only via `smart_search` |
| Suspended | <10% success (min 10 calls) | Removed from all layers |

- **Rolling 7-day window** — fixed APIs recover naturally
- **Manual override** — `[Unsuspend]` button in Telegram menu
- **Never auto-delete** — suspended APIs stay in registry

## i2p v3 Enrichment

New `api_hints` field in step definitions:

```json
{
  "step": "1.5_lite",
  "title": "competitors_research",
  "tools_hint": ["smart_search", "read_file"],
  "api_hints": ["market_data", "app_store"]
}
```

**How it works**: Workflow expander reads `api_hints`, passes to `fast_resolver.enrich_context()` which fetches relevant data and injects into task context. Research/analysis phases (1-3) get real data. Coding phases (7+) unaffected.

**Fallback**: `api_hints` referencing a category with no APIs → silently skipped.

## Discovery & Growth

### Timing
- **8:30am daily** — before 9am morning briefs. If new APIs/MCPs found, summary line in brief: "Discovered 12 new APIs, 2 MCP tools."
- **Catch-up** — if last discovery >36h ago (system down, missed schedule), run on next orchestrator startup before processing tasks.
- **Once per day cap** — never more than once in 24h.

### Keyword index rebuild
After every discovery run, re-tokenize all API descriptions/tags, update `api_keywords` table. Fast — string splitting + DB inserts.

### Sources (existing)
1. `public-apis` GitHub repo
2. `free-apis.github.io` JSON
3. MCP server registry
4. ClawHub skill descriptions

## Observability — Telegram Menu

Command: `/smartsearch` (or `/apistats`)

```
Smart Search Stats
---------------------
Queries today: 34
  Layer 0 (fast-path):   12  (35%)
  Layer 1 (enriched):     8  (24%)
  Layer 2 (smart_search): 9  (26%)
  Fell through to web:    5  (15%)

Top APIs (7d)
  wttr.in         -- 23 calls, 100% success
  ExchangeRate    -- 14 calls, 93% success
  pharmacy        -- 8 calls, 100% success

Worst Performers (7d)
  some-random-api -- 3/15 success (20%) demoted
  flaky-endpoint  -- 5/12 success (42%) warning

MCP Tools (7d)
  fetch           -- 5 calls, 80% success
  github          -- 2 calls, 100% success

Skills injected: 47 (7d)
  Matched tasks: 31/89 (35%)
  Top: currency_lookup (12x), weather_check (8x)

Registry: 1,247 APIs | 3 MCP servers
Last discovery: today 08:30 (+8 new)
```

Inline buttons: `[Refresh Now]` `[Discovery Log]` `[Top Failures]` `[Unsuspend]`

## DB Schema Changes

### New tables

```sql
-- Keyword index for API matching
CREATE TABLE IF NOT EXISTS api_keywords (
    api_name TEXT NOT NULL,
    keyword TEXT NOT NULL,
    source TEXT DEFAULT 'description',  -- 'description', 'tag', 'category'
    UNIQUE(api_name, keyword)
);
CREATE INDEX idx_api_keywords_kw ON api_keywords(keyword);

-- Turkish category patterns (thin localization layer)
CREATE TABLE IF NOT EXISTS api_category_patterns (
    category TEXT PRIMARY KEY,
    pattern TEXT NOT NULL  -- regex pattern for Turkish terms
);

-- Smart search usage tracking
CREATE TABLE IF NOT EXISTS smart_search_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime')),
    query TEXT NOT NULL,
    layer INTEGER NOT NULL,  -- 0, 1, 2, 3
    source TEXT,             -- api name, mcp tool, skill, web
    success INTEGER DEFAULT 1,
    response_ms INTEGER
);

-- API call reliability tracking
CREATE TABLE IF NOT EXISTS api_reliability (
    api_name TEXT PRIMARY KEY,
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    last_success TEXT,
    last_failure TEXT,
    status TEXT DEFAULT 'active'  -- 'active', 'warning', 'demoted', 'suspended'
);
```

### Modified tables

- `free_api_registry` — no schema change, but discovery now also populates `api_keywords`

## Files Changed

| File | Change |
|------|--------|
| `src/core/fast_resolver.py` | **NEW** — Layer 0 + Layer 1 logic, keyword matching, parameter extraction, response formatting |
| `src/tools/smart_search.py` | **NEW** — Layer 2 tool, internal routing through APIs → MCP → skills → web |
| `src/tools/__init__.py` | Register `smart_search`, fix MCP lazy connect method name, add MCP stub registration |
| `src/agents/base.py` | Inject Layer 1 enriched context in `_build_context()`, fix skill logging levels |
| `src/memory/skills.py` | Fix `WHERE success_count > 0` filter, update ranking formula |
| `src/core/orchestrator.py` | Call `fast_resolver.try_resolve()` before dispatch, fix auto-skill regex escaping, add discovery scheduling (8:30am + catch-up) |
| `src/tools/free_apis.py` | Add keyword index building after discovery, add reliability tracking calls |
| `src/infra/db.py` | New tables, keyword index queries, reliability tracking queries |
| `src/app/telegram_bot.py` | `/smartsearch` command + inline menu |
| `src/workflows/engine/expander.py` | Read `api_hints` field, call `enrich_context()` |
| `src/workflows/i2p/i2p_v3.json` | Add `api_hints` to relevant steps, add `smart_search` to `tools_hint` |
| `mcp.yaml` | No change (stubs parsed from it at import time) |

## Out of Scope

- Changing existing tool interfaces (`web_search`, `api_call`, etc.)
- Auto-connecting MCP at startup (respects no-auto-connect rule)
- Prompt engineering to make LLMs prefer `api_call` over `web_search` (Layer 0/1/2 solve this better)
- Shopping scraper integration into `smart_search` (separate system, works fine)
