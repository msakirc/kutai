# Smart Resource Integration — System Guide

How KutAI agents find and use APIs, MCP tools, and skills to answer queries without relying solely on web search.

## Architecture: Layered Resolution

When a task arrives, it passes through layers in order. Each layer is progressively more expensive:

```
Task → Layer 0 (no LLM) → Layer 1 (LLM + data) → Layer 2 (LLM + tool) → Agent
```

### Layer 0: Fast-Path (No LLM)

**File:** `src/core/fast_resolver.py` → `try_resolve(task)`
**Called from:** `src/core/orchestrator.py` in `process_task()`, after classification, before agent dispatch.

Matches task text against the API keyword index + Turkish category patterns. If a strong match (score >= 0.6), calls the API directly, formats the response, and returns it as the task result. No LLM involved.

Example: "Istanbul hava durumu" → Turkish pattern matches weather → calls wttr.in → formats response → done.

### Layer 1: Context Enrichment (LLM Gets Pre-fetched Data)

**File:** `src/core/fast_resolver.py` → `enrich_context(task)`
**Called from:** `src/core/orchestrator.py`, after workflow pre-hooks, before agent dispatch.
**Injected via:** `src/agents/base.py` `_build_context()` reads `task_ctx["api_enrichment"]`.

Medium match (0.3-0.6): fetches API data and injects it as facts in the agent's context. The LLM reasons over real data instead of hallucinating.

### Layer 2: smart_search Tool

**File:** `src/tools/smart_search.py` → `smart_search(query: str)`
**Registered in:** `src/tools/__init__.py` as an optional tool in `TOOL_REGISTRY`.

When an agent needs to search, it calls `smart_search(query)` instead of `web_search`. Internally routes:
1. API registry (keyword match → call_api)
2. MCP tools (Fetch for URL extraction)
3. Web search (Brave/GCSE/DuckDuckGo fallback)

Returns the best result with source attribution.

## Matching System

### Keyword Index

**Built by:** `src/tools/free_apis.py` → `build_keyword_index()`
**Storage:** `api_keywords` table (api_name, keyword, source)
**Rebuilt:** On startup (`src/app/run.py`) and after discovery runs.

Each API's description is tokenized into keywords (stop words removed). Matching is keyword overlap: how many of the task's keywords appear in the API's keyword set.

### Turkish Category Patterns

**Defined in:** `src/tools/free_apis.py` → `TURKISH_CATEGORY_PATTERNS`
**Storage:** `api_category_patterns` table
**Coverage:** 14 categories (weather, currency, pharmacy, earthquake, fuel, gold, prayer_times, time, news, translation, map, travel, holiday, sports)

Regex patterns for Turkish terms that don't appear in English API descriptions. Score 0.8 (strong signal).

### Ranking

```
final_score = relevance_score (keyword + category + Turkish pattern)
```

Reliability penalizes bad APIs: warning → 0.5x, demoted → 0.3x, suspended → excluded.

## Reliability Tracking

**Table:** `api_reliability` (api_name, success_count, failure_count, status)
**Updated by:** `src/infra/db.py` → `record_api_call(api_name, success)`

Auto-demotion thresholds:
- Warning: <50% success (5+ calls) → ranking -50%
- Demoted: <25% success (5+ calls) → excluded from Layer 0/1
- Suspended: <10% success (10+ calls) → excluded from all layers

Manual override via `/smartsearch` → [Unsuspend All].

## API Discovery

**Function:** `src/tools/free_apis.py` → `discover_new_apis(source)`
**Scheduled:** 8:30am daily, catch-up if >36h missed
**Sources:** public-apis GitHub, free-apis.github.io, MCP registry, ClawHub

After discovery, keyword index is rebuilt automatically.

## MCP Integration

**Config:** `mcp.yaml` (3 servers: fetch, sequential_thinking, github)
**Stubs:** Registered at import time in `src/tools/__init__.py` → `_register_mcp_stubs()`
**Connection:** Lazy — first tool call triggers actual MCP server connection

Agents see MCP tools in their tool list (stubs). On first call, `execute_tool()` connects the MCP server, then the real function replaces the stub.

## Skill Injection

**File:** `src/memory/skills.py` → `find_relevant_skills(task_text, limit=3)`
**Injected in:** `src/agents/base.py` `_build_context()` (Phase 13.2)

24 seed skills with hand-crafted Turkish/English trigger patterns. Auto-captured skills from task execution (quality varies — see `docs/superpowers/specs/2026-04-03-skill-system-overhaul-findings.md`).

## i2p Workflow Integration

**api_hints field:** Steps in `i2p_v3.json` can specify `api_hints: ["market_data"]`.
**Extraction:** `src/workflows/engine/expander.py` copies api_hints to task context.
**Enrichment:** `src/workflows/engine/hooks.py` pre-fetches API data for those hints in `pre_execute_workflow_step()`.

16 research steps in phases 0-5 have `smart_search` in tools_hint. 12 steps have api_hints.

## Observability

**Command:** `/smartsearch` in Telegram
**Shows:** Layer hit rates, top/worst performing APIs, source breakdown, registry size
**Buttons:** [Refresh Now], [Top Failures], [Unsuspend All]
**Logging:** `smart_search_log` table tracks every query with layer, source, success, response_ms

## Key Files

| File | Purpose |
|------|---------|
| `src/core/fast_resolver.py` | Layer 0 + Layer 1 resolution |
| `src/tools/smart_search.py` | Layer 2 unified search tool |
| `src/tools/free_apis.py` | API registry, keyword index, Turkish patterns, discovery |
| `src/tools/__init__.py` | Tool registration, MCP stubs, lazy connect |
| `src/infra/db.py` | Tables + helpers (api_keywords, api_reliability, smart_search_log) |
| `src/memory/skills.py` | Skill matching and injection |
| `src/core/orchestrator.py` | Calls fast_resolver, schedules discovery |
| `src/agents/base.py` | Injects enrichment + skills into agent context |
| `src/workflows/engine/expander.py` | Extracts api_hints from workflow steps |
| `src/workflows/engine/hooks.py` | Pre-fetches data for api_hints |
| `src/app/telegram_bot.py` | /smartsearch command |

## Known Limitations

- Response formatting is generic (raw JSON) — needs per-category formatters
- Reliability uses all-time counters, not rolling 7-day window
- Parameter extraction uses string replacement on example_endpoint — fragile for discovered APIs
- Layer 0 failure doesn't fall through to Layer 1 enrichment for same match
