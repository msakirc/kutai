# Shopping Plan Analysis — Current Architecture Comparison

## Summary
plans_shopping.md describes a 13-phase Turkish e-commerce intelligence system.
This analysis maps each phase against what exists in kutay, identifies what to reuse,
what to build, and what to defer.

---

## Phase 0: Data Models & Schema
**Status**: Build new, but leverage existing DB patterns

| Step | Reuse | Build | Notes |
|------|-------|-------|-------|
| 0.1 Product Data Model | - | `src/shopping/models.py` | Standard dataclasses |
| 0.2 Product Cache | `src/infra/db.py` patterns | New tables in same DB or separate | TTL invalidation is new |
| 0.3 Request Tracking | - | New tables | Feeds anti-detection |
| 0.4 Turkish Text Utils | - | `src/shopping/text_utils.py` | Critical, build early |
| 0.5 Config | `src/app/config.py` pattern | `src/shopping/config.py` | YAML-driven |

**Recommendation**: Start here. Models + text utils are dependency-free.

---

## Phase 1: Scraping Tools
**Status**: Build new, but reuse web infrastructure

| Step | Reuse | Build | Priority |
|------|-------|-------|----------|
| 1.1 Base Scraper | `aiohttp` from web_search.py | Abstract base class | HIGH — everything depends on this |
| 1.2 Akakce | - | New scraper | HIGH — primary price aggregator |
| 1.3 Trendyol | - | New scraper | HIGH — largest Turkish e-commerce |
| 1.4 Hepsiburada | - | New scraper | MEDIUM — hardest, defer initially |
| 1.5 Amazon TR | - | PA-API integration | LOW — requires affiliate account |
| 1.6 Forums | - | New scraper | LOW — nice-to-have |
| 1.7 Eksi Sozluk | - | New scraper | LOW |
| 1.8 Sikayetvar | - | New scraper | MEDIUM — good for warnings |
| 1.9 Grocery | - | New scrapers | MEDIUM |
| 1.10 Scrapling | - | Fallback layer | LOW — only if simple requests fail |
| 1.11 Google CSE | `web_search.py` | Extend existing | MEDIUM |
| 1.12 Sahibinden | - | New scraper | LOW |
| 1.13 Koctas/IKEA | - | New scraper | LOW |

**Recommendation**: Start with Akakce + Trendyol only. Both have APIs/light protection.
Add others incrementally based on user needs.

---

## Phase 2: Remote Execution (GitHub Actions)
**Status**: Skip initially

This adds massive complexity (GitHub Actions workflows, dispatchers, artifact management)
for marginal benefit. The user's home IP is fine for Akakce/Trendyol.
Revisit only if anti-bot becomes a real problem.

---

## Phase 3: Knowledge Base
**Status**: Build as static files, load via existing RAG

| Step | Reuse | Build |
|------|-------|-------|
| 3.1 Turkish Market | RAG/memory system | Markdown file |
| 3.2 Store Profiles | RAG/memory system | Markdown file |
| 3.3 Compatibility | - | JSON files |
| 3.4 Substitutions | - | JSON file |
| 3.5 Search Terms | - | JSON file |
| 3.6 Installments | - | JSON file |
| 3.7 Brand Service | - | Markdown file |

**Recommendation**: Write these as markdown/JSON, ingest via existing `/ingest` command.
LLM reads them as context. No code needed beyond the files themselves.

---

## Phase 4: Intelligence Modules
**Status**: Build as agent prompts + small Python modules

| Step | Reuse | Build |
|------|-------|-------|
| 4.1 Query Analyzer | task_classifier.py pattern | New LLM prompt |
| 4.2 Search Planner | planner agent pattern | New LLM prompt |
| 4.3 Alternative Generator | - | LLM prompt + substitutions.json |
| 4.4 Substitution Engine | - | JSON lookup + LLM |
| 4.5 Constraint Checker | - | Python module |
| 4.6 Value Scorer | - | Python module |
| 4.7 Market Timing | - | LLM + turkish_market.md |
| 4.8 Combo Builder | - | LLM prompt |
| 4.9 Review Synthesizer | - | LLM prompt |
| 4.10 Product Matcher | - | Python module |
| 4.11 Installment Calc | - | Python + installments.json |
| 4.12 Delivery Comparison | - | Python module |
| 4.13 Return Policy | - | store_profiles.md lookup |

**Recommendation**: Most of these are LLM prompts, not code. Build 4.1, 4.2, 4.5, 4.6
first. The rest follow naturally.

---

## Phase 5: Shopping Agents
**Status**: Extend existing agent framework

| Step | Reuse | Build |
|------|-------|-------|
| 5.1 Shopping Advisor | `BaseAgent` | New agent subclass |
| 5.2 Product Researcher | `ResearcherAgent` | Extend with shopping tools |
| 5.3 Deal Analyst | `BaseAgent` | New agent subclass |
| 5.4 Clarification Agent | Clarification flow in base.py | Enhance existing |

**Recommendation**: The existing `ResearcherAgent` with shopping tools registered
covers 80% of this. Add a `ShoppingAgent` that orchestrates the workflow.

---

## Phase 6: Shopping Workflows
**Status**: Use existing workflow engine

| Step | Reuse | Build |
|------|-------|-------|
| 6.1 Main Shopping WF | WorkflowRunner, JSON definitions | New workflow JSON |
| 6.2 Quick Search | - | Simplified workflow |
| 6.3 Combo Research | - | Sub-workflow |
| 6.4 Price Watch | scheduled_tasks | Cron + scraper |
| 6.5 Gift Recommendation | - | Workflow variant |
| 6.6 Exploration | Clarification flow | Guided conversation |

**Recommendation**: Define as JSON workflows like `i2p_v2`.
The engine already handles dependencies, phases, and artifact passing.

---

## Phase 7: Memory Extensions
**Status**: Extend existing memory system

| Step | Reuse | Build |
|------|-------|-------|
| 7.1 User Profile | memory/feedback.py patterns | New DB table |
| 7.2 Price Watch | todo_items pattern | New DB table |
| 7.3 Shopping Session | task context | Extend context |
| 7.4 Purchase History | - | New DB table |

---

## Phase 8: Output Formatting
**Status**: Build for Telegram

- Comparison tables: Telegram doesn't support real tables. Use monospace blocks.
- Product cards: InlineKeyboardButton with product links
- Reuse existing `send_notification` pattern

---

## Phase 9: Integration Points
**Status**: Mostly exists, just wire up

| Step | Reuse | Status |
|------|-------|--------|
| 9.1 Task Classifier | task_classifier.py | Add "shopping" type |
| 9.2 Router | router.py | Add shopping workflow routing |
| 9.3 Orchestrator | orchestrator.py | Already has hooks |
| 9.4 Telegram | telegram_bot.py | Add /price, /watch, /deals |
| 9.5 Perplexica | web_search.py | Already integrated |
| 9.6 Notifications | reminders.py | Extend for price drops |
| 9.7 Researcher | researcher.py | Add shopping tools |
| 9.8 Web Tools | tools/__init__.py | Register scraper tools |

---

## Phase 10-11: Resilience & Special Intelligence
**Status**: Defer

These are advanced features (seasonal patterns, fraud detection, etc.).
Build after the core works. The knowledge files from Phase 3 provide
90% of the intelligence without code.

---

## Phase 12-13: Testing & Maintenance
**Status**: Build alongside each phase

Follow the existing test patterns (unittest + pytest).
Add scraper canary tests that verify selectors still work.

---

## Recommended Build Order

1. **Phase 0**: Models + text utils (foundation)
2. **Phase 1.1**: Base scraper (architecture)
3. **Phase 1.2-1.3**: Akakce + Trendyol (minimum viable)
4. **Phase 3**: Knowledge files (write markdown/JSON)
5. **Phase 9.1-9.4**: Integration points (wire into existing system)
6. **Phase 4.1-4.2**: Query analyzer + search planner
7. **Phase 6.1-6.2**: Main + quick workflows
8. **Phase 5.1**: Shopping agent
9. **Everything else**: Incrementally based on usage

## Key Architecture Decisions

1. **Scrapers register as tools** in TOOL_REGISTRY — agents can call them directly
2. **Shopping sessions are missions** — use existing workflow engine
3. **Knowledge is context** — markdown/JSON files loaded via RAG, not hardcoded logic
4. **Start with 2 scrapers** — Akakce (price aggregation) + Trendyol (largest market)
5. **Skip GitHub Actions** — local scraping is fine for low-risk sites
6. **Price watch uses scheduled_tasks** — same infrastructure as todo reminders
