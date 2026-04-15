# KutAI Classification System — Refactor Spec

**Date**: 2026-04-15  
**Status**: Findings documented, design pending  
**Goal**: Unify the classification into one reliable system

---

## Problem Statement

KutAI's task classification is unreliable. A user typing "kahve makinesi" (coffee machine) gets a FastAPI web app built instead of a product search. The root cause isn't a missing keyword — it's an architectural problem: two independent classifiers with scattered rules, no shared signals, and no reconciliation.

---

## Current Architecture (What's Broken)

### Two Independent Classifiers

**Layer 1 — Telegram Message Classifier** (`src/app/telegram_bot.py`)
- `_classify_message_by_keywords()` (line 3965): keyword matching → 12 message types
- `_classify_user_message()` (line 3913): LLM fallback with MESSAGE_CLASSIFIER_PROMPT (line 3879)
- Shopping keywords (line 4002-4008): only intent words ("fiyat", "almak istiyorum"...), no product nouns
- Output: message type string ("shopping", "task", "question", "mission"...)
- If "shopping" → creates task with `agent_type="shopping_advisor"`
- If anything else → creates task with `agent_type="executor"` (default)

**Layer 2 — Task Classifier** (`src/core/task_classifier.py`)
- `classify_task()` (line 227): called by orchestrator when `agent_type="executor"`
- `_classify_with_llm()` (line 248): LLM with CLASSIFIER_PROMPT (line 52) → 14 agent types
- `_classify_by_keywords()` (line 348): keyword fallback → same 14 agent types
- Only runs if Layer 1 produced "executor" — if Layer 1 got it right, Layer 2 never fires

**Layer 3 — Dispatch Regex Fallback** (`src/workflows/engine/dispatch.py`)
- `should_start_shopping_workflow()` (line 79): regex patterns for buying intent
- Called by orchestrator (line 1530) as last-resort shopping detection
- Only matches intent verbs ("almak istiyorum", "satın al"), never bare nouns

### The Fatal Sequence for "kahve makinesi"

```
User types: "kahve makinesi"
  → Telegram keywords: NO MATCH (no intent words) → type="task"
  → Telegram LLM: returns "task" (bare noun, no action context)
  → Task created: agent_type="executor"
  → Orchestrator classify_task(): LLM returns "coder" confidence=0.4
  → Confidence 0.4 < 0.7 threshold: classification IGNORED
  → Dispatch regex: NO MATCH (no buying verbs)
  → Agent: executor → gemma builds a FastAPI app
```

### 10 Specific Problems

1. **Dual classifiers don't communicate** — message classification context lost at task creation
2. **No product recognition** — all layers match intent keywords only, not product nouns
3. **Scattered keyword rules** — shopping keywords in 4+ files with different subsets
4. **LLM unreliable on small models** — gemma/local models misclassify bare nouns
5. **Confidence threshold gap** — 0.4-0.7 range: LLM result used but not overridden by keywords
6. **No keyword-LLM reconciliation** — if LLM returns low confidence, keywords aren't consulted
7. **Regex overfits to Turkish verbs** — dispatch.py patterns require explicit buying intent
8. **Context not propagated** — chat history, user preferences not available to task classifier
9. **Keyword duplication** — "fiyat", "compare", etc. repeated across task_classifier.py, telegram_bot.py, dispatch.py
10. **Shopping workflow mismatch** — message type "shopping" doesn't map to workflow name

---

## Key Files

| File | Lines | Role |
|------|-------|------|
| `src/app/telegram_bot.py` | 3397, 3723, 3879-3962, 3965-4061, 4002-4008 | Message classification + routing |
| `src/core/task_classifier.py` | 52-96, 227-310, 315-375 | Task classification (LLM + keywords) |
| `src/core/orchestrator.py` | 1489-1505, 1522-1540 | Classification gate + shopping fallback |
| `src/workflows/engine/dispatch.py` | 30-56, 69-90 | Shopping regex patterns |
| `src/shopping/intelligence/query_analyzer.py` | — | Query analysis (NOT used for routing) |

### Agent Types (18 registered in `src/agents/__init__.py`)

planner, architect, coder, implementer, fixer, test_generator, reviewer, visual_reviewer, researcher, analyst, writer, summarizer, assistant, executor, shopping_advisor, product_researcher, deal_analyst, shopping_clarifier

### Message Types (Telegram bot)

status_query, todo, load_control, bug_report, feature_request, casual, mission, task, question, shopping, feedback, ui_note, progress_inquiry, followup, clarification_response

### Confidence Scores (inconsistent)

- LLM classification: 0.85
- Keyword match: 0.4  
- Default fallback: 0.3
- Message classifier LLM fallback threshold: < 0.4
- Task classifier override threshold: >= 0.7

---

## Shopping Pipeline Status (Fixes Applied 2026-04-15)

These are on `main` now (merged from `fix/shopping-pipeline-relevance-filter`):

### Committed Today
- **Relevance filtering**: `_relevance_score()` and `_filter_relevant()` in `pipeline.py` — scores products against query tokens, filters irrelevant results
- **Product matcher wired**: `_match_and_flatten()` in `pipeline.py` — calls `product_matcher.py` for cross-source deduplication
- **Per-scraper timeout**: 20s limit in `fallback_chain.py` — one blocked scraper can't starve others
- **Pipeline timeout**: 30s → 45s for product search
- **Test import fix**: `error_recovery` → `scraper_failure_handler` in test_resilience.py
- **Classifier hack** (temporary): product nouns added to keyword rules + LLM prompt hint

### Committed (from 2026-04-14 session)
- **Brotli fix**: vecihi fetchers strip `br` from Accept-Encoding
- **Akakce CSS**: fallback selectors `a.iC`, `span.pt_v9`, `span.pb_v8`
- **Eksisozluk URL**: `/?q=` → `/basliklar/ara?searchForm.Keywords=`
- **Price parsing**: strip "+16 FİYAT" suffix
- **Hooks crash**: guard `validate_artifact_schema` against non-dict keys
- **Orchestrator**: `_stop_server()` → `shutdown()`, `current_litellm_name` → `current_model`
- **Health check**: add `_health_check()` to `local_model_manager.py`

### Still Broken / Not Addressed
- **Scraper coverage**: only Trendyol returned results in morning test — Akakce/Hepsiburada/others may still fail at runtime (needs live testing with running orchestrator using venv Python)
- **GPU timeout reviewer task**: "reviewer task kept requesting GPU access, timing out at 60s each" — identified but not fixed
- **Intelligence modules not wired**: `value_scorer`, `review_synthesizer`, `fake_discount_detector`, `delivery_compare` exist but aren't called
- **Classification is a hack**: product noun keywords are whack-a-mole, need the full refactor

---

## Design Direction (For Next Session)

**Goal**: Unify into one classification system with these properties:

1. **Single source of truth** — one classifier, one set of rules, one confidence model
2. **Product-aware** — can distinguish "kahve makinesi" (shopping) from "state machine" (coding) without enumerating every product
3. **Context-aware** — knows whether user came from a shopping button, chat history, etc.
4. **Graceful degradation** — when LLM is unreliable, keyword/heuristic fallback is ALWAYS consulted
5. **Testable** — classification decisions logged and reproducible

Possible approaches to explore:
- **Negative classification**: instead of listing all products, detect code/tech signals and default everything else to a lightweight intent check
- **Two-pass with reconciliation**: fast keyword pass + LLM pass, merge results with weighted confidence
- **Embedding-based**: use the existing multilingual-e5-base embeddings to classify by similarity to category exemplars
