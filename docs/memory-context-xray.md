# KutAI Memory & Context Management

> How the system stores knowledge, retrieves it, and decides what to inject into agent prompts — and why each piece works the way it does.

---

## The Core Constraint

KutAI runs 8B-14B parameter local LLMs with ~8K context windows. Context is the scarcest resource. Every token injected into a prompt displaces a token the model could use for reasoning. Injecting irrelevant context doesn't just waste space — it actively degrades output quality because small models can't distinguish signal from noise.

This constraint shapes every design decision below: precision over coverage, static gating over LLM routing, binary grading over numeric scores, piggybacked extraction over separate LLM calls.

---

## Architecture Overview

```
Task arrives
    │
    ├─ Context Policy ──► Which layers does this task type need?
    │   (context_policy.py)    Static map + heuristic overrides
    │
    ├─ Budget Calculator ──► How many tokens per layer?
    │   (context_policy.py)    40% of model context, weighted distribution
    │
    ├─ Layer Injection ──► Each active layer fetches its content
    │   (base.py)             Truncated to its budget
    │   │
    │   ├─ deps ──► Dependency task results
    │   ├─ prior ──► Inline prior step results
    │   ├─ skills ──► Matched execution recipes (vector search)
    │   ├─ rag ──► Retrieved knowledge (vector search + reranker)
    │   ├─ convo ──► Recent conversation exchanges
    │   ├─ board ──► Mission blackboard
    │   ├─ profile ──► Project profile
    │   ├─ ambient ──► Ambient context (time, system state)
    │   ├─ api ──► API enrichment data
    │   ├─ prefs ──► User preferences
    │   └─ memory ──► Project memory (key-value)
    │
    └─ Agent Execution ──► Model sees task + gated context
        │
        └─ Grading ──► Binary verdict + piggybacked learning
            (grading.py)    Skills, preferences, insights extracted
```

### Key Files

| File | Responsibility |
|------|---------------|
| `src/memory/context_policy.py` | Policy map, heuristic overrides, budget calculator |
| `src/agents/base.py:_build_context()` | Assembles the prompt from active layers |
| `src/memory/rag.py` | RAG pipeline: query, rank, rerank, deduplicate, format |
| `src/memory/skills.py` | Skill matching, injection, A/B tracking |
| `src/core/grading.py` | Unified grading with piggybacked learning extraction |
| `src/memory/vector_store.py` | ChromaDB wrapper — 7 specialized collections |
| `src/memory/embeddings.py` | Embedding generation (multilingual-e5-base, 768d, CPU) |
| `src/memory/episodic.py` | Task result storage + insight storage |
| `src/memory/conversations.py` | Exchange storage + periodic summarization |
| `src/memory/preferences.py` | Preference storage and retrieval |
| `src/memory/decay.py` | Memory lifecycle — importance-weighted decay |

---

## Context Gating

### Why It Exists

Without gating, `_build_context()` unconditionally injects all 11 context layers into every prompt — roughly 15K tokens. On an 8K context model, this literally doesn't fit. Even on larger contexts, a shopping price check doesn't need codebase error patterns, and a code review doesn't need conversation history.

### How It Works

**Static policy map** — each of the 17 task profiles declares which context layers it needs:

```python
CONTEXT_POLICIES = {
    "executor":         {"deps", "skills", "api"},
    "coder":            {"deps", "skills", "profile", "rag"},
    "shopping_advisor": {"skills", "convo"},
    "reviewer":         set(),          # zero extra context
    "assistant":        {"convo", "rag", "memory", "prefs"},
    ...
}
```

**Heuristic overrides** — four runtime adjustments on top of the static policy:
- Workflow steps with `tools_hint` → add `skills` + `api`
- Tasks with `depends_on` → add `deps`
- Follow-up tasks (`is_followup`) → add `convo`
- Mission tasks (`mission_id`) → add `board`

**Token budget** — 40% of the model's context window is allocated to injected context (60% reserved for system prompt + model reasoning). Active layers share this budget by priority weight:

| Layer | Weight | Rationale |
|-------|--------|-----------|
| deps | 5 | Dependency results are the most critical context |
| prior | 4 | Inline prior step results, high relevance |
| skills | 3 | Execution recipes guide agent behavior |
| rag | 3 | Retrieved knowledge, precision-filtered |
| convo | 2 | Conversation context for follow-ups |
| board | 2 | Mission state for multi-task coordination |
| profile | 1 | Project metadata, small and stable |
| ambient | 1 | Time/state, always small |
| api | 1 | API enrichment, pre-formatted |
| memory | 1 | Key-value project facts |
| prefs | 1 | User preferences, small |

Example: an `executor` task on an 8K model activates `{deps, skills, api}` with weights 5+3+1=9. Available budget: 3276 tokens. `deps` gets ~1820, `skills` ~1092, `api` ~364.

### Why Static, Not LLM-Routed

An LLM classifier to decide context layers would consume the budget it's trying to save. The static map is zero-cost, predictable, and covers 17 task profiles. The heuristic overrides handle the four dynamic conditions that matter. No measured failure case justifies adding an LLM call to the routing path.

---

## RAG Pipeline

### What It Does

Retrieves relevant knowledge from ChromaDB collections and formats it for prompt injection. Entry point: `retrieve_context()` in `rag.py`.

### Pipeline Steps

```
1. Build query text from task title + description
2. Decompose multi-part queries (split on "and", ";", etc.)
3. Query collections gated by agent type (2 results per collection)
4. Rank all results: relevance × 0.4 + recency × 0.25 + importance × 0.2 + access_freq × 0.15
5. Filter by minimum relevance (cosine ≥ 0.72)
6. Cross-encoder rerank (if 3+ results survive filtering)
7. Deduplicate (Jaccard ≥ 0.85)
8. Format within token budget
```

### Collection Gating

Not every agent type needs every collection. A coder never needs shopping data. A shopping advisor never needs error patterns:

```python
RAG_COLLECTIONS = {
    "coder":            ["errors", "codebase"],
    "researcher":       ["web_knowledge", "semantic", "conversations"],
    "shopping_advisor": ["shopping"],
    "assistant":        ["semantic", "conversations"],
    ...
}
# Unknown types default to: ["episodic", "semantic"]
```

### Thresholds

| Parameter | Value | Why |
|-----------|-------|-----|
| min_relevance | 0.72 cosine | On multilingual-e5-base, 0.72 = clearly same topic. Below is noise. |
| top_k per collection | 2 | With hard budgets, can't use 5. Two high-quality results per collection. |
| dedup threshold | 0.85 Jaccard | Catches near-duplicates without over-filtering. |

### Cross-Encoder Reranker

**Model:** `ms-marco-MiniLM-L-6-v2` — runs on CPU, ~200ms for 5 results, no VRAM cost.

**Why it exists:** Bi-encoder cosine similarity (what ChromaDB uses) compares independent embeddings. A cross-encoder sees query and document together, producing significantly more accurate relevance scores. On tight budgets, the difference between rank 1 and rank 3 matters.

**When it fires:** Only when 3+ results survive the relevance filter. Reranking 1-2 results has no value. Lazy-loaded on first use — if `sentence-transformers` isn't available, silently falls back to unranked results. The `predict()` call runs via `asyncio.to_thread()` to avoid blocking the event loop.

### What Was Removed and Why

**Fake HyDE** — The system had a "Hypothetical Document Embeddings" implementation that generated a hardcoded template (`"The task was completed successfully..."`) instead of calling an LLM. This biased every query toward generic boilerplate language. Raw query embedding produces better retrieval. Real HyDE requires an LLM call before retrieval — not viable on local hardware.

---

## Unified Grading

### What It Does

Every completed task gets graded by an LLM call (dispatched as `OVERHEAD` — won't trigger model swaps). The grading prompt asks 8 questions:

```
RELEVANT: YES or NO        ─┐
COMPLETE: YES or NO         ├─ Binary verdict (robust, always works)
VERDICT:  PASS or FAIL     ─┘
SITUATION: one line         ─┐
STRATEGY:  one line          ├─ Skill extraction (optional, best-effort)
TOOLS:     comma list       ─┘
PREFERENCE: one line or NONE ─┐
INSIGHT:    one line or NONE ─┘─ Piggybacked learning (optional)
```

### Why One Prompt, Not Two

The system previously had two separate graders: a rich JSON grader (fragile with small LLMs) and a binary YES/NO grader (robust). Most tasks used the binary grader via deferred grading, which meant they never captured skill metadata. The unified prompt puts binary fields first (what small LLMs handle reliably) and optional extraction fields after. If the model can't produce SITUATION/STRATEGY/TOOLS, grading still works — only skill extraction degrades to a mechanical fallback.

### Parsing Cascade

The parser handles progressively degraded LLM output:

1. Parse all 8 fields via regex → full data
2. If SITUATION/STRATEGY/TOOLS fail → grade valid, skill fields empty
3. If RELEVANT/COMPLETE fail → derive from VERDICT alone
4. If VERDICT line not found → scan for bare `PASS` or `FAIL` anywhere
5. Nothing parseable → raise ValueError (grader incapable)

Step 4 is critical: small LLMs that ramble or break structure almost always emit "PASS" or "FAIL" somewhere in their output.

**Multiline support:** The `_parse_text_field` regex captures across line breaks until the next `KEY:` marker or end of string, then collapses internal whitespace. This handles small LLMs that wrap long TOOLS lists across lines.

**NONE normalization:** Fields like PREFERENCE and INSIGHT use `_is_none_value()` to catch common LLM refusal variants: "NONE", "N/A", "No preference observed", "Not applicable", "nil", "-". Without this, refusal strings would be stored as real preferences/insights in the vector space.

### GradeResult

```python
@dataclass
class GradeResult:
    passed: bool
    relevant: Optional[bool] = None
    complete: Optional[bool] = None
    situation: str = ""       # what problem was solved
    strategy: str = ""        # what approach worked
    tools: list[str] = []     # which tools were effective
    preference: str = ""      # user preference signal
    insight: str = ""         # reusable learning
    raw: str = ""             # raw grader output
```

No numeric score. No backward compatibility. PASS/FAIL is the only verdict. Skill quality is measured by injection success rate (real-world signal from A/B tracking), not grader scores.

### What Happens After Grading

**On PASS:**
1. Task transitions to `completed`
2. Model quality feedback recorded
3. Skill extraction — rich (from verdict fields) or mechanical fallback (tools + iterations)
4. Preference stored (if non-empty, via `store_preference()`)
5. Insight stored (if non-empty, via `store_insight()`)
6. Injection success tracked for previously injected skills
7. Telegram notification (non-silent tasks)

**On FAIL:**
1. Generating model added to exclusion list
2. Retry timing computed (exponential backoff with jitter)
3. If retries remain → task transitions to `pending` with delay
4. If terminal → task transitions to `failed`, quarantined to DLQ

---

## Piggybacked Learning

### The Insight

The grading LLM call already reads the full task description and response. Instead of making separate LLM calls for preference detection, insight extraction, or skill capture, we expand the grading prompt to also ask for these signals. Cost: ~30 extra prompt tokens, ~20 extra output tokens, zero additional LLM calls.

### Three Extraction Channels

**Skills** (`SITUATION` / `STRATEGY` / `TOOLS`) — captured for every PASS verdict where iterations ≥ 2 and tools were used. Stored in the skill library with description, strategy summary, and effective tools. When the grader can't produce these fields, falls back to mechanical extraction from task metadata.

**Preferences** (`PREFERENCE`) — captured when the grader observes a user preference signal in the task. Stored with `confidence=0.8` via `store_preference()` in the semantic collection with deterministic doc_id (prevents duplicates). Only injected for `assistant` and `writer` task profiles — code-focused profiles get preference guidance through skills instead.

**Insights** (`INSIGHT`) — captured when the grader identifies a reusable learning. Stored with `importance=7` in the semantic collection as `type: cross_agent_insight`. Available to any agent type whose RAG collections include `semantic`.

### What Was Replaced

**Keyword preference detection** — scanned feedback text for keywords like "python", "typescript", "snake_case". If "python" appeared in an error message, it stored "user prefers Python" with high confidence. ~50% false positive rate. Replaced by grader-observed preferences that have full task context.

**Fake insight extraction** — reformatted task titles as `f"Insight from {agent_type}: Task '{title}' — {result_preview}"`. Not real extraction — just stored the preview with an "insight" label. Polluted the semantic collection with noise. Replaced by grader-extracted insights that contain actual learnings.

---

## Vector Store

### Collections

ChromaDB with 7 specialized collections. Each stores embeddings via `multilingual-e5-base` (768 dimensions, CPU):

| Collection | What It Stores | Who Queries It |
|------------|---------------|----------------|
| `episodic` | Task results (success/failure, model used, tools, iterations) | fixer, error_recovery, planner, architect |
| `semantic` | Facts, insights, preferences, web knowledge summaries | researcher, assistant, writer, planner, architect, analyst |
| `errors` | Error patterns with prevention hints and fix records | coder, fixer, implementer, error_recovery, test_generator |
| `codebase` | Code context, file summaries | coder, fixer, implementer, test_generator |
| `shopping` | Products, reviews, shopping sessions | shopping_advisor |
| `conversations` | User-AI exchanges + periodic summaries | assistant, researcher |
| `web_knowledge` | Web search results, scraped content | researcher, analyst |

### Conversation Summaries

Every 10 exchanges per chat, `maybe_summarize()` creates a ~300 character summary covering topics discussed and key exchange previews. These are stored in the `conversations` collection (not `semantic`) with `type: conversation_summary`. They're retrieved through the normal RAG pipeline when an agent type's `RAG_COLLECTIONS` includes `conversations`. This is more context-efficient than injecting raw exchanges — a summary of 10 exchanges costs fewer tokens than 2 raw exchanges.

### Memory Decay

Facts decay over time with a 30-day half-life. Important facts (error recovery patterns, user preferences with `importance ≥ 8`) are protected from decay. Access frequency provides a secondary signal — frequently retrieved facts stay relevant longer.

---

## What Doesn't Exist Yet

### Temporal Validity

Facts never expire. "User lives in Ankara" stored in January competes with "User lives in Istanbul" stored in March. The decay system helps (old facts lose weight over time) but can't handle outright contradictions — both facts coexist in the vector space. Fix requires `valid_from`/`valid_to` fields on memory entries and contradiction detection at store time.

### Bilingual Query Expansion

Turkish queries miss English-stored knowledge. `multilingual-e5-base` handles some cross-language similarity but isn't perfect for domain-specific terms. A lightweight dictionary of ~200 common term pairs (Turkish ↔ English for shopping, tech, common domains) would bridge the gap without LLM calls.

### Precomputed Essentials

Every task runs the full retrieval pipeline even for trivial queries. A ~200 token daily-refreshed identity brief (language, timezone, communication style, active project context) would serve as a baseline for all tasks. Simple tasks would get this and nothing else, skipping the retrieval pipeline entirely.

### RAG Weight Validation

The ranking formula (`relevance × 0.4 + recency × 0.25 + importance × 0.2 + access_freq × 0.15`) is unvalidated. Different task types likely need different weight profiles. Validating requires instrumentation: for completed PASS tasks, check whether injected context was actually referenced in the output. This needs production data from the current system.

---

## Design Decisions Reference

| Decision | Chose | Over | Why |
|----------|-------|------|-----|
| Context routing | Static policy map | LLM classifier | Zero-cost, predictable. LLM classifier wastes the budget it's saving. |
| Grading format | Binary PASS/FAIL | Numeric 1-5 score | Small LLMs produce noisy scores. PASS/FAIL is reliable. |
| Skill quality signal | Injection success rate | Grader score | Real-world signal beats grader opinion. A/B testing built in. |
| Retrieval precision | High thresholds (0.72) | Low thresholds (0.5) | On 8K context, one relevant result beats five vaguely related ones. |
| Preference learning | Piggybacked on grading | Keyword detection | Grader has full task context. Keywords have ~50% false positive rate. |
| Insight extraction | Piggybacked on grading | Reformatted titles | Grader produces real insights. Titles are noise in the vector space. |
| Conversation memory | Summaries via RAG | Raw exchange injection | 300-char summary of 10 exchanges < 2 raw exchanges in tokens. |
| Reranker | Cross-encoder on CPU | Embedding similarity only | Significantly more accurate for ranking. ~200ms latency, no VRAM cost. |
| Context budget | 40% of model window | Unbounded | 60% reserved for system prompt + reasoning. Hard limit prevents overflow. |
| HyDE | Removed | Fake template | Real HyDE needs LLM call (not viable). Fake template biases queries. |
