# Memory Redesign — Decisions, Reasoning & Constraints

Date: 2026-04-07
Purpose: Context dump for any agent picking up parallel work on the memory subsystem.

---

## The Core Problem

KutAI runs local LLMs (8B-14B) with tight context windows. The hard cap is ~8K tokens, often tasks need to work with less. Every injected token competes with the model's reasoning space.

The current memory/context system was built incrementally — 13 context layers added over time, each independently deciding what to inject. Nothing gates whether a layer should fire. Result: 15K+ tokens of context injected into prompts that can only hold 8K total. Small LLMs can't distinguish noise from signal, so irrelevant context actively degrades output quality.

This isn't "nice to optimize" — it's "the system literally doesn't fit in the context window."

## Hardware Constraints

- **GPU**: Single NVIDIA GPU shared between llama-server and optional Ollama
- **VRAM**: Limited — context size is kept small on purpose to use more GPU layers for the model itself. More layers on GPU = faster inference. Larger context = fewer layers = slower.
- **RAM**: Limited — no room for large in-memory indexes or caches
- **Cloud models**: Not connected yet. Everything runs local.
- **Embedding**: sentence-transformers on CPU (multilingual-e5-base, 768d). No GPU cost but adds latency.

## LLM Constraints

- Small models (8B-14B) can't reliably produce structured JSON output
- The old rich grading prompt (1-5 score + JSON with situation/strategy/tools in `router.py`) was replaced with binary YES/NO grading (`grading.py`) specifically because small LLMs kept breaking the JSON format
- This fixed grading reliability but lost skill extraction as collateral — deferred grades (majority of tasks) never produce skill metadata
- Any structured output format must degrade gracefully — if the model can't produce optional fields, the core function (grading, skill capture, etc.) must still work
- Small LLMs are better at keyword output (PASS/FAIL) than structured formats (JSON, YAML)
- Using different keywords (PASS/FAIL vs YES/NO) helps last-resort parsing — regex for bare PASS or FAIL anywhere in output is nearly impossible to miss

## Key Decisions Made

### 1. No backward compatibility
Backward compat always complicates systems. When we kill the old score-based grader, we kill it completely. No `score` field preserved "just in case." All downstream code that checks `>= 4.0` gets rewritten to check `verdict.passed`.

### 2. Binary grading with progressive extraction (not two separate graders)
We unify `router.py:grade_response()` and `grading.py:grade_task()` into one prompt. The prompt asks RELEVANT/COMPLETE/VERDICT (binary, robust) followed by SITUATION/STRATEGY/TOOLS (optional, for skill capture). Parsing is progressive — if structured fields fail, the grade still works. This fixes the deferred grading gap without adding LLM calls.

### 3. No additional LLM calls for memory improvement
Running an extra LLM pass before/after every task is too much load on the limited GPU. Any memory improvement must piggyback on existing LLM calls (grading) or use heuristics. This rules out: real HyDE (needs LLM), LLM-based preference detection, LLM-based query reformulation, LLM-based insight extraction.

### 4. Context gating via static policy map + lightweight heuristics
We considered three gating approaches:
- **Static routing** (task profiles define which layers): Simple, predictable, zero overhead
- **Heuristic routing** (rules based on task metadata): Slightly more adaptive
- **LLM-assisted routing** (classifier decides): Most adaptive but wastes the budget it's trying to save

Decision: Static policy map (keyed by task profile name from existing `TASK_PROFILES` in capabilities.py) with heuristic overrides. Heuristics add value for workflows (structured metadata like tools_hint, phase, depends_on) without LLM cost.

### 5. Precision over coverage for retrieval
On 8K context, injecting one highly relevant result beats injecting five vaguely related ones. The model can't sort signal from noise. Thresholds raised aggressively: RAG 0.5→0.72, skills 0.6→0.75, top_k 5→2. Risk of missing useful context is accepted — it's a better failure mode than drowning the model in noise.

### 6. RAG collection gating by task type
Currently RAG queries all 5 core collections regardless of task. A shopping task gets code error patterns; a code task gets shopping reviews. Wasteful and noisy. Each task type maps to 1-2 relevant collections.

### 7. Hard token budget with weighted distribution
Available context = 40% of model context window (60% reserved for system prompt + reasoning). Active layers compete for this budget via priority weights. Dependencies get highest weight (they're actual task prerequisites), then skills and RAG, then everything else. Each layer truncates to its allocation.

### 8. Keep RELEVANT + COMPLETE + VERDICT (three binary questions)
The binary grader's three questions are better than a single PASS/FAIL because:
- RELEVANT catches off-topic hallucinated responses
- COMPLETE catches "here's a plan for how I would do this" non-deliverables
- VERDICT is the final call, can override (e.g., RELEVANT=YES COMPLETE=NO but VERDICT=PASS for tasks where a plan IS the deliverable)
- Small LLMs handle YES/NO reliably

### 9. Modular issue tracking for parallel sessions
Instead of one massive redesign, issues are documented individually in `docs/issues/memory-subsystem-findings.md` with severity, independence, and suggested order. Independent fixes (#1-4, #9-11) can be tackled in parallel sessions. The main design only covers the critical path (#6-8).

## What the Main Design Covers

1. **Unified grading** — merge two graders, progressive extraction, fix deferred grade skill capture gap
2. **Context gating** — policy map per task profile, heuristic overrides, hard token budget
3. **Threshold tuning** — RAG/skill thresholds raised, collection gating, top_k reduced

## What the Main Design Does NOT Cover (Parallel Fixes)

See `docs/issues/memory-subsystem-findings.md` for full details on each:

| # | Issue | Quick Summary |
|---|-------|--------------|
| 1 | Fake HyDE | Remove hardcoded template, raw queries are better |
| 2 | Dead conversation summaries | Built, never queried — remove or wire in |
| 3 | Keyword preference detection | ~50% precision, actively misleading — remove or replace |
| 4 | Fake cross-agent insights | Just reformats titles — remove |
| 5 | Disabled reranker | Enable after gating ships |
| 9 | No temporal validity | Facts never expire — add valid_from/valid_to |
| 10 | No bilingual query expansion | Turkish queries miss English knowledge — add dictionary |
| 11 | No precomputed essentials | Simple tasks run full retrieval — add cached brief |
| 12 | Unvalidated RAG/decay weights | Hardcoded, never tested — needs instrumentation |
| 13 | Reranker enablement | Cross-encoder on CPU, enable after gating |

## File References

| File | Relevance |
|------|-----------|
| `docs/issues/memory-subsystem-findings.md` | All 13 issues with details, severity, fix options |
| `docs/issues/context-layers-reference.md` | What each of the 13 context layers provides, sources, token costs |
| `docs/skill-system.md` | Skill system design doc — capture, injection, tracking, thresholds |
| `src/agents/base.py:347-577` | `_build_context()` — the method being refactored |
| `src/core/grading.py` | Binary grader (being unified) |
| `src/core/router.py:1498-1579` | Rich grader (being killed) |
| `src/core/orchestrator.py:2239-2285` | Skill extraction from completed tasks |
| `src/memory/rag.py` | RAG pipeline — thresholds, HyDE, ranking |
| `src/memory/skills.py` | Skill matching, formatting, injection |
| `src/models/capabilities.py:82-372` | `TASK_PROFILES` — 17 profiles used as policy map keys |
| `src/memory/vector_store.py` | ChromaDB wrapper — 7 collections |
| `src/memory/decay.py` | Memory lifecycle management |
| `src/memory/preferences.py` | Preference detection (noisy) |
| `src/memory/episodic.py` | Task results + fake insights |
| `src/memory/conversations.py` | Conversation storage + dead summarization |

## MemPalace Comparison (What We Took, What We Didn't)

Evaluated https://github.com/milla-jovovich/mempalace — a zero-API local memory layer for LLMs.

**Took the philosophy, not the code:**
- Progressive memory loading (don't retrieve unless you need to) → our context gating
- Restraint principle (less is more on tight context) → our threshold tuning + hard budgets

**Didn't take:**
- AAAK compression dialect — KutAI runs local models (token cost is zero), compression fragile with small LLMs
- Palace hierarchy (wing/room/hall) — KutAI's 7 ChromaDB collections already partition better
- MCP server integration — KutAI has its own agent system, not a stateless MCP consumer
- Auto-save hooks — KutAI manages its own context lifecycle

**Deferred for later evaluation:**
- Temporal knowledge graph (valid_from/valid_to) → documented as issue #9
- Precomputed identity brief (L0/L1 layers) → documented as issue #11
- Cross-domain bridge queries → not documented, low value for KutAI's use case
