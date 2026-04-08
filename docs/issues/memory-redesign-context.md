# Memory Redesign — Full Reasoning & Roadmap

Date: 2026-04-07
Purpose: Complete reasoning arc from MemPalace evaluation through current system audit to the vision for a sophisticated memory system. Read this first if you're picking up any memory-related work.

---

## 1. The Spark: MemPalace

We evaluated [MemPalace](https://github.com/milla-jovovich/mempalace) — a zero-API local memory system for LLMs. It's a pure memory layer (no agents, no tasks, no tools) that gives stateless LLM assistants persistent memory via MCP.

### What MemPalace does well

**Progressive memory loading (4-layer wake-up):**
- L0: Identity (~100 tokens) — core facts, always loaded
- L1: Essential story (~800 tokens) — precomputed top-importance moments
- L2: On-demand — filtered retrieval by domain when needed
- L3: Deep search — full semantic search, only when L0-L2 don't cover it

Most queries never need deep search. This is the key insight: **don't retrieve unless you need to**.

**Temporal knowledge graph:** SQLite RDF triples with `valid_from`/`valid_to` windows. Facts expire. "User lives in Istanbul" stored in January can be invalidated when new info arrives. Enables "what was true in March?" queries.

**AAAK compression dialect:** Custom symbolic shorthand (~30x token compression) that LLMs can read. Entity names become 3-char codes, emotions become compact symbols.

**Zero-LLM indexing:** All mining, routing, entity detection, room classification done with heuristics. LLMs only consume memory, never produce the indexing. Zero API cost for memory operations.

### What MemPalace lacks (where KutAI is already stronger)

- No agent execution, task management, or tool use
- Single ChromaDB collection with metadata tags vs KutAI's 7 specialized collections
- Flat vector search only — no HyDE, no query decomposition, no reranking
- No skill system, no A/B testing, no feedback loops
- No memory decay or lifecycle management
- No preference learning or self-improvement analysis

### What we took from MemPalace

**The philosophy, not the code.** Two principles:

1. **Restraint** — ask "does this context deserve injection?" before injecting. MemPalace's layered approach means most tasks get 100-800 tokens of context, not 15K. This directly inspired our context gating system.

2. **Progressive loading** — don't run the full retrieval pipeline for every task. Simple tasks need zero or minimal context. Complex tasks get deeper retrieval. This inspired the policy map where `reviewer` gets zero extra context while `researcher` gets skills + RAG + API.

### What we didn't take and why

- **AAAK compression** — KutAI runs local models where token cost is zero. The constraint is context window size, not token cost. AAAK is also fragile with small LLMs that may not understand the symbolic encoding.
- **Palace hierarchy** (wing/room/hall/drawer) — KutAI's 7 ChromaDB collections already partition better than metadata tags on one collection.
- **MCP server integration** — KutAI is an autonomous agent system, not a stateless MCP consumer.
- **Auto-save hooks** — KutAI manages its own context lifecycle through episodic memory and skill capture.

### What we deferred for later

- **Temporal validity** (valid_from/valid_to on facts) → issue #9
- **Precomputed identity brief** (L0/L1 cache) → issue #11
- **Cross-domain bridge queries** → low value for KutAI's architecture

---

## 2. The Audit: What's Actually Broken

The MemPalace comparison triggered a deep audit of KutAI's memory system. The audit was brutal — the system looks sophisticated on paper but has serious practical issues.

### The core problem

KutAI runs local LLMs (8B-14B) with ~8K context windows. The memory system was built incrementally — 13 context layers added over time, each independently deciding what to inject. Nothing gates whether a layer should fire. Result: **15K+ tokens of context injected into prompts that can only hold 8K total.**

Small LLMs can't distinguish noise from signal. Irrelevant context doesn't just waste tokens — it actively degrades output quality.

### Hardware constraints

- **GPU**: Single NVIDIA, shared between llama-server and Ollama
- **VRAM**: Limited. Context size kept small on purpose — fewer context tokens = more model layers on GPU = faster inference
- **Cloud models**: Not connected yet. Everything runs local.
- **Embedding**: multilingual-e5-base on CPU (768d). No GPU cost but adds latency.

### What the audit found

**Working well:**
- Vector store fundamentals (ChromaDB, 7 collections, metadata handling)
- Memory decay system (running, reasonable thresholds)
- Episodic memory (task results properly captured)
- Error recovery patterns (well-designed, actually used)
- Skill deduplication (prevents explosion of similar skills)

**Half-implemented or broken:**
- **Fake HyDE** — hardcoded template (`"The task was completed successfully..."`) instead of actual hypothetical document embedding. Worse than no HyDE because it biases retrieval toward generic language.
- **Dead conversation summaries** — summarization built every 10 exchanges but never queried by any code path.
- **Keyword preference detection** — `if "python" in combined: patterns.append("Prefers Python")`. ~50% precision. "Python" in an error message = "user prefers Python."
- **Fake cross-agent insights** — just `f"Insight from {agent_type}: Task '{title}'"`. Not extraction, just reformatting. Pollutes semantic collection.

**Fundamentally broken architecture:**
- **Zero context gating** — `_build_context()` unconditionally injects 13 layers. No task-type awareness.
- **Two disconnected grading systems** — binary grader (robust, used by deferred grading) and rich JSON grader (fragile, used by immediate grading). Deferred grades (majority of tasks) never produce skill metadata.
- **Loose retrieval thresholds** — RAG at 0.5 cosine similarity means "vaguely related." Skill matching at 0.6 matches irrelevant skills 30-40% of the time.

### The two-sided problem

The memory system has problems on **both sides**:
1. **What gets stored** — fake insights, noisy preferences, mechanical skill entries pollute the vector space
2. **What gets injected** — everything gets injected regardless of task type or relevance

Fixing only one side isn't enough. Better storage with ungated injection still bloats prompts. Gated injection with noisy storage still retrieves garbage.

---

## 3. The Design: What We Built (Phase 1)

### Approach selection

We considered three approaches:
- **A) Context diet only** — just stop injecting noise. Doesn't fix what gets stored.
- **B) Unified grading + context diet** — fix both sides. One grader does double duty (grading + skill extraction). Context gating stops the bloat.
- **C) Full memory overhaul** — everything in B plus temporal validity, precomputed essentials, bridge queries.

**Chose B.** A leaves skills broken. C is too large for one pass and adds features that don't help the core problem (prompt bloat on 8K context).

### The piggybacking insight

The grading LLM call already reads the full task + response. Instead of adding separate LLM calls for memory improvement (preference detection, insight extraction, skill capture), we **expanded the grading prompt** to also ask for SITUATION/STRATEGY/TOOLS after the binary verdict.

This adds ~50-100 tokens to grading output but **zero additional LLM calls**. The parsing is progressive — if the model can't produce the optional fields, grading still works.

This pattern extends further. When parallel fixes tackle preferences (#3) and insights (#4), they can add more optional fields to the same grading prompt:
- `PREFERENCE: one-line user preference signal or "none"`
- `INSIGHT: one-line reusable learning or "none"`

The progressive parsing infrastructure already handles this — empty fields are silently ignored. One LLM call, multiple memory operations.

**What piggybacking rules out:** Real HyDE (needs LLM *before* retrieval, not after), LLM-based query reformulation, LLM-based conversation summarization. These need their own LLM calls and aren't viable on current hardware.

### What was implemented

1. **Unified grading** — one prompt, one code path. RELEVANT/COMPLETE/VERDICT (binary, robust) + SITUATION/STRATEGY/TOOLS (optional, for skill capture). Progressive parsing cascade: try all fields → try verdict only → scan for bare PASS/FAIL keyword. Kills old `router.py:grade_response()` and numeric scores entirely.

2. **Context gating** — `src/memory/context_policy.py` with policy map for 17 task profiles. Each profile declares which context layers it needs. Heuristic overrides for workflows (tools_hint), follow-ups (is_followup), and missions (mission_id). Hard token budget: 40% of model context window, distributed by priority weights.

3. **Threshold tuning** — RAG relevance 0.5→0.72, skill matching 0.6→0.75, top_k per collection 5→2. RAG collection gating by task type (coder only queries errors+codebase, shopping only queries shopping).

### Key design decisions

**No backward compatibility.** Backward compat always complicates systems. Old `score` field killed entirely. All downstream code rewritten.

**Binary grading, not numeric scores.** Small LLMs produce noisy 1-5 scores. PASS/FAIL is reliable. Skill quality uses injection success rate (real-world signal) instead of grader scores.

**Precision over coverage.** On 8K context, one highly relevant result beats five vaguely related ones. The model can't sort signal from noise. Missing useful context is a better failure mode than drowning in noise.

**Static policy + heuristics, not LLM routing.** An LLM classifier to decide context layers wastes the budget it's trying to save. Static map is zero-cost and predictable. Heuristics add value for workflows without LLM calls.

**Three binary questions, not one.** RELEVANT catches off-topic results. COMPLETE catches "here's a plan" non-deliverables. VERDICT is the final call. Small LLMs handle YES/NO reliably.

---

## 4. The Gaps: What a Sophisticated Memory System Still Needs

After the Phase 1 redesign, prompt bloat is solved and skill extraction works. But the **quality of stored knowledge** is still mediocre, and several memory capabilities are missing entirely. Here's what remains, organized by impact.

### Tier 1: Fix the noise sources (high impact, independent)

These directly degrade retrieval quality. Every fake insight or noisy preference in the vector space makes every future query slightly worse.

| Issue | What's Wrong | Fix Approach |
|-------|-------------|-------------|
| **#3 Keyword preferences** | 50% false positives actively mislead tasks | Remove keyword detection. Piggyback on grading: add `PREFERENCE:` field. |
| **#4 Fake insights** | Reformatted titles pollute semantic collection | Remove the `extract_and_store_insight()` call. Piggyback on grading: add `INSIGHT:` field. |
| **#1 Fake HyDE** | Generic template biases every query toward boilerplate | Remove entirely. Raw queries are better. Revisit when cloud models enable real HyDE. |
| **#2 Dead summaries** | Built every 10 exchanges, never queried. Wastes storage. | Remove or wire into RAG as conversation_summary type. |

### Tier 2: Add missing memory capabilities (medium impact)

These are features KutAI doesn't have at all. They'd make the memory system qualitatively better.

| Issue | What's Missing | Fix Approach |
|-------|---------------|-------------|
| **#9 Temporal validity** | Facts never expire. Stale facts retrieved with high confidence. | Add `valid_from`/`valid_to` to memory entries. Contradiction detection on store. |
| **#10 Bilingual expansion** | Turkish queries miss English knowledge. | Lightweight dictionary-based term expansion. No LLM needed. |
| **#11 Precomputed essentials** | Every task runs full retrieval even for trivial queries. | Daily-refreshed ~200 token identity/project brief. Inject as baseline for all tasks. |
| **#15 Grading test coverage** | apply_grade_result has zero unit tests for PASS/FAIL paths. | Add mocked tests for skill extraction, retry logic, DLQ. |

### Tier 3: Optimize retrieval quality (lower impact, depends on Tier 1)

These improve how well the system finds what it's looking for. Less impactful if the stored data is still noisy (fix Tier 1 first).

| Issue | What's Suboptimal | Fix Approach |
|-------|------------------|-------------|
| **#13 Reranker** | Top-k ordered by embedding similarity, not actual relevance. | Enable existing cross-encoder on CPU. ~200ms latency. |
| **#12 Weight validation** | RAG ranking weights (0.4/0.25/0.2/0.15) are arbitrary. | Instrument retrieval quality after gating ships. Tune with data. |
| **#16 Single-line parsing** | TOOLS field can't span multiple lines. | Multiline-aware regex or document the single-line constraint. |
| **#14 Stale docs** | skill-system.md still references removed functions. | Text-only cleanup. |

### The vision: what "sophisticated" looks like for KutAI

A truly sophisticated memory system for KutAI on local hardware would have:

1. **Zero-waste context injection** ✅ — every injected token is relevant to the specific task. Gating + budgets + high thresholds ensure this. *Implemented.*

2. **Self-improving skill library** ⚠️ — skills captured from every successful task, ranked by real-world injection success, with rich descriptions. *Partially implemented.* The unified grader now captures skill data for all tasks (immediate and deferred), but quality depends on small LLM output. Progressive extraction makes this best-effort gracefully.

3. **Temporal awareness** ❌ — facts expire, contradictions detected, "what was true then?" queries work. *Not implemented.* Issue #9.

4. **Clean vector space** ❌ — no fake insights, no noisy preferences, no generic HyDE templates polluting embeddings. Only real knowledge stored. *Not implemented.* Issues #1-4.

5. **Bilingual retrieval** ❌ — Turkish queries find English knowledge and vice versa. *Not implemented.* Issue #10.

6. **Efficient retrieval** ⚠️ — reranking, validated weights, collection gating. *Partially implemented.* Collection gating and threshold tuning done. Reranker and weight validation pending (#12, #13).

7. **Piggybacked learning** ⚠️ — preferences, insights, and skills all extracted from the single grading LLM call with zero additional cost. *Infrastructure built* (progressive parsing), but only skill extraction is wired in. Preferences and insights pending (#3, #4).

8. **Precomputed essentials** ❌ — a tiny cached brief that replaces broad retrieval for simple tasks. *Not implemented.* Issue #11.

### Recommended execution order

```
Phase 1: Context diet + unified grading     ✅ DONE
    ↓
Phase 2: Clean the vector space             ← NEXT (issues #1-4)
    ↓                                          Independent, parallelizable
Phase 3: Piggyback preferences + insights   ← After #2 (issues #3, #4 option B)
    ↓                                          Adds fields to grading prompt
Phase 4: Enable reranker                    ← After #2 (issue #13)
    ↓                                          Clean data + reranking = quality jump
Phase 5: Temporal validity                  ← Independent (issue #9)
    ↓                                          Medium effort, high value
Phase 6: Bilingual expansion + essentials   ← Independent (issues #10, #11)
    ↓                                          Low effort each
Phase 7: Weight validation                  ← Needs data from phases 2-4
                                               Instrument and tune with real usage
```

Phases 2-6 are largely independent and can run in parallel sessions. Phase 7 needs the system to run with the earlier changes to collect meaningful data.

---

## 5. File References

### Documentation
| File | What it contains |
|------|-----------------|
| `docs/issues/memory-subsystem-findings.md` | 16 issues with details, severity, fix options, priority matrix |
| `docs/issues/context-layers-reference.md` | What each of 13 context layers provides, sources, token costs |
| `docs/issues/memory-redesign-context.md` | This document — full reasoning and roadmap |
| `docs/superpowers/specs/2026-04-07-memory-context-redesign-design.md` | Design spec for Phase 1 |
| `docs/superpowers/plans/2026-04-07-memory-context-redesign.md` | Implementation plan for Phase 1 |
| `docs/skill-system.md` | Skill system design — capture, injection, tracking |

### Code (post-Phase 1)
| File | What it does |
|------|-------------|
| `src/core/grading.py` | Unified grading — prompt, parsing cascade, GradeResult, apply_grade_result with skill extraction |
| `src/memory/context_policy.py` | Context gating — policy map, heuristics, budget calculator |
| `src/agents/base.py:347-577` | `_build_context()` — policy-gated context injection with token budgets |
| `src/memory/rag.py` | RAG pipeline — config dict, collection gating, thresholds |
| `src/memory/skills.py` | Skill matching and injection — updated threshold |
| `src/memory/vector_store.py` | ChromaDB wrapper — 7 collections |
| `src/memory/decay.py` | Memory lifecycle management |
| `src/memory/preferences.py` | Preference detection (broken — issue #3) |
| `src/memory/episodic.py` | Task results + fake insights (issue #4) |
| `src/memory/conversations.py` | Conversation storage + dead summarization (issue #2) |
| `src/models/capabilities.py:82-372` | `TASK_PROFILES` — 17 profiles used as policy map keys |
