# Memory & Context Management Redesign

Date: 2026-04-07
Status: Design
Scope: Unified grading, context gating, threshold tuning

## Problem

KutAI's context injection system (`_build_context()`) unconditionally injects ~13 layers totaling 15K+ tokens. The system runs on local LLMs with ~8K context windows. Result: prompts overflow, small models can't distinguish signal from noise, output quality degrades.

Additionally, two disconnected grading systems exist — the binary grader (robust, used by deferred grading) lost skill extraction when it replaced the rich JSON grader (fragile with small LLMs). Most tasks never get meaningful skill capture.

## Goals

1. Context injection respects model context limits — hard budget, never exceed
2. Tasks only receive context layers relevant to their type
3. Single grading pathway that captures skill data without sacrificing robustness
4. Higher retrieval precision — fewer, better results

## Non-Goals

- Fixing broken subsystems (fake HyDE, keyword preferences, dead summaries, fake insights) — documented in `docs/issues/memory-subsystem-findings.md` for parallel sessions
- Adding new memory features (temporal validity, bilingual expansion, precomputed essentials) — documented in same file
- Cloud model integration
- Changing embedding model or ChromaDB architecture

---

## Part 1: Unified Grading with Progressive Extraction

### Current State

Two graders:
- `src/core/router.py:grade_response()` — returns 1-5 score + JSON dict with `situation_summary`, `strategy_summary`, `tool_template`. Fragile with small LLMs.
- `src/core/grading.py:grade_task()` — returns YES/NO on RELEVANT/COMPLETE/VERDICT. Robust. Used by deferred grading (majority of tasks).

Deferred grades never produce `grader_data`. Skill extraction falls back to mechanical `f"Used {tools} in {iterations} iterations"`.

### Design

**Kill `router.py:grade_response()`** and the `GRADING_PROMPT` in router.py. One grading function: `grading.py:grade_task()`.

**New unified prompt:**

```
Evaluate this task result.

Task: {title}
Description: {description}
Result: {response}

RELEVANT: YES or NO
COMPLETE: YES or NO
VERDICT: PASS or FAIL
SITUATION: one line, what type of problem was solved
STRATEGY: one line, what approach worked
TOOLS: comma-separated list of tools used effectively
```

Binary fields (RELEVANT, COMPLETE, VERDICT) come first — these are what small LLMs handle reliably. Skill extraction fields (SITUATION, STRATEGY, TOOLS) come after — optional bonus.

**New `GradeResult` dataclass:**

```python
@dataclass
class GradeResult:
    passed: bool
    relevant: Optional[bool] = None
    complete: Optional[bool] = None
    situation: str = ""
    strategy: str = ""
    tools: list[str] = field(default_factory=list)
    raw: str = ""
```

No `score` field. No backward compatibility. All code checking `score >= 4.0` or `verdict.score` rewrites to `verdict.passed`.

**Parsing cascade** (most structured → least):

1. Parse all 6 fields via regex `KEY:\s*(.+)` per line
2. If SITUATION/STRATEGY/TOOLS fail → accept grade, skill extraction fields stay empty
3. If RELEVANT/COMPLETE fail → derive from VERDICT alone
4. If VERDICT line not found → scan for bare `PASS` or `FAIL` keyword anywhere in output
5. If nothing parseable → raise ValueError (grader incapable)

Step 4 is the critical robustness measure: small LLMs that ramble or break structure almost always emit "PASS" or "FAIL" somewhere.

**Skill extraction in `apply_grade_result()`:**

After PASS verdict:
- If `verdict.situation` non-empty → call `add_skill()` with extracted situation, strategy, tools
- If `verdict.situation` empty → mechanical fallback (tools + iterations, same as current)
- Skill extraction fires for ALL passed tasks (immediate and deferred) — fixing the current gap

**What gets removed:**
- `router.py:grade_response()` function
- `router.py:GRADING_PROMPT` constant
- All callers of `grade_response()` — rewire to `grading.py:grade_task()`
- `grader_data` dict passing through dispatcher/orchestrator
- Score-based thresholds everywhere (`>= 4.0`, `< 3.0`, etc.)

### Changes Required

| File | Change |
|------|--------|
| `src/core/grading.py` | New prompt, new GradeResult fields, new parsing cascade |
| `src/core/router.py` | Remove `grade_response()`, `GRADING_PROMPT` |
| `src/core/orchestrator.py` | Rewrite skill extraction to use `verdict.situation/strategy/tools` instead of `grader_data` dict. Remove score-based checks. |
| `src/core/llm_dispatcher.py` | Remove `grade_response` tuple handling if any |
| `src/memory/skills.py` | `add_skill()` caller passes `verdict.situation` as description, `verdict.strategy` as strategy_summary |
| `src/agents/base.py` | Remove any score references in result handling |
| `docs/skill-system.md` | Update to reflect unified grading |
| Tests | Update grading tests, add progressive parsing tests |

---

## Part 2: Context Gating

### Current State

`_build_context()` in `src/agents/base.py:347-577` has 13 layers. Almost all are unconditionally injected via try/except blocks. No total budget enforcement. See `docs/issues/context-layers-reference.md` for full layer inventory.

### Design

**New module: `src/memory/context_policy.py`**

Contains the policy map, heuristic overrides, and budget calculator. Keeps `_build_context()` clean — it calls the policy module to decide what to inject and how much.

**Context policy map** — keyed by task profile name (from `TASK_PROFILES` in capabilities.py):

```python
CONTEXT_POLICIES: dict[str, set[str]] = {
    "executor":         {"deps", "skills", "api"},
    "coder":            {"deps", "skills", "profile", "rag"},
    "implementer":      {"deps", "prior", "skills", "profile", "board"},
    "fixer":            {"deps", "skills", "rag", "profile"},
    "researcher":       {"skills", "rag", "api", "convo"},
    "shopping_advisor": {"skills", "convo"},
    "assistant":        {"convo", "rag", "memory"},
    "writer":           {"deps", "convo", "memory"},
    "planner":          {"deps", "board", "ambient", "memory"},
    "architect":        {"deps", "profile", "board", "rag"},
    "reviewer":         set(),
    "summarizer":       {"deps"},
    "analyst":          {"deps", "rag", "board"},
    "error_recovery":   {"deps", "rag", "skills"},
    "router":           set(),
    "visual_reviewer":  set(),
    "test_generator":   {"deps", "skills", "profile"},
}

DEFAULT_POLICY = {"deps", "skills", "rag"}
```

**Heuristic overrides:**

```python
def apply_heuristics(task: dict, policy: set[str]) -> set[str]:
    p = set(policy)

    # Workflow steps with tools_hint → ensure skills + api
    if task_context.get("tools_hint"):
        p.add("skills")
        p.add("api")

    # Tasks with dependencies always get deps
    if task.get("depends_on"):
        p.add("deps")

    # Follow-up tasks always get conversation
    if task_context.get("is_followup"):
        p.add("convo")

    # Mission tasks always get blackboard
    if task.get("mission_id"):
        p.add("board")

    return p
```

**Hard token budget:**

```python
def compute_layer_budgets(model_context: int, active_layers: set[str]) -> dict[str, int]:
    # Reserve 60% for system prompt + model reasoning
    available = int(model_context * 0.40)

    WEIGHTS = {
        "deps": 5, "prior": 4, "skills": 3, "rag": 3,
        "convo": 2, "board": 2, "profile": 1, "ambient": 1,
        "api": 1, "memory": 1, "prefs": 1,
    }

    active_weights = {k: WEIGHTS[k] for k in active_layers if k in WEIGHTS}
    total_weight = sum(active_weights.values()) or 1

    return {
        layer: int(available * w / total_weight)
        for layer, w in active_weights.items()
    }
```

Example on 8K model with `executor` policy (`deps`, `skills`, `api`):
- Available: 3200 tokens
- deps: ~1780, skills: ~1070, api: ~350

**Refactored `_build_context()`:**

```python
async def _build_context(self, task: dict) -> str:
    parts: list[str] = []

    # Always: task core + task context (ungatable)
    parts.append(format_task_core(task))
    parts.extend(format_task_context(task))

    # Determine active layers
    agent_type = task.get("agent_type") or self.name
    policy = get_context_policy(agent_type)
    policy = apply_heuristics(task, policy)

    # Compute budgets
    model_ctx = get_model_context_length(task)
    budgets = compute_layer_budgets(model_ctx, policy)

    # Inject only active layers, each respecting its budget
    if "deps" in policy:
        parts.extend(await fetch_dependencies(task, max_tokens=budgets["deps"]))
    if "prior" in policy:
        parts.extend(format_prior_steps(task, max_tokens=budgets["prior"]))
    if "convo" in policy:
        parts.extend(format_conversation(task, max_tokens=budgets["convo"]))
    if "ambient" in policy:
        parts.extend(await fetch_ambient(task, max_tokens=budgets["ambient"]))
    if "profile" in policy:
        parts.extend(await fetch_profile(task, max_tokens=budgets["profile"]))
    if "board" in policy:
        parts.extend(await fetch_blackboard(task, max_tokens=budgets["board"]))
    if "skills" in policy:
        parts.extend(await fetch_skills(task, max_tokens=budgets["skills"]))
    if "api" in policy:
        parts.extend(format_api_enrichment(task, max_tokens=budgets["api"]))
    if "rag" in policy:
        parts.extend(await fetch_rag(task, agent_type, max_tokens=budgets["rag"]))
    if "prefs" in policy:
        parts.extend(await fetch_preferences(max_tokens=budgets["prefs"]))
    if "memory" in policy:
        parts.extend(await fetch_memory(task, max_tokens=budgets["memory"]))

    return "\n\n".join(parts)
```

Each `fetch_*` / `format_*` function is a small helper that handles its own try/except, truncation to `max_tokens`, and returns `list[str]` (empty list if nothing to inject or layer fails).

### Changes Required

| File | Change |
|------|--------|
| `src/memory/context_policy.py` | **New file.** Policy map, heuristics, budget calculator. |
| `src/agents/base.py` | Refactor `_build_context()` to use context_policy. Extract layer logic into helper functions. |
| `src/memory/rag.py` | `retrieve_context()` accepts `max_tokens` parameter (already partially does), respects it strictly. |
| `src/memory/skills.py` | `find_relevant_skills()` + formatting respects `max_tokens` budget. |

---

## Part 3: Threshold Tuning

### Current Thresholds

| Threshold | Current | Location |
|-----------|---------|----------|
| RAG relevance minimum | 0.5 | `src/memory/rag.py` |
| Skill match minimum | 0.6 | `src/memory/skills.py` |
| RAG top_k per collection | 3-5 | `src/memory/rag.py` |
| RAG dedup (Jaccard) | 0.85 | `src/memory/rag.py` |
| Skill dedup (cosine) | 0.85 | `src/memory/skills.py` |

### New Thresholds

| Threshold | New | Rationale |
|-----------|-----|-----------|
| RAG relevance minimum | 0.72 | On multilingual-e5-base, 0.72 cosine similarity = clearly same topic. Below this is noise on small models. |
| Skill match minimum | 0.75 | Skills must be specifically relevant. "currency lookup" must not match "weather check". |
| RAG top_k per collection | 2 | With hard budgets, can't use 5 anyway. 2 high-quality results per collection. |
| RAG dedup | 0.85 | Keep — works fine. |
| Skill dedup | 0.85 | Keep — works fine. |

### RAG Collection Gating

Currently RAG queries all 5 core collections. New: each task type maps to relevant collections only.

```python
RAG_COLLECTIONS: dict[str, list[str]] = {
    "coder":            ["errors", "codebase"],
    "fixer":            ["errors", "codebase", "episodic"],
    "implementer":      ["errors", "codebase"],
    "researcher":       ["web_knowledge", "semantic"],
    "shopping_advisor": ["shopping"],
    "assistant":        ["semantic", "conversations"],
    "writer":           ["semantic"],
    "error_recovery":   ["errors", "episodic"],
    "test_generator":   ["errors", "codebase"],
    "planner":          ["episodic", "semantic"],
    "architect":        ["episodic", "semantic"],
    "analyst":          ["semantic", "web_knowledge"],
}
RAG_DEFAULT_COLLECTIONS = ["episodic", "semantic"]
```

### Changes Required

| File | Change |
|------|--------|
| `src/memory/rag.py` | Update thresholds, add collection gating map, reduce top_k. Consolidate all thresholds into a config dict at top of file. |
| `src/memory/skills.py` | Update match threshold. Consolidate thresholds into config dict at top of file. |

---

## Migration Plan

### Phase 1: Unified Grading
1. Add new fields to GradeResult, write new prompt and parsing cascade
2. Update `apply_grade_result()` to extract skills from verdict fields
3. Rewire all callers from `router.py:grade_response()` to `grading.py:grade_task()`
4. Remove `grade_response()`, old GRADING_PROMPT, score-based checks everywhere
5. Update docs and tests

### Phase 2: Context Gating
1. Create `src/memory/context_policy.py` with policy map, heuristics, budget calculator
2. Extract each context layer in `_build_context()` into a helper function with `max_tokens` param
3. Refactor `_build_context()` to use policy + budget system
4. Test with different task types — verify layers activate/deactivate correctly

### Phase 3: Threshold Tuning
1. Update RAG thresholds and add collection gating
2. Update skill match threshold
3. Consolidate all thresholds into config dicts at top of respective files

### Phase 4: Validation
1. Run full test suite
2. Test with actual tasks on local models — verify context fits in 8K
3. Check that skill extraction fires for deferred grades
4. Verify grading parsing cascade handles all edge cases

---

## Supporting Documents

- `docs/issues/memory-subsystem-findings.md` — 13 issues, parallel fix targets
- `docs/issues/memory-redesign-context.md` — all decisions, reasoning, constraints
- `docs/issues/context-layers-reference.md` — layer inventory with sources and token costs
- `docs/skill-system.md` — skill system design (capture, injection, tracking)
