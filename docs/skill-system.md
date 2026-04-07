# Skill System

The skill system learns what execution strategies work and injects them into future similar tasks. It answers one question: "we've seen a task like this before — here's what worked."

## Why It Exists

KutAI has three systems that decide **where** a task goes: the classifier picks the agent, the fast resolver handles API-answerable queries, and the router picks the model. None of them capture **how** the agent should approach the work once it starts.

When a shopping comparison task succeeds after searching three stores separately and building a comparison table, that strategy is lost. The next similar task starts from scratch. The skill system captures that strategy and feeds it back as context.

## What a Skill Is

A skill is a named execution recipe with:
- A **description** of the situation it applies to (used for vector matching)
- One or more **strategies** — ranked approaches that worked, each with tool sequences, iteration counts, and success tracking

Skills are cross-cutting. An "app store search" skill can help both a shopping_advisor comparing apps and a researcher doing competitor analysis. Skills have no agent_type — the description and vector similarity determine relevance.

There are two types:
- **Seed skills** (24): Hand-crafted, cover common Turkish use cases (weather, currency, shopping, pharmacy, etc.). Created at startup.
- **Auto-captured skills**: Learned from tasks that pass unified grading. The grader generates the description and strategy summary.

## How It Works

### Capture (after task completion)

```
Task completes → Unified grader evaluates (PASS/FAIL)
  │
  ├─ FAIL → nothing captured
  │
  └─ PASS → grader also outputs (when model can produce them):
       SITUATION: "Comparing laptop prices across Turkish stores"
       STRATEGY: "Search each store separately then compare"
       TOOLS: smart_search, web_search
             │
             ▼
       Check ChromaDB: does a similar skill exist? (cosine similarity >= 0.85)
             │
             ├─ Yes → add this strategy to existing skill's strategy list
             │         (prune to max 5, drop worst-performing proven strategies first)
             │
             └─ No  → create new skill, embed description in ChromaDB
```

**Graceful degradation**: If SITUATION/STRATEGY/TOOLS are empty (small LLM couldn't produce them), falls back to task metadata (title + agent_type + tools used). Parsing is progressive — binary PASS/FAIL always works, skill fields are bonus.

**Unified grading**: Both immediate and deferred grading use the same prompt and parser. The deferred grading gap (where most tasks got only mechanical skill entries) is fixed.

### Injection (before task execution)

```
Task about to execute → base.py._build_context()
  │
  ▼
Embed task text (title + description) → vector search ChromaDB
  │
  ▼
Top 5 candidates with similarity >= 0.75
  │
  ▼
Rank by: similarity * 0.5 + injection_success_rate * 0.5
  │
  ▼
Adaptive injection depth:
  │
  ├─ Top skill highly trusted (>= 80% success, >= 5 injections)?
  │     → inject just that one, full verbose format
  │
  └─ Uncertain?
        → inject up to 3 in compact format, scaled by context budget:
            < 2048 tokens → 1 skill
            < 4096 tokens → 2 skills
            >= 4096 tokens → 3 skills
```

**Tool injection**: If a high-confidence skill (>= 70% success, >= 5 injections) recommends a tool the agent doesn't have in its `allowed_tools`, that tool is temporarily added for this execution. This lets proven strategies override static tool restrictions.

### Tracking (feedback loop)

At injection time: `injection_count += 1` on each injected skill.

After task completes with a PASS grade: `injection_success += 1` on all skills that were injected into that task.

This creates a feedback loop. The initial grading score is just a rough gate (noisy, from small LLMs). The **injection success rate** — "when this skill was injected, did the task succeed?" — is the real ranking signal. It self-corrects over time: skills that actually help rise, skills that don't contribute fall behind.

## Design Decisions

These decisions were made through iterative discussion. The alternatives are documented in `docs/superpowers/specs/2026-04-03-skill-system-overhaul-design.md` with full reasoning.

### Skills are execution recipes, not routing hints

The classifier, fast resolver, and router already handle routing. Adding a fourth routing system creates conflicts. Skills capture **strategy** — the thing no other system captures.

### Vector matching, not regex or tags

The old system split task titles into words to create regex patterns. "coffee machine" became `coffee|machine` — matching nothing useful. Tag-based matching was considered but requires reliable tag extraction at query time, which small LLMs can't do consistently.

Vector matching on rich descriptions works because: the grader writes descriptions at capture time (after seeing the full execution), and the task text at query time (title + description) is usually 10-30 words — enough for meaningful embeddings with multilingual-e5-base.

### Cross-cutting skills with no agent_type filter

An "app store search" skill should serve both a shopping task and an i2p competitor research step. Storing agent_type on skills and filtering by it would prevent this. Instead, the task text naturally includes agent context in the embedding — "shopping_advisor: find laptop alternatives" will match differently than "researcher: competitor analysis for mobile apps" even against the same skill, because the embedding captures the full context.

### Binary grading, not numeric scores

Small LLMs grading their own work produce noisy scores. Binary PASS/FAIL is far more reliable than a 1-5 scale. Within passed tasks, ranking uses injection success rate — real-world validation from deployment.

### Accumulate strategies, never overwrite

Each skill stores up to 5 strategies. New successful executions are added, not replacing existing ones. Strategies earn ranking through injection success, not through the original grade. A noisy high grade can't overwrite a genuinely good strategy — it just gets added as a candidate and has to prove itself.

### Adaptive injection depth

One highly trusted skill gets full verbose context (situation, strategy, numbered steps, track record). Uncertain situations get multiple compact one-line hints. Small context models get fewer skills. This prevents wasting context on unproven advice while giving proven strategies full attention.

## Files

| File | What it does |
|------|-------------|
| `src/memory/skills.py` | Core module. Vector search, strategy accumulation, dedup, adaptive injection, tracking. All skill logic lives here. |
| `src/memory/seed_skills.py` | 24 seed skill definitions + `seed_skills()` function. Called at startup. |
| `src/infra/db.py` | `skills` table schema + 5 helper functions: `upsert_skill`, `get_all_skills`, `get_skill_by_name`, `increment_skill_injection`, `increment_skill_success`. |
| `src/agents/base.py:479-525` | Injection point in `_build_context()`. Calls `find_relevant_skills`, formats output, injects tools, tracks injections, stores skill names in task context. |
| `src/core/orchestrator.py:~2172-2230` | Capture point after task completion. Reads `grader_data`, calls `add_skill`. Also calls `record_injection_success` for tasks that had skills injected. |
| `src/core/grading.py` | Unified grading module — single prompt and parser used by both immediate and deferred grading paths. Progressive skill field extraction alongside binary PASS/FAIL. |
| `src/core/llm_dispatcher.py` | Passes `grader_data` tuple through the grading pipeline. `grade_response` returns `(bool, dict)`. |
| `src/memory/context_policy.py` | Context gating — decides which layers to inject per task type, with token budgets. |
| `src/memory/vector_store.py` | ChromaDB wrapper. Skills use the `semantic` collection with `{"type": "skill"}` metadata. |

## DB Schema

```sql
CREATE TABLE skills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,          -- "currency_lookup" or "auto:shopping_advisor:laptop comparison"
    description TEXT NOT NULL,           -- situation description, used for vector matching
    skill_type TEXT DEFAULT 'auto',      -- 'seed' or 'auto'
    strategies TEXT DEFAULT '[]',        -- JSON array of strategy objects
    injection_count INTEGER DEFAULT 0,   -- how many times injected into tasks
    injection_success INTEGER DEFAULT 0, -- how many of those tasks succeeded
    created_at TEXT,
    updated_at TEXT
);
```

### Strategy object (inside strategies JSON)

```json
{
    "summary": "Search each Turkish e-commerce site separately, compare prices",
    "tool_template": ["smart_search({product} trendyol)", "smart_search({product} hepsiburada)", "compare"],
    "tools_used": ["smart_search", "web_search"],
    "avg_iterations": 4,
    "source_grade": "great",
    "source_task_id": 1234,
    "injection_count": 8,
    "injection_success": 6,
    "created_at": 1712345678.0
}
```

## Thresholds

| Constant | Value | What it controls |
|----------|-------|-----------------|
| `DEDUP_SIMILARITY_THRESHOLD` | 0.85 | Minimum similarity to merge into existing skill instead of creating new |
| `MATCH_SIMILARITY_THRESHOLD` | 0.75 | Minimum similarity to consider a skill relevant for injection |
| `HIGH_CONFIDENCE_THRESHOLD` | 0.8 | Injection success rate needed for "trusted" status (verbose injection) |
| `MIN_INJECTIONS_FOR_CONFIDENCE` | 5 | Minimum injections before trusting the success rate |
| `MAX_STRATEGIES_PER_SKILL` | 5 | Cap on strategies per skill |
| `TOOL_INJECTION_THRESHOLD` | 0.7 | Success rate needed before a skill can inject tools into allowed_tools |
| `TOOL_INJECTION_MIN_COUNT` | 5 | Minimum injections before tool injection is allowed |

These are starting values. Tune based on observed behavior — especially `MATCH_SIMILARITY_THRESHOLD` (too low = irrelevant skills injected, too high = skills rarely match).

## What Agents Need to Know

**If you're modifying the skill system:**
- All skill logic is in `skills.py`. DB helpers are in `db.py`. Don't put business logic in db.py.
- Vector operations use ChromaDB's `semantic` collection with `{"type": "skill"}` metadata. The `doc_id` format is `skill:{name}`.
- `_find_duplicate_skill` and `_vector_search_skills` convert ChromaDB's L2 distance to similarity as `max(0, 1.0 - distance)`. This is approximate but consistent across the codebase.
- The `upsert_skill` DB helper accepts `strategies` as either a list (auto-serialized) or a JSON string. Check the current implementation if unsure.

**If you're modifying the grading system:**
- `grade_response` in `grading.py` returns `tuple[bool, dict]`. The bool is PASS/FAIL; the dict contains progressive skill fields. All callers must unpack the tuple.
- The unified grading prompt always asks for SITUATION/STRATEGY/TOOLS. Skill fields are parsed opportunistically — absence is gracefully handled.
- Both immediate and deferred grading (via GradeQueue) use the same prompt and parser. There is no longer a grader-data loss path for deferred tasks.

**If you're modifying agent execution:**
- Skill injection happens in `base.py:_build_context()`. It's wrapped in try/except — failures are non-critical.
- `injected_skills` list is stored in `task["context"]` JSON for tracking after completion.
- Tool injection modifies `self.allowed_tools` in-place. The agent's `finally` block in `execute()` restores `_original_allowed_tools` if workflow `tools_hint` was used, but skill-injected tools are NOT restored — they persist for the execution.

**If you're adding new seed skills:**
- Add to `SEED_SKILLS` list in `seed_skills.py`. Required fields: `name`, `description`, `strategy_summary`, `tools_used`.
- Descriptions should be rich and contain keywords users would actually type (including Turkish terms). They're used for vector matching.
- `seed_skills()` is idempotent — it skips skills that already exist by name.

## Known Limitations

1. **Embedding model dependency**: Vector matching requires ChromaDB + multilingual-e5-base. If ChromaDB is unavailable, all skill matching silently returns empty. Skills are still captured to SQLite (they just won't match until ChromaDB is back).

2. **Cold start**: New skills have `injection_count = 0`, so their success rate is capped at 0.5 (neutral). They need 5+ injections before the real rate is trusted. Until then, they compete on vector similarity alone.

3. **No skill deletion via Telegram**: Skills can only be managed through direct DB access or code. A `/skillstats` command exists in the old system but needs updating for the new schema.

## Future Work

- **Evaluate seed skill relevance**: Many of the 24 seed skills may be redundant with the fast resolver (which handles weather, currency, etc. directly). After the system runs for a while, check which seed skills actually get injected and remove dead weight.
- **`/skillstats` Telegram command**: Show injection counts, top performers, strategy counts. The data is all there — just needs a Telegram handler.
- **Skill decay**: Skills that haven't been injected in 30+ days could have their confidence lowered, preventing stale strategies from crowding out new ones.
