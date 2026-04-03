# Skill System Overhaul — Design Spec

**Date**: 2026-04-03
**Status**: Approved design
**Depends on**: Smart Resource Integration (fully implemented)
**Related**: `2026-04-03-skill-system-overhaul-findings.md` (problem statement)

## Problem

163 skills in DB. 24 seed skills (well-crafted but blocked by bugs, now fixed). 139 auto-captured skills — 132 are i2p step-name garbage with broken regex, 7 are overly specific. Auto-capture splits task titles into words, producing patterns like `coffee|machine` that don't generalize.

Even with the bug fixes from smart-resource-integration (success_count filter removed, re.escape applied, logging upgraded), the fundamental auto-capture logic produces low-value skills because it memorizes task instances, not reusable strategies.

Meanwhile, routing is already handled by three other systems:
- **Classifier** — picks agent_type (shopping_advisor, researcher, etc.)
- **Fast resolver** — resolves API-answerable queries without an agent
- **Router** — picks which LLM model to use

Skills should not duplicate routing. They should capture **execution strategy** — how to approach a type of problem once routing has already decided where it goes.

## Design Decisions (with alternatives considered)

### Decision 1: Skills are execution recipes, not routing hints

**Chosen**: Skills capture strategy — tool sequences, iteration patterns, approach descriptions — that help agents execute better.

**Alternatives rejected**:
- **A) Routing hints** ("this query → this agent + these tools"): Redundant. Classifier, fast resolver, and router already handle routing. Adding a fourth routing system creates conflicts and confusion.
- **C) Category mappings** ("shopping queries → shopping_advisor"): Also redundant with classifier. The classifier already maps intent to agent_type.

**Why B (execution recipes)**: No other system captures what actually worked during execution. After a successful GPU comparison task, the system knows the agent used smart_search 3 times for different stores then built a comparison table. That strategy is genuinely new information that helps future similar tasks.

### Decision 2: Grader generates skill metadata (no extra LLM call)

**Chosen**: Expand the existing grading prompt to also output `situation_summary`, `strategy_summary`, and `tool_template` fields. Same LLM call, two extra fields in JSON response.

**Alternative rejected**:
- Separate LLM call for skill extraction after grading. Wastes an inference cycle when the grader already has full conversation context.

**Why**: The grader sees the entire execution — task, tool calls, results, final answer. It's the best positioned component to summarize what happened and why it worked.

### Decision 3: Vector matching with task context, no tag system

**Chosen**: At query time, embed the full task text (title + description, which naturally contains agent_type, domain keywords, etc.) and vector-search against skill descriptions in ChromaDB. No pre-filter by agent_type field on skills. Skills are cross-cutting — a "app store search" skill can serve both shopping_advisor and researcher agents.

**Alternatives considered and rejected**:

1. **Tag extraction by grader + tag matching at query time**: Grader extracts tags like `["app-search", "comparison"]` at capture time. At query time, extract tags from incoming task. Problem: query-side tag extraction requires either another LLM call or classifier expansion. Both add complexity and depend on LLM reliability for a critical matching step.

2. **Tag extraction by grader + vector fallback**: Tags on skill side only, embedded into description text. Avoids query-side extraction but tags become just part of the vector representation anyway — no real benefit over a well-written description.

3. **Regex matching for auto-skills** (current system): Title-word-splitting produces patterns that don't match real queries. Fundamentally broken for auto-captured skills.

4. **Agent_type pre-filter + vector**: Store agent_type on skills, filter by it before vector search. Problem: skills should be cross-cutting. A shopper and researcher can both use "app store search." Pre-filtering by agent_type prevents this.

5. **Pure vector matching with no structured context**: Works well for i2p tasks (long descriptive text = strong embeddings) but poorly for short shopping queries ("coffee machine" = weak embedding). However, the task text always includes surrounding context (title + description) which is usually enough for meaningful embeddings, even for short queries.

**Why vector-only with rich descriptions**: Simplest approach that works for both long (i2p) and short (shopping) task texts. No extraction logic, no tag maintenance, no agent_type coupling. The quality depends on skill descriptions — grader-written descriptions that capture situation + strategy naturally embed well. The risk (weak matches for very short queries) is acceptable because those queries come through structured paths anyway and carry enough context in title + description.

### Decision 4: Coarse quality buckets, not precise grades

**Chosen**: Three buckets instead of ranking by precise grade scores:
- **Great** (grade >= 4.0): Capture as strategy
- **OK** (3.0–3.9): Ignore, not worth learning from
- **Bad** (< 3.0): Don't capture

Within the "great" bucket, rank by **injection success rate** — real-world validation — not by the original grade number.

**Alternative rejected**:
- Rank strategies by precise grade (4.1 > 3.8). Small LLMs grading their own work produce noisy scores. A 3.8 vs 4.1 distinction is meaningless. Real-world injection success (did future tasks also succeed when this skill was injected?) is a stronger signal that self-corrects over time.

### Decision 5: Adaptive injection depth by confidence and context budget

**Chosen**: Three-tier injection strategy:
1. If top skill is highly trusted (>= 0.8 injection success rate, 5+ injections): inject only that one, full verbose format
2. If uncertain and context budget allows: inject up to 3 skills in compact format
3. Scale by model context size — large context models get more skills, small models get fewer

**Alternatives rejected**:
- Always inject 3 skills: Strains context for small LLMs (300-500 tokens for 3 verbose blocks). Unnecessary when one highly trusted skill exists.
- Always inject 1 skill: Misses complementary strategies in uncertain situations.

### Decision 6: Accumulate strategies, never overwrite

**Chosen**: Each skill stores a ranked list of strategies (up to 5). New successful executions are added to the list. Ranking is by injection success rate. A new strategy cannot displace an existing one until it earns its position through real-world use.

**Alternative rejected**:
- Single strategy per skill, overwritten by better grades. Risk: a noisy high grade from a mediocre execution overwrites a genuinely good strategy. Accumulate-and-rank prevents this.

**Tie-breaking between equally successful strategies**: Not a real problem in practice. A single technique rarely accumulates multiple genuinely different high-performing strategies. If it happens, both are valid — inject the one with more data (higher confidence from more injections).

### Decision 7: Skill deduplication by vector similarity, not name equality

**Chosen**: When capturing a new skill, embed the situation_summary and check ChromaDB for existing skills with similarity >= 0.85. If match found, add strategy to existing skill instead of creating a new one.

**Why**: Grader may name the same technique differently ("app-store-comparison" vs "play-store-search"). Vector similarity catches semantic equivalence regardless of naming.

## Data Model

### Skills table (modified)

```sql
-- Replace current skills table
CREATE TABLE IF NOT EXISTS skills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    description TEXT NOT NULL,          -- grader-written situation description (used for vector matching)
    skill_type TEXT DEFAULT 'auto',     -- 'seed' or 'auto'
    strategies TEXT DEFAULT '[]',       -- JSON array of strategy objects (ranked)
    injection_count INTEGER DEFAULT 0,
    injection_success INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime')),
    updated_at TEXT DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime'))
);
```

Dropped columns: `trigger_pattern` (regex matching removed entirely), `tool_sequence` (replaced by strategies), `success_count`/`failure_count` (replaced by injection tracking), `examples` (grader description replaces this).

Seed skills are migrated to the new schema with `skill_type = 'seed'` and their tool_sequence converted to a single strategy entry. No regex preserved.

### Strategy object (JSON in strategies column)

```json
{
    "summary": "Search each major Turkish e-commerce site separately, extract price/seller/shipping, build comparison table",
    "tool_template": [
        "smart_search({product} site:trendyol.com)",
        "smart_search({product} site:hepsiburada.com)",
        "compare: price, seller, rating, shipping"
    ],
    "tools_used": ["smart_search", "web_search"],
    "avg_iterations": 4,
    "source_grade": "great",
    "injection_count": 0,
    "injection_success": 0,
    "created_from_task": 1234,
    "created_at": "2026-04-03 14:30:00"
}
```

### ChromaDB collection

Collection `skills` stores embeddings of skill descriptions with metadata `{"type": "skill", "skill_name": "..."}`. Used for:
1. Query-time matching (find relevant skills for a task)
2. Capture-time dedup (find existing skill with similar description, threshold 0.85)

## Capture Flow

### Trigger

Task completes AND grader assigns grade >= 4.0 ("great" bucket).

### Grading prompt expansion

The existing grading prompt is expanded to include three additional output fields:

```json
{
    "grade": 4.2,
    "feedback": "...",
    "situation_summary": "Comparing mobile app alternatives across stores by features and ratings",
    "strategy_summary": "Search each store separately with smart_search, extract structured data per app, build comparison table with ratings/reviews/price",
    "tool_template": [
        "smart_search({category} play store)",
        "smart_search({category} app alternatives)",
        "structure comparison: ratings, reviews, size, price"
    ]
}
```

### Graceful degradation

Small LLMs may botch the JSON formatting. Degradation path:

1. **Full parse succeeds**: All three fields present → full skill capture
2. **Partial parse**: Some fields missing or malformed → capture what we can. At minimum, we always have task metadata (agent_type, tools actually used, iteration count) — enough for a basic strategy entry with `summary` auto-generated from task title + tools
3. **Grader response completely unparseable**: Skip skill capture for this task. No harm, we miss one data point.

Never crash, never store garbage.

### Dedup check

1. Embed the `situation_summary` (or auto-generated summary if degraded)
2. Query ChromaDB for existing skills with cosine similarity >= 0.85
3. If match found → add strategy to existing skill
4. If no match → create new skill

### Strategy ranking

New strategies start with `injection_count: 0, injection_success: 0`. They earn ranking position through real-world use. Within a skill, strategies are ordered by injection success rate (with minimum 5 injections before rate is considered meaningful — below that, newest first).

Maximum 5 strategies per skill. When a 6th is added, the lowest-performing strategy (worst injection success rate, minimum 5 injections) is dropped. Strategies with fewer than 5 injections are never dropped (still proving themselves).

### Multiple capture

One task execution = one skill/strategy entry. Even if a complex task used multiple techniques, it's captured as one combined strategy. Splitting requires understanding which execution segments map to which technique — too hard for the grader to do reliably.

## Injection Flow

### Query

1. Embed task text (title + description) — naturally contains agent_type, domain keywords, context
2. Vector search ChromaDB `skills` collection, top 5 candidates
3. Filter by minimum similarity threshold (0.6)
4. Rank by: `vector_similarity * 0.5 + injection_success_rate * 0.5` (injection_success_rate = injection_success / max(injection_count, 1), capped at 0.5 for skills with < 5 injections)

### Adaptive injection depth

```
top_skill = ranked_results[0]
top_confidence = top_skill.injection_success_rate  (0.5 if < 5 injections)

if top_confidence >= 0.8 and top_skill.injection_count >= 5:
    # Highly trusted — just this one, full format
    inject top_skill best_strategy (verbose)

else:
    # Uncertain — multiple hints, compact format
    context_budget = estimate_available_tokens(model)
    max_skills = 1 if context_budget < 2048 else (2 if context_budget < 4096 else 3)
    inject top max_skills skills (compact format)
```

### Prompt format — verbose (single trusted skill)

```
## Proven Strategy

### {skill_name}
**Situation**: {description}
**Strategy**: {best_strategy.summary}
**Steps**:
  1. {tool_template[0]}
  2. {tool_template[1]}
  3. {tool_template[2]}
**Track record**: {injection_success}/{injection_count} successful uses
**Tools**: {tools_used joined}
```

### Prompt format — compact (multiple uncertain skills)

```
## Strategy Hints
- {skill1_name}: {best_strategy.summary} (tools: {tools}, {success_rate}% success)
- {skill2_name}: {best_strategy.summary} (tools: {tools}, {success_rate}% success)
```

### Tool injection

If any injected skill has injection success rate >= 0.7 AND injection_count >= 5, and its strategy recommends a tool the agent doesn't have in `allowed_tools`:
- Temporarily add the tool to `allowed_tools` for this execution only
- Only applies to high-confidence skills — new/unproven skills only hint

### Tracking

At injection time: increment `injection_count` on skill AND on the specific strategy used.

After task completes: if task grades "great" (>= 4.0), increment `injection_success` on all injected skills AND their used strategies.

This creates a feedback loop: skills that actually help tasks succeed rise in ranking. Skills that get injected but don't help eventually fall behind.

## Cleanup

### Wipe garbage auto-captured skills

1. Delete all 132 i2p step-name skills (broken regex, never match, `name LIKE 'auto:%'` with i2p step patterns)
2. Evaluate the 7 non-i2p auto-captured skills individually:
   - Check if tool_sequence metadata contains genuinely useful strategy info
   - Keep any that do, convert to new schema
   - Delete the rest
3. Seed skills (24): Migrate to new schema with `skill_type = 'seed'`, convert to vector-only matching like auto-skills. No backward compatibility with old regex patterns. **Full evaluation of seed skill relevance deferred to after overhaul is live** — many may be redundant with fast_resolver.

### Migration

1. Back up current skills table
2. Create new schema
3. Migrate seed skills: convert tool_sequence to strategy format, set `skill_type = 'seed'`
4. Migrate any surviving non-i2p auto-skills similarly
5. Drop everything else
6. Rebuild ChromaDB `skills` collection from surviving skills

## Files Changed

| File | Change |
|------|--------|
| `src/memory/skills.py` | New schema, rewrite `find_relevant_skills()` (vector-only with adaptive injection), rewrite `add_skill()` (strategy accumulation, dedup), new `format_skill_verbose()` and `format_skill_compact()`, injection tracking |
| `src/memory/seed_skills.py` | Update seed data to new format (strategies list instead of tool_sequence) |
| `src/core/orchestrator.py` | Rewrite auto-capture (~L2138-2190): use grader output instead of title splitting. Update grading prompt to request situation/strategy/template fields. Injection tracking on task completion. |
| `src/agents/base.py` | Rewrite skill injection in `_build_context()` (~L479-496): adaptive depth, verbose/compact formatting, tool injection for high-confidence skills. Context budget estimation. |
| `src/infra/db.py` | New skills schema, migration logic, strategy CRUD queries |

## Out of Scope

- Seed skill relevance evaluation (deferred to after overhaul is live)
- `/skillstats` Telegram command (can be added separately)
- Changing how skills interact with i2p workflow steps (skills are injected at agent level, workflow is above that)
- Skill sharing across KutAI instances (single instance only)
- Manual skill creation/editing via Telegram (auto-capture only for now)

## Observability

For debugging and tuning after launch:
- Log every skill injection at INFO level: which skills matched, similarity scores, which format used
- Log every skill capture at INFO level: new skill vs strategy added to existing
- Log dedup matches at DEBUG level: what similarity score triggered a merge
- Existing `/skillstats` command (from skills.py `list_skills()`) should be updated to show new fields (injection counts, strategy counts, top performers)
