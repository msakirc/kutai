# Agent Usage Audit — 2026-05-08

**Source:** `kutai.db` (`tasks` + `model_pick_log`, last 60 days).

## Headline

`tasks` table undercounts real activity because most LLM iterations log to `model_pick_log` not `tasks`. `model_pick_log` is the truth source.

## Tasks table (60d, status='completed' = done)

| agent | tasks | done% | notes |
|---|---|---|---|
| mechanical | 2669 | 1.00 | salako, non-LLM |
| reviewer | 116 | 0.85 | |
| overhead | 89 | 0.99 | dispatch category not agent |
| grader | 73 | 1.00 | |
| analyst | 57 | 0.44 | many active/in-flight |
| summarizer | 29 | 0.97 | |
| self_reflection | 29 | 1.00 | dispatch category |
| artifact_summarizer | 22 | 1.00 | |
| executor | 20 | 0.00 | mostly active in current dev session |
| writer | 14 | 0.50 | |
| architect | 12 | 0.00 | |
| researcher | 11 | 0.55 | |
| coder | 10 | 0.00 | |
| implementer | 6 | 0.00 | |
| fixer | 5 | 0.00 | |
| test_generator | 3 | 0.00 | |

**Zero task rows in 60d:** assistant, visual_reviewer, code_reviewer, planner (planner runs as overhead, not own task), shopping_*, product_researcher, deal_analyst.

## model_pick_log (60d) — REAL traffic

| agent | picks | rank |
|---|---|---|
| (null) | 17374 | n/a — untagged calls |
| **implementer** | **13509** | ⭐ workhorse |
| **test_generator** | **11529** | ⭐ workhorse |
| **executor** | **10138** | ⭐ workhorse |
| **planner** | **5114** | ⭐ heavy |
| reviewer | 2035 | |
| analyst | 231 | |
| coder | 216 | |
| fixer | 126 | |
| researcher | 112 | |
| self_reflection | 107 | dispatch cat |
| writer | 104 | |
| shopping_advisor | 47 | |
| summarizer | 35 | |
| deal_analyst | 18 | |
| shopping_pipeline_v2 | 10 | workflow id, not agent |
| shopping_clarifier | 2 | nearly dead |

**Zero picks:** assistant, visual_reviewer, code_reviewer, artifact_summarizer, product_researcher, grader (logged elsewhere), architect (logged as analyst maybe).

## i2p_v3 step distribution (workflow JSON)

| agent | steps |
|---|---|
| analyst | 70 ⭐ dominant |
| executor | 22 |
| reviewer | 21 |
| writer | 15 |
| implementer | 14 |
| researcher | 12 |
| architect | 12 |
| coder | 12 |
| mechanical | 6 |
| test_generator | 6 |
| fixer | 6 |
| summarizer | 3 |
| planner | 2 |

## Key findings

1. **`implementer` is THE code workhorse**, not `coder`. 13509 picks vs 216. i2p_v3 routes most code emit to implementer (single-file, structured-spec).
2. **`coder` exists for ad-hoc multi-file `/task`** flow + 12 i2p steps where full project build is needed.
3. **`test_generator` is heavy** (11529 picks, 6 i2p steps × many iterations).
4. **`executor` is heavy** (10138 picks) — used as fallback AND in 22 i2p steps.
5. **`analyst` dominates i2p** (70 steps) but moderate pick count — short tasks per step.
6. **Cluster overlap real:**
   - coder vs implementer vs fixer — distinct prompts (project build / single file / fix-from-feedback). Traffic patterns confirm distinct usage. Mergeable with mode flag, but the three roles are real.
   - reviewer vs code_reviewer — `code_reviewer` has 0 traffic. Pure dead alias.
   - summarizer vs artifact_summarizer — both used (29 + 22 tasks); artifact_summarizer 0 picks (probably runs as mechanical post-hook).
   - shopping cluster: advisor 47 / clarifier 2 / deal_analyst 18 / product_researcher 0. Advisor is the only live one.
7. **`product_researcher` referenced in 4 shopping workflow JSONs but 0 picks** — those workflows (exploration, gift_recommendation, price_watch) rarely or never execute. Confirms shopping_pivot decision (2026-04-20) deprioritized them.
8. **`assistant` 0 traffic** — referenced only in `telegram_bot.py`. Used for inline chat, possibly bypasses task pipeline.
9. **`visual_reviewer` 0 traffic** — referenced in router/classifier/vision code but no real picks.

## Recommendations (drop/merge/keep)

### KEEP separate (distinct roles + real traffic)
- analyst, executor, implementer, test_generator, reviewer, planner, researcher, architect, writer, summarizer, mechanical (salako)
- coder — keep for ad-hoc /task multi-file build, distinct from implementer
- shopping_advisor

### MERGE
- **fixer → coder** with `mode: fix` payload flag. Same toolset, similar iteration budget. 126 picks/60d justifies a mode but not a separate file.
- **deal_analyst → shopping_advisor** with `phase: analyze` flag. 18 picks doesn't warrant own agent.
- **artifact_summarizer → summarizer** — 22 tasks but 0 LLM picks suggests post-hook role; absorb prompt with `artifact_type` flag.

### DROP (alias for back-compat)
- **code_reviewer** — 0 traffic, fold any references into `reviewer`.
- **product_researcher** — 0 picks, only referenced in 3 shopping workflows that don't run. Alias to `shopping_advisor` for back-compat.
- **shopping_clarifier** — 2 picks. Alias to `shopping_advisor` with `phase: clarify`. Light use can be a mode.
- **visual_reviewer** — 0 traffic. If telegram vision flow needs it, keep; otherwise drop. Verify before deleting.
- **assistant** — 0 task-pipeline traffic but live in telegram inline-chat. KEEP if telegram routes through it; investigate.

### Investigate before action
- `visual_reviewer` — check `tools/vision.py` and telegram image handlers.
- `assistant` — check telegram_bot.py inline chat routing.
- `product_researcher` workflows — confirm they're truly dead before aliasing.

## Target

21 agents → **~15 active agents** (4 dropped to aliases, 2 absorbed as modes/phases).
- Drop: code_reviewer, product_researcher, shopping_clarifier (alias) — possibly visual_reviewer + assistant after verify.
- Merge as modes: fixer → coder; deal_analyst → shopping_advisor; artifact_summarizer → summarizer.
- Keep: 15 agents with distinct roles + real traffic.

## Schema gaps noted

- `tasks` has no `classifier_picked_agent` column — can't measure classifier-vs-actual divergence directly. Add column in a future migration if classifier hardening (Plan Phase 4) needs A/B.
- `model_pick_log.agent_type` is null for 17k entries (~38%) — overhead/system calls bypass agent tagging. Worth wiring before consolidation A/B.
