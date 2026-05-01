# Handoff — Session 2026-04-29

User: sakircimen@gmail.com
Caveman mode: drop articles/filler/pleasantries. Fragments OK. Code/commits/security normal.
Approach: surface options before destructive change. No unilateral architecture decisions.

---

## What shipped this session (chronological)

```
007dd09  fix(i2p_v3 4.8): align third_party_service_selection + audit script comma-list trigger
824434b  fix(i2p_v3 3.4): align data_requirements (8 per-entity nested + 4 aggregates)
d36a24a  fix(schema): legacy auto-normalize → presence-only (was "string" default) + 6.1 migrated
94bb2fa  fix(base): persist step-refresh to DB so post-execute hook validates against live schema
```

Plus user-side parallel work (cherry-pick): `5ee2cba feat(nerd_herd): rewrite pressure_for as 10-signal orchestrator` + 8 prior nerd_herd commits.

---

## Critical incident: stale `.pyc` survived source deletion

The nerd_herd refactor (5ee2cba and the 8 commits before it) deleted
`packages/fatih_hoca/src/fatih_hoca/scarcity.py`. Python's `__pycache__/scarcity.cpython-310.pyc`
remained. Per PEP 488, Python loads source-less .pyc → stale bytecode kept the
`pool_scarcity` reference alive → every `select()` call raised
`NameError("name 'pool_scarcity' is not defined")` → every LLM admission
denied → only mechanical tasks could run.

**Resolution**: nuked all `__pycache__/` and `*.pyc` under `packages/` and
`src/`. User needs to restart KutAI to recompile fresh bytecode.

**Lesson**: when refactor deletes source files, also nuke `__pycache__/`.
Add to handoff checklist: after `git pull` of significant refactor,
`find . -type d -name __pycache__ -exec rm -rf {} +`.

**Defensive option (deferred)**: write a pre-commit / deploy hook that
deletes orphan .pyc files (those without matching .py source). Otherwise this
hits again.

---

## Architecture decisions made

### Schema dialect E1 (canonical, single source of truth)
Locked yesterday (0856cd5). Today extended with:
* `optional: true` flag for fields that may legitimately be empty arrays/objects.
* Auto-normalize for legacy schemas: untyped fields → `{}` rule = presence-only check (was `{type: "string"}` — caused "expected string, got list" rejections for actually-array fields like `user_stories`, `dependencies`).
* Validator short-circuits to `None` when rule has no `type`.

### Step-refresh persists to DB
Was in-memory only; post-hook (`workflow_engine.advance`) re-fetches the task and validated against pre-refresh stored context. Now writes back via `update_task` whenever fields change. Side-effect: sweep, /queue, DLQ inspection, telemetry all see the same schema the agent ran against.

### Migrated 6.1 to canonical E1 dialect
`epics` array of objects with mixed-type sub-fields (string/number/array). `dependencies` marked `optional: true` (empty `[]` no longer rejected). `work_breakdown.tasks` deeply nested.

---

## Deferred — high priority (verify on next mission tick)

### (P1) 6.1 verification post-restart
After Python cache nuke + restart:
* `/dlq retry 4450` (or check if already pending — last seen status=pending, attempts=0).
* Watch logs for `[Task #4450] step-refresh: ... re-synced` AND DB `tasks.context` actually showing canonical form (not legacy).
* If schema validation finally passes, 6.1 clears. Other 79 pending M57 tasks may bite next.

### (P1) Prompt simplification (carried from previous handoff)
Still open. See `docs/handoff/2026-04-28-prompt-simplification.md`. 90k input observed. Hot section: `## Results from Previous Steps` (5-30k). Investigation steps in that handoff.

### (P2) Grader vs schema unification
User's stated next architectural priority after E1. Make grader prompt receive the schema as authoritative ground truth; instruction text becomes quality guidance only. Eliminates "schema passes, grader rejects" drift. ~50 lines + grader prompt rewrite. Not started.

---

## Deferred — medium priority

### Audit script gaps
`scripts/audit_schema_instruction.py` upgraded today: comma-list after triggers, recursive `fields` walking. Still has gaps:
* False positives for example-value tokens (`emerging`, `mature` as `maturity_level` enum values; `transaction`, `rollback`, `headers`, `file` as discussed concepts).
* Aggregate fields requested by instruction but absent from schema (4.8 pattern: "calculate total monthly cost"). Audit doesn't detect these — manual review needed.
* Deeply-nested fields below `items.fields.items.fields` may not surface in flagged candidates.

### Other audit-flagged steps (not yet fixed)
Today's audit re-run flagged 13. Reviewed each:
* **0.5** human_clarification_request — clarify-action gated, validator skipped (commit 6aaedaf). Skip.
* **1.4** indirect_competitor_identification — flagged `collect` (false positive: trigger word itself).
* **1.10** technology_trend_research — flagged `emerging/mature/experimental` (false positive: enum values).
* **2.6, 4.1** — random words.
* **5.11a** — markdown content within sections, not headers.
* **9.2, 9.3, 10.2** — discussed concepts not field names.
* **2.2, 4.5b** — already migrated nested form; audit's recursive walker may still surface them.

If any of these bite during M57, fix surgically.

---

## Deferred — explicitly deprioritized

(Per user, not interesting yet)

* **DLQ retry preserves `failed_models`** (`dead_letter.py:308-309` wipes history). User said: "Post dlq related suggestions are currently not important."
* **Auto-difficulty bump on DLQ retry** — same.
* **Groq compound tool-calling rejection** — proposed adapter capability flag, user reverted. Compound family will keep crashing tool-using agents until upstream fix or different mechanism.
* **Force-kill llama-server on every swap** — Windows console signal limitation, accepted.
* **Registry path warning `packages/src/models/models.yaml`** — user said skip.
* **m46 orphan empty artifacts** — needs cancel-or-resume decision.
* **aiohttp Unclosed session warnings** — litellm internal, cosmetic.
* **HF_TOKEN missing, GigaChat3.1 unmatched, Docker unavailable** — cosmetic startup warnings.

---

## Architecture facts (don't relearn)

* **DB_PATH**: `C:\Users\sakir\ai\kutai\kutai.db` per `.env`. `data/kutai.db` is orphan.
* **Workflow JSON** has mtime cache (`loader.py`). Edits propagate without restart.
* **Python code** changes need restart. `.pyc` survives source deletion (PEP 488 source-less loading).
* **Task context** is stored in DB `tasks.context` (JSON string). `base.py` step-refresh now writes back on change. Post-hook `workflow_engine.advance` calls `get_task(id)` so reads from DB.
* **E1 dialect canonical form**: `{type, fields, items, required_sections, min_items, max_items, min_keys, min_length, optional}`. Recursive. See `src/workflows/engine/schema_dialect.py` for helpers (validate_value, translate_rule, make_example, render_checklist, iter_required_paths, check_presence_top_level).
* **`_normalize_rule`** auto-converts legacy form to canonical at every helper entry. Legacy fields default to `{}` (presence-only), NOT `{type: "string"}`.
* **`is_empty_required_value`** rejects: None, "", whitespace, "...", {}, []. Skipped for `optional: true` fields.
* **`prompt_versions` DB table** is runtime source of truth for agent prompts (Phase 13.1). Hardcoded `get_system_prompt` strings are FROZEN REFERENCES — runtime ignores them. Edit via `save_prompt_version(activate=True)`.

---

## Stuff to NEVER do

* NEVER `taskkill llama-server`. NEVER force-kill Yaşar Usta.
* NEVER `pytest` without timeout: `timeout 30 pytest <targeted>`.
* NEVER call `call_model()` directly — use `LLMDispatcher.request()`.
* NEVER commit `.worktrees/` (the pool-pressure worktree is from a parallel branch).
* NEVER edit prompt strings in `src/agents/*.py` expecting them to take effect — use `save_prompt_version`.
* DON'T flip a failing task to completed manually.
* DON'T mass-rewrite all schemas — surface, surgical, batch only when justified.

---

## Mission 57 state at handoff

* completed: 546
* pending: 79
* skipped: 9
* processing: 0 (orchestrator can't admit due to stale .pyc bug — fixed pending restart)
* failed/ungraded: 0
* DLQ history: 6+ tasks recovered through dlq retry; 4450 last to bite (6.1).

---

## Where to start (fresh session)

1. **Confirm KutAI restarted** after the .pyc nuke. Check `logs/kutai.jsonl` for fresh `Startup` event AFTER 2026-04-29T17:50 UTC. If not, ask user to restart.
2. **Watch first dispatches** — should NOT see `pool_scarcity` NameError. If it returns, .pyc cache survived somewhere (check venv site-packages, not just packages/).
3. **6.1 task #4450** — should pick up + step-refresh + persist + validate cleanly.
4. **Other 79 pending tasks** — likely uneventful but watch for new schema-instruction mismatches.
5. **If quiet**, move to **prompt simplification** (P1 carry-over) or **grader-schema unification** (P2).

---

## Memory writes deferred

This session wasn't recorded to `MEMORY.md` index. Consider adding:
* `project_session_20260429_arc.md` — describes the .pyc cache incident, step-refresh persistence fix, normalize default change.
* Update `feedback_*.md` if new corrections emerge.

---

## Honest counter

Mission 57 has been on retry-recovery loops for 3 days. Each fix unblocks one class of issue, then the next class surfaces. This is normal for a deep workflow with strict validation; M57 was started before any of these fixes existed. Once it completes, future missions should run noticeably cleaner.

If 6.1 retries STILL fail post-restart, it means there's a third layer of the bug we haven't found yet (likely in workflow_engine.advance fetching or post-hook invocation). Don't assume the fix was wrong — re-trace through DB state + step-refresh logs.
