# KutAI Debugging Session Handoff — 2026-04-27

User: sakircimen@gmail.com
Caveman mode: drop articles/filler/pleasantries. Fragments OK. Code/commits/security normal.
Approach: deep dive, structural root-cause fixes, no band-aids. "No sells. Honest truth with counter-arguments." Don't take destructive action without confirmation. Don't decide architecture unilaterally — surface options and let user pick.

---

## Architecture facts (don't relearn, don't break)

- **DB_PATH**: `C:\Users\sakir\ai\kutai\kutai.db` per `.env`. `src/app/config.py` calls `load_dotenv` at import — standalone scripts hit the same DB as the running KutAI. Don't trust `data/kutai.db` (orphan fallback, currently locked stale).
- **WORKSPACE_DIR**: `C:\Users\sakir\Dropbox\Workspaces\kutay\workspace` (NOT `src/workspace` — that's a stale historical path). Folder named `Dropbox` but not actually syncing — pure local. Docker uses Hyper-V backend on Windows.
- **Sandbox bind-mount**: `validate_or_recreate_sandbox` runs at orchestrator startup (commit e9aff93). On stale mount source, force-removes the container so the next shell call recreates with current `WORKSPACE_DIR`. One-shot, not per-call. Path comparison normalized (case-insensitive, separator-agnostic, trailing-slash-agnostic).
- **Agent prompts**: `prompt_versions` DB table is **runtime source of truth** (Phase 13.1). Hardcoded `get_system_prompt` strings are FROZEN REFERENCES — runtime ignores them. To change a live prompt: `save_prompt_version(activate=True)`.
- Boot-time auto-seed REMOVED (commit 82c37d9). New agent → manual `save_prompt_version`.
- Block comment in `BaseAgent.get_system_prompt` warns: editing prompt strings does nothing live, use DB.
- **Writer agent**: DB row v2 = `_INLINE_MARKDOWN_PROMPT` active. v1 `_FILE_WRITE_PROMPT` deactivated.
- **Dispatcher** owns in-flight via `src/core/in_flight.py`. **Nerd Herd** observer-only HTTP sidecar (port 9881). Don't add Beckman→Dispatcher imports.
- **Workflow JSON loader** has mtime-keyed cache. JSON edits propagate without restart.
- **`base.py` step-refresh** at dispatch resyncs: `description, done_when, input_artifacts, output_artifacts, artifact_schema, tools_hint, difficulty`, free-form `context.*`.
- **`orchestrator._dispatch` agent_type-refresh** resyncs `agent_type` from live JSON before agent class lookup.
- **Schema validation gate**: producer-only. `executor != "mechanical"` AND `agent_type not in (mechanical, grader, artifact_summarizer)`. Empty output on producer = real failure.
- **`_is_empty_required_value` guard**: rejects empty `{}` / `[]` / `""` / `None` for required fields — prevents constrained-decoding from satisfying PRESENCE with placeholder values.
- **Clarify-action skip on schema validation** (commit 6aaedaf): `triggers_clarification` steps that emit clarify-action skip the validator (the artifact IS the question, lives in `result.question`/`result.clarification`).
- **`workflow_engine.advance`** loads `task.result` + `task.status` from DB before calling post_execute_workflow_step.
- **Retry-hint per-artifact checklist** in `BaseAgent._build_context`. Walks schema vs parsed `_prev_output`. `[x]` = field present, `[ ]` = missing.
- **Envelope unwrap before checklist parse** (commit f08c9c0): `_prev_output` may carry the raw envelope `{"action":"final_answer","result":"..."}`; checklist now calls `_unwrap_envelope` first so it walks artifact keys, not envelope keys.
- **`_unwrap_envelope` tolerates list/dict input** (commit b7cf388): coerces non-string via `json.dumps` before string ops. Prevents `'list' object has no attribute 'strip'` crashes when agent emits bare array/object directly.
- **`_is_disguised_failure` is negation-aware** (commit 6cea4ed): scans the SENTENCE preceding a failure phrase for negation tokens (`not, no, n't, non-, without, never, isn't, ...`). `"is not a critical blocker"` no longer trips. Bare `shell tool` removed from hard list.
- **Retry policy**: quality failures retry IMMEDIATELY (no backoff). Availability keeps exponential ladder `[0,10,30,120,600]s`.
- **`/queue` shows retry-pending section** with attempt count + category + ETA.
- **Local pressure ladder** (`_local_pressure`):
  - in_flight (any local entry): -1.0 (saturated)
  - swapping: -0.5
  - loaded busy idle=0: +0.5 (telemetry stale-tolerant)
  - loaded idle 1-60s: +0.5 → +1.0 linear
  - loaded idle 60s+: +1.0 (peak)
  - cold (no model): +0.5 (capacity, swap cost ahead)
  - **in_flight is GROUND TRUTH** for "slot free"; `idle_seconds` telemetry can be stale.
- **`tasks.result` always TEXT**: `_apply_complete` JSON-stringifies dict/list payloads before binding.
- **`tasks.error` cleared on completion** (commit cf1e049): success path resets error/error_category/next_retry_at/retry_reason/failed_in_phase so completed rows don't show ghost failure metadata.
- **Sweep guards** (commit e9c2398):
  - Section 1 honors `is_task_in_flight()` before flipping `processing→pending`.
  - Section 8: pending tasks past `worker_attempts >= max_worker_attempts` get force-DLQ'd.
  - Section 9: auto-resolves DLQ rows for decommissioned mechanical actions (`api_discovery`, `daily_digest`, etc).
- **Admission cap-guard** (commit e9c2398): rejects + DLQs candidates past cap before Hoca pick. Mechanical bypass.
- **Double-dot expander fix** (commit d7e27a2): `step_id = f"{prefix.rstrip('.')}.{tpl_step_id}"`. Both calling conventions (`"8.<fid>."` and `"8.<fid>"`) now produce identical step_ids. Mission 46/57 phase 8 had every task titled `[8.<fid>..feat.X]` before this.
- **Feature-template idempotency** (commit 0e5ee35): single SQL pre-query collects `[8.<fid>.]` titles for the mission, builds already-expanded set, per-feature loop skips fids in it.
- **vector_store.query self-heals** corrupt HNSW segments by drop+recreate.
- **Reviewer agents skip grade post-hook** (commit 9137ccd): `reviewer` joined `grader` and `artifact_summarizer` in `_NO_POSTHOOKS_AGENT_TYPES`. 20 i2p_v3 reviewer steps no longer double-judged. Pseudo-review steps with non-reviewer agents (0.6 writer, 1.7 researcher, 12.4 analyst, 13.11 executor) still graded.
- **Constrained decoding (Phase A + B + permissive-type fix + skip-when-clean)** wired end-to-end:
  - `src/workflows/engine/json_schema_translator.py` converts artifact_schema → strict JSON Schema. Properties typed as `["string","number","boolean","array","object","null"]` so decoder samples real leaf values (NOT bare `{}` which strict-mode reads as "empty object").
  - `BaseAgent._maybe_constrained_emit` runs OVERHEAD fix-up call after main agent loop, with `response_format: json_schema`, when step has constrainable schema (object/array). Markdown unconstrainable → handled by validator + writer schema-aware prompt.
  - **Skip-when-clean** (commit d838eef): if draft already parses with all required artifact keys present, skip emit entirely — re-emit risks compression. Validator catches genuine gaps next.
  - **Bumped budgets** (commit d838eef): draft input cap 12000→30000 chars, output token cap 12000→16000, floor 1000→2000.
  - Gates: status in (None, completed); draft non-empty string; is_workflow_step; constrainable schema; capable model. On error/empty/non-JSON output → returns draft unchanged, never regresses.
- **Empty-artifact guard** (commit e9aff93): `ArtifactStore.store` rejects empty/whitespace/None writes with WARN log, returns False. Prevents silent blackboard corruption.
- **Failed-model tracking** (commit d7238a9): `general_beckman.on_task_finished` reads `result.generating_model` / `result.model`, persists to ctx, and on `status=='failed'` appends to `ctx.failed_models`. Idempotent. Was the missing piece for R1/R2 (model exclusion at attempts >= 3 + difficulty bump) — they were wired but never had data.

---

## Stuff to NEVER do

- NEVER `taskkill llama-server`. NEVER kill Yaşar Usta wrapper. Killing hung orchestrator OK — wrapper auto-restarts.
- NEVER `pytest` without timeout (`timeout 30 pytest <targeted>`, `timeout 120 pytest tests/`).
- NEVER call `call_model()` directly — use `LLMDispatcher.request()`.
- NEVER edit prompt strings in `src/agents/*.py` expecting them to take effect. Use `save_prompt_version`.
- DON'T blanket-apply changes without surfacing risk. Don't unilaterally flip a failing task to completed.
- DON'T manually bump priority on tasks to work around admission deadlocks. Fix the deadlock.
- DON'T `docker rm -f orchestrator-sandbox` mid-mission unless the user confirms — kills in-flight shell calls.
- DON'T mass-sweep i2p_v3 fixes when 1-2 specific steps are failing — the broader pattern often hasn't materialized; fix-as-they-bite is usually correct.

---

## Mission 57 status (the canary)

- i2p_v3 mission building "AnyList"-style collaborative shopping app
- Started 2026-04-26 ~22:00, ran ~17h with multiple restarts/dlq retries
- 290 completed / 103 pending / 9 skipped / 4 failed at last check
- Phase 4-7+ blocked by 4 DLQ'd tasks in phase 3-5 — restart + dlq retry should clear most:
  - **4409 [3.5] integration_requirements**: schema gap + low token budget — both fixed (1080e64 + bf64070)
  - **4410 [3.6] platform_and_accessibility_requirements**: schema gap + diff bump — fixed
  - **4417 [3.11] requirements_review**: reviewer grader-FAIL — fixed by reviewer-no-grade (9137ccd)
  - **4441 [5.4b] forms_and_states**: multiple issues
    - Constrained_emit was COMPRESSING good drafts (fixed: d838eef skip-when-clean + bumped budgets)
    - failed_models never accumulated → same model picked every retry (fixed: d7238a9)
    - Per-artifact checklist falsely marked all fields `[ ]` due to envelope-vs-artifact-keys mismatch (fixed: f08c9c0)
- All four canaries hit before the latest fixes shipped. After restart + dlq retry, expect clean run.
- Cloud providers being added soon — will help on long-context steps (e.g. 4417 had 33kB prompt) and on attempts >=3 difficulty bump escalations.

---

## What got shipped this session — chronological

```
d7e27a2  expander: strip trailing dot from prefix (handoff D supplement)
3314ff7  delete dead pre_execute_workflow_step + enrich_task_description (handoff D)
6cea4ed  hooks: negation-aware _is_disguised_failure
6aaedaf  hooks: skip schema validation for clarify-action on triggers_clarification steps
b7cf388  hooks: _unwrap_envelope tolerates list/dict input
9137ccd  beckman: reviewer agents skip grade post-hook (handoff D / item C)
1080e64  i2p_v3: tighten 3.5/3.6 schemas to match instruction intent
85e90df  audit script for schema-instruction mismatches
bf64070  i2p_v3: bump output tokens + difficulty for confirmed truncation steps
d838eef  constrained-emit: skip when draft is clean + bump budgets
d7238a9  beckman: persist generating_model + accumulate failed_models in on_task_finished
f08c9c0  retry-feedback: unwrap envelope before per-artifact checklist parse
e9aff93  artifact-store empty-guard + sandbox bind-mount validation (earlier in arc)
e9c2398  beckman: sweep+admission guards (R+N+A from prior handoff)
cf1e049  beckman: clear stale failure metadata on completion (B from prior handoff)
f9507bb  base: missing-artifact NOTE in _build_context (D partial from prior handoff)
5c31db7  i2p_v3: token budget for coder/implementer/fixer object/array steps (J)
5377d2e  sweep: auto-resolve decommissioned mechanical-action DLQ rows (M)
90acdd2  base: recency-order schema + retry blocks at prompt tail (Q)
60a4131  base: drop skills+prior-steps blocks on retry >= 3 (O)
0e5ee35  hooks: feature-template expansion idempotency (P)
368c68c  run: try/finally around orchestrator start (L)
ab8c2e1  shell: cache docker-down state (K)
```

---

## Deferred items — REMAINING (priority order)

### PRIORITY 1 — pending live verification

**(canary) Mission 57 retry behavior post-fix**
- Restart + `/dlq retry 4409 4410 4417 4441` to validate the d7238a9 + f08c9c0 + d838eef chain.
- Watch logs for:
  - `tracked failed_model` (proves d7238a9 firing)
  - `step-refresh: ...` showing diff/exclude_models updates at attempts >=3
  - `constrained_emit skipped — draft parses` (proves d838eef short-circuit)
  - `[x] forms`-style accurate checklist marks (proves f08c9c0)
  - 0 `Schema validation: empty placeholder` events on 5.4b — d838eef + f08c9c0 chain
- If 4441 still loops despite all fixes, deeper investigation needed (possibly grader rubric, possibly model-capability ceiling).

### PRIORITY 2 — quality / observability (defer until live data)

**(E) `idle_seconds` telemetry stays at 0.0 on loaded local model**
- Code logic looks correct: ``InferenceCollector`` polls llama-server `/metrics` every 5s, tracks `_idle_since_ts` when `llamacpp_requests_processing` drops to 0; `nerd_herd.snapshot()` overlays the live value.
- Yet pressure formula's 0.5→1.0 ramp over 60s never fires.
- Suspect: llama-server build doesn't emit `llamacpp_requests_processing`, OR sidecar's `InferenceCollector` poll task isn't actually starting.
- Diagnostic: ``curl http://127.0.0.1:8080/metrics | grep requests_processing`` while a model is loaded but idle.
- Pressure floor at +0.5 is the workaround; not user-visible.

**(F) `may_need_clarification` over-trigger**
- Pre-fix, analyst asked human even on valid 6795-char MoSCoW input.
- Likely partly dissolved by retry-staleness fix + reviewer-no-grade. Verify with mission run.

**(G) Event-loop perf profile**
- Instrumentation already shipped (commit fdf7fe5): 250ms WARN with task_id, step_id, output_chars when `post_execute_workflow_step` exceeds threshold.
- Watch for accumulated `post_execute slow` events; only 2 observed in last 17h. If a specific offender emerges, profile + move into `run_in_executor`.

**(H-extension) Auto-bump tokens proportional to schema weight at workflow load**
- 117 i2p_v3 steps have structured schema with no token override. Many will be fine — schema weight is a rough proxy. Specific mismatches fixed reactively (3.5, 3.6, 5.4b).
- If broader truncation pattern recurs, consider a workflow-loader-time synthesizer that bumps `estimated_output_tokens` proportional to required-fields weight.

**(I) Architect agent prompt prose-bias**
- Architect's ``get_system_prompt`` produces design narrative; for 4.x architect+object steps that fails object schema.
- Tactically fixed 4.3, 4.4 with explicit JSON skeleton. Other architect+object steps untouched.
- Phase B constrained-decoding likely fixes the rest at runtime — verify by watching schema fail rate on 4.5b, 4.6, 4.7, 4.9-4.13.

### PRIORITY 3 — architectural / scope decisions

**(R-class retry recovery) Improvements still on the shelf**
The session shipped R1/R2 wiring (commit d7238a9). Other ideas surfaced and deferred:
- **R3 multi-signal trust**: schema-pass + grader-fail same insight 3× → trust schema, count grader as advisory. Risk: lets stylistically-bad-but-schema-valid output ship.
- **R4 insight-deduped retry budget**: track grader insight hash; same insight repeats → don't increment worker_attempts. Risk: needs hard ceiling regardless.
- **R5 exit-on-no-progress**: edit-distance between attempts < 20% AND fail same way → bail with last-best-draft. Risk: cuts legitimate slow-improvement cycles short.
- User's preferred direction was D (workflow design audit) over E/G changes. R-class items can be reconsidered if even with d7238a9+f08c9c0 the DLQ rate stays meaningful.

**Schema dialect nesting**
- Audit script (`scripts/audit_schema_instruction.py`) flagged 5 i2p_v3 steps with nested-field instruction-vs-schema gaps the flat dialect can't express:
  - `-1.3 artifact_synthesis`: artifact_name (sub-field of items)
  - `2.2 value_proposition_canvas`: gain_creator, pain_reliever (in value_map)
  - `6.4 sprint_planning`: task_id, estimated_effort (in sprint_plans[].tasks)
  - `7.6 test_infrastructure`: coverage_tool (in backend/frontend_test_config)
  - `8.arch_check`: design_was_wrong (classification value in deviations item)
- Phase B's JSON-Schema translator handles `items.required` so arrays partially benefit. Object-sub-fields aren't enforced by the dialect.
- Extending dialect to support nested `required_fields` is a breaking change; defer until specific step keeps biting.

**Grader vs schema separation (handoff item C reframed)**
- Concern: tasks DLQ when worker produces same garbage 5x and grader reproduces same complaint each time.
- d7238a9 fixes the model-monoculture half (now rotates after attempts >=3).
- f08c9c0 fixes the feedback-correctness half (checklist no longer lies).
- If those two land but DLQs persist, R3/R4/R5 are next options. User pushback: "all options degrade or complicate." Still on the shelf.

### PRIORITY 4 — minor / cleanup

**(L-residual) Unclosed aiohttp session warnings**
- Cosmetic. Cosmetic leak comes from litellm internals — out of scope.

**(M-cont) m46 orphan empty design artifacts**
- Mission 46 blackboard has `screen_specifications, prd_final, business_rules, design_handoff, ...` stored as empty strings (pre-fix-era). If you ever resume mission 46, downstream cascade-fails immediately.
- Either: explicitly delete those keys via SQL OR cancel mission 46 outright OR leave as-is and never resume m46.

**(audit) Phase 8/9/10/etc agent_type sweep follow-ups**
- Earlier sweep (commit 9252e81) fixed 51 agent/schema mismatches. Current validator: 0 warnings. New mission JSON edits could re-introduce — re-run `audit_schema_instruction.py` after any workflow JSON change.

---

## Where to start (fresh session)

The pipeline has just had a coordinated chain of fixes shipped that need to be observed in production:

1. **Restart KutAI** (or assume already restarted). Picks up d7238a9 + f08c9c0 + d838eef + 9137ccd + 6cea4ed + 6aaedaf + b7cf388 + 3314ff7 + d7e27a2.
2. **Retry the 4 mission 57 DLQs** (`4409 4410 4417 4441`) via Telegram or `/dlq retry`.
3. **Watch logs** for the indicators listed in PRIORITY 1.
4. If clean run → connect cloud providers; mission 57 pushes through to completion.
5. If 4441 still DLQs → deeper investigation (capability ceiling for 5.4b's heavy output? schema demands content unrealistic for any local model?).

**Behavior to keep:** don't decide architecture unilaterally. Surface options, let user pick. The user is in charge.

**Lessons re-learned this session:**
- Retry recovery had R1/R2 wired but no data flowing in (failed_models stayed []). The architectural mechanism was in place; the persistence was missing.
- Per-artifact checklist was lying for 5+ retries because of envelope-vs-artifact-key confusion. Fix-once at parse-time.
- Phase B constrained-emit can BACKFIRE on already-clean drafts (compresses content). Skip-when-clean was the architectural correction.
- Workflow JSON edits propagate without restart (mtime cache). Code edits need restart.
- Empty artifacts silently-corrupting blackboard → ArtifactStore.store now refuses empty writes (loud signal).
