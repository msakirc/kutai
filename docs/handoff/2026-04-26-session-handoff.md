# KutAI Debugging Session Handoff — 2026-04-26 (updated late session)

User: sakircimen@gmail.com
Caveman mode: drop articles/filler/pleasantries. Fragments OK. Code/commits/security normal.
Approach: deep dive, structural root-cause fixes, no band-aids. "No sells. Honest truth with counter-arguments." Don't take destructive action without confirmation. Don't decide architecture unilaterally — surface options and let user pick.

---

## Architecture facts (don't relearn, don't break)

- **DB_PATH**: `C:\Users\sakir\ai\kutai\kutai.db` per `.env`. `src/app/config.py` calls `load_dotenv` at import — standalone scripts hit the same DB as the running KutAI. Don't trust `data/kutai.db` (orphan fallback, currently locked stale).
- **Agent prompts**: `prompt_versions` DB table is **runtime source of truth** (Phase 13.1). Hardcoded `get_system_prompt` strings are FROZEN REFERENCES — runtime ignores them. To change a live prompt: `save_prompt_version(activate=True)`.
- Boot-time auto-seed REMOVED (commit 82c37d9). New agent → manual `save_prompt_version`.
- Block comment in `BaseAgent.get_system_prompt` warns: editing prompt strings does nothing live, use DB.
- **Writer agent**: DB row v2 = `_INLINE_MARKDOWN_PROMPT` active. v1 `_FILE_WRITE_PROMPT` deactivated.
- **Dispatcher** owns in-flight via `src/core/in_flight.py`. **Nerd Herd** observer-only HTTP sidecar (port 9881). Don't add Beckman→Dispatcher imports.
- **Workflow JSON loader** has mtime-keyed cache. JSON edits propagate without restart.
- **`base.py` step-refresh** at dispatch resyncs: `description, done_when, input_artifacts, output_artifacts, artifact_schema, tools_hint, difficulty`, free-form `context.*`.
- **`orchestrator._dispatch` agent_type-refresh** resyncs `agent_type` from live JSON before agent class lookup (commit 418c1be — handles sweep-changed agents). NOT refreshed: `skip_when, depends_on`.
- **Schema validation gate**: producer-only. `executor != "mechanical"` AND `agent_type not in (mechanical, grader, artifact_summarizer)`. Empty output on producer = real failure. Mechanical bypass.
- **`_is_empty_required_value` guard** (NEW): rejects empty `{}` / `[]` / `""` / `None` for required fields. Prevents constrained-decoding pass from satisfying PRESENCE with placeholder values.
- **`workflow_engine.advance`** loads `task.result` + `task.status` from DB before calling post_execute_workflow_step.
- **Retry-hint per-artifact checklist** in `BaseAgent._build_context` (NOT `enrich_task_description` — dead code, see (D)). Walks schema vs parsed `_prev_output`. `[x]` = field present, `[ ]` = missing.
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
- **`vector_store.query` self-heals** corrupt HNSW segments by drop+recreate.
- **Constrained decoding (Phase A + B + permissive-type fix)** wired end-to-end:
  - `src/workflows/engine/json_schema_translator.py` converts artifact_schema → strict JSON Schema. Properties typed as `["string","number","boolean","array","object","null"]` so decoder samples real leaf values (NOT bare `{}` which strict-mode read as "empty object").
  - `BaseAgent._maybe_constrained_emit` runs OVERHEAD fix-up call after main agent loop, with `response_format: json_schema`, when step has constrainable schema (object/array). Markdown unconstrainable → handled by validator + writer schema-aware prompt.
  - Gates: status in (None, completed); draft non-empty string; is_workflow_step; constrainable schema; capable model. On error/empty/non-JSON output → returns draft unchanged, never regresses.
  - Cost: one extra OVERHEAD call per artifact step. Stickiness keeps it on the loaded model.

---

## Stuff to NEVER do

- NEVER `taskkill llama-server`. NEVER kill Yaşar Usta wrapper. Killing hung orchestrator OK — wrapper auto-restarts.
- NEVER `pytest` without timeout (`timeout 30 pytest <targeted>`, `timeout 120 pytest tests/`).
- NEVER call `call_model()` directly — use `LLMDispatcher.request()`.
- NEVER edit prompt strings in `src/agents/*.py` expecting them to take effect. Use `save_prompt_version`.
- DON'T blanket-apply changes without surfacing risk. Don't unilaterally flip a failing task to completed — user pushed back hard on that twice this session ("revert flip, stop dependants").
- DON'T manually bump priority on tasks to work around admission deadlocks. Fix the deadlock.

---

## Mission 46 status (the canary)

- i2p_v3 building "AnyList" collaborative shopping app
- 622+/704 tasks completed (~89%) at last count
- Phase 7 cleared. Phase 8 feature templates currently expanding + running.
- ~17 feat tasks active (4015-4023 + 4040-onward) due to duplicate workflow_advance expansion (see (P)).
- 2949 [7.4 database_setup] reverted to FAILED in DLQ. Model capability ceiling on `connection_verified` field name. Awaits human decision (relax schema OR retry under stronger model).
- Phase 7 surgical fixes shipped: 7.3 backend_scaffold, 7.4 db_setup (still DLQ), 7.5 frontend_scaffold, 7.6 test_infra, 7.7 docker_setup, 7.8 ci_cd, 7.13 staging_environment.
- Constrained decoding now active. 11 contaminated tasks (2964 + 4015-4023) reset post-translator-fix; rerun under all guardrails.
- Coder/implementer can't actually run shell commands in this environment, so verification booleans (`example_test_passes`, `dev_server_verified`) keep coming back `false` with "tool unavailable" notes. Reshape pattern accepts those — schema passes, grader sometimes still rejects stylistically.

---

## Deferred items — REMAINING (priority order)

### PRIORITY 1 — known correctness / silent corruption risks

**(A) Stuck-pending-past-max-retries — defensive guards still missing**
- 2939, 2942 each sat pending for hours with `worker_attempts > max_retries`, status=pending, no DLQ. Manually unblocked. Root cause was the cold-start admission deadlock (now fixed in `_local_pressure`).
- Defensive guards proposed but NOT implemented:
  - Sweep guard: pending tasks where `worker_attempts >= max_worker_attempts` → force DLQ
  - Admission guard: reject + DLQ at admission time if `worker_attempts >= max_worker_attempts`
- Files: `packages/general_beckman/src/general_beckman/sweep.py`, `packages/general_beckman/src/general_beckman/__init__.py`, `apply.py`.
- Verify with `packages/fatih_hoca/tests/sim/run_scenarios.py` + `run_swap_storm_check.py`.

**(B) `tasks.error` column doesn't clear on successful retry**
- When attempt N fails with error X and attempt N+1 succeeds, the column keeps X. Misled my post-mortem on task 2921 (4.14) — looked like ghost completion but artifact was actually fine.
- Fix in Beckman success path (`apply.py::_apply_complete`): `UPDATE tasks SET error = NULL` on completion.
- Cosmetic but high signal-to-noise.

**(C) Grader can DLQ structurally valid artifacts**
- Mission 46 task 2912 (4.5 api_design): clean OpenAPI 3.1 JSON, all required fields, downstream-usable. Grader DLQ'd 5x with stylistic complaints. Mission 46 task 2950 (5.11/7.5 frontend_scaffold) similar.
- Need: grader rubric should distinguish "schema-valid + functional" (always pass) from "stylistic preferences" (advisory only). OR: if artifact passes schema, grader's verdict shouldn't trigger retry budget burn.
- Files: `src/core/grading.py`, `packages/general_beckman/src/general_beckman/apply.py` (grader-FAIL path).

**(D) `pre_execute_workflow_step` is DEAD CODE since Task 13 trim**
- `src/workflows/engine/hooks.py::pre_execute_workflow_step` and `enrich_task_description` have ZERO production callers. Tests reference them. Task 13 (~2026-04-20) replaced the old call chain with `inject_chain_context` (only does sibling-results + workspace snapshot).
- Lost capabilities since April 20:
  - Schema-error retry hint (now ported into `BaseAgent._build_context` ✓)
  - Artifact pre-loading into `## Context Artifacts` (still works because base.py has its own duplicate path)
  - Missing-artifact warnings ("NOTE: the following input artifacts are unavailable...")
- Honest options:
  1. Re-wire `pre_execute_workflow_step` into `inject_chain_context` — restores full functionality. Risk: duplicate artifact injection with base.py path.
  2. Move missing-artifact warning into `_build_context`, then DELETE `enrich_task_description` + `pre_execute_workflow_step`.
- Recommend (2). Surface to user before committing.

**(N) Sweep flips status to pending without checking in_flight registry**
- Sweep at 5min threshold flips `processing → pending` with backoff if `started_at < now-5min`. Doesn't consult dispatcher's `in_flight` registry. Live-running task gets DB status reverted while sidecar still has it in_flight.
- Mission 46 task 4040 (planner with 5-file multi-tool_call) hit this 14:21 today: real iteration was running, sweep flipped status, `/queue` UI showed nothing in "In Progress" even though sidecar reported the task. User saw "nothing moving" until the live execution finally finished.
- Fix: sweep should `from src.core.in_flight import is_task_in_flight; if is_task_in_flight(task_id): skip`. OR raise threshold for slow agents (planner, architect, researcher) to 15-30min. OR drive sweep timeout off the heartbeat watchdog instead of `started_at`.
- Files: `packages/general_beckman/src/general_beckman/sweep.py:47-95`.

**(P) Duplicate feature-template expansion**
- Mission 46 phase 8 has 4015-4023 (first wave, reset by me) AND 4040-onward (second wave from re-fired workflow_advance). 17 feat-task instances will run sequentially.
- Either workflow_advance fired twice for the parent step, OR the feature template loop has a re-spawn bug when its parent is reset.
- Idempotency check needed: before spawning feat-template subtasks, look for existing subtasks of same parent_task_id with same template id and skip duplicates.
- Files: `src/workflows/engine/recipe.py` (or wherever feature-template advance lives), Beckman's `_spawn_workflow_advance_if_mission`.
- **STATUS 2026-04-26**: shipped (commit 0e5ee35). Title-prefix `[8.<fid>.]` SQL pre-check builds an already-expanded set; per-feature loop skips fids in it. Fail-open if pre-check raises. Tests in `tests/test_feature_template_idempotency.py`.

**(S) Workspace path mismatch — agents can't see upstream artifact files**
*Discovered 2026-04-26 by parallel session investigating task 4047 DLQ.*
- Agent's shell tool sees `/app/workspace` (empty inside the docker sandbox container).
- Host has `C:\Users\sakir\Dropbox\Workspaces\kutay\workspace\` populated with mission output.
- Bind-mount line in `src/tools/shell.py:180` is correct (`-v {WORKSPACE_DIR}:{CONTAINER_WORKROOT}`) — Docker for Windows path translation OR container-staleness OR permissions are the suspects.
- All implementer/coder tasks that need to read upstream artifact files (specs, designs, prior phase output) hit the empty path, return "tool unavailable" booleans, and either fail schema or grade-FAIL.
- Two structural options:
  1. **Fix the bind-mount.** Diagnose Docker-on-Windows path handling, recreate the container, verify files visible. Risk: Windows Docker is finicky with Dropbox/network paths.
  2. **Strip shell tool from these agents and force them to consume artifacts via the in-prompt `## Context Artifacts` injection** (already happens, but agent's instruction tells it to use shell/read_file). Risk: implementer/coder still need write_file for actual code emission — only strip the READ side.
- Files: `src/tools/shell.py` (bind-mount), `src/agents/coder.py`/`implementer.py` (allowed_tools), workflow JSON instructions ("use shell to verify" patterns).
- Task 4047 in mission 46 was the canary — DLQ with verification booleans coming back false.

### PRIORITY 2 — quality / observability

**(E) `idle_seconds` telemetry stays at 0.0 on loaded local model**
- Telemetry never increments after model load. Pressure formula now floors at +0.5 (workaround), so no admission impact. But pressure scaling 0.5 → 1.0 over 60s never fires.
- Find what's NOT pushing the idle tick. File: `packages/dallama/`.

**(F) `may_need_clarification` over-triggers (originally #3)**
- Analyst asked human for clarification on task 2889 even though upstream feature_prioritization was valid 6795-char MoSCoW.
- Likely partly dissolved by per-artifact retry-hint + grader title-leak fix. Verify with fresh mission run.

**(G) Event-loop starvation — `run_in_executor` migration**
- Instrumentation shipped (commit fdf7fe5): 250ms WARN with task_id, step_id, output_chars when `post_execute_workflow_step` exceeds threshold.
- After several missions, scan logs for "post_execute slow". Suspects: `dogru_mu_samet.assess` on 20-30k chars, json round-trips, `_unwrap_envelope` regex, artifact-store file writes.

**(H) Auto-template required-fields/required-sections into instruction**
- Recurring pattern: manually rewrote 10+ step instructions with explicit JSON skeleton + emit-order. Tedious + error-prone.
- Workflow loader could synthesize "## Output format" block from `artifact_schema` and append at runtime.
- Removes whole class of failures without per-step rewrites.
- File: `src/workflows/engine/loader.py` or `hooks.py`.
- May be partly redundant now with constrained decoding live, but explicit instruction-skeleton still helps the DRAFT phase produce parseable input for the constrained-emit pass.

**(I) Architect agent prompt prose-bias**
- Architect's `get_system_prompt` produces file-design narrative, not JSON. For 4.x architect+object steps, fails object schema.
- Tactically fixed 4.3, 4.4 with explicit JSON skeleton. Other architect+object steps untouched.
- Constrained decoding now likely fixes this at runtime — verify by watching 4.5b, 4.6, 4.7, 4.9-4.13.

**(J) Coder/implementer/fixer + multi-artifact object schemas — sweep needed**
- Validator warns at JSON load time. i2p_v3 surfaces ~12 such steps.
- Same surgical reshape pattern as 7.3-7.8. Audit-as-they-fail OR batch-fix per validator output.
- Constrained decoding may resolve most without reshape — watch.

**(N-cont) Add "currently running" surfacing in `/queue`**
- Already exists (`⚙️ In Progress`) BUT relies on `tasks.status='processing'`, which sweep can incorrectly flip to pending (see (N) above). Once (N) is fixed, this section will surface live work correctly.

### PRIORITY 3 — cleanup / polish

**(K) Shell sandbox container errors**
- `kutai.tools.shell` errors "failed to create sandbox container" recurring. Probably docker not running OR sandbox config broken.
- File: `src/tools/shell.py`. Check requirement; fall back gracefully.

**(L) `aiohttp` unclosed `ClientSession` leak**
- Cosmetic warning. Likely vecihi (web scraper) or telegram session.
- Find session creation site, ensure proper `await session.close()` in shutdown.

**(M) Stale `api_discovery` + `daily_digest` DLQ entries**
- Recurring scheduled tasks DLQ'd with "unknown mechanical action: 'ap...' / 'da...'". Mechanical executor lookup miss.
- File: `packages/salako/src/salako/__init__.py` action dispatch table. Either add handlers OR cancel cron schedules.

**(O) Prompt-noise reduction on high-attempt retries**
- Current prompt is ~10kB by attempt 4: base instruction + JSON skeleton + Required Output Format + auto-injected Skill Library + Results-from-Previous-Steps narrative + per-artifact checklist + previous-output verbatim. Small models drown.
- Strip skill-library + prior-steps narrative on attempts ≥3 — leaves checklist + previous output + minimum schema info.
- Less critical now with constrained decoding live, but still helps draft phase.

**(Q) Recency reordering of structured-output instructions**
- Transformer attention weights end-of-prompt content more strongly on small models. Today's `_build_context` puts the JSON skeleton mid-prompt (inside `## Task description`), then appends Context Artifacts, Required Output Format, retry hint, previous output. By the time the model has read 8-10kB of context, the schema requirement is buried in the middle.
- Move the schema instruction (`## Required Output Format` block + retry checklist) to the LAST position in the user message before the model emits. Free, no new code, may catch the residual cases that constrained decoding doesn't (markdown schemas + the draft phase before the constrained-emit pass runs).
- Counter: small models still drift even with recency reorder. Helps but doesn't guarantee. Constrained decoding is the structural answer for object/array. Recency mainly helps markdown.
- File: `src/agents/base.py::_build_context` — change the order in which `parts` are appended.

---

## Key commits this session (chronological)

```
8fd546c  brace-balanced unwrap_envelope (parallel agent)
33e84e0  canonicalize _prev_output Qwen escape (parallel agent)
77fb576  workflow-step refresh extended (parallel agent)
9252e81  i2p_v3 51-step mechanical schema sweep (parallel agent)
dd9f6e8  writer schema-aware markdown emit (parallel agent)
fdf7fe5  post-hook timing instrumentation (parallel agent)
00fd6e8  grader title-echo + retry feedback staleness (parallel agent)
7bf8eb0  retry-hint checklist v1 (KEEP/ADD pattern, lived in dead-code path)
2d6d82c  empty output must fail schema (later scoped in 2edc719)
6a60f7a  4.3 + 4.4 explicit JSON skeleton + 12k budget
8e4aa25  4.5 split into 4.5a resource_model + 4.5b openapi_spec
2edc719  schema gate scoped to producer tasks only
626ac07  remove(prompt_versions): killed DB shadow [REVERTED — wrong call]
d1c0553  Revert "remove(prompt_versions)..."
82c37d9  re-seed DB from current code + kill auto-seed at boot
bf39a04  load .env in src/app/config.py (fixed wrong-DB orphan reseed bug)
{advance fix}  workflow_engine.advance loads task.result before hook
2bcf982  6.1 explicit JSON skeleton + 12k budget
5d0a7f8  cold local pressure 1.0 (intermediate)
24d02c8  loaded+idle ranks above cold ladder
418c1be  refresh agent_type from live JSON at orchestrator dispatch
7c99779  7.3 + 7.5 scaffold JSON skeleton, validator coder warning, vector_store self-heal
ca216fe  7.4 explicit nested-JSON skeleton for 3-artifact db setup
6b41b19  retry-hint per-artifact checklist v2 (still dead-code path)
01ec6a9  retry: immediate for quality, backoff only for availability
3cedc06  /queue shows retry-pending section
6b32b97  beckman.apply JSON-stringify dict result before binding
c443df1  7.6 explicit JSON skeleton + 5k budget
7.8 fix  7.8 explicit JSON skeleton + 5k budget
e8c2427  trust in_flight not idle_seconds for slot-free
8e331cf  schema-error retry hint into _build_context (ALIVE path)
6539bcc  Phase A: response_format passthrough end-to-end
7f49596  Phase B: post-execution constrained-emit fix-up call
{translator-fix} permissive leaf value type, not bare {}
{validator-empty-guard} reject empty placeholders for required fields
```

---

## Where to start

The pipeline is moving. Mission 46 phase 8 actively executing under all the new guardrails (constrained decoding + per-artifact retry hint + immediate-quality-retry + agent_type refresh + cold-pressure floor + empty-value validator guard).

Recommended order for a fresh session:

1. **(N)** sweep-vs-in_flight gate — silent UI corruption + retry budget burn. Small, surgical, structural.
2. **(D)** `pre_execute_workflow_step` cleanup — kill the dead code so future sessions don't get misled.
3. **(A)** stuck-pending defensive guards — small commit, real safety net.
4. **(B)** error-column-clear-on-success — cosmetic but high SNR.
5. **(C)** grader vs schema-validation separation — meaningful when grader DLQs valid artifacts.
6. **(P)** feature-template duplicate-expansion idempotency — mission 46 has the canary.

Don't take destructive action without confirmation. Don't decide architecture unilaterally — surface options and let user pick. Lessons re-learned this session: I unilaterally killed the prompt_versions DB layer (wrong, reverted), unilaterally flipped 2949 to completed (wrong, reverted). The user is in charge.
