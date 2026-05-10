# Cross-cutting concerns + sandboxing + demo deliverable

## Frame

Concerns that appear in every zone but are nobody's primary
responsibility. Trust calibration, time awareness, reversibility framing,
cost transparency, provenance. Plus two infra pieces that don't fit a
single zone: per-mission sandboxing/containerization, and the end-of-
mission demo deliverable.

This doc is the "shared spine" — every other zone doc must conform to
the patterns established here. Independent agent picking this up should
expect to drive design decisions that ripple across the whole folder.

## Current state (2026-05-08) — operational reality

Two passes of evidence: (1) what tables/fields exist; (2) **what actually breaks today**. Pass 2 is the load-bearing one — many fields exist but never gate or attribute correctly.

### Severity-ranked gaps

| Axis | Severity | Real failure (not "field missing") |
|---|---|---|
| Reversibility | **HIGH** | DB schema migration `goals → missions` at `src/infra/db.py:244` fired once with no rollback wrapper; `file_locks` table at `db.py:483-493` has no release-on-task-failure path (orphan locks); `git_commit` mr_roboto verb does best-effort `git push` with no atomic local-commit↔push pairing — orphans on push fail |
| Sandbox escape | **HIGH** (partly Z10-T3B) | `src/tools/shell.py` mode=`local` used to fall back silently with no `sandbox_mode` field; T3B logs the *resolved* `sandbox_mode` and gates per-mission `local` requests via `sandbox_local_mode` confirmation. T3B adds `_is_blocked_argv` semantic guard catching `["dd", "if=/dev/sda", ...]` style argv tricks. Per-mission container `kutai-mission-{id}` via `ensure_mission_container` with `--memory/--cpus/--pids-limit`. Egress whitelist file `config/egress_allowlist.txt` + `broader_egress` confirmation. Still **partial**: egress filtering is whitelist-by-prompt only — no iptables in-container drop rule (full netns control is post-v1); per-mission DB/Chroma scoping is T3C |
| Provenance | **HIGH** | `model_call_tokens` has `task_id` but no `mission_id` (`db.py:428-447`); `cost_budgets.scope` is opaque text (`db.py:469-480`); cannot answer "which model produced `src/x.py`" — only joins are file→commit (git) and task→model (DB), no bridge between them |
| Cost attribution | **MED** | Tokens captured per-call, but `cost_budgets` uses generic `scope`/`scope_id` strings — no FK to `missions`. `iteration_n` field exists on `model_call_tokens` but **unused** in any rollup; retry-cost vs first-pass cost is uncomputable today |
| Confidence gating | **MED** | `min_confidence` defined `src/agents/base.py:85`, set on Coder/Researcher/ShoppingAdvisor (=3) — but no `if confidence < min_confidence: escalate()` in execute path. Field is **cosmetic**: never blocks a task |
| Demo / "shipped" | **MED** | i2p `15.14` final step is `roadmap_update` (quarterly planning), not delivery; `15.10` `pre_launch_verification` exits via `needs_clarification` (manual human sign-off). Zero automated proof "the product works" |
| Telegram threading | **MED** | All output flat to `TELEGRAM_ADMIN_CHAT_ID`; `_pending_clarifications` keyed by `task_id` only (`telegram_bot.py:372-384`); two concurrent missions interleave indistinguishably; chats >1000 msgs lose `_clarification_msg_ids` reverse-lookups |
| Cross-mission interference | **MED** | `_tx_lock = asyncio.Lock()` in `db.py:40` is a **global** writer — Mission A's large INSERT blocks Mission B's `add_task` until 60s timeout → `OperationalError "database is locked"`. `chroma_data` shared (no per-mission namespace). Single sandbox container = global `$PWD`, `$SHELL_HISTORY`, env |
| Time awareness | **LOW** | Only `tasks.created_at/started_at/completed_at`. No `step_started_at`, no per-phase boundary timestamps, no `phase_time_budget`. `failed_in_phase` text exists but no time-budget partner |
| Trust calibration loop | **LOW** | `model_pick_log` has no `confidence_score` or outcome-correctness column to correlate with picks. Loop has no input |

### Reversibility taxonomy — drafted from real verbs

Eyeballed list across `packages/mr_roboto/src/mr_roboto/actions.py` (12+ verbs) + `src/agents/tools/` + `src/tools/`. Classification proposed for v1:

| Verb / tool | Source | Tag | Rationale |
|---|---|---|---|
| `git_commit` | mr_roboto | `full` | `git revert`/`reset` recovers; no external state |
| `git_push` (paired w/ commit) | mr_roboto | `partial` | shared-branch push: revert is a new commit, history preserved |
| `workspace_snapshot` | mr_roboto | `full` | additive only |
| `verify_artifacts` | mr_roboto | `full` | read-only |
| `check_grounding` | mr_roboto | `full` | read-only |
| `http_check` | mr_roboto | `full` | read-only GET |
| `run_cmd` (sandboxed) | mr_roboto | `partial` | depends on cmd; default `partial`, override per-command |
| `cloud_refresh` | mr_roboto | `full` | cache write |
| `kdv_persist` | mr_roboto | `full` | DB upsert, idempotent |
| `parse_og_tags` | mr_roboto | `full` | read-only |
| `notify_user` (Telegram) | mr_roboto | `irreversible` | message sent, can't unsend (after edit window) |
| `todo_reminder` | mr_roboto | `irreversible` | same as notify |
| `price_watch_check` | mr_roboto | `full` | read-only |
| `read_file` / `file_tree` | tools | `full` | read-only |
| `write_file` | tools | `full` | overwrite — restorable from git if pre-committed; tag `partial` if path outside repo |
| Shell `rm` / `mv` | shell | `partial` | recoverable from git for tracked, lost otherwise |
| Shell network / external API | shell | `irreversible` | side effects external |
| DB migration (DDL) | db.py | `partial`/`irreversible` | `DROP COLUMN`/`RENAME` partially recoverable from snapshot only; `INSERT` `full` |
| Vendor adapter publish (06) | future | `irreversible` | app-store / email-send / payment |
| `record_demo` (future) | future | `full` | additive artifact |

**Decision rule for v1.** Default tag per verb registry; `partial`/`irreversible` triggers Phase G confirmation flow. `run_cmd` and shell verbs need a per-invocation override mechanism (the LLM tags its intent; mechanical executor enforces).

### Three real reversibility incidents (concrete)

1. **Schema rename `goals → missions`** (`src/infra/db.py:244`). `ALTER TABLE` ran once at startup, no `BEGIN/COMMIT` wrapper; partial-fail leaves DB in inconsistent state. No migration ledger.
2. **`file_locks` orphans** (`db.py:483-493`). Acquired but no release on task-crash path. Prevents Beckman re-issuing the task.
3. **Silent `git push` fail** (mr_roboto `git_commit.py`). Local commit succeeds; push fails (auth/network); commit orphaned locally; next run double-commits. No tx around commit↔push pair.

### Confidence gate is unplugged

`base.py:85` defines `min_confidence`; agent configs set it (3 for Coder/Researcher/ShoppingAdvisor); **no caller compares step output `confidence` against `min_confidence`**. Reviewer escalation runs on its own criteria. To make Phase A meaningful, the gate must actually block — currently any "feature" built on it is theatre.

### Cost attribution is a phantom

`cost_budgets(scope, scope_id, daily_limit, total_limit, spent_today, spent_total)` (`db.py:469-480`) — `scope`/`scope_id` are free text, never set to `mission`/`<mission_id>` in current code. `model_call_tokens.iteration_n` exists but is never aggregated. Until both are wired, "per-mission cost" is a label without a number.

### Provenance join — physical demonstration

Question: "which model produced `src/x.py`?"
- Step 1: `git log --follow src/x.py` → commits + timestamps. ✅
- Step 2: `tasks WHERE started_at < commit_ts AND completed_at > commit_ts` → candidate task_ids.
- Step 3: `model_call_tokens` joined on task_id → models touched. ✅
- **Step 4 (broken)**: which of those models *wrote that file*? No artifact↔task join. The model that wrote a tool call producing the file isn't tagged. Phase A must add `artifact_provenance(path, task_id, step_id, model_id, retry_n, reviewer_verdict_id)`.

### Demo gap — what "shipped" means today

`i2p_v3.json` step `15.10 pre_launch_verification` exits to `needs_clarification` waiting for human go/no-go. `15.14 roadmap_update` — quarterly planning. **There is no automated "the product works" check.** Phase F isn't a vanity feature; it's the only proof.

### Cross-mission interference — measured

- Global `_tx_lock` (`db.py:40`): A single 1s INSERT blocks every other writer for 1s; cumulative under load.
- WAL 60s timeout → `OperationalError "database is locked"` observed in logs (need to grep for frequency, but pattern is real).
- `chroma_data`: no namespace per mission. Embedding writes from two missions can interleave on the same collection.
- Sandbox `CONTAINER_NAME` is constant (`shell.py:20`) — `pwd`/env shared.

These don't show up as gaps in a schema review; they show up as "missions get slower/flakier with concurrency." Phase E must address container-per-mission **and** per-mission DB scoping (or sharded WAL).

**Key insight (revised).** Phase A is leveraged not because the schema is missing — it's because the existing fields are *unenforced*. Phase B's biggest unlock is wiring `cost_budgets.scope = 'mission'` and aggregating `iteration_n` (1-2 days, mostly query work). Phase E is the highest-risk, highest-payoff: today's "single sandbox + global tx_lock + shared chroma" makes mission concurrency physically unsafe.

## Concerns

### A. Trust calibration

**Gap.** Agent doesn't know what it doesn't know. No "I'm 30% confident
in this architectural choice; here's why." Confidence is implicit /
absent; founders can't tell which decisions warrant scrutiny.

**Solution.**
- Every artifact-producing step gains a `confidence` field (low / medium / high) + `reasoning`.
- Every architectural decision (ADR) gains explicit `confidence` + `reversal_cost`.
- UI surfaces low-confidence + irreversible decisions for explicit human sign-off.
- Confidence calibration training: feed past confidence vs actual-correctness back into prompts ("models that say 'high confidence' here are wrong 40% of the time — be more conservative").
- Per-domain reliability score: which areas of decision-making does this agent actually do well in? Track over time.

### B. Time awareness

**Gap.** Mission has no temporal pacing. "We've been at this 3 weeks" not surfaced. No deadline pressure, no scope-cut judgment when launch date approaches.

**Solution.**
- Mission declares: target launch date, total time budget, per-phase budget.
- Real-time pacing dashboard: where are we vs plan? burndown.
- Approaching deadlines surface tradeoff conversations: "phase 8 backlog has 12 features; at current pace you'll hit launch with 8. Cut which 4?"
- Long-running concerns scheduled (backups, key rotation, dependency updates) cross-ref [08-operations.md](08-operations.md).
- Calendar integration: founder availability, demos scheduled (cross-ref [07-humanish-layers.md](07-humanish-layers.md)).

### C. Reversibility framing

**Gap.** "I'm about to migrate the database schema" should feel different from "I'm about to add a button." Today they look the same in the agent's confirmation flow.

**Solution.**
- Every action gets a reversibility annotation:
  - `full` — undo by file revert / git revert
  - `partial` — undo possible but data may be lost (e.g. dropped column with backfill)
  - `irreversible` — once done, it's done (publish to app store; send email to all users; spin up paid resource)
- High-stakes actions (`partial` / `irreversible`) require explicit founder confirmation in Telegram thread.
- Audit log captures reversibility tag per action.
- "Roll back to last green" primitive (cross-ref [10 sandboxing](#h-sandboxing--reset-to-green) below).

### D. Cost transparency

**Gap.** Founder doesn't know mission cost until invoice. Tokens (LLM
calls), infra (vendor adapters), human-time (own + counsel), opportunity
(months of work) — all opaque.

**Solution.**
- Per-mission real-time cost gauge: tokens × model-rate, vendor API costs, projected-cost-to-completion.
- "This feature will cost ~$X to build" estimates upfront.
- Cost surfaces at decision points: "this multi-pass review iteration will cost ~$2; continue?"
- Per-mission budget ceiling; threshold alerts at 50% / 75% / 90%.
- Quick-vs-thorough mode dial trades cost for quality; dial visible in mission setup.
- Long-tail costs surface (subscription that auto-renews; storage that grows monthly).

### E. Provenance

**Gap.** "Where did this code come from? Which model? Which iteration?
Why?" — currently opaque. Audit trail exists in DB but isn't queryable
from the artifact end.

**Solution.**
- Every artifact tagged with the chain that produced it:
  - Source step ID
  - Model + iteration count + retry count
  - Decisions referenced (ADRs, lessons applied)
  - Reviewers + verdicts
  - Founder approvals (if any)
- Queryable: "show me the provenance of `backend/services/billing.py`" returns the full chain.
- Useful for incident response ("this bug shipped from mission 47, model X, after 3 retries — model X has been having issues; rollback").

### F. Cross-mission memory (re-emphasized — its home is [02-build-foundation.md](02-build-foundation.md) but it's cross-cutting)

- Memory schema (`mission_lessons`) maintained in 02.
- Cross-cutting use: every other zone reads from + writes to it.
- Founder profile + product profile persist across missions; same founder's second mission inherits voice / brand / stack preferences.
- Memory pruning policy: TTL + occurrences-weighted retention (cross-ref 02 open questions).

### G. Per-mission Telegram thread

- Persistent thread per mission_id (cross-ref [07-humanish-layers.md](07-humanish-layers.md) for support patterns).
- Posts: `[milestone]`, `[blocker]`, `[asking]`, `[confirmation_required]`, `[cost_alert]`.
- Founder reactions become typed events (`approve`, `reject`, `comment`).
- Comments → revision tasks against the relevant artifact.
- request_review action surfaces here.

### H. Sandboxing + reset-to-green

- **Per-mission container.** Mission writes go through Docker container with mounted workspace. Lets us drop safety rails on `shell` without risk to host. Default Docker; firecracker / nsjail later if perf needs.
- **Reset-to-green primitive.** Every commit-after-green is known-good restore point. `/rollback_mission <id>` returns to last green commit + state.
- **Resource limits per mission.** CPU / memory / disk caps prevent runaway mission consuming the whole host.
- **Network policy.** Egress limited to whitelisted vendor APIs by default; broader requires founder approval.

### I. End-of-mission demo deliverable

- Final playwright run with `--video on` capturing the core flow.
- 30s MP4 attached to mission deliverable.
- Forces "running demo" to be a real exit criterion, not a checkbox.
- Useful for founder review + investor updates + marketing.
- Mobile equivalent (cross-ref [05-build-mobile-track.md](05-build-mobile-track.md)) records device-screen video.

## Founder territory

- Trust calibration: founder ultimately decides which agent decisions to trust how much. Confidence-and-reasoning helps, but final call is taste.
- Cost cap setting: founder sets ceiling; agent stays under.
- Reversibility judgment: agent can tag, but founder makes the irreversible call.
- Time pressure: founder owns the launch date.

## Proposed direction

Sequencing rationale: A first (everything else needs the schema + audit hook); B next (machinery is half-built — finish surfacing); C/D in parallel (independent); E grows out of existing Docker shell; F/G ride on top.

### Phase A — Reversibility + provenance + plug confidence gate (foundational; ship early)

**Effort.** ~4-5 days (raised from 3-4: confidence-gate wiring + file_locks fix + DDL migration ledger added).

**Tasks.**
1. **Verb registry as code.** New `packages/mr_roboto/src/mr_roboto/reversibility.py` with the taxonomy table above as a dict; `Action` (`actions.py:23`) gains `reversibility` field, populated from registry on dispatch. Shell `run_cmd` accepts override from caller's intent.
2. **Plug `min_confidence`.** In coulson step runner, after the LLM emits a result with `confidence`, compare against agent's `min_confidence` (`base.py:85`); fail-closed → reviewer escalation. Currently cosmetic — make it real before adding more confidence fields.
3. **Schema additions.** `tasks` gains `confidence_categorical` (`low|med|high`), `confidence_numeric REAL`, `reasoning TEXT`, `reversibility` (`full|partial|irreversible`). New `artifact_provenance(path TEXT, task_id, step_id, model_id, retry_n, reviewer_verdict_id, written_at)` (single row per write event; multiple rows per file = full chain). Indexed on `path`.
4. **Audit extension.** Extend `registry_events` (`db.py:847`) with `reversibility`, `mission_id`, `task_id`, `verb` columns + new scope `action`. Lean: extend not split (open question resolved — see below); split if query patterns diverge.
5. **DDL migration ledger.** New `schema_migrations(version, applied_at, sql, reversal_sql)` table; the existing `goals → missions` rename (`db.py:244`) is the canary — wrap all future DDL in `BEGIN/COMMIT` with ledger entry. Provides reversal_sql for partial undos.
6. **Fix `file_locks` orphans** (`db.py:483-493`): add `expires_at` + sweeper job (every 60s) releasing locks held by tasks in non-running states. Closes a real reversibility hole.
7. **Atomic `git_commit` ↔ `git_push`.** mr_roboto `git_commit.py`: fail the action if push fails after local commit succeeds; provide `--rollback-local` flag to undo the orphan via `git reset HEAD~1`.
8. **Provenance query API.** `db.get_artifact_provenance(path: str) -> list[ProvenanceRow]` joins `artifact_provenance` → `tasks` → `model_call_tokens` → `reviewer_verdicts`.
9. **Confirmation flow.** In coulson dispatcher, before dispatching `partial`/`irreversible`, post `[confirmation_required]` to mission thread (Phase G) and block on founder reaction.

**Acceptance.**
- `min_confidence` gate provably escalates a low-confidence Coder output (test: synthetic confidence=1, min=3 → reviewer task created).
- `db.get_artifact_provenance("src/agents/base.py")` returns ≥1 row with model + retry + reviewer.
- A scripted irreversible action (`notify_user` external) blocks until a 👍 reaction lands.
- `file_locks` sweeper releases ≥1 orphan in a fault-injection test.
- DDL migration ledger has ≥1 row after first run; rollback dry-run produces matching `reversal_sql`.

### Phase B — Cost: wire mission scope + iteration aggregation (mostly query work)

**Effort.** ~2 days (revised down: machinery exists, fix is plumbing).

**Tasks.**
1. **Wire mission scope into `cost_budgets`.** Today `scope`/`scope_id` are unused free text. Standardize: every mission row gets a `cost_budgets` row with `scope='mission', scope_id=<mission_id>`. Existing `get_mission_total_cost()` (`db.py:2949`) becomes the read-side; writes happen via existing token-accounting hook.
2. **Aggregate `iteration_n`.** Already captured per call in `model_call_tokens` (`db.py:428-447`). New view: `cost_by_iteration(mission_id, iteration_n, prompt_tokens, completion_tokens, cost_usd)`. Surfaces retry-tax (currently invisible).
3. **Per-mission gauge in Telegram.** `/mission_cost <id>` + auto-post on `[milestone]`. Show first-pass vs retry split.
4. **Budget ceiling + threshold alerts.** `missions.budget_ceiling_usd`; Beckman scheduled job checks 50/75/90% breaches → `[cost_alert]` (Phase G).
5. **Upfront per-step estimate.** Compute from model's per-1K rate × historical avg tokens for that step kind (`model_pick_log` × `model_call_tokens`). Store on `tasks.estimated_cost_usd`. Compare with actual on completion → calibration data for future estimates.
6. **Cost-at-decision.** Multi-pass actions estimated `>$1` post `[asking]` before execution.
7. **Quick-vs-thorough dial.** `missions.quality_mode` (`quick|balanced|thorough`) wires to retry caps (coulson) + reviewer rounds (03) + fatih_hoca scoring weights. Default `balanced`.
8. **Vendor API cost.** 06's adapters expose `record_cost(mission_id, vendor, usd, line_item)` → same `cost_budgets` rows.

**Acceptance.**
- `/mission_cost 47` returns `tokens: $X (first-pass $A, retries $B), vendor: $Y, total: $Z` with all four populated from real data.
- Mission with `budget_ceiling_usd=10` triggers alerts at $5/$7.50/$9 in test.
- `cost_by_iteration` view returns ≥1 row showing retry-tax > 0 for any past mission.
- `quality_mode=quick` measurably reduces retry+reviewer rounds in `model_pick_log` deltas.

### Phase C — Time awareness

**Effort.** ~2 days.

**Tasks.**
1. Schema: `missions.target_launch DATE`, `missions.time_budget_hours`, `phases.time_budget_hours` (in workflow JSON).
2. Pacing computation: total elapsed = sum(`tasks.completed_at - started_at`); remaining = sum of unfinished phase budgets; burndown computed daily.
3. Pacing dashboard surfaces in `/mission <id>` Telegram command + the per-mission thread (Phase G).
4. Tradeoff prompt: at >75% time-budget burn with >25% scope remaining, post `[asking]` listing remaining features ranked by founder preference + suggesting 30% to cut.
5. Schedule integration: long-running maintenance (08) shows up as time consumed against budget.

**Acceptance.**
- `/mission 47` shows "elapsed: 18h / 40h budget; on pace for 2026-05-22 vs target 2026-05-20."
- A simulated mission with depleted budget posts the cut-prompt with concrete features.

### Phase D — Telegram thread (per-mission)

**Effort.** ~3 days (Telegram topics + state plumbing).

**Tasks.**
1. Use Telegram **forum topics** (one chat, one topic per mission_id). On mission create, `bot.create_forum_topic(name=f"Mission {id}: {title}")`; persist `missions.telegram_thread_id`.
2. Typed event API in `telegram_bot.py`: `post_event(mission_id, kind, payload)` with kinds `milestone|blocker|asking|confirmation_required|cost_alert`. All Phase A/B/C posts route through this.
3. Reaction handler: 👍 = `approve`, 👎 = `reject`, ✏️ comment-reply = `comment` → spawned revision task on referenced artifact.
4. `request_review` action (cross-ref 03) posts in-thread + blocks task until verdict reaction.
5. Backfill: replace ad-hoc `pending_clarifications` dict (`telegram_bot.py:33`) usage with the typed-event flow where it overlaps.

**Acceptance.**
- Mission 47 has its own topic; all events for it land there.
- Founder reaction translates to typed event row in DB.
- Comment-reply on an artifact post creates a revision task linked to that artifact.

### Phase E — Sandboxing + concurrency safety (HIGH-RISK, HIGH-PAYOFF)

**Effort.** ~5-6 days (raised: cross-mission state isolation is non-trivial).

**Tasks.**
1. **Per-mission container.** Extend `src/tools/shell.py:20-33` — `CONTAINER_NAME` becomes `kutai-mission-{id}`; lazy-create on first shell call per mission; teardown on mission complete. Resource caps `--memory`, `--cpus`, `--pids-limit` from per-mission config.
2. **Kill `SANDBOX_MODE=local` silently.** Today `local` mode bypasses Docker with no audit. Either: (a) require explicit founder opt-in per-mission with `[confirmation_required]`; or (b) deprecate. Lean (a) for dev ergonomics. All `_run_quiet` (`shell.py:125-133`) log lines must include `sandbox_mode`.
3. **Tighten `BLOCKED_PATTERNS`** (`shell.py:60-77`) — currently regexes fixed strings. Add semantic guard: parse argv, check argv[0] against allow/deny list (`dd`, `mkfs`, `nc -l`, etc.). Belt-and-suspenders against argv-construction tricks like `["dd", "if=/dev/sda", ...]`.
4. **Egress whitelist.** `--network` per-mission bridge; iptables drops outbound except vendor-adapter hostnames (06 owns list). Broader egress → `[asking]` approval.
5. **Cross-mission DB safety.** Today `_tx_lock = asyncio.Lock()` in `db.py:40` is **global writer** — Mission A's INSERT blocks Mission B's `add_task`. Two routes: (a) shard `_tx_lock` per-mission (most writes are mission-scoped); (b) move per-mission write-heavy tables to per-mission attached SQLite files. Lean (a) v1; (b) if WAL contention persists.
6. **Per-mission Chroma namespace.** `chroma_data` collection-per-mission — `mission_{id}_<collection>`. Prevents embedding-index interleave between concurrent missions.
7. **Reset-to-green primitive.** On every green CI run, tag `green-{mission_id}-{task_id}` in workspace git **and** dump mission-scoped DB rows + Chroma collection snapshot to `data/snapshots/mission-{id}-{task_id}/`. `/rollback_mission <id>` restores all three.
8. **Bridge with Phase A migration ledger.** Reset-to-green also rewinds the `schema_migrations` ledger if rollback crosses a DDL boundary.

**Acceptance.**
- Two missions running concurrently each have own container + Chroma namespace; `add_task` latency for Mission B is unaffected by Mission A's heavy writes (load test).
- `/rollback_mission 47` restores workspace + mission-scoped DB rows + Chroma collection.
- `SANDBOX_MODE=local` invocation without explicit per-mission opt-in raises `[confirmation_required]`.
- Argv-injection attempt (`["dd", "if=/dev/sda", ...]`) is rejected by tightened `BLOCKED_PATTERNS`.
- Egress to a non-whitelisted hostname is dropped at the bridge layer.

### Phase F — Demo deliverable

**Effort.** ~1-2 days.

**Tasks.**
1. New mr_roboto verb: `record_demo(scenario_path)` runs `playwright test --headed --video on` against `scenario_path`, then `ffmpeg -t 90 -i input.webm -c:v libx264 demo.mp4`.
2. i2p workflow gains a final phase step: `record_demo` with `produces: demo.mp4`, gated on green tests.
3. Mission deliverable bundle: `demo.mp4` + final commit hash + provenance summary, posted to mission thread on completion.
4. Mobile equivalent (cross-ref 05): `record_demo_mobile(scenario)` uses Maestro/Detox screen-record verb.

**Acceptance.**
- A toy mission completes with `demo.mp4` attached to its Telegram thread.
- Demo recording failure blocks mission completion (forces it to be real, not a checkbox).

### Phase G — Trust calibration loop

**Effort.** ~2-3 days (mostly analytics; no UI).

**Tasks.**
1. Outcome attribution: when a step's downstream produces a green review or a regression, write back `outcome_correct: bool` linked to the original confidence claim (extend `model_pick_log` or new `confidence_outcomes` table).
2. Per-domain reliability score: nightly job computes `P(correct | confidence=high, domain=X)` per domain (domain = task_kind tag).
3. Prompt-builder feedback: `requirements_builder` (now in `packages/fatih_hoca/`) injects a calibration line for picked model+domain (e.g. "your high-confidence picks in `db_schema` correlate 0.4 with downstream success — be more conservative").
4. `/calibration` Telegram command surfaces the matrix.

**Acceptance.**
- `/calibration` returns a per-(model, domain) reliability table after ≥30 outcomes.
- Generated prompts include the calibration line for the active task.

## Human-in-loop pattern

| Concern | Agent surfaces | Founder decides | Reversibility tag |
|---|---|---|---|
| Confidence | low-confidence decisions for review | trust / override | full |
| Time | budget burn, scope-cut suggestions | which features to cut | full |
| Cost | per-decision estimate, ceiling alerts | quick-vs-thorough, budget cap | full |
| Reversibility | irreversible-action confirmation | sign-off | by definition |
| Provenance | full chain on request | n/a (read-only) | n/a |
| Sandbox | resource alerts | container reconfig | full |
| Demo | drafts video | re-records / approves | full pre-publish |

## Dependencies

- **Inbound:** every other zone's actions need to honor these patterns.
- **Outbound:** every other zone reads these patterns at design time.
- **Especially tight coupling with:** [02-build-foundation.md](02-build-foundation.md) (memory + cost), [06-real-world-bridge.md](06-real-world-bridge.md) (cost + reversibility on real-world actions), [08-operations.md](08-operations.md) (audit + reversibility on on-call actions).
- **Z0 inputs consumed (assumed once z0 lands):** founder profile (cost ceiling, idle-vs-confirm policy, irreversible-action thresholds), vendor vault (Phase E egress whitelist seed), north-star metric (Phase C pacing dashboard reference).

## Contradictions to surface (for 00-README + zone owners)

- **Reversibility tag on agent verbs vs founder approval flow.** Phase A says irreversible blocks on confirmation, but 08 (on-call) implies the agent must act fast on incidents. Resolution: 08's incident actions need an `incident_override` flag that downgrades the confirmation requirement — owner: 08 author.
- **Per-mission Telegram topic vs single-bot dispatch.** Phase D forum topics need careful keyboard handling — `_reply()` helper in `telegram_bot.py` defaults to a global REPLY_KEYBOARD that should adapt per-thread. Owner: 07 + this doc.
- **Sandbox per mission vs current single shared sandbox.** Existing `src/tools/shell.py` assumes one container; agents reaching across missions (e.g. cross-mission memory writes) need a path that bypasses the sandbox or uses a "shared services" container. Owner: 02 + this doc.
- **Cost dial vs Fatih Hoca scoring.** Phase B's `quality_mode` modifies fatih_hoca weights; that's an extension of the Phase 2d utilization equation. Owner: this doc proposes; fatih_hoca owner ratifies.

## Open questions

- **Confidence-field schema.** Numeric (0-1) or categorical (low/med/high)? **A:** Both — categorical surface, numeric storage. (See Phase A task 1.)
- **Reversibility taxonomy.** 3 buckets (full/partial/irreversible) or finer? **A:** 3 v1.
- **Cost surfacing frequency.** Per-action vs per-step vs per-phase? **A:** Per-step normally; per-action for `>$1`.
- **Telegram thread vs separate chat.** Topics within one chat or per-mission separate? **A:** Topics — keeps founder in one place.
- **Container runtime.** Docker vs firecracker vs nsjail? **A:** Docker v1; revisit if boot >5s becomes a bottleneck.
- **Provenance storage.** Inline in artifact metadata vs separate table? **A:** Separate `artifact_provenance` table; artifact references provenance_id.
- **Audit-log shape.** Extend `registry_events` (one append-only log) or split per-action `action_audit`? Open — extending keeps query surface small but mixes scopes; splitting cleaner but adds a join. Lean: extend with scope discriminator, revisit if query patterns diverge.
- **Demo recording length.** Fixed 30s vs variable? **A:** Variable, capped 90s.
- **Trust-calibration loop scope.** Per-mission vs cross-mission? **A:** Cross-mission — sample size matters.
- **Reset-to-green snapshot scope.** Just workspace git, or include DB rows for the mission? Lean: both, paired by green-tag (Phase E task 5).

## Agent task brief

When picking up this doc:
1. Read 00-README + every other zone doc (skim — this doc affects all).
2. Phase A: confidence + reversibility schema + audit-log extension + provenance query.
3. Phase B: cost gauge + budget cap + quick-vs-thorough dial.
4. Phase C: time-awareness fields + pacing dashboard.
5. Phase D: Telegram thread + typed event flow.
6. Phase E: sandboxing template + reset-to-green primitive.
7. Phase F: demo recording verb + mission integration.
8. Phase G: trust-calibration loop scaffolding.
9. Surface any contradictions with other zone docs back into 00-README.
10. Add `## Updates` entry.

## Updates

- 2026-05-08 — initial doc; absorbs cross-cutting concerns from 2026-05-08 round + Wave 6 (sandboxing + demo) from `docs/plans/2026-05-07-i2p-capability-expansion.md`.
- 2026-05-08 — refinement pass: added Current state table with codebase evidence (file:line refs across 10 axes); concretized Phase A-G with effort estimates, task lists, and acceptance criteria; surfaced 4 cross-zone contradictions; resolved 7 open questions, opened 2 new ones (audit-log shape, reset-to-green snapshot scope). Z0 inputs flagged in Dependencies (assumes z0 lands first; tighten when z0 doc settles). Z1 still in flight — Z1 outputs (executable spec format) feed Phase A artifact schema; revisit confidence/reasoning fields once Z1 spec shape is final.
- 2026-05-08 — deep-dive pass: replaced field-level "what's missing" with operational reality. Severity-ranked gaps (Reversibility/Sandbox/Provenance HIGH; Cost/Confidence/Demo/Telegram/Concurrency MED; Time/Calibration LOW). Drafted concrete reversibility taxonomy across all mr_roboto verbs + agent tools. Surfaced 3 real reversibility incidents (DDL rename, file_locks orphans, silent git_push). Documented `min_confidence` is unplugged (cosmetic), `cost_budgets.scope` is phantom (free text never set), provenance join breaks at artifact↔task (Step 4 of join missing). Found `i2p_v3` step 15.10 = manual sign-off, 15.14 = roadmap_update — no automated "product works" check. Documented cross-mission interference: global `_tx_lock`, shared `chroma_data`, single sandbox `CONTAINER_NAME`. Phase A raised 3-4d → 4-5d (added: confidence-gate plug, file_locks sweeper, atomic commit↔push, DDL migration ledger). Phase B lowered 2-3d → 2d (mostly plumbing). Phase E raised 3-4d → 5-6d (added: per-mission tx_lock shard, Chroma namespace, kill silent local mode, semantic argv guard).
