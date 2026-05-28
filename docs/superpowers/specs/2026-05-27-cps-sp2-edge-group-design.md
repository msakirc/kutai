# Design — CPS SP2: Edge group migration

**Date:** 2026-05-27
**Status:** ready for planning
**Owner:** founder + agent
**Parent spec:** `docs/superpowers/specs/2026-05-27-cps-migration-design.md` (rev. 2, approved)
**Substrate:** SP1 (landed `main`, commits ef2df712..cbb44b04)
**Inventory:** `docs/superpowers/specs/2026-05-27-cps-migration-call-site-inventory.md` (rows #1–#6)

---

## Problem

SP1 landed a durable continuation substrate (`enqueue(spec, on_complete=..., on_error=..., cont_state=...)` → atomic `continuations` row → claim-then-fire on terminal child → restart-reconcile from `tasks.result`). The substrate is unexercised in production: only the two pre-existing `on_complete` callers (`analytics_digest`, `classify_signals`) drive it, both of which are mr_roboto-internal. SP2 migrates the six **edge-group** `await_inline=True` sites — telegram + jobs — off the synchronous primitive. Edge sites hold no cap-counted lane slot, so this is the lowest-risk SP and the right place to validate the substrate against real production traffic before SP3 touches the deadlock surface.

SP2 also closes a latent bug discovered during the call-site re-audit (see §Per-site design, row #3): `src/app/interview.py` reads `task_result.output`, a field that does not exist on `TaskResult` (it has only `status`, `result`, `error`). The current code silently writes empty interview summaries on every run. The CPS migration fixes this by routing through the documented `result["result"]["content"]` shape.

---

## Scope guard

**In scope (exactly these 6 sites — re-audited 2026-05-27 against `main`):**

| # | Inventory row | Symbol | File | Current line |
|---|---|---|---|---|
| 1 | row #1 | `_enqueue_inline_chat` (called by `_classify_user_message` and `_handle_casual`) | `src/app/telegram_bot.py` | helper at 92; callers at 7377 (`_classify_user_message`) and 7907 (`_handle_casual`) |
| 2 | row #2 | `_enqueue_inline_classifier` (called by `_classify_with_llm`) | `src/core/task_classifier.py` | helper at 259; caller at 305 inside `_classify_with_llm` (currently line 296+) |
| 3 | row #3 | `summarize_interview` | `src/app/interview.py` | `await_inline=True` at 260 |
| 4 | row #4 | `_call_llm_meeting_brief` | `src/app/meetings.py` | `await_inline=True` at 398 |
| 5 | row #5 | `_llm_cluster_draft` | `src/app/jobs/faq_regen.py` | `await_inline=True` at 159 |
| 6 | row #6 | `_call_llm_anomaly_hypothesis` (via `_enqueue_overhead` wrapper) | `src/app/jobs/investor_bullets.py` | `await_inline=True` at 204 — **DEFERRED to SP5+** per founder decision 2026-05-27; see §Site 6 deferral below |

**Out of scope (deferred):**
- SP3 sites — `grading.py`, `code_review.py`, `workflows/engine/hooks.py` (`_llm_summarize`), `llm_dispatcher.request` shim. These hold cap-counted lane slots and are where the actual deadlock lives.
- SP4 sites — `tools/vision.py`, all `mr_roboto/{crisis_draft_holding,demo_storyboard,incident_draft_update,press_kit_assemble,reviews_{classify,draft_reply}}.py`, `yalayut/discovery/synthesize.py`, both posthook handlers.
- SP5 — deletion of `await_inline` / `resolve_inline` / `_inline_waiters` / `INLINE_TIMEOUT`. The primitive **stays fully intact** through SP2; SP2 only stops *calling* it from the six sites listed.

**Non-negotiable project rules baked in:**
- All test invocations use `timeout 60 pytest tests/... -v` (zombie pytest crash-loops KutAI).
- SQLite datetime is `strftime('%Y-%m-%d %H:%M:%S')`, never `isoformat()`.
- `rtk` prefix on git commands.
- Lazy cross-module imports inside functions to avoid circulars.
- `[[feedback_singular_dispatcher_caller]]` — only Beckman calls `LLMDispatcher.request`. SP2 introduces no new caller.
- `[[feedback_verify_verdict_roundtrip]]` — wiring is only complete when the resume's output reaches the destination; each site's test must drive the full enqueue → terminal-fire → side-effect path against a temp DB.
- `[[feedback_audit_call_sites]]` — inventory annotations are *starting points*; this spec re-audits each site against current `main`.
- `[[feedback_no_agent_modes]]` — no `use_cps` flag. Each site switches wholesale; if CPS is wrong it gets reverted, not toggled.
- `[[feedback_zero_traffic_not_dead]]` — fallbacks (keyword classifier, JSON-regex rescue, `return ""`, `return None`, `return [],[]`) are **proven** as natural `on_error` handlers, not deleted blindly.

---

## The keystone: result-delivery pattern for CPS edge sites

Every site in SP2 today follows the same synchronous shape:

```python
# 1. enqueue + block
tr = await enqueue(spec, await_inline=True)
# 2. check status, fall back on failure
if tr.status != "completed": return <fallback>
# 3. extract content, parse
content = tr.result.get("content")  # (or .output, .raw — see §Result-field shape)
data = parse(content)
# 4. perform side-effect (DB write OR Telegram reply OR return-to-caller)
deliver(data)
# 5. return value to caller (or None)
return data
```

The CPS inversion replaces this with:

```python
# 1. enqueue + return (no block, no lane slot held)
child_id = await enqueue(
    spec,
    on_complete="<module>.<resume_name>",
    on_error="<module>.<on_error_name>",   # optional; lift step-2 fallback here
    cont_state={"<everything needed in step 4>": ...},
)
# Caller returns immediately; no `tr` to inspect.

# Elsewhere, registered at import time:
async def <resume_name>(child_task_id: int, result: dict, state: dict) -> None:
    # result has shape {"status": "completed", "result": <agent_output>, "error": None, ...}
    # For raw_dispatch LLM calls, <agent_output> is the dispatcher response dict
    # with "content"/"model"/"usage"/etc.
    agent_output = (result or {}).get("result") or {}
    content = agent_output.get("content", "") if isinstance(agent_output, dict) else ""
    data = parse(content)
    deliver(data, **state)  # state carries the delivery target

async def <on_error_name>(child_task_id: int, result: dict, state: dict) -> None:
    # Same fallback the synchronous code did on tr.status != "completed".
    deliver_fallback(**state)
```

**Resume / on_error co-location.** Each site's resume + on_error live in the **same module as the original synchronous code**. Reason: the resume needs to call the same internal helpers (e.g. `_parse_brief_llm_response`, `_emit_faq_founder_action`, `compose_brief_md`) and reads/writes the same DB tables. Co-location preserves the read locality reviewers depend on, avoids forcing those helpers into module-level cycles, and makes each migration a single-file diff plus one line in `_HANDLER_MODULES`.

**Naming convention.** `<module_prefix>.<resume_action>` / `<module_prefix>.<resume_action>_err`. The `<module_prefix>` matches the existing convention in `mr_roboto.executors.analytics_digest` (`growth.store_weekly_digest`). Concrete names: §Per-site design.

**Registration.** Each migrated module exports `def register_continuations() -> None` and calls it at import time (mirroring the SP1 update to `analytics_digest.py`). Its name is added to `_HANDLER_MODULES` in `packages/general_beckman/src/general_beckman/continuations.py` — without this, restart-reconcile cannot find the handler and silently drops the continuation (the exact gotcha SP1 §Restart called out).

---

## Result-field shape (re-audit finding — load-bearing)

`TaskResult` (`packages/general_beckman/src/general_beckman/__init__.py:69`) has exactly three fields: `status`, `result`, `error`. **There is no `.output` field.** Three sites confused themselves about this:

- `meetings.py:411` — reads `task_result.result["content"]`. **Correct.**
- `investor_bullets.py:222` — reads `task_result.result["content"]`. **Correct.**
- `faq_regen.py:171-173` — reads `task_result.result["content"]` (with `.text` / `.response` fallbacks). **Correct.**
- `interview.py:265` — reads `getattr(task_result, "output", None) or ""`. **Always returns `""`.** Summary, quotes, insights, action_items are always empty. Latent bug; CPS migration fixes it incidentally by routing through `result["result"]["content"]`.
- `telegram_bot._enqueue_inline_chat` (line 124) and `task_classifier._enqueue_inline_classifier` (line 287) — both helpers unwrap `tr.result` themselves and `json.loads` strings before returning; callers read `response.get("content")`. After CPS migration, the helpers themselves are deleted; the resumes read `result["result"]["content"]` directly.

**The resume-handler contract.** The `result` arg passed to the resume is the full envelope `on_task_finished` saw — `{"status": ..., "result": <agent_output>, "error": ..., "cost_usd": ...}`. The inner `<agent_output>` for `raw_dispatch` LLM calls is the dispatcher response dict (`llm_dispatcher.py:949+`: `{"content", "model", "model_name", "cost", "usage", "tool_calls", "latency", "thinking", "is_local"}`). Resumes must defensively unwrap because:
- On restart-reconcile, `result` is reconstructed from `tasks.result` (a JSON string) — SP1 reconcile parses it into a dict; if parsing fails the substrate sets `{"result": <raw_string>}`. Resumes must tolerate `result["result"]` being either a dict or a string.
- For non-`raw_dispatch` agents (e.g. `agent_type=summarizer` in interview), the inner result shape depends on the agent's executor. Each site's resume defends with `isinstance(agent_output, dict)`.

---

## Per-site design

Column legend:
- **resume name** — registered with `general_beckman.continuations.register_resume`, also passed as `on_complete`.
- **on_error name** — same registry; lifts the current fallback branch verbatim.
- **cont_state** — minimum dict serialized into `continuations.state_json`; everything the side-effect step needs.
- **Resume action** — what the resume does after parsing `result["result"]["content"]`.
- **Caller-return note** — what the synchronous-style caller now sees in place of the awaited value.

| # | Site | resume name | on_error name | cont_state keys | Resume action | on_error action | Caller-return note |
|---|---|---|---|---|---|---|---|
| 1a | `telegram_bot._classify_user_message` (caller at 7176, in `handle_message`) | `telegram.message_route_resume` | `telegram.message_route_err` | `chat_id`, `text`, `_pending_action` snapshot (for race idempotence — see §Idempotence), and a `flow="message_route"` discriminator | Parse classification JSON → call the **same routing block** currently at `telegram_bot.py:7185-7254` (refactored into `_route_classified_message(chat_id, text, classification)`) which performs `add_task`, `_handle_user_input`, `_handle_status_query`, `_handle_casual`, etc. Reply via `bot.app.bot.send_message(chat_id, ...)` — **NOT** `self._reply(update, ...)` (the `update` object is dead by the time the resume fires). | Fall back to `_classify_message_by_keywords(text)` and route via the same `_route_classified_message`. | `handle_message` no longer awaits a classification — it acknowledges receipt (no reply, or a "thinking..." typing-indicator) and returns. The user sees the eventual reply when the resume fires. |
| 1b | `telegram_bot._handle_casual` (caller at 7227, in `handle_message`) | `telegram.casual_reply_resume` | `telegram.casual_reply_err` | `chat_id`, original `text` (for logging) | Extract `result["result"]["content"]`, send via `bot.app.bot.send_message(chat_id, content[:1000])`. | Send `"Hey! Send me a task or mission to work on."` via `bot.app.bot.send_message(chat_id, ...)`. | `_handle_casual` itself becomes the enqueue-and-return shell; the reply is asynchronous. |
| 1c | `cmd_mission._classify_user_message` (caller at 2204) | (reuses 1a) | (reuses 1a) | adds `flow="mission_route"`, includes `description` | If classification → `workflow=="i2p"`, the resume creates a workflow mission; else it creates a plain mission via `add_mission` then prints the link. | Treat as no-workflow: create a plain mission. | Same as 1a — `cmd_mission` returns immediately after enqueueing. |
| 2 | `task_classifier._classify_with_llm` (caller at 305 inside; entry `classify_task` at 238) | **n/a — see §Site 2 special case below** | n/a | n/a | n/a | n/a | This site **cannot be CPS-migrated wholesale** without changing `classify_task`'s synchronous contract. See §Site 2. |
| 3 | `interview.summarize_interview:260` | `interview.summary_persist_resume` | `interview.summary_persist_err` | `note_id`, `product_id` | Parse JSON from `result["result"]["content"]` (apply the existing regex-rescue), compute `bullets / quotes / insights / action_items`, write the same `UPDATE interview_notes SET summary_md=?, quotes_json=?, insights_md=?, action_items_json=? WHERE note_id=?` already at line 290. | Log warning; leave `summary_md` NULL (matches current behaviour when LLM returns junk — see §Site 3 caller pattern). | `summarize_interview` returns `{"ok": True, "queued": True, "note_id": note_id}` immediately. mr_roboto sees it as completed; the actual summary lands later. **See §Site 3 caller pattern for the mr_roboto handshake.** |
| 4 | `meetings._call_llm_meeting_brief:398` | `meetings.brief_persist_resume` | `meetings.brief_persist_err` | `meeting_id`, `product_id`, `ctx` (only the fields `compose_brief_md` reads — `contact`, `meeting`, `interactions`, `follow_ups`, `changelog`, `mentions`) | Parse content via `_parse_brief_llm_response`; build `brief_md` via `compose_brief_md(ctx, talking_points, suggested_asks)`; `UPDATE meetings SET brief_md=?, brief_generated_at=strftime(...) WHERE meeting_id=?`; surface as Telegram notify via `get_telegram().app.bot.send_message`. | Build `brief_md` via `compose_brief_md(ctx, llm_unavailable=True)`; same UPDATE; same Telegram notify. | `_call_llm_meeting_brief` is renamed to `enqueue_meeting_brief(ctx, meeting_id, product_id) -> int` (returns child task id). The mr_roboto `meetings/brief` action (in `packages/mr_roboto/src/mr_roboto/__init__.py:3955-…`) is restructured to enqueue + return `Action(status="completed", result={"queued": True, "meeting_id": meeting_id})` rather than awaiting the brief. **See §Site 4 caller pattern.** |
| 5 | `faq_regen._llm_cluster_draft:159` | `faq_regen.draft_persist_resume` | `faq_regen.draft_persist_err` | `lang`, `cluster_size`, `mission_id` (always `0` in current code, kept for forward-compat) | Parse the JSON object from `result["result"]["content"]` (with `text` / `response` fallbacks); attach `lang` to the dict; call `_emit_faq_founder_action(mission_id=0, entry=draft, cluster_size=cluster_size)`. | Log warning + skip — matches the existing `return None` fallback. | `_draft_faq_entry` (the only caller of `_llm_cluster_draft`) → `_llm_cluster_draft` becomes `enqueue_cluster_draft(cluster, lang) -> int`. `_draft_faq_entry` returns `None` (queued). `run_faq_regen` (line 326) loses its per-language `total_drafts` accounting — see §Site 5 caller pattern. |
| 6 | `investor_bullets._call_llm_anomaly_hypothesis:204` | **n/a — DEFERRED to SP5+** | n/a | n/a | n/a | n/a | See §Site 6 deferral below. |

---

### Site 2 special case — `task_classifier._classify_with_llm`

This site cannot be migrated to CPS *without changing the public contract of `classify_task`*. `classify_task` is called from:
- `src/infra/db.py::add_task` (the dedup-tier resolution path) — **synchronous in the request thread**;
- `general_beckman` priority/ordering — synchronous;
- `src/app/telegram_bot.py:2204` — synchronous caller flow;
- a dozen tests.

Every one of those callers consumes the returned `TaskClassification` dataclass immediately to make a routing/tiering decision. Converting any of them to "enqueue + return None, finalize later" would require redesigning the *task admission contract* — far beyond SP2.

**SP2 decision:** Site #2 stays on `await_inline=True` for now. It is an **edge site holding no lane slot** (it runs from `add_task` which is not under cap), so it does not deadlock and `await_inline` is safe to keep. SP5 deletes `await_inline` — by then either:
- (a) `_classify_with_llm` has been moved to a pre-`add_task` synchronous LLM call (no Beckman round-trip), or
- (b) `classify_task` itself becomes async-deferred (an architectural change requiring its own spec).

This is an explicit deferral, documented here so it's not lost: **Site #2 will be excluded from the SP5 deletion guard until the contract is renegotiated.** Adding the row "telegram-classifier" (site #1a — `_classify_user_message`) is **not** the same as removing `_enqueue_inline_classifier` (#2) — they share no code today.

(Inventory row #2 is downgraded from "SP2" to "SP5-or-later" with this rationale. The plan reflects the reduced site count: 5 of 6 in SP2.)

> **Open question for founder before plan execution:** Should we instead spend one extra task in SP2 to *delete the `_classify_with_llm` LLM path entirely* and rely on the keyword classifier (which already covers ~80% of incoming text in inventory analysis), eliminating site #2 by removing the LLM call, not by CPS-migrating it? This is a behaviour change but resolves the SP5 deletion-guard gap cleanly.

---

### Site 3 caller pattern — `summarize_interview`

`summarize_interview` is called from `packages/mr_roboto/src/mr_roboto/__init__.py:4130` inside the `interview/summarize` action handler, which wraps the result in `Action(status="completed", result=res)`. The mechanical executor's task is itself a Beckman task, so today it holds a cap-counted slot for the LLM call duration — but mechanical agents are cap-exempt (ea0d5b2d), so no deadlock.

After CPS migration, `summarize_interview` enqueues the LLM child and returns immediately. The mechanical action returns `Action(status="completed", result={"queued": True, "note_id": note_id, "child_task_id": child_id})`. The actual DB write happens in `interview.summary_persist_resume` when the child completes. The mr_roboto verb is **decoupled from the LLM result** — it just kicks off the work.

**Trade-off accepted:** the mr_roboto-level retry / `Action(status="failed")` semantics no longer cover LLM failure. Instead, `on_error` does its own logging + leaves `summary_md` NULL (today's behaviour on parse failure is exactly this). Surfaceable failures still flow through the child task's own DLQ.

### Site 4 caller pattern — `_call_llm_meeting_brief`

Same shape as Site 3. `meetings/brief` mechanical action enqueues the child and returns immediately. The resume composes `brief_md`, writes it, and sends the Telegram notify. **All side-effects move from the mechanical action body into the resume.**

The Telegram notify happens from the resume (no `Update` object available) via `get_telegram().app.bot.send_message(chat_id, ...)`. The `admin_chat = os.getenv("TELEGRAM_ADMIN_CHAT_ID")` lookup is identical to today.

### Site 5 caller pattern — `_llm_cluster_draft` / `run_faq_regen`

`run_faq_regen` currently loops per language and increments `total_drafts` based on the synchronous draft return. After CPS migration, `_draft_faq_entry` returns `None` immediately for queued drafts (none were ever returned anyway when LLM failed); the founder_action emission moves into the resume. `run_faq_regen` returns `{"ok": True, "drafts": 0, "queued": N}` instead of the actual draft count.

**This is a behaviour change in the return shape.** The caller is mr_roboto's `faq_regen` executor (in `packages/mr_roboto`); a grep for `run_faq_regen` shows it's not consumed beyond logging the `drafts` count, so this is safe. The plan's Task 5 includes a `rtk grep "run_faq_regen"` regression check to confirm.

### Site 6 deferral — `_call_llm_anomaly_hypothesis`

**SP2 decision (founder, 2026-05-27): defer site #6 to SP5+ alongside site #2.**

Rationale captured at decision time: the anomaly-hypothesis path is **already unreachable in production** per the 2026-05-17 Z7 unwired-features handoff — `missions.product_id` is never set by any code, so the metric fetchers (`_fetch_z6_metrics`, `_fetch_review_density`, `_fetch_support_metrics`) all JOIN against a NULL column, return `{}`, no outliers are detected, and `_call_llm_anomaly_hypothesis` never fires. The deadlock pressure is therefore zero, and the architectural cost of CPS-ing it (either ~80 LOC + a `investor_bullet_pending` schema migration for fan-in, or capping at 1 anomaly and re-architecting `run_investor_bullets` into kickoff + finalize passes) is not justified while the upstream producer is missing.

When SP5 deletes `await_inline`, site #6 will either need:
- (a) the upstream `metric_emit` producer + `missions.product_id` wiring to have shipped — making the multi-anomaly hypothesis count empirically observable, at which point fan-in (option 6A) can be designed against real data, or
- (b) the entire anomaly-hypothesis path to be removed from `run_investor_bullets` if it remains unreachable.

This is an explicit deferral, documented here so it's not lost: **Site #6 will be excluded from the SP5 deletion guard** until one of (a)/(b) is resolved, identically to site #2.

(Inventory row #6 is downgraded from "SP2" to "SP5-or-later". The plan reflects the reduced site count: **4 of 6 in SP2**.)

---

## Cross-cutting decisions

### Handler registration

Each migrated module exports a single `def register_continuations() -> None` that calls `general_beckman.continuations.register_resume(...)` for every name it owns, and invokes it at module-import time (the SP1 `analytics_digest.py` pattern). The module name is added to `_HANDLER_MODULES` in `packages/general_beckman/src/general_beckman/continuations.py`. After SP2:

```python
_HANDLER_MODULES = (
    "mr_roboto.executors.analytics_digest",
    "mr_roboto.executors.classify_signals",
    "src.app.telegram_bot",           # site #1 (a/b/c)
    "src.app.interview",              # site #3
    "src.app.meetings",               # site #4
    "src.app.jobs.faq_regen",         # site #5
    # site #6 (investor_bullets) deferred to SP5+ — see §Site 6 deferral
)
```

Restart-reconcile imports these in order. Failures are swallowed and logged (existing SP1 behaviour) — a missing dependency in one module doesn't kill the reconcile pass.

**Telegram-module gotcha.** `src.app.telegram_bot` registers continuations at import time. `register_continuations()` must NOT trigger the heavy import path that constructs `TelegramInterface` (the constructor calls `Application.builder().token(...)` which requires `TELEGRAM_BOT_TOKEN`). The registration block lives at module level near the existing module-level constants (after imports, before the class definition), and uses lazy lookups inside the resume bodies — never at registration time. The resume bodies look up `get_telegram()` lazily so the import order during reconcile is irrelevant.

### The `_pending_action` re-entry race

`telegram_bot._pending_action[chat_id]` is mutated by `handle_message`. If the user sends message A (classified slowly), then message B (mutates `_pending_action`) before message A's resume fires, the resume must NOT trample B's state. Resolution:

- `cont_state` captures a **snapshot** of the pending-action state at enqueue time (`flow`, `pending_action_id`, etc.), not a reference.
- The resume reads the state from `cont_state` but compares against current `_pending_action[chat_id]` before mutating. If the chat has moved on (`_pending_action` shape changed, or `chat_id` is gone from `user_last_task_id`), the resume only sends a reply — it does not mutate session state.
- Specifically for site #1a (`_classify_user_message` → routing): the resume always uses `bot.app.bot.send_message(chat_id, ...)` for output, never `self._reply(update)`. It only writes `self.user_last_task_id[chat_id] = task_id` if `chat_id` is still active (`chat_id in self.user_last_task_id` after the new `add_task`). This matches today's idempotence-by-accident — no observable change.

### Idempotence — "user moved on"

If `chat_id` is no longer present (user blocked the bot, chat deleted), the resume's `bot.app.bot.send_message` raises a `telegram.error.Forbidden` or similar. The resume's outer `try/except Exception` logs at `debug` and returns — no DLQ, no retry. The continuation has already been marked `fired` (SP1 claim-then-fire), so it cannot re-trigger. This matches the existing `await_inline` behaviour where a stale chat just dropped the reply silently.

### Helper de-duplication — `_send_telegram_via_resume`

Sites 1a/1b/1c, 4 (the Telegram notify), and potentially 6 all need the same primitive: "send a Telegram message from a continuation context, given only `chat_id` + `text`." The plan introduces a single helper in `telegram_bot.py`:

```python
async def _send_telegram_via_resume(chat_id: int, text: str, *, parse_mode: str | None = None) -> bool:
    """Send a Telegram message from a CPS resume context. Returns False if
    the bot is uninitialised or the chat is unreachable (silent drop —
    matches the await_inline 'reply got lost' behaviour)."""
```

Used by every resume that needs to send Telegram output. Not exported (underscore prefix). Kept in `telegram_bot.py` because it touches the singleton.

### No mode flags

Per `[[feedback_no_agent_modes]]`: this migration is wholesale per site. No `use_cps=True/False` parameter, no env var, no feature flag. If a site's CPS migration is wrong, revert the commit. The substrate (SP1) and the primitive (`await_inline`, deleted in SP5) coexist at the library level — that is the only knob.

### Fallback lift — proof of liveness

Per `[[feedback_zero_traffic_not_dead]]`: each site's on_error handler is the *same code* the synchronous fallback executes today. Concretely:

- #1a fallback today: `return self._classify_message_by_keywords(text)`. After migration: same call, in `telegram.message_route_err`. **Same code, different invoker.**
- #1b fallback today: `await self._reply(update, "Hey! Send me a task...")`. After: `await _send_telegram_via_resume(chat_id, "Hey! Send me a task...")`. **Equivalent.**
- #3 fallback today: regex JSON-rescue → empty dict → empty UPDATE. After: same regex path, same UPDATE. **No behaviour change.**
- #4 fallback today: `return [], []` → `compose_brief_md(ctx, llm_unavailable=True)` → UPDATE + Telegram notify. After: identical sequence inside `meetings.brief_persist_err`. **No behaviour change.**
- #5 fallback today: `return None` (caller skips emitting founder_action). After: `on_error` logs + skips. **No behaviour change.**
- #6 — DEFERRED, no migration in SP2.

This is the proof that no live path is being silently amputated.

---

## Testing strategy

Each site gets a host-path, DB-isolated test pair following the SP1 pattern (`tests/beckman/test_continuations_durable.py::_fresh_db`):

1. **Happy path:** enqueue via the migrated callsite → manually drive `on_task_finished(child_id, {"status": "completed", "result": {"content": "<JSON>"}})` → assert the side-effect landed (DB row, Telegram mock invoked, etc.).
2. **Failure path:** drive `on_task_finished(child_id, {"status": "failed", "error": "..."})` → assert the on_error handler ran the documented fallback.
3. **No regression:** the existing legacy `tests/app/test_telegram_classifier_enqueue.py` + `tests/app/test_telegram_casual_chat_enqueue.py` + `tests/z7/test_b4_meeting_brief.py` + `tests/z7/test_a8_faq_flywheel.py` + `tests/z7/test_a9_investor_bullets.py` + `tests/z7/test_b7_interview_pipeline.py` must be updated to assert the new shape — or kept as a contract test against `await_inline` if the site has a behaviour-flag fallback (none do here).

**Restart reconciliation (mandatory).** One end-to-end restart test on **site #5 (`faq_regen`)** — chosen because it's a pure cron with no Telegram coupling, no in-process state, and the simplest cont_state. Test pattern:
- Enqueue a draft via the migrated `_llm_cluster_draft` → confirm `continuations` row written with `status='pending'`.
- Simulate restart: clear the in-memory registry (`general_beckman.continuations._HANDLERS.clear()`), drop the DB connection singleton.
- Set the child task `status='completed'`, `result=json.dumps({"content": "{\"question\": \"Q?\", \"answer\": \"A.\"}"})` directly.
- Re-import `src.app.jobs.faq_regen` (or call `register_startup_handlers()`).
- Call `reconcile_continuations()` → assert the resume fired and `_emit_faq_founder_action` was invoked.

This is the verdict-round-trip test from `[[feedback_verify_verdict_roundtrip]]` adapted for SP2.

---

## Acceptance

- `rtk grep "await_inline=True" src/app/telegram_bot.py src/app/interview.py src/app/meetings.py src/app/jobs/faq_regen.py` returns **zero** matches. (`src/core/task_classifier.py` and `src/app/jobs/investor_bullets.py` keep their `await_inline=True` matches — see §Site 2 and §Site 6 deferrals.)
- `rtk grep "await_inline" packages/general_beckman/src/general_beckman/__init__.py` still returns matches — the primitive body is untouched.
- `_HANDLER_MODULES` contains the five new module names.
- Each migrated site exports `register_continuations()` and calls it at import time.
- All SP1 continuation tests pass (`timeout 120 pytest tests/beckman/test_continuations_durable.py tests/beckman/test_continuations.py -v`).
- All new SP2 tests pass (one happy + one failure + one restart-reconcile, ~11 tests total across the 5 sites).
- No regressions in `tests/app/` or `tests/core/` (`timeout 180 pytest tests/app/ tests/core/ tests/z7/ -v`).
- Founder receives a casual reply within ~30s of sending a casual Telegram message (the user-observable acceptance check; covered by the manual smoke test in Task 8).

---

## Out of scope (explicit deferrals)

- **SP3:** `src/core/grading.py:373`, `src/core/code_review.py:179`, `src/workflows/engine/hooks.py:84`, `src/core/llm_dispatcher.py::request` shim. These are the in-task / cap-counted sites where the deadlock lives.
- **SP4:** `src/tools/vision.py:93`, `packages/mr_roboto/src/mr_roboto/{crisis_draft_holding,demo_storyboard,incident_draft_update,press_kit_assemble,reviews_classify,reviews_draft_reply}.py`, `packages/yalayut/src/yalayut/discovery/synthesize.py`, both `posthook_handlers/` sites.
- **SP5:** deletion of `await_inline` / `resolve_inline` / `_inline_waiters` / `INLINE_TIMEOUT`. Note: SP5 must explicitly carve out site #2 (`task_classifier._enqueue_inline_classifier`) AND site #6 (`investor_bullets._call_llm_anomaly_hypothesis`) — see deferral sections above.
- **Investor-bullets anomaly-hypothesis CPS migration** (all of site #6, both 6A multi-anomaly fan-in and 6B cap-at-1 variants). Deferred unconditionally pending upstream `metric_emit` / `missions.product_id` producer work.