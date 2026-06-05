# CPS SP4a — husam-inline for legit LLM consumers

**Date:** 2026-06-05
**Parent:** `docs/superpowers/specs/2026-05-27-cps-migration-design.md` (umbrella, rev2)
**Predecessors (merged to `main`):** SP1/SP1.1, SP2, SP3, SP3b.
**Kickoff:** `docs/handoff/2026-05-30-sp4-kickoff.md`. This spec splits that kickoff's SP4 into **SP4a** (this doc) and **SP4b** (deferred — see §6).

---

## 1. Goal

Remove `await_inline=True` from the four **legitimate LLM-consumer** callers by routing each LLM hop through `husam.run(spec)` (the non-agentic single-call worker) instead of the blocking `general_beckman.enqueue(..., await_inline=True)` primitive. This is the `mr_roboto.critic_gate.produce_verdict` pattern, applied to consumers that must return a result synchronously.

After SP4a, the only `await_inline=True` users left are: the six mr_roboto LLM executors (→ **SP4b**), the SP5 carve-outs (`task_classifier`, `investor_bullets`), and the shopping-only `dispatcher.request()` shim.

### Non-goals
- The six **mr_roboto** LLM executors. Founder ruling: *mr_roboto is mechanical-only; LLM execution does not belong in it.* These need the LLM **extracted** to an admitted agent step, not a husam-inline swap. That is architectural and is deferred to **SP4b** (§6).
- Deleting the `await_inline` primitive (SP5).
- Touching the coulson ReAct loop.

## 2. Why husam-inline (not CPS) for these four

| Property | husam-inline (`husam.run`) | `await_inline=True` |
|----------|----------------------------|---------------------|
| Touches the pump | **No** — direct select→execute→map | Yes — admit → pump picks → inline waiter resolves |
| Sibling-deadlock when called from a running continuation | **Impossible** | Possible (the kickoff-flagged nested-`await_inline` hazard) |
| Returns synchronously | Yes (returns the legacy dict) | Yes (returns `TaskResult`) |
| Beckman admission (pool_pressure / in_flight / quota look-ahead) | **Bypassed** — reaches the worker un-admitted | Enforced |

The four callers all need a **synchronous** result and are **not** mr_roboto mechanicals — they are legitimate LLM consumers (a tool, two post-hook handlers, a discovery synthesizer). For them:

- **CPS is wrong/impossible** — `vision` returns a string mid-ReAct to the coulson loop; the two post-hook handlers and yalayut all need the value inline to finish their own synchronous return contract.
- **husam-inline is the sanctioned precedent** (`critic_gate.produce_verdict`, SP3b Task 8) and additionally **kills the sibling-deadlock** for the two post-hook handlers, because `husam.run` never waits on a pump slot.

**Accepted trade:** husam-inline reaches the worker un-admitted (no Beckman lifecycle / failure-tracking / quota look-ahead). This is the documented SP3b stopgap; acceptable for tool/overhead consumers. Not acceptable for mr_roboto mechanicals — hence SP4b's different treatment.

## 3. The substrate (reuse; do not reinvent)

- **`husam.run(task: dict) -> dict`** (`packages/husam/src/husam/worker.py:49`). Takes the same `{"context": {"llm_call": {...}}, "kind": ...}` spec the callers already build. Does select (`fatih_hoca.select`) → execute (dumb-pipe dispatcher) → map. Returns the **legacy response dict** `{"content", "model", "cost", "usage", "tool_calls", ...}` (`mapping.result_to_dict`).
- **husam.run RAISES on failure** (`ModelCallFailed` / `RuntimeError` / `SelectionFailure`) — it does **not** return a failed `TaskResult`. Every migration must wrap the call in `try/except` and map the raise onto the caller's existing failure fallback (the `task_result.status == "failed"` branch each caller has today).
- **husam install:** `husam` is already an `import` dependency at these sites only transitively today (vision/brand_voice/yalayut import `general_beckman`). After SP4a they `import husam` directly → the venv needs `pip install -e packages/husam` to be live. Already in the founder prerequisite list (kickoff item 1); restate in §7 "done".

## 4. Per-caller migration matrix

Each caller today: builds a `raw_dispatch` spec → `enqueue(await_inline=True)` → reads `_task_result_to_request_response(task_result).get("content")` (or `.get("result")`) → applies its own post-processing. The spec each builds is **unchanged**; only the call + result-read + failure-handling change.

| # | Caller | File:line | Today's result-read | After |
|---|--------|-----------|---------------------|-------|
| A1 | **vision tool** | `src/tools/vision.py:90` | `_task_result_to_request_response(tr).get("content")`, then `dogru_mu_samet.assess` degenerate-check | `resp = await husam.run(spec)` → `resp.get("content","")`. Keep the degenerate-check + the existing `try/except` returning `"Error analyzing image: ..."`. On husam raise → existing `except` returns the error string. Drop the `status=="failed"` branch + the `_task_result_to_request_response` import. |
| A2 | **brand_voice_lint** | `posthook_handlers/brand_voice_lint.py:411` (`_run_llm_tone_pass`) | `_task_result_to_request_response(tr).get("content")`, JSON-parse, score | `resp = await husam.run(spec)` → `resp.get("content","")`. On raise → existing `tone_pass_skipped` info finding. **This also removes the nested-`await_inline` sibling-deadlock** (handler runs inside a post-hook continuation). |
| A3 | **copy_compliance_review** | `posthook_handlers/copy_compliance_review.py:498` | builds a **plain agent task** (`agent_type:"classifier"`, `description:prompt`, **no** `context.llm_call`), `enqueue(await_inline=True, lane="oneshot")`, reads `result.raw.get("result"/"answer"/"content")` | **Heavier than A1/A2/A4**: husam.run rehydrates `context.llm_call`, which this spec lacks. Build a raw_dispatch overhead spec — `messages=[{"role":"user","content":prompt}]`, `call_category:"overhead"`, `agent_type:"classifier"`, low `difficulty` — then `resp = await husam.run(spec)` → `resp.get("content","")`. Keep the fence-strip + JSON-parse + the existing `try/except` fallback. Same nested-deadlock fix as A2. |
| A4 | **yalayut synthesize** | `yalayut/discovery/synthesize.py:105` (`_llm_synthesize`) | `result.get("result")` then `json.loads` | `resp = await husam.run(spec)` → `resp.get("content")` then `json.loads`. On raise → existing `except` returning empty manifest. Remove the now-stale `lane="oneshot"` comment about the pump. |

**Spec shape note:** all four already set `context.llm_call.raw_dispatch=True` with `messages`, `call_category`, `agent_type`, `difficulty`, token estimates — exactly what `husam.run` rehydrates. No spec changes needed beyond dropping `lane`/`parent_id`/`await_inline` kwargs (husam takes only the spec dict; `mission_id` rides top-level on the spec if present).

## 5. `_task_result_to_request_response` fate

Consumers after SP4a: `grading` (beckman) + the six SP4b mr_roboto executors still use it via the `TaskResult` path. It therefore **stays** for now. SP4a only stops vision/brand_voice_lint from importing it (copy_compliance reads `result.raw`, not this helper; yalayut reads `.get("result")`). Do **not** retire it — SP4b/SP5 own that.

## 6. SP4b (deferred — Group B, separate spec/plan)

The six **mr_roboto** LLM executors (`reviews_draft_reply`, `reviews_classify`, `crisis_draft_holding`, `incident_draft_update`, `press_kit_assemble`, `demo_storyboard`) violate the founder principle (mechanical-only). Each must be split:

- **LLM producer leaves mr_roboto** → an admitted agent step (`agent_type` already exists: `reviewer`, etc.) → pump → worker (the sanctioned admitted path).
- **mr_roboto keeps only the mechanical sink** — DB read, persist draft, enforce the "never auto-post" contract — taking the produced text as input, making **no** LLM call.
- **Orchestration:** workflow-step callers (i2p_v3 `agent:mechanical` steps + the `__init__.py` action router at ~4151–4663) become producer→confirm pairs (`depends_on` + result passing); cron callers (`reviews_poll_daily`) enqueue the producer + a CPS `on_complete` → mechanical persist.

**Why deferred, not a shortcut:** SP4b's producers run **on the pump** — the exact post-hook/admission substrate the founder prerequisite (kickoff items 1–3) has **never validated in prod** (phantom-lane bug). SP4b is *physically blocked* on that validation. SP4a's husam-inline path is **not** (no pump), so it ships and validates independently. SP4b gets its own `docs/superpowers/specs/YYYY-MM-DD-cps-sp4b-design.md` + plan, sequenced after the substrate is proven.

## 7. Testing & "done"

- Per-caller test: asserts (a) **no** `await_inline=True` in the module, (b) `husam.run` is called with the expected spec, (c) the success path returns the post-processed value (degenerate-check / JSON-parse intact), (d) the **raise path** maps to the caller's existing fallback (error string / `tone_pass_skipped` / empty manifest).
- Existing behavioral tests for these four keep passing (mock `husam.run` instead of `enqueue`).
- Suites run **sequentially**, `timeout`-prefixed, `-p no:cacheprovider`, worktree-absolute python — **never two pytest at once** (shared-DB deadlock crash-loops live KutAI). `tests/` and `packages/*/tests/` in separate invocations.
- `.venv\Scripts\pip install -e packages/husam` noted as a live-deploy step (new direct `import husam` sites).
- **Done:** no `await_inline=True` reachable from vision, brand_voice_lint, copy_compliance_review, yalayut synthesize; each migrated per §4; the two post-hook nested-deadlocks gone; tests green; SP4b spec stub written so the remaining scope is captured.

## 8. Execution

Subagent-driven in a fresh worktree (`git worktree add -b feat/cps-sp4a .claude/worktrees/cps-sp4a main`). One implementer per caller (4 are independent — no shared state), two-stage review per task, sequential pytest. Merge `--no-ff`, re-verify on the merged tree. Push to `main`.
