# Handoff — i2p mission debugging (2026-05-23)

**For:** the next session continuing the first real end-to-end i2p mission run.
**TL;DR:** A long debugging arc (mission 71 → 72 → 73) traced a Phase-0 stall to a
**single keystone bug (Bug G)** plus a chain of contributors. All fixes are in,
tested, and **uncommitted**, batched for your **next restart + a FRESH mission**.
No mission has still completed end-to-end — that's the next goal.

---

## §0 — Do this first
1. **Commit decision is the founder's.** Nothing is committed yet. Touched files
   listed in §3.
2. **Full restart** (wrapper = Yaşar Usta, not just `/restart` which only bounces
   the orchestrator) so sidecars (nerd_herd on **:8081**) + all code load.
   - Verify: `Get-NetTCPConnection -LocalPort 8081` → `llama-server.exe`;
     nerd_herd cmdline `--llama-url …:8081`; mitmproxy still owns :8080.
3. **Start a NEW mission** (`/mission … --workflow`). DO NOT keep retrying #73 —
   its tasks have `agent_type`, context, and the broken pre-fix config **baked at
   expansion**. Workflow-structure fixes only reach a fresh expand.
4. **Watch Phase 0** — the intake hybrid (`0.0a.draft` → `0.0a`) is the first live
   test of Bug C + Bug G. Expect `mission_<id>/.intake/intake_todo_draft.json` to
   land on disk and its grounding check to pass.

---

## §1 — What was fixed (8 fixes, all tested, batched)
| # | Bug | Fix | Tests |
|---|---|---|---|
| A | `spec_consistency_check` used `Path.cwd()` → vacuous pass | fallback → canonical `WORKSPACE_DIR` | mr_roboto 15 |
| B | `[9.0z]` depended on recurring `8.arch_check` → ran Phase 0 | repoint → `8.spike.git_commit` + dep-integrity guard | 3 |
| C | intake generic (dead `use_llm`, forbidden HK call) | analyst-draft + canonical-merge hybrid (then →writer) | 10 |
| D | mitmproxy squats :8080 → local loads fail | llama → :8081 via `LLAMA_SERVER_PORT` (.env + wrapper + nerd_herd + dlq) | nerd 245 / dallama 59 |
| ctx | undersized loaded model reused | dispatcher `loaded_ctx_insufficient` reload + `BASELINE_LOCAL_CTX=16384` floor | 4 |
| F | local `timeout=0.0` "no-cap" → `max(10,t-5)`=10s → APITimeout | `_http_timeout`: `<=0` → `None` | 3 / hk 65 |
| **G** | **engine wrote `<name>.md` at mission ROOT, ignoring the `produces` path** → `.intake/…json` never landed → grounding DLQ | **`hooks.py`: also persist `output_value` to the declared produces path (subdir+ext), only when missing** + instruction "return JSON, engine persists, no write_file" | 2 |
| tests | 7 stale `test_i2p_v3` failures (count range, `skip_when` list-only validator, mechanical steps lack difficulty/tools_hint) | widened ranges, validator accepts str, mechanical carve-outs | 46 |

**Bug G is the keystone** — everything else cascaded from it (see §2).

---

## §2 — The cascade (recognize it instantly next time)
```
produces file written to WRONG path (root <name>.md, not .intake/...json)
  → grounding check finds nothing → DLQ
  → agent never "completes" cleanly → max_iterations / retries
  → checkpoint-resume RE-ACCUMULATES the failed conversation every retry (react.py:206-214; only _schema_error skips it)
  → prompt balloons to ~100k tokens
  → effective_context_needed ~100k → ctx-eligibility filters out ALL cheap free models (groq TPM 6-8k, openrouter free 32k ctx)
  → only gemini/cerebras (huge ctx) survive → hammered → PER-MINUTE rate-limits (mislabeled "daily_exhausted")
  → falls to slow local thinking model → "## Analysis…" narration (it has NO tools: write_file auto-stripped for schema'd steps, Allowed:[])
  → loops for hours
```
The "narration" was **not** thinking-leak (checked: 0 `<think>` markers; HK separates `reasoning_content`). It was the model coherently explaining `Tool 'write_file' not available. Allowed: []`.

---

## §3 — Files touched (uncommitted)
- `packages/mr_roboto/src/mr_roboto/spec_consistency_check.py` (A)
- `src/workflows/i2p/i2p_v3.json` (B `9.0z` dep; C `0.0a.draft` step + `0.0a` rewire; instruction wording)
- `packages/mr_roboto/src/mr_roboto/generate_intake_todo.py` (C: draft-merge, no LLM)
- `.env` + `kutai_wrapper.py` + `packages/nerd_herd/src/nerd_herd/__main__.py` + `src/infra/dlq_analyst.py` (D: `LLAMA_SERVER_PORT`)
- `src/models/local_model_manager.py` (ctx: `loaded_context_length`, `BASELINE_LOCAL_CTX`)
- `src/core/llm_dispatcher.py` (ctx: `loaded_ctx_insufficient` in `_ensure_local_model`)
- `packages/hallederiz_kadir/src/hallederiz_kadir/caller.py` (F: `_http_timeout`)
- `src/workflows/engine/hooks.py` (**G: produces-path persist**)
- `src/workflows/engine/loader.py` (test fix: `skip_when` accepts str)
- Tests: `tests/core/test_ctx_reload_guard.py`, `tests/workflows/test_i2p_v3_dep_integrity.py`,
  `tests/workflows/test_produces_persist.py`, `tests/i2p/test_intake_todo.py`,
  `packages/hallederiz_kadir/tests/test_http_timeout.py`,
  `packages/mr_roboto/tests/test_spec_consistency_check.py`, `tests/test_i2p_v3.py`

---

## §4 — Debugging playbook (hard-won this session — saves hours)
- **Logs are UTC; the DB's wall-clock differs.** A timezone slip made me grep the
  wrong hour and falsely conclude "no dallama load events." Always reconcile.
- **Live LLM/dispatch logs are in `logs/kutai.jsonl`** (NOT orchestrator.jsonl).
  Selector picks: `fatih_hoca.selector "selector pick: model=… min_time=…"`.
- **Trust running rows/logs over labels.** `daily_exhausted` was actually
  per-minute; "narration" was empty-toolset; "exhausted quotas" was a ctx filter.
- **Scratch tools (kept):** `scripts/_probe_missions.py <mission_id>` (live state
  snapshot), `scripts/_probe_toolstrip.py` (proves write_file→`[]` for schema'd
  steps). Use read-only `file:…?mode=ro` against `C:\Users\sakir\ai\kutai\kutai.db`.
- **`/restart` ≠ wrapper restart.** Orchestrator-only restart loads code but NOT
  wrapper-spawned sidecars (nerd_herd). Founder does full restarts.
- **Re-expansion rule:** workflow-JSON changes (deps, new steps, agent, instruction)
  only reach a FRESH mission; existing tasks bake them at expansion.
- **Verify red→green on a host-path test**, never a unit mock — every sweep bug
  this project has shipped passed its unit test with the bug live.

---

## §5 — Open items / watch on the next mission
1. **Nothing validated end-to-end yet.** A+B exercise at Phase-7+ spec gates; C+G
   at Phase-0 intake; D/F/ctx on any local call. Watch all.
2. **Checkpoint-resume accumulation (Bug-G amplifier)** — `react.py:206-214` resumes
   the full prior conversation for non-`_schema_error` failures. If a task ever
   fails repeatedly again, this still re-accumulates. Consider **progress-gated
   resume** (only resume when real tool/artifact progress was made) as a deeper fix.
3. **Pre-existing test debt (NOT this session, isolated via stash):**
   `tests/core/test_dispatcher_records_swap.py::test_dispatcher_records_swap_after_swap`
   + `tests/test_llm_dispatcher.py::…test_retries_on_call_error_then_succeeds` fail on
   a pick_log `task_id` bind / `record_swap` mock. Test-harness debt.
4. **4 Phase-15 non-recurring→recurring deps** (`15.11→15.8`, `15.12→15.7`,
   `15.10b_record_demo→15.10`, `15.14b_deliverable_bundle→15.14`) — suspected Bug-B
   class; far off, verify when Phase 15 is reached.
5. **Watch `model_pick_log`** for groq/openrouter actually being picked now that
   prompts won't balloon — they were filtered only because of the 100k ctx need.

Full per-bug detail + evidence: `docs/handoff/2026-05-22-mission71-bug-ledger.md`.
