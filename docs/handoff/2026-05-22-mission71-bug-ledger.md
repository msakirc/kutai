# Mission 71 — live e2e run bug ledger

**Mission:** #71 "todo app → gamified habit builder" (i2p_v3, lifecycle=active)
**Launched:** 2026-05-22 10:32 (founder, via Telegram). First-ever end-to-end run.
**Mode:** KUTAI_ENV unset → vendors mocked. LLM = **cloud** (`gemini/gemini-2.5-flash`
main_work, `gemini-2.5-flash-lite` overhead). llama-server idle (502 proxy).
**DB:** `C:\Users\sakir\ai\kutai\kutai.db`. **Workspace:** `<repo>\workspace\mission_71`.

Whole workflow expanded up front: 283 tasks, phases 0–14 all present as `pending`;
Beckman gates dispatch by `depends_on`. No DLQ, no orchestrator-log errors so far.
Artifacts landing fine (`reverse_pitch.md` 2985 B, real content).

---

## BUG A (systemic) — spec_consistency_check looks in the wrong dir → vacuous pass

**Failure mode:** 4 (trust-gate bypassed) + §4 "workspace_path mismatch zeroes payload".

- Caller `packages/mr_roboto/src/mr_roboto/__init__.py:2858` passes
  `workspace_path=payload.get("workspace_path")`.
- The expander never injects `workspace_path` into the mechanical payload →
  payload.workspace_path is `None` for **all 6** spec_consistency_check tasks
  (`[7.0z]`,`[8.0z]`,`[9.0z]`,`[10.0z]`,`[11.0z]`,`[12.0z]`).
- `spec_consistency_check.py:73-74` `_resolve_workspace_root(None,…)` falls back to
  `Path.cwd()` = repo root → `mission_dir = <repo>/mission_71` (real: `<repo>/workspace/mission_71`).
- `:579-589` dir-missing → returns `ok:True, spec_artifacts_present:[]` (vacuous).

**Evidence:** task 147134 result
`{"ok":true,…,"warnings":["mission_dir_missing:C:\\…\\kutay\\mission_71"]}`.

**Root fix (not symptom):** expander must propagate the mission `workspace_path`
into mechanical step payloads (same value the writer/coulson path already resolves
to write `workspace/mission_71`). `init_mission_github_repo` (`__init__.py:2889`)
reads the same `payload.get("workspace_path")` → likely same hole. Do NOT just
default `_resolve_workspace_root` to `cwd()/workspace` — find the canonical
workspace base resolver and reuse it.

**Test trap (§0):** `test_spec_consistency_check.py:155-156` *asserts* `ok is True`
+ `mission_dir_missing` warning — the bug is baked into the unit test. Host-path
test required: drive the real expander→mechanical payload, assert the check reads
the populated `workspace/mission_<id>`.

---

## BUG B (data, one-line) — [9.0z] missing depends_on → ran in Phase 0

**Failure mode:** ordering / missing edge.

- `[9.0z] spec_consistency_check_phase_9` (task 147134) has `depends_on: []`.
- Every sibling has an edge: `[7.0z]→147099`, `[8.0z]→147124`,
  `[10.0z]→147146`, `[11.0z]→147157`, `[12.0z]→147163`.
- Empty deps → Beckman admitted it immediately → it ran during Phase 0 (combined
  with Bug A = doubly vacuous).

**Root fix:** in `src/workflows/i2p/i2p_v3.json`, give the phase-9
`spec_consistency_check` step the same wave-start `depends_on` as its siblings
(point at the phase-9 anchor step, mirroring `[8.0z]`/`[10.0z]`).

---

## Watch status

- Phase 0 in progress: `[0.0z] reverse_pitch_draft` done (file written),
  `[0.1] product_charter` (writer) pending next.
- Grading lag: completed tasks show `quality_score=None`, `confidence_*=None`,
  status flipping `ungraded`↔`completed` — watch whether Z10 calibration /
  grading ever populates (classic empty-table scaffold; verify, don't assume bug).
- `mission_lessons` — confirm rows appear on DLQ/posthook-fail paths.

---

## FIXES APPLIED (2026-05-22, awaiting founder restart)

Both fixes are code/data only — the **running** orchestrator already imported the
old modules, so they take effect on next restart. Founder will restart + reset
the failed task chain.

- **Bug A** — `spec_consistency_check.py:69-83` `_resolve_workspace_root` fallback
  changed `Path.cwd()` → canonical `WORKSPACE_DIR` (lazy import from
  `src.tools.workspace`). Tests: `test_spec_consistency_check.py` +3
  (`test_resolver_falls_back_to_workspace_dir_not_cwd`,
  `test_check_resolves_workspace_when_payload_omits_path`) → **15 passed**.
- **Bug B** — `i2p_v3.json` `9.0z.depends_on` `["8.arch_check"]` →
  `["8.spike.git_commit"]` (last concrete phase-8 step; arch_check is recurring).
  Tests: `tests/workflows/test_i2p_v3_dep_integrity.py` (no missing dep refs;
  consistency gates depend on concrete non-recurring steps) → **2 passed**.

**NOTE — Bug A semantic inconsistency (NOT fixed, logged):** `spec_consistency_check`
treats `workspace_path` as the **base** (`workspace/`, then appends `mission_<id>`),
but `init_mission_github_repo` (`__init__.py:2889`) treats it as the **per-mission
dir** (reads `workspace_path/charter.md`). So the expander cannot inject one
`workspace_path` value safely. Fixed Bug A at the executor default instead. A
proper unification (one canonical "mission workspace" contract across mechanicals)
is a follow-up.

## Live-run evidence (mission 71, ~46 tasks in)

- **DLQ chain (3) all = task 147134**: `[9.0z]` + 2 `Grounding check for #147134`.
  `[9.0z]` failed `check_grounding: missing=['mission_71/spec_drift_report.md']`
  — the **G-grounding L2 post-hook caught the vacuous pass** (no report written
  because Bug A made it look in the wrong dir) and DLQ'd it instead of letting it
  pass silently. Defense-in-depth held. After restart+reset, 9.0z runs at the
  right time (Bug B) in the right dir (Bug A) and produces the report.
- **`waiting_human` = `[0.0z.confirm] reverse_pitch_confirm`** (146973) — parked
  for **founder confirmation in Telegram**. Phase 0's clean path is blocked here.
- **Odd model ids in `model_pick_log`:** `gemini/gemma-4-31b-it`,
  `gemini/gemma-4-26b-a4b-it` (alongside the valid `gemini-2.5-flash*`). Verify
  these resolve on the provider — if fabricated catalog entries, calls 404.
  (Watch item; tasks are still completing so most picks are valid.)

## Follow-ups surfaced (not fixed this session)

1. **4 Phase-15 non-recurring→recurring deps** (suspected same class as Bug B):
   `15.10b_record_demo→15.10`, `15.11→15.8`, `15.12→15.7`,
   `15.14b_deliverable_bundle→15.14`. Verify whether Phase 15 one-shot steps
   actually materialise (vs whole-phase cron) before fixing.
2. **`resolve_dependencies` warn-and-drop** (runner.py:53-63) leaves a gate
   unguarded when a dep doesn't materialise — it warns but admits. Consider a
   loud/blocking variant for gate-class steps.
3. **Reset on restart:** clear DLQ/failed for task 147134 + its 2 grounding
   children so the corrected 9.0z chain re-runs (`/dlq retry` or DB reset).

---

## BUG C (design+wiring) — intake_todo was generic-by-accident, not by-design

**Founder caught it:** the `[0.0a]` intake questions are fully generic, ignoring
the HabitLoop pitch. Investigation:

- Payload had `use_llm: True` but `inputs: None`, `paths: None` — the expander
  declared `input_artifacts` in context but never wired them into the mechanical
  `payload.inputs/paths` (same class as Bug A: declared-but-unwired).
- `_llm_builder` called `hallederiz_kadir.call(messages=, max_tokens=, profile=)`
  — but the real signature is `call(model, messages, tools, timeout, task,
  needs_thinking, …, call_category=)`. Stale kwargs → **TypeError before any
  network call** → swallowed → silent deterministic fallback. So `use_llm` was a
  **dead no-op** (no tokens burned, but no specialisation either).
- A mechanical calling HK directly also **violates `feedback_no_direct_dispatcher`**.

**Fix (hybrid — founder-approved direction):**
- New analyst step `[0.0a.draft]` (`i2p_v3.json`) reads raw_idea + strategic_context
  + reverse_pitch (agents DO get `input_artifacts` injected; mechanicals don't —
  that was the starvation) and writes `intake_todo_draft.json`: 14 `{n,category,
  question}` items reworded for the product.
- `generate_intake_todo.py`: removed `_llm_builder` (dead + forbidden); added
  `_load_analyst_draft` + `_build_intake` that **merges the draft onto the fixed
  14-slot canonical skeleton** — a slot uses the analyst wording only when its
  declared category matches the canonical category for that slot, else canonical.
  → coverage + ordering can't regress even with a weak model; specialisation is
  upside-only. No LLM call in the mechanical.
- `[0.0a]` re-pointed `depends_on: ["0.0a.draft"]`, dropped `use_llm`.
- Chain verified: `0.0z.confirm → 0.0a.draft (analyst) → 0.0a (merge+confirm) → 0.1`.
- Tests: `tests/i2p/test_intake_todo.py` +4 (specialise / category-mismatch-reject
  / invalid-json-fallback / no-draft-canonical); existing 4 preserved.

**Design rationale (small-LLM safety):** deciding *which* dimensions to ask = hard
for small models (they drop Compliance/Non-goals) → keep the static contract.
*Phrasing* each dimension for the product = easy → let the LLM do only that.
Founder-confirm gate is the final safety net.

**⚠️ Re-expansion caveat:** this is a workflow-definition change. Mission 71's
tasks were already expanded from the OLD JSON (no `0.0a.draft`), and a restart does
NOT re-expand an existing mission. So #71 keeps the generic intake unless it is
re-run from Phase 0. The hybrid applies to the **next** mission (fresh expand) or a
reset-to-Phase-0 of #71.

## BUG D (env/infra) — mitmproxy squats port 8080, all local loads fail

Surfaced on the **2nd mission** (fresh expand, gets A+B+C). Writer `0.0z` failed:
`All models failed for 'writer': Failed to load local model Qwen3.5-9B-...`.

**Root cause (NOT KutAI/model):** `mitmdump.exe -p 8080 -s capture_addon.py`
(PID 14060, running since 05-18) holds port **8080** — the exact port DaLLaMa
binds llama-server to. DaLLaMa's orphan-kill only targets `llama-server.exe`, so
it cannot evict mitmproxy → every local load aborts. dallama.jsonl:
`Port 8080 already in use … taskkill(llama-server.exe) returned 128: not found …
Port 8080 still in use after orphan kill — aborting start`. Load failures began
05-20 (after the squatter appeared); explains why mission #71 silently ran on
**cloud** the whole time (local was unloadable).

GGUF (5.56 GB) + llama-server.exe both present and fine. The "odd" model names
(`gemma-4-26B-A4B`, `Qwen3.5-9B/27B/35B`) are **real local GGUFs** — watch item
cleared.

**Founder decision:** keep mitmproxy on 8080, **move llama-server to 8081.**

**Fix (standardise on `LLAMA_SERVER_PORT`, default 8080):**
- `.env`: `LLAMA_SERVER_PORT=8081` (DaLLaMa bind + api_base already honour it via
  `local_model_manager.py:69,84`).
- `kutai_wrapper.py:145`: nerd_herd `--llama-url` derived from the env var.
- `nerd_herd/__main__.py`: `--llama-url` default derived from the env var.
- `src/infra/dlq_analyst.py:143`: health-check URL derived from the env var
  (+ `import os`).
- Verified: compile clean; all sites resolve to `127.0.0.1:8081`; dallama config
  default stays 8080 (tests intact).

**Needs full restart** — the wrapper reads `LLAMA_SERVER_PORT` at launch; running
wrapper/nerd_herd/orchestrator still hold 8080. After restart llama binds 8081,
mitmproxy keeps 8080, both coexist. The new mission's failed `0.0z` then retries
on a loadable local model (or cloud).

## BUG E (DEFERRED — observe after restart, do NOT fix blind)

Mission 72 (`/mission` re-run), `0.0z.verify` (task 149467) → DLQ:
`missing=['headline','sub_head','customer_quote','founder_quote','faq']`.

The artifact `mission_72/.charter/reverse_pitch.md` (680 B) contains the **raw
agent envelope**, not unwrapped markdown:
````
```json
{ "action": "final_answer", "result": "## Headline\n\nHabitQuest...<truncated mid-Customer-Quote>" }
```
````
Compounding factors:
1. **Truncated** output (cut mid-sentence) — the 0.0z `result` even holds a
   reviewer verdict `"issues"`: *"markdown file was severely truncated … missing
   ## Founder Quote and ## FAQ."*
2. **Envelope-as-content** — the fenced `final_answer` JSON was written to the
   `.md` instead of the unwrapped/unescaped `result`.
3. **Reviewer flagged but `0.0z` completed anyway** (trust-gate bypass flavour);
   the downstream mechanical verify is what caught it.

**Why deferred:** this happened **pre-restart**, while Bug D made local loads
fail — so 0.0z ran on a degraded fallback model that truncated + emitted a fenced
envelope. Mission 71 on healthy gemini produced clean markdown. Likely a symptom
of the degraded model scramble, not a standalone defect. **Restart first** (A–D
live, local models on 8081, healthy selection), then re-observe. If truncation/
envelope reproduces with healthy models, investigate: (a) the write_file content
source / final_answer unwrap at the produces boundary, (b) why a reviewer "issues"
verdict didn't block 0.0z completion. `generating_model` + `tool_calls` are in the
task context for the post-restart triage.

## BUG (ctx) — undersized loaded local model reused without reload

Mission 73 `[0.0a.draft]` (analyst) DLQ'd with `context_overflow` on the local
Qwen3.5-9B. Root: `calculate_dynamic_context` (registry.py:486,509) bounds ctx by
*instantaneous free system RAM* (CPU-offloaded KV) → same 9B loaded at ctx 5786
one load, 48293 another. The 9B was loaded small (RAM-tight), then **reused** for
a ~14162-token analyst prompt. `_ensure_local_model`'s ctx guard existed but was
unreachable: `llm_dispatcher._ensure_local_model`'s `needs_reload` only checked
model identity + thinking/vision, never context, and short-circuited
(`return True, False`) before `ensure_model`. Eligibility (`selector.py:446`)
checks the model's *registry* ctx (~32k), not the *loaded* ctx (5786).

**Fix + hardening (implemented, awaiting restart):**
- `local_model_manager.py`: new `loaded_context_length` property; `BASELINE_LOCAL_CTX=16384`
  floor on every local load (capped at registry ceiling; env `LLAMA_BASELINE_CTX`).
- `llm_dispatcher.py`: `needs_reload` now includes `loaded_ctx_insufficient`
  (`this_actually_loaded and estimated_context>0 and manager.loaded_context_length < estimated_context`).
- Tests: `tests/core/test_ctx_reload_guard.py` (4) green.
- llama-server cannot resize n_ctx at runtime (subagent-verified) → reload is the only path; the baseline floor makes reloads rare.

## BUG F — `timeout=0.0` "no-cap" sentinel collapses to a 10s HTTP timeout

Mission 73 `[0.0a.draft]` then DLQ'd with `litellm.Timeout: APITimeoutError` on ALL
models. Root: `llm_dispatcher.execute:717` passes `timeout=0.0` for local models
(intended "no wall-clock cap — stream watchdog governs"), but
`hallederiz_kadir/caller.py:580` did `http_timeout = max(10.0, timeout - 5.0)` =
**10s** for the 0.0 sentinel. A slow local thinking model (selector-estimated
`min_time=100.6s`) got a 10s HTTP deadline → APITimeoutError (the model WAS
generating — full responses landed ~35s later). Compounded by cloud being
`daily_exhausted` (no cloud escape) and a `-thinking` variant emitting
"## Analysis…" narration instead of the intake JSON (selection-tuning, deferred
per founder).

**Fix (implemented):** `caller.py` `_http_timeout(timeout)` helper — returns `None`
(no HTTP cap; stream watchdog governs) when `timeout <= 0`, else `max(10, t-5)`.
Tests: `packages/hallederiz_kadir/tests/test_http_timeout.py` (3) green.

NOTE: `#150067` grounding-missing is purely downstream of the analyst failing —
no independent bug; resolves when the analyst completes.

PRE-EXISTING (NOT this session, isolated via stash): `test_dispatcher_records_swap`
+ `test_retries_on_call_error_then_succeeds` fail on a pick_log `task_id` bind /
`record_swap` mock — fail with my dispatcher change reverted too. Test-harness debt.

## BUG G (the keystone) — engine persists artifacts to `<name>.md` at root, ignoring the declared `produces` path

The long intake loop (mission 73, ~hours, 100k-prompt spiral) traced to ONE bug,
proven by `scripts/_probe_toolstrip.py`:

- **write_file is auto-stripped for ALL schema'd agent steps** — repro shows BOTH
  `0.0z` (reverse_pitch) and `0.0a.draft` (intake) resolve to `allowed_tools=[]`
  after `_apply_auto_strip`. So the agent CANNOT write its produces file; the
  engine must (the intended "strip write_file, engine saves output" design).
- **The engine DOES write to disk** (`hooks.py:1466-1479`) — but **always as
  `{name}.md` at the mission root**, ignoring the step's declared `produces`
  path (subdir + extension). So intake (`produces: .intake/intake_todo_draft.json`)
  got `intake_todo_draft.md` at the **root** → the declared `.json` path never
  existed → grounding DLQ + `generate_intake_todo` (reads `.intake/...json`)
  found nothing. reverse_pitch only "worked" because its produces ext is `.md`.

This was the source of EVERYTHING downstream: missing file → grounding DLQ →
non-completing agent → checkpoint-resume accumulation → 100k prompt → ctx filter
eliminates cheap free models → only gemini/cerebras → per-minute rate-limits
(mislabeled "daily_exhausted") → slow local → narration. **Not quotas, not
thinking-leak, not a context-bound problem.**

**Fix (generic — every schema'd+produces step):**
- `hooks.py` — after the legacy `<name>.md`-at-root write, ALSO persist
  `output_value` to each **declared `produces` path** (real subdir + ext),
  **only when the file is missing** (never clobber an agent-written file).
- `i2p_v3.json` `[0.0a.draft]` instruction — "RETURN the JSON as your final
  answer; the engine persists it; do NOT call write_file; output ONLY the JSON"
  → stops the "## Analysis…write_file unavailable" narration.
- Tests: `tests/workflows/test_produces_persist.py` (writes to `.intake/…json`;
  preserves an existing agent-written file) — 2 pass; dep-integrity 3 pass.

**Investigation artifacts left:** `scripts/_probe_toolstrip.py` (proves both
steps strip to `[]`), `scripts/_probe_missions.py` (live mission monitor).

NOTE: needs a FRESH expand (new mission) — 149782's context + agent_type are
baked; the engine-persist fix is global and live on restart.

(Append further findings below as the run proceeds.)
