# Handoff — m90: narration-clobber VALIDATED fixed; two NEW roots found + fixed (2026-07-18)

**UPDATE 2026-07-18:** The "just /restart + re-reset a finished fix" plan below was WRONG. After restart both tasks re-DLQ'd for reasons this handoff did not anticipate. The narration-clobber class IS genuinely fixed (5.0c disk file is now clean markdown, not `## Analysis`). But two SEPARATE roots surfaced and are now fixed at root with TDD — **committed `95a62136` on branch `feat/yasar-usta-multiproject-hub`** (the repo was switched to the Yaşar-Usta hub branch during session time-gaps; user chose to leave the m90 fix on feat — it reaches main at hub merge). Still **restart-gated** (editable installs). Full write-up: memory `project_m90_verdict_repr_userflow_repair_20260718`.

- **567426 [3.11 requirements_review] "no parseable review verdict"** = repr-serialization, NOT schema/fleet. Reviewer returned a valid `{status:fail, issues:[real gaps]}` but `coulson/parsing.py:329` (legacy `status` branch) stored `str(parsed)` = Python repr (single quotes) → the verdict parser's `json.loads` failed → malformed → DLQ. **Fixed:** `json.dumps(parsed, ensure_ascii=False)` + `apply.py::_parse_review_result` `ast.literal_eval` fallback.
- **567452 [5.0c user_flow] "degenerate repeat"** = model non-compliance on the MECHANICAL shell (instruction template is correct). `cerebras/gemma-4-31b` authored a correct flow graph but dropped the `surfaces:` frontmatter line + the ```mermaid fence → `verify_user_flow_shape` correctly rejected → deterministic identical repeat → DLQ. **Fixed:** `verify_user_flow_shape.py::normalize_user_flow` deterministically injects `surfaces:` (from surfaces.json) + fences bare mermaid before validation, writing the repaired renderable artifact back.

## ⭐ DO THIS FIRST (updated)
1. **`/restart`** via Telegram (loads commit `95a62136`).
2. **Re-reset the 2 tasks** — LEDGER-CLEARING SQL (plain reset instant-DLQs as "degenerate repeat" against the stale ledger hash):
   ```sql
   UPDATE tasks SET status='pending', task_state=NULL, result=NULL, error=NULL,
     worker_attempts=0, grade_attempts=0, next_retry_at=NULL, exhaustion_reason=NULL,
     retry_reason=NULL, sleep_state=NULL,
     context=json_remove(context,'$._rejection_ledger','$._schema_error','$._schema_error_for_attempt','$._prev_output')
     WHERE id IN (567452, 567426) AND status IN ('failed','waiting_human');
   ```
   (rw connect: `sqlite3.connect(db, timeout=10)` + `PRAGMA busy_timeout=8000`.)
3. **Watch.** 567452 → disk `workspace/mission_90/.flow/user_flow.md` should be repaired (starts `---`, has `surfaces: [web]`, fenced ```mermaid) and complete. 567426 → the fail verdict now PARSES and routes to the requirements_spec producer to add the missing falsification triples (correct — the reviewer found real gaps), instead of DLQ.
4. **Push** is the hub branch's call (m90 fix rides feat). ⚠️ Repair does NOT extend to sibling 5.0d/6.5z — their frontmatter fields (`chunks`, `shared_components`) are semantic, not mechanically repairable; if they fail post-restart the lever is model capability, not more repair code.

---

## Mental model (read this before touching anything)
The bug class: on i2p_v3 workflow steps that AUTHOR a markdown file, the produced `.md` ended up as an agent's narration report (`## Analysis / Summary / Findings`, or garbled JSON) instead of the clean document → shape/grade gates rejected it → degenerate-repeat DLQ. It recurred for **many** sessions because each fix treated a downstream symptom.

**The real root (3 layers, all now fixed):**
1. **The agent had no `write_file`.** These 4 steps shipped `tools_hint: []`, and `coulson._apply_tools_hint` OVERWRITES `allowed_tools` with the hint verbatim → empty tool set. Unable to write a file, the analyst dumped the doc into `final_answer`, where it narration-wraps. (Object/array schemas ALSO auto-strip write_file, compounding it.)
2. **The schema lied about the artifact form.** Steps producing `.md` carried `type: object`/`array` schemas (to validate frontmatter fields). That (a) triggered write-strip, (b) made the grade schema gate's field-NAME substring fallback false-reject clean markdown, (c) told the analyst to emit JSON. **Schema type is a VALIDATION concern; the `.md` produces extension is the AUTHORING concern — they must not be conflated.**
3. **The reset kept the stale rejection ledger**, so any re-pend instant-DLQ'd as "degenerate repeat" against a pre-reset output hash — masking whether any fix worked.

**The key correction from the sub-investigation (don't re-learn this the hard way):** fixing the schema→markdown only RELOCATES the rejection from the schema gate to `verify_*_shape`; it does NOT make the artifact PASS. The remaining live blocker is **analyst non-compliance** — the erratic small model the degraded fleet hands this step (`gemma/gemma-4-26b-a4b-it`) sometimes omits the `surfaces:` frontmatter or uses an unfenced `graph TD`, which `verify_user_flow_shape` CORRECTLY rejects. **Proven live:** given `write_file`, the analyst DID write a clean compliant doc — so the structural fix works; completion now depends on the markdown schema (no JSON confusion) + a capable model. If 567452 still fails after restart, check WHICH model ran it (`context.generating_model`) and whether the fleet is exhausted (all capable cloud models rate-limited → falls to a tiny model).

---

## What's DONE — committed, do NOT redo
| Commit | What |
|---|---|
| `734b1c96` | Write-tool stripping keys off the `.md` produces FORM, not the schema TYPE (`coulson._write_tools_redundant`, used by `_apply_auto_strip` + `materialize_produces.write_stripped`). |
| `6871f9aa` | Invariant `coulson._ensure_write_tools_for_markdown_produces`: a `.md` produces gets `write_file` regardless of `tools_hint`. Runs in `execute()` AFTER `_refresh_workflow_step_config`, before the react loop. |
| `197f7130` | **The completing set (sub-validated):** (1) `schema→markdown` for the AUTHORING steps **5.0c/5.0d/6.5z** in `i2p_v3.json` + `tools_hint:[write_file]` + an explicit frontmatter/```mermaid template in the instruction; (2) `reset_workflow_step` json_removes `_rejection_ledger`/`_schema_error`/`_schema_error_for_attempt`/`_prev_output` (dabidabi); (3) `produces_markdown` plumbed through the GRADE-path schema gate (`mr_roboto/schema_gate.py` + `general_beckman/apply.py:~1905`); (4) invariant refined to SKIP structured-return `.md` steps (4.14). Guard test `test_v3_md_produces_steps_use_markdown_schema`. |
| `706b9b7d` | Review fix: `verify_shared_shell_shape` asserts `applicable_to_surfaces` (the schema flip had dropped that validation). |

**4.14 (ADR register.md) is DELIBERATELY NOT in the class.** Its instruction says "RETURN your decision as ADR JSON; do NOT write any files; the workflow rebuilds register.md automatically." It is a structured-RETURN step (array schema is correct); the engine (`verify_adr_register`) builds register.md from the returned JSON. The invariant SKIPS it (no write_file), and the guard test exempts it. Do not "fix" 4.14 to markdown — that breaks its contract.

**Sub-review verdict: SHIP.** Verified: no structured downstream consumers (artifacts are stored as raw markdown STRINGS in the blackboard — the object schema was "a fiction"); `produces_markdown` skips ONLY the prose fallback (`hooks.py:906-911`), never real structured validation; regen ledger-clear does not weaken normal auto-retry degenerate detection (`reset_workflow_step`'s only caller is the Regenerate button; normal retry appends via `apply.py` `_stamp_retry_feedback`).

---

## The two failed m90 tasks
m90 now: **415 completed, 2 failed (567426, 567452), 140 pending, 19 skipped.** Clearing 567452 should unblock the whole flow phase (5.0d/6.5z and downstream are pending behind it).

- **567452 [5.0c user_flow_lock]** — the narration case. After restart + re-reset it should complete IF a capable model authors it. Its DB row is currently hand-patched (schema=markdown, tools_hint=[write_file]) from this session's live testing — that now MATCHES the JSON, so it's consistent; the re-reset above is still needed to clear its stale ledger and give a fresh attempt.
- **567426 [reviewer — "Review requirements_spec against prd_final"]** — **NOT narration, NOT a gate bug.** All 5 capable models were in `failed_models` (transient rate-limit) so it fell to `gpt-oss-20b:free`, which degenerated (96% of its 6000-char output was the literal character `!`). The schema reject was correct. Fix = operational: re-reset (the SQL above clears the exclusion history via context wipe) when the fleet is healthy so a capable model does the review. If it re-degenerates, check `context.failed_models` / fleet capacity.

---

## Key mechanisms & gotchas (save yourself the rediscovery)
- **`_refresh_workflow_step_config` (coulson) refreshes `artifact_schema`/`tools_hint`/`produces` from the LIVE i2p_v3.json on every worker run AND persists it back to `tasks.context`.** So hand-patching a task's stored context is FUTILE — the JSON wins. Edit i2p_v3.json, not the DB row, for config. (The loader is `(path, mtime)`-keyed, so JSON edits are picked up live WITHOUT a restart — but CODE changes need `/restart`.)
- **Degenerate-repeat masks real failures.** `apply.py` emits "degenerate repeat: identical output across attempts" when the fresh output hash == the last ledger hash. A reset that does NOT clear `_rejection_ledger` will instant-DLQ a legit retry at worker_attempts≤1. ALWAYS use the ledger-clearing reset SQL above when re-testing.
- **Grade gate vs producer gate** both validate the artifact but read different sources; both now pass `produces_markdown` for `.md` produces. The grade gate reads `source_ctx` (the stored task context) at grade time.
- **Two artifact "sources":** the agent's `write_file` to disk vs its `final_answer`. `materialize_produces` picks between them ([disk, output_value] for non-write-stripped .md → disk-first). If the agent didn't call write_file (`context.tool_calls == []`), there's no disk file and the narration wins — that's the failure signature to check first.

## Inspect (read-only — safe while the bot is live)
```python
# .venv/Scripts/python.exe ; set sys.stdout.reconfigure(encoding='utf-8',errors='replace')
c = sqlite3.connect(r"file:C:\Users\sakir\ai\kutai\kutai.db?mode=ro", uri=True, timeout=3)
c.execute("PRAGMA busy_timeout=3000")
# task state, model, tool calls:
#   SELECT status,worker_attempts,generating_model? (in context), error FROM tasks WHERE id=?
#   context keys of interest: tool_calls, generating_model, failed_models, _rejection_ledger, artifact_schema, tools_hint, produces
```
Workspace artifacts: `C:\Users\sakir\Dropbox\Workspaces\kutay\workspace\mission_90\.flow\` (NOT under `ai\kutai\` — that path is empty). Artifacts are also stored as strings in the `blackboards` table (`data` JSON, keyed by mission_id).

## Environment hazards
- **Zombie pytest deadlock:** a parallel session leaves many `pytest … -p no:cacheprovider` procs holding the SQLite write lock. A full-suite run (e.g. `packages/coulson/tests`) DEADLOCKS. Run TARGETED single-file tests FOREGROUND with `timeout` (pure-python tests don't take the lock). Kill only your OWN hung run (match the exact command line via `Get-CimInstance Win32_Process`). Do NOT mass-kill — the parallel session is active (its commits `3dd5f54d`/`b39baa48`/`61c952e6` are in this history).
- **Conftest collision:** mixing a package test dir (`packages/*/tests`) with root `tests/` in ONE pytest invocation errors ("Plugin already registered"). Run each package separately.
- **Bash auto-backgrounds long/looping commands** (sleep-poll loops especially) → you get a task-notification when done; don't re-Read the output file in a tight loop (wasted calls).

## Verify the fix set (per-file, foreground + timeout)
```
.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider packages/coulson/tests/test_auto_strip_schema_scope.py
.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider packages/mr_roboto/tests/test_schema_gate.py tests/i2p/test_shared_shell.py
.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/test_i2p_v3.py tests/test_schema_gate_markdown_produces.py tests/workflows/test_materialize_produces.py --deselect "tests/test_i2p_v3.py::TestV3WorkflowLoading::test_v3_all_steps_have_artifact_schema"
.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider packages/general_beckman/tests/test_task_write_api.py::test_reset_workflow_step_clears_rejection_ledger
```
Session regression totals (all green): coulson+mr_roboto 63 · beckman 37 · root 125.

## Open / deferred (not blocking)
- **Pre-existing red test (NOT this work):** `tests/test_i2p_v3.py::TestV3WorkflowLoading::test_v3_all_steps_have_artifact_schema` fails on step `13.14` (a `reviewer` agent with no `artifact_schema`). The test exempts `mechanical` agents but not `reviewer`. Fix = add `reviewer` to the exemption, or give reviewer steps a schema. Present since before this session.
- **Deferred review MINOR:** the grade-gate `produces_markdown` defer does not assert a `verify_*_shape` check EXISTS — a future object+`.md` step without a shape check would silently lose validation. Mitigated by the guard test (forces markdown schema on new `.md` steps). Consider asserting shape-check-presence when deferring.
- **The real unknown:** whether a CAPABLE model reliably produces `verify_*_shape`-compliant markdown for these steps. If completion is still flaky post-restart, the lever is model selection (this step draws a "medium"-difficulty analyst; the fleet was exhausted), NOT more gate code.

## Memory
`project_narration_clobber_produces_form_root_20260628` (full detail incl. the sub-investigation UPDATE), `project_advisory_complete_grade_override_20260703` (parallel session, shares the `_write_tools_redundant` seam), `project_m90_three_gate_fixes_20260627`.
