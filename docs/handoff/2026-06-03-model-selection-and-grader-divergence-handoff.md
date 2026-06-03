# Handoff — i2p DLQ deep-dive: stale-description divergence (fixed) + the local-fallback gate bug (root-caused, NOT fixed)

**Date:** 2026-06-03
**Predecessor:** `docs/handoff/2026-06-01-verify-steps-as-posthooks-handoff.md`
**HEAD after this session:** `ee76628c`
**Live build:** all 3 commits below were committed AFTER the last restart — they need the NEXT restart to load. The founder restarted + DLQ-retried mid-session (that retry is the evidence in §2/§4) and later **purged mission 80** for a fresh run, so the DB rows referenced here (#259351, #259369) no longer exist — but the logs do (see §6).

Trigger: founder asked "did recent CPS/post-hook updates break something?" — two i2p tasks DLQ'd with grader `COMPLETE: NO`:
- #259351 `[1.3] direct_competitor_identification` (researcher)
- #259369 `[2.4] user_personas_and_journeys` (analyst)

Answer: **No, CPS/post-hook work did not break these.** The post-hook fixes unblocked earlier DLQ points so a fresh mission (#80) reached phase 1/2 for the first time, exposing **older latent bugs**. Three were real and fixed; one triaged bug turned out not to exist; one deep bug is root-caused and left for a plan.

---

## 1. What shipped this session (3 commits, all TDD, all green)

| Commit | Fix | Tests |
|--------|-----|-------|
| `f7d82de5` | **analyst no longer forced through the researcher URL rule.** `src/models/models.py` `_AGENT_TYPE_CATEGORY` mapped `analyst → "research"`; the research output rule (`models.py:282`) demands a URL or source-keyword. Personas/journeys/positioning are structured deliverables with no URL → #259369 rejected every attempt for "missing URL". Removed analyst from the map (now returns `[]` = no task-type validation). researcher keeps the URL rule. | `tests/test_iteration_exhaustion.py` (+2; renamed `test_analyst_same_rules`) 14 pass |
| `3a026fa8` | **i2p 1.3 instruction aligned to schema.** The `1.3.instruction` told the researcher to collect `founding_year, funding_total, team_size_estimate, primary_market` per competitor; the `artifact_schema` validates only `[name, website_url, one_line_description, platforms, status]`. Confirmed via JSON scan: those 4 heavy fields appear **only** in 1.3's own instruction — zero downstream consumers, zero code refs. The agent chased one web_search per competitor to source them → blew the 6-iteration cap → DLQ, and fabricated the numbers pre-search (the `search_required` guard correctly caught that). Rewrote the instruction to ask for exactly the 5 schema fields and forbid researching the others. | (instruction text — schema/instruction field-set checked by hand) |
| `ee76628c` | **persist the refreshed workflow-step description (THE big one — see §2).** `packages/coulson/src/coulson/__init__.py` `_refresh_workflow_step_config` re-syncs a step's instruction from live JSON into the task each run, but persisted only `context`, never `description`. Now persists `description` too when changed. | `packages/coulson/tests/test_step_refresh_persists_description.py` (+2) — 69 coulson pass |

Note: the analyst+1.3 fixes alone did NOT make #259351/#259369 pass on retry — see §2 for why.

---

## 2. THE ROOT CAUSE that mattered most — stale-description divergence (FIXED, `ee76628c`)

**Symptom:** after the 1.3 instruction edit was live, retrying #259351 STILL DLQ'd with grader `COMPLETE: NO`, even though the worker produced a flawless artifact (exactly 5 competitors, all 5 schema fields, valid JSON).

**Ground truth from the grade-child logs** (tasks `279737/279768/279800/279833/279880/279966`, all `reviewer` overhead children of #259351): every grader verdict failed it citing the **OLD** requirements — *"missed mandatory fields founding_year, funding_total, team_size_estimate, primary_market"* and *"failed minimum 5 competitors"*. The self_reflect children (`279722/279760/...`) were even re-injecting funding fields via `corrected_result`.

**Why:** a workflow step has **two readers of the instruction, one stale:**
- **WORKER** rebuilds its prompt from the **live workflow JSON** every run via `_refresh_workflow_step_config` (coulson `__init__.py:301`, called at line ~72 when `is_workflow_step`). It got the NEW light instruction. (Log: `00:23:40 step-refresh: description re-synced from live JSON (step=1.3)`.)
- **GRADER** (`src/core/grading.py::build_grading_spec`, ~line 290) and **self_reflect** (`src/core/reflection_posthook.py::build_reflect_messages`, ~line 306) read `tasks.description` **straight from the DB** — frozen at expander time (`src/workflows/engine/expander.py:494` sets `description = step.instruction` ONCE at mission creation).

`_refresh_workflow_step_config` updated `task["description"]` in memory and was BUILT for exactly this (docstring cites "grader kept citing 'use cases'"), but its DB write persisted only `context` (line ~401). So the grader kept reading the stale heavy-field description forever → eternal `COMPLETE: NO` on a correct artifact. Confirmed: #259351's `tasks.description` column still literally contained `founding_year/funding_total/...`.

**Fix:** persist `description` alongside `context` when it changed. One source of truth.

**This finally answers the recurring "do I need a fresh mission?" question:**
- **Fresh mission = clean.** `description` is written from the CURRENT (fixed) JSON at creation; worker and grader agree from the start. This divergence only ever appears AFTER a JSON instruction edit on an already-created mission.
- **Pre-edit rows = stale until re-run.** With `ee76628c` they converge on the next retry (the refresh now persists). That is why DLQ-retrying #259351 kept failing pre-fix.

**This also dissolved triaged "bug #2"** — see §3.

---

## 3. Triaged "bug #2" (availability retry ladder) — NOT A BUG, retracted

Earlier triage claimed availability failures DLQ at the 6-cap instead of riding the 15-step transient ladder. **False.** Verified the whole path:
- `packages/general_beckman/src/general_beckman/retry.py`: `TRANSIENT_CATEGORIES` includes availability; `effective_max_attempts` returns `max(cap, 15)` for transient.
- `apply.py::_apply_failed` (line 457) resolves category preferring this attempt's result, then **sniff-overrides** to `availability` from the error text (`_classify_availability_text`).
- `apply.py::_retry_or_dlq` persists `error_category=category` on **retry** (line 596); `_dlq_write` persists it on **DLQ** (line 698). (The earlier subagent claim that `_dlq_write` doesn't persist error_category was WRONG.)
- `dead_letter_tasks` table is empty; no task anywhere died transient-at-6.

#259351's `category=quality` DLQ was the grader's stale-spec `COMPLETE: NO` (= bug §2/the divergence), and the 00:23:44 availability blip **recovered** at 00:28. I had conflated a transient blip with the quality DLQ. No fix needed; the ladder machinery is correct (hardened by `7958475e`/`7d8f1e44`/sniff-override).

---

## 4. THE DEEP OPEN BUG — local-model fallback gate (ROOT-CAUSED, NOT fixed)

**Founder's complaint (correct):** "if cloud exhausted, KutAI could have used local." During #259351's run, **every** model call was `local=False` — a local model was never dispatched once, even when cloud rate-limited.

**Confirmed root cause** (logs `kutai.jsonl.1`, 00:23:44, task=researcher):
```
selector eligibility: candidates=53 providers=[openrouter=25, local=11, gemini=10, groq=5, cerebras=2]
rank_candidates result: count=53 top=groq/qwen3-32b(11.7) > cerebras/zai-glm-4.7(4.0) > groq/gpt-oss-120b(3.9) ...
selector: all candidates below pressure threshold urgency=0.50 threshold=-0.75
   scalars=[groq/qwen3-32b=-0.83, cerebras/zai-glm-4.7=-0.95, groq/gpt-oss-120b=-0.95, cerebras/gpt-oss-120b=-0.95, groq/llama-4-scout=-0.95]
ModelCallFailed: No model candidates available (category=availability)
```

- **11 local models passed eligibility and were ranked** — NOT demoted, NOT eligibility-filtered. (The only locals filtered were `Qwen3-Coder` (coding_mismatch/demoted), `Qwen3.5-27B` (demoted), `Apriel` (no_function_calling) — none of these are the FC-capable general locals `Qwen3.5-35B-A3B` / `Qwen3.5-9B`, which DID get admitted fine for other tasks at 00:29.)
- The **pool-pressure gate** (`packages/fatih_hoca/src/fatih_hoca/selector.py:317-360`) is a **hard veto with no floor**:
  ```python
  threshold = max(-1.0, -0.5 - 0.5 * urgency)          # urgency 0.50 → -0.75
  scored_after = [s for s in scored if s.urgency >= threshold and s.urgency > -1.0]
  if scored_after: scored = scored_after
  else: return None        # ← nukes a non-empty, serviceable pool
  ```
  Cloud scalars ≈ −0.95 (correctly penalized — rate-limited/depleted). Under the 00:23 concurrent load, the locals also dipped below −0.75. The gate filtered out the **entire 53-candidate field** and returned None → "no candidates (availability)" → the task burned an attempt with **11 ready local models idle**.

**Design intent vs bug:** the gate treats `−1.0` as "truly dead, never admit" (VRAM-full local / fully-depleted quota) and the `−0.75…−1.0` band as "pressured but alive." When EVERY candidate is in that pressured-but-alive band it returns None instead of keeping the best serviceable one. The gate's own comment forbids "silently relaxing the threshold" (fear of re-picking dead cloud) — but that conflates **rate-limited cloud** (genuinely can't serve) with **local, which has no quota to deplete and CAN serve**.

**Proposed fix direction (for the plan — do NOT blind-patch):** when the gate would empty the pool, fall back to the **best serviceable LOCAL candidate** (`is_local` and `urgency > -1.0`) rather than returning None. Rationale: a pressure-dampened local can actually run; a −0.95 cloud would just re-fail. Preserves the −1.0 hard veto (VRAM-full/busy local stays out). Only return None when there's no serviceable local either (all local at −1.0 + cloud depleted) → then backoff is correct. This delivers "cloud exhausted → use local".

**Mandatory validation (CLAUDE.md Phase 2d rule):** any change to this gate MUST re-run `packages/fatih_hoca/tests/sim/run_scenarios.py` and `run_swap_storm_check.py`. Design reasoning: `docs/architecture/fatih-hoca-phase2d-equilibrium.md`.

**Secondary question to answer in the plan:** *why* did loaded local models dip to ≤ −0.75 under concurrent load at 00:23? If the local utilization/scarcity scalar is itself miscomputed (e.g., local "busy" pressure over-penalizing loaded models that could still serve), that's a related but separate issue. The gate-floor fix is robust either way, but the scalar should be sanity-checked (`fatih_hoca/scarcity.py`, `capability_curve.py`, `ranking.py::_apply_utilization_layer`).

---

## 5. Producer-quality residue (NOT code-fixable) — #259369 journeys

After the analyst-URL fix, #259369's remaining failure was `gemini/gemini-flash-latest` emitting `user_journey_map.stages` as a **prose paragraph** instead of the structured stage array the schema needs (`user_personas` was present and fine). Grader correctly said `COMPLETE: NO`. This is the "weak model emits wrong shape" category. Options if it recurs on the fresh run: structured/constrained emit on step 2.4, exclude gemini-flash for that artifact, or coerce `stages`. Separate from the bugs above.

---

## 6. Environment / how to reproduce the §4 evidence

- Live DB: `C:\Users\sakir\ai\kutai\kutai.db` (DB_PATH in `.env`). Mission 80 + #259351/#259369 were PURGED — gone. Use the logs.
- Logs: `C:\Users\sakir\Dropbox\Workspaces\kutay\logs\`. The §4 selection trace is in **`kutai.jsonl.1`** (rotation covering 2026-06-01T17:22 → 2026-06-03T01:55 UTC). The events are ~00:13–00:32 UTC on 2026-06-03 (= 03:13–03:32 local, UTC+3). Grep `kutai.jsonl.1` for `259351` + `pressure`/`rank_candidates`/`model filtered task=researcher`.
- Grade-child verdicts for §2 are DB task rows `279708..279966` (now purged); the verdict TEXT is also echoed in the logs as `grader:task#259351:...` reviewer outputs.
- venv python: `C:/Users/sakir/Dropbox/Workspaces/kutay/.venv/Scripts/python.exe`. ALWAYS `-p no:warnings` + a timeout. `tests/` and `packages/*/tests/` collide on conftest — run separately. DB-integration tests load the embedding model (~19s) — slow, not hung.
- Restart KutAI via Telegram; never taskkill llama-server.

---

## 7. Suggested first moves next session

1. `git log --oneline -4` to confirm `ee76628c`/`3a026fa8`/`f7d82de5`. Restart KutAI if it hasn't picked them up.
2. **Run a FRESH i2p mission** (founder already purged for this). Watch phase 1/2: 1.3 competitors should now pass (worker + grader read the same fresh light spec via `ee76628c`); 2.4 personas should pass the URL gate (analyst fix) — but may still hit §5 if a weak model mis-shapes the journey.
3. **Plan the §4 local-fallback gate fix** (the real architectural bug). Confirm the §4 secondary question (why loaded locals dipped below threshold) first, then implement the local-floor fallback in `selector.py:317-360`, TDD'd, and re-run the Phase 2d simulator (`run_scenarios.py` + `run_swap_storm_check.py`) — non-negotiable per CLAUDE.md.

---

## 7b. CORRECTION (appended 2026-06-03, later session) — §4 diagnosis was WRONG

A follow-up session verified §4 against the raw `in_flight` + pressure logs in `kutai.jsonl.1` and **§4's root cause and proposed fix are both wrong. Do NOT implement them.**

- **Locals were NOT in a −0.75…−0.95 "alive band."** They pegged at **exactly −1.0** (see the `task=analyst` pressure lines at log offsets ~29326 / ~302829, which print the local scalars verbatim: `Qwen3.5-35B-A3B=-1.00, Qwen3.5-9B=-1.00, gemma-4-26B=-1.00`). The §4 researcher line only printed the top-5 (all cloud) — locals sat below, unprinted. −1.0 = the S9 `LOCAL_BUSY_PENALTY` hard veto, which is **correct**: task **#259413** held the single GPU continuously **00:12:55→00:27:08 (~14 min, gemma-4-26B)** with a clean release (no leak). The §4 failure at 00:23:44 is inside that window.
- **The §4 fix (admit best local `urgency > −1.0`) is dead on arrival:** locals ARE at −1.0 (excluded anyway), and forcing one in runs 2 concurrent local inferences on `--parallel 1` = GPU thrash. The hard veto must stay.
- **§4 caused ZERO terminal failures.** `dead_letter_tasks` is empty for transient; the availability-None just burned retries and recovered at ~00:28. #259351's actual death was the §2 stale-description grader bug — already fixed in `ee76628c`.

**The REAL issue (founder reframed):** "No model candidates available" fires **too often** (16× in this rotation). Root cause:
- Pool-pressure gate (`selector.py:317-360`) returns `None` whenever every candidate is below `threshold = −0.5 − 0.5·urgency`. The gate's own comments call −1.0 the only "dead" value and −0.75…−1.0 "pressured but **alive**" — yet it vetoes the entire alive band.
- Urgency is effectively fixed at **0.50** (`compute_urgency`, `general_beckman/admission.py:22` = `priority/10 + age·0.05 + unblocks·0.05`; age ≤+0.05/24h) → threshold **−0.75**. A serviceable cloud candidate at −0.83 is rejected.
- **No urgency escalation across Beckman retries** — the gate comment promises "back off OR escalate urgency" but NO caller implements escalate (`accelerate_retries`/`capacity_restored` still unimplemented). On retry, urgency≈same → same `None`. The within-ReAct `+0.1` bump (`husam/worker.py:189`, `coulson/dispatch_helpers.py:61`) doesn't help the FIRST selection.

**Fix directions (need plan + TDD + mandatory Phase 2d sim re-run):** (A) escalate urgency by `worker_attempts`; (B) "best-serviceable floor" — never `None` on a non-empty pool with a `>−1.0` candidate; (C) hybrid. Preserve the −1.0 hard veto. Re-run `run_scenarios.py` + `run_swap_storm_check.py` per CLAUDE.md. Memory: `project_no_models_available_gate_20260603`.

## 8. Lessons

- **Read the full evidence before theorizing.** The grader verdict's `SITUATION`/`INSIGHT` text (truncated to 140c in `tasks.error`) named the exact stale fields — the whole §2 root cause was sitting in the grade-child logs. Two subagents produced plausible-but-wrong fix hypotheses (schema-injection for §2-grader, `_dlq_write` persistence gap for §3) that the ground-truth logs/DB disproved.
- **Two readers, one source of truth.** Any time a worker rebuilds config from a live source, every downstream reader (grader, reflect, advance) must read the SAME source or persist the refresh. The expander-frozen `tasks.description` is the recurring trap (docstring already cited a prior instance: "use cases").
- **A gate is a soft preference, not a kill switch.** The pressure gate (§4) returning None on a non-empty serviceable pool is the same class of error as the verify-step dead-ends from the prior handoff: a mechanism meant to *steer* instead *blocks*.
- **Instruction↔schema drift** (§1) is worth a periodic audit, but the snake_case token-diff heuristic is too noisy (flagged 34→97 steps, mostly false positives: input-artifact refs, enum values, post-hook verb names, nested object sub-fields). Only flat-array `item_fields` vs instruction-leaf-fields where the extras require per-entity external research is the harmful pattern (1.3 was the sole confirmed instance; 1.7/1.8 over-ask but the extras come from the same single read, no spiral).
