# Handoff — the LLM grader pre-empts the deterministic shape check (false DLQ of compliant producers)

**Date:** 2026-06-04
**Predecessor:** `docs/handoff/2026-06-03-model-selection-and-grader-divergence-handoff.md`
**HEAD:** unchanged from prior session (`ee76628c`) — **nothing shipped this session; this is a root-cause-only handoff. No code written.**

Trigger: fresh mission **#81** DLQ'd `[0.1] product_charter` (task **#289700**, agent `writer`) with:
```
RELEVANT: YES  COMPLETE: NO  VERDICT: FAIL  WELL_FORMED: FAIL  COHERENT: FAIL/PASS
```
This is a **NEW, distinct bug** from anything in the prior handoff — not the stale-description divergence (that's fixed) and not the model-selection gate (see §"related"). It is the deepest instance of this session's recurring theme.

---

## 1. The one-line root cause

**A stochastic LLM grader enforces a DETERMINISTIC structural constraint ("EXACTLY five `## ` sections in order") and is sequenced BEFORE the deterministic checker that exists for exactly that — so the grader's hallucinated section-count DLQs a structurally-compliant artifact before the reliable check ever runs.**

LLMs cannot reliably count sections. Proof: 6 grade children, 6 inconsistent verdicts (§3).

---

## 2. The artifact was COMPLIANT

Step 0.1 instruction (i2p_v3.json:919) requires EXACTLY these five `## ` sections, in order:
1. Product Positioning
2. Brand Keywords (≥5 `**name** — desc` bullets)
3. Core Problem / JTBD
4. Goals & Mission (`Mission:` line + `Desired Outcomes` sub-header, 4-7 bullets)
5. Solutions We Own (3-7 `### <name>` blocks, each with `What it solves`/`Typical path`/`Outcome for the user`/`Boundaries`/`Guiding principles`)

#289700's artifact (`tasks.result`, ~4.9KB) had **exactly those 5 sections, in order**, 3 fully-formed solution blocks (each with all 5 labeled fields), ≥5 brand keywords, no placeholder text. It satisfies the instruction AND the `artifact_schema.required_sections`. **It is compliant.** `verify_charter_shape` (the deterministic check) would PASS it.

---

## 3. The grader is unreliable at structural counting (evidence)

Six `reviewer`/grade-child overhead tasks of #289700 (`289937, 289945, 289953, 289961, 289972, 289982`), all `COMPLETE: NO VERDICT: FAIL`, but their INSIGHTs **contradict each other**:
- `289937/289972`: "added a sixth section" / "added Goals & Mission as a sixth section" — **wrong, Goals & Mission is required #4**
- `289961`: `WELL_FORMED: PASS` but "included extra sections (Goals & Mission, Solutions We Own) not requested" — **both ARE required (#4, #5); grader didn't read the instruction's list**
- `289945/289953/289982`: generic "failed section count/order constraint", `WELL_FORMED: FAIL`, one also `COHERENT: FAIL`

Different models each round (`gemini/gemma-4-31b-it`, `openai/Qwen3.5-9B`, etc.) → different hallucinated counts. The constraint is deterministic; the judge is not.

---

## 4. Why the deterministic check never saved it

Step 0.1 DOES carry the deterministic validator — but in the **`checks` pot**, not `post_hooks` (i2p_v3.json:942-955):
```json
"post_hooks": ["find_similar_missions", "index_idea_fingerprint"],
"checks": [{
  "kind": "verify_charter_shape",
  "payload": {"action": "verify_charter_shape",
              "charter_paths": ["mission_{mission_id}/.charter/product_charter.md"],
              "min_solutions": 3, "max_solutions": 7, "min_brand_keywords": 5}
}]
```
`verify_charter_shape` exists and is real (`packages/general_beckman/src/general_beckman/posthooks.py:504`, `_shape_check_spec`). It was added by the recent "checks pot + convert 34 verify steps to producer post-hooks" work (commit `78ad0c16`).

**But there is NO `verify_charter_shape` execution in the logs for #289700.** The grade gate runs on the producer's output and DLQ's it on `WELL_FORMED: FAIL` first; `checks` evidently run only on/after producer completion, which never happens. The reliable check is sequenced where it cannot pre-empt the unreliable one.

Sequence per attempt (logs `kutai.jsonl`, ~20:10-20:15 UTC 2026-06-03, 6 worker attempts):
1. writer emits charter (e.g. `20:12:42 Raw response 5243c` — compliant markdown).
2. LLM grade gate → `WELL_FORMED: FAIL` (hallucinated section count).
3. retry; 6 attempts; DLQ at the worker cap (`worker_attempts=5/6`, `grade_attempts=0`, `failed_in_phase=worker`, `error_category=None`).

---

## 5. Fix direction (NEEDS A PLAN — do not band-aid)

The principle: **deterministic, countable constraints must be owned by deterministic `checks`; the LLM grader must NOT fail structural axes a `checks` validator owns.** Two shapes to evaluate in the plan:

- **(A) Grader defers structural axes.** For a step that has a shape `checks` entry, the grade prompt/parse should not let `WELL_FORMED`/section-count drive a FAIL — the grader judges only semantic quality (RELEVANT + meaning-level COHERENT). Structure is the check's job.
- **(B) Run `checks` before the grade-fail decision.** A passing deterministic shape-check should prevent (or override) a grader structural FAIL, so the reliable signal wins. Requires reordering: `checks` participate in the producer's pass/fail BEFORE the LLM grade verdict can DLQ.

(B) is the more general fix (the grader stops being able to veto a deterministically-valid artifact), but touches Beckman's grade gate + `checks` execution ordering — find where the grade gate fires vs where `checks` run. Start in `packages/general_beckman/src/general_beckman/apply.py` (grade verdict application, `_apply_posthook_verdict` / grade-child resume) and `hooks.py`/`posthooks.py` (`checks` execution). `src/core/grading.py::build_grading_spec` builds the grader prompt with the WELL_FORMED axis ("no repeated sections, structurally sound") — that axis is what's hallucinating.

Blast radius: this likely DLQs **every structurally-constrained producer** whose shape lives in `checks` — charter (0.1), ADRs (verify_adr_shape), interview script, screen plans, etc. — not just 0.1. High value.

**Secondary (tighten, not the killer):** the writer inconsistently prepends the JSON envelope INTO the `.md` across attempts (e.g. `20:14:25` emitted `{"_schema_version"...}\n\n# Product Charter...`). The instruction says "Also emit a JSON envelope alongside the markdown" — ambiguous; the model sometimes inlines it. Graders cited section count, not this, so it's not the DLQ cause, but the envelope should be a separate field/line, not concatenated into the charter body.

---

## 6. Reproduce / evidence pointers

- Mission **#81**, task **#289700** (live DB `C:\Users\sakir\ai\kutai\kutai.db`). Status `failed`, `worker_attempts=5/6`, `grade_attempts=0`, `failed_in_phase=worker`, `error_category=None`.
- Grade children: tasks `289937/289945/289953/289961/289972/289982` (title `grader:task#289700:*`), `agent_type=reviewer`/overhead — read `tasks.result` for the full verdicts.
- Logs: `logs/kutai.jsonl` (current; ~20:10-20:15 UTC 2026-06-03). Grep `289700`. No `charter_shape` hit = the check never ran.
- Step def: `src/workflows/i2p/i2p_v3.json:919` (0.1). Shape verb: `posthooks.py:504`.
- venv: `C:/Users/sakir/Dropbox/Workspaces/kutay/.venv/Scripts/python.exe`; pytest `-p no:warnings` + timeout; `tests/` vs `packages/*/tests/` conftest collision → run separately. Restart KutAI via Telegram only.

---

## 7. What's verified vs assumed

- **Verified:** artifact is compliant (read it); 6 grade verdicts contradict each other (read them); `verify_charter_shape` exists + is configured in 0.1 `checks`; no check-execution in logs; producer DLQ'd on the grade path.
- **Assumed / confirm in the plan:** exact ordering of grade-gate vs `checks` execution (where does the grade FAIL fire relative to `checks`?). This determines whether fix (A) or (B) is right. Trace it before coding.

---

## 8. Related (from this session, for context)

- **Stale-description divergence** — FIXED `ee76628c` (worker reads live JSON, grader/reflect read frozen `tasks.description`; persist the refresh). See [[project_stale_description_divergence_20260603]].
- **"No models available" gate** — separate model-SELECTION bug; the prior handoff's §4 "no local fallback" diagnosis was CORRECTED (locals were at −1.0 = correct S9 busy veto from a 14-min GPU hold by #259413; real issue is the −0.75…−1.0 alive-band veto + an unwired urgency-escalation safety valve). See [[project_no_models_available_gate_20260603]]. NOT the same as this charter bug.
- **Recurring meta-theme across all of it:** mechanisms meant to *steer/validate* (utilization dampener, output validators, LLM grader) are acting as *hard vetoes* on correct work. The charter bug is the grader-as-veto-of-deterministic-truth instance.

---

## 9. Suggested first moves

1. Trace grade-gate vs `checks` execution order in `general_beckman` (§7 "confirm"). Decide fix (A) vs (B).
2. TDD the fix: a structurally-compliant charter with a `verify_charter_shape` `checks` entry must NOT be DLQ'd by a grader `WELL_FORMED: FAIL`. Use a real-chain test (the prior handoff's lesson: don't unit-test a synthetic shape the real path strips).
3. Generalize: confirm the same gate covers other `checks`-bearing producers (ADR shape, interview, screen plans).
4. Then re-run the affected i2p phases on a fresh mission.
