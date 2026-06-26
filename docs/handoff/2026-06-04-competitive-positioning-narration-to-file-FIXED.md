# Handoff — competitive_positioning DLQ root cause was NARRATION-TO-FILE, not truncation (FIXED)

**Date:** 2026-06-04
**Sibling of:** `docs/handoff/2026-06-04-grader-truncation-root-cause-FIXED.md` — same *family*
(a correct-enough artifact false-DLQ'd by a handling defect) but a **different mechanism**. Not truncation.

---

## 1. Real root cause (proven against live data, mission 81 / task #289715)

Step `1.4a competitive_positioning_lock` (agent=`analyst`, difficulty=hard) DLQ'd after 5 worker attempts:
```
❌ verify_competitive_positioning_shape: missing=['Landscape', 'Value Thesis', 'Strengths']
❌ Schema validation: 'competitive_positioning' missing sections: ['Our Differentiators', 'Switching Costs', 'Notes']
```
The two contradictory missing-halves were the tell: two gates read **two different bad copies**.

What actually happened (from the DB row, not inference):
1. The analyst's `final_answer.result` (6376 chars) is a chat-style report —
   `## Analysis / ### Summary / ### Corrected Artifact Content` — with the **correct, complete
   6-section document buried inside a ```yaml fenced block**.
2. The agent's **only** `write_file` call (`context.tool_calls` n=1) wrote a *different*
   **1419-char narration** ("Findings / Recommendations", zero document) to the declared path
   `mission_81/.prd/competitive_positioning.md`.
3. `verify_competitive_positioning_shape` (check payload names the `.prd/` path) read that narration
   → no `## Landscape/Value Thesis/Strengths` headers, no front-matter → `missing=[…]`, `named=[]` → FAIL.
4. The flat `mission_81/competitive_positioning.md` (= the 6376-char result) is an earlier-attempt
   artifact — **red herring**; the check reads the `.prd/` path explicitly.

**Why both safety nets missed it:**
- `coulson/grounding.py::autopersist_candidate` rescues only a **totally unwritten** produces path
  (gate: `unmatched_produces(...) == produces`). The path WAS written (with garbage), so it skipped.
- The L1/L2 grounding guard only asks *"was the produces path written by some write call?"* — never
  *"is the written content the artifact, vs a report about it?"* A write of pure narration satisfies it.

So a producer that composed the right content but (a) buried it in a fence and (b) wrote a status
report to the file slipped through every gate, and a deterministic verifier correctly rejected the
report. Same class as the truncation bug: the *handling*, not the content, caused the false DLQ.

---

## 2. Fix shipped (Opt 1 — "canonicalize override", deterministic, no new LLM call)

Principle: **the engine guarantees the canonical file IS the best artifact the agent produced,
before any gate reads it.** Generalized auto-persist from "rescue unwritten" to also "replace a
written-but-wrong file" — conservatively.

| File | Change |
|------|--------|
| `packages/coulson/src/coulson/grounding.py` | + `unwrap_fenced_artifact(result)` (pure): extracts the most-substantial artifact-looking ```` ``` ```` fence body (front-matter/header/JSON kept, narration + fence markers stripped). + `recanonicalize_candidate(produces, written, result, *, disk_content, schema_ok)` (pure): for a single written `.md`/`.json` produces path, returns `(path, content)` to OVERWRITE **only when** on-disk content FAILS `schema_ok` AND the result candidate (unwrapped fence, else raw) PASSES it. `schema_ok` injected → module stays pure (no schema/workspace import). |
| `packages/coulson/src/coulson/react.py` | After the existing AUTO-PERSIST block, added a CANONICALIZE OVERRIDE block: builds `schema_ok` from `ctx.artifact_schema` via `src.workflows.engine.hooks.validate_artifact_schema`, reads the on-disk produces file, calls `recanonicalize_candidate`, writes the unwrapped artifact + records a synthetic `recanonicalized: True` write. Fully fail-soft (swallows all exceptions). |

**Conservative by construction** — overrides ONLY when disk fails the schema and result passes, so a
deliberately-written valid file is never clobbered. No-op when `artifact_schema` absent
(`validate_artifact_schema(x, {})` returns `(True, "")` → disk "already good" → skip).

### Tests (TDD)
`packages/coulson/tests/test_recanonicalize.py` (12 new): unwrap (fence body / no-fence / non-str /
largest-block), override (disk-fails+result-passes / disk-valid-no-clobber / unwritten-skip /
candidate-fails-skip / multi-file-skip / non-text-skip / json-parse-guard). Existing
`test_autopersist_candidate.py` (8) still green — `autopersist_candidate` untouched.

### Proven on live data (#289715)
`disk passes schema? False | unwrapped passes? True | override fired? True →
mission_81/.prd/competitive_positioning.md | verify gate ok? True missing [] named 5`.
Note: the **raw** result passes `validate_artifact_schema` (the header scan finds `## Landscape`
*inside* the fence) but FAILS `verify_competitive_positioning_shape` (front-matter `---` not at file
start). **Unwrapping is what makes the file pass both** — writing raw result would not have fixed it.

**Not live until KutAI restart** (via Telegram). Re-run i2p phase 1 (or retry #289715) after restart.

---

## 3. The moonshot (scoped, NOT built — follow-up spec)

Real disease: the LLM owns **two jobs** — *compose* content (good at) and *materialize* it into the
right file / shape / path / front-matter (flaky at). Fence-burial, wrong-path drift, missing
`_schema_version`, narration-to-file are all symptoms of job #2 leaking to the model.

**Separate composition from materialization.** Agent's contract = *produce content*, however messy.
A deterministic **materializer** becomes the **sole writer** of `produces` paths: unwrap fences,
shape-validate, resolve path, and **stamp front-matter the engine already knows** — `mission_id`
(in ctx) + `_schema_version` (in `artifact_schema`). Agent `write_file` demotes to a *hint*, not the
source of truth. Then narration-wrap / wrong-path / fence-burial / forgotten front-matter become
**structurally impossible**, for every artifact type. This fix (Opt 1) is the first brick.

Open design questions for the spec: (a) does the materializer run in-loop (coulson, like this fix) or
as a mechanical post-hook before verify? (b) JSON already has `constrained_emit`; markdown has no
reshape — fold both into one materializer? (c) front-matter stamping must not double-stamp when the
agent already wrote correct front-matter.

---

## 4. Related weakness worth a look (not fixed here)

`validate_artifact_schema` (markdown branch) accepts a fence-buried document because it scans for
`## Section` substrings anywhere in the text, including inside a ```` ``` ```` block. So the in-loop
`autopersist_candidate` (unwritten case) could persist a fence-wrapped raw result that passes the
schema check yet fails the stricter `verify_*` front-matter gate. The override path here dodges it by
unwrapping; consider making `autopersist_candidate` unwrap too (or aligning the two checks).
