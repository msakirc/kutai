# Handoff — LATENT: grade-gate validates result text, engine gate validates materialized file

**Date:** 2026-06-21
**Status:** LATENT / unconfirmed. Static finding only — **did NOT manifest in mission 87**. Recorded so it is neither missed nor overstated. Verify before treating as a bug; do not "fix" speculatively.

## The seam
Two schema gates run for an i2p step, with **different validation targets**:

1. **Engine post-exec gate** — `src/workflows/engine/hooks.py::_post_execute_workflow_step_impl`, gate at ~:1552. Validates the value returned by `materialize_produces` (`hooks.py:1531` → `:272`), which for a single declared `produces` path **returns the on-disk canonical file content** (`canonical_out`). So this gate validates the FILE.

2. **Grade-posthook gate** — `packages/general_beckman/src/general_beckman/apply.py:1747`:
   ```python
   _draft = source.get("result")          # the agent's final_answer text
   _sg = schema_gate(output_value=_draft, schema=_art_schema)
   ```
   So this gate validates the agent's RESULT STRING.

`materialize_produces` does **not** write its canonical output back into `tasks.result`; `result` stays the agent's final_answer (`coulson/react.py:954` `result = parsed.get("result", content)`). The artifact STORE and the produces FILE get the materialized content; `tasks.result` does not.

## Why it can (in principle) diverge
For a `produces`-file step whose instruction is "**write** the artifact to the produces path" (e.g. `[1.0a]`, `[1.0c]`), a model may write a perfectly valid file and return a short narration as its final_answer ("Wrote prior_art_queries.json with 4 queries"). Then:
- Engine gate (file) → PASS.
- Grade gate (`result` = narration) → text-fallback keyword/section scan on the narration → could FALSE-REJECT.

For "your response text IS the artifact" steps (e.g. `[1.3]`), `result` legitimately carries the artifact, so gate (2) is correct there.

## Why it is only LATENT (the honest part)
In mission 87 the produces-file steps **passed**: `[1.0a]` 524362 completed, `[1.4a]` 524369 completed. So either (a) the materializer/grade ordering already prevents the divergence in practice, (b) the models emitted the artifact inline AND wrote the file, or (c) grade gate (2) is short-circuited for produces steps by something upstream. **This was not run to ground.** Do not assume the bug is live.

## Verify before fixing
1. Find a real `produces`-file step where the agent wrote a valid file but returned narration as `result`, and confirm whether the grade gate (`apply.py:1750`) actually rejects it. Search `model_pick_log`/task history or construct a unit test feeding `schema_gate(output_value="<narration>", schema=<produces step schema>)` and a separately-valid file.
2. If it CAN reject: the fix is to make the grade gate validate the **materialized artifact** (file/store content) for `produces`-file steps, not `source.result` — mirroring gate (1). One option: have `materialize_produces` (or the post-exec hook) persist the canonical content into `tasks.result` (or a dedicated `result_materialized` field the grade gate reads) before the grade posthook runs.
3. If it CANNOT reject (some upstream guard makes them agree): document why and close this as a non-issue.

## Files
- `src/workflows/engine/hooks.py:272` (`materialize_produces`), `:1531` (call), `:1552` (engine gate).
- `packages/general_beckman/src/general_beckman/apply.py:1747` (grade gate target).
- `packages/coulson/src/coulson/react.py:954` (`result` = final_answer text).
- `packages/mr_roboto/src/mr_roboto/schema_gate.py`.

## Do NOT
Do not change the gate target speculatively — `[1.3]`-style "response IS the artifact" steps rely on the grade gate reading `result`. Any fix must be conditional on `produces` being declared.
