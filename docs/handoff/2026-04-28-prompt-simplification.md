# Handoff — Prompt Simplification (next session)

User: sakircimen@gmail.com
Caveman mode: drop articles/filler/pleasantries. Fragments OK. Code/commits/security normal.

---

## The trigger

User just observed **90k tokens of input on a single LLM call**. Unacceptable. Fix.

That's a researcher / coder agent on i2p_v3 step. Likely 4423 (4.5b openapi_spec, 27kB seen historically) or similar. Models choke on 90k input — grok rejects, local models get slow, retry budget burns.

Goal next session: **prompt fits in 20-30k tokens for typical step**. Compression without losing signal.

---

## Where prompts live

All agent prompts assembled in `src/agents/base.py::_build_context(self, task)` (~line 682+). Sections (in current emit order):

| Block | Source | Typical size |
|-------|--------|--------------|
| `## Task (PRIMARY ...)` | task title + description | 0.5-1k |
| `## Additional Context` | task_context.* free-form | 0.5-3k |
| `## Results from Previous Steps` | input_artifacts injected as full JSON dump | **5-30k** ← hot |
| `## Resources` | step.resources from workflow JSON | 1-5k |
| `## Retrieved Knowledge (reference only)` | RAG hits from semantic + episodic | 0.5-3k |
| `## Required Output Format` | schema_dialect.make_example + nudge | 0.5-2k |
| `## IMPORTANT: Previous Output Was Invalid` | retry context | 0.5-2k |
| `## Your Previous Output (fix this, don't start over)` | last attempt full dump | **2-15k** ← hot on retry |
| Skills + prior-steps blocks | dropped on attempts >= 3 (60a4131) | varies |
| System prompt (separate) | DB row from `prompt_versions` | 1-3k |

Probable 90k culprits:
1. **Multiple input_artifacts injected in full** — 4.5b takes `api_resource_model + system_architecture + database_schema + functional_requirements`. Each can be 5-15kB. Total: 20-60kB.
2. **Previous Output dump on retry** — 4.5b's prior output was 30k chars after constrained_emit grew it. On retry attempt 5, full prev dump is in prompt.
3. **No upstream artifact summarization** — artifact_summarizer agents exist (post-hooks) but their summaries aren't always used in place of full artifact in downstream steps.

---

## What's already shipped this week (don't redo)

Recent commits informing prompt build:
- `90acdd2` — recency-order: schema + retry blocks at prompt tail
- `60a4131` — drop skills+prior-steps blocks on retry >= 3
- `f9507bb` — missing-artifact NOTE in `_build_context`
- `f08c9c0` — envelope unwrap before per-artifact checklist parse
- `b7cf388` — `_unwrap_envelope` tolerates list/dict input
- `0856cd5` (just shipped) — E1 nested schema dialect — example block now richer/structured
- `3139a1c` — schema fixes for 6 i2p_v3 steps + prose nudge

So: schema block + retry feedback are NOT the bloat source. Look upstream.

---

## Architecture facts (don't relearn)

- **DB_PATH**: `C:\Users\sakir\ai\kutai\kutai.db` per `.env`. `data/kutai.db` is orphan.
- **Workflow JSON** has mtime cache. Edits propagate without restart. Code edits need restart (`/restart` via Telegram).
- **Workflow loader** is `src/workflows/engine/loader.py`.
- **base.py step-refresh** at dispatch resyncs description/done_when/input_artifacts/output_artifacts/artifact_schema/tools_hint/difficulty + free-form `context.*`.
- **Artifact storage**: `ArtifactStore` (`src/workflows/engine/artifact_store.py`) writes to mission blackboard table. Empty writes refused (e9aff93).
- **Artifact summaries**: `artifact_summarizer` post-hook agent runs on most steps; produces `<artifact>_summary` as separate artifact. Some downstream steps ALREADY use `_summary` (e.g. 3.5 uses `prd_final_summary` not `prd_final`).
- **input_artifacts in workflow JSON** lists what to inject. If both `X` and `X_summary` exist, downstream usually points at the lighter one — but not always consistent.

---

## Investigation starting points

1. **Find the 90k call**: grep `kutai.jsonl` for `User context (` lines → find largest. The number in parens is the prompt size.
   ```
   grep -oP 'User context \(\d+ chars\)' logs/kutai.jsonl | sort -t'(' -k2 -n -r | head
   ```

2. **Audit input_artifacts vs available summaries**: for each i2p_v3 step, check whether `input_artifacts` references full or `_summary` form. Where summary exists and is sufficient, switch the reference.

3. **Cap per-artifact injection size in `_build_context`**: each artifact's full body should have a soft cap (e.g. 8kB). Truncate with `... [truncated, fetch via read_blackboard]` marker. Preserves signal, prevents tail blowups.

4. **Drop `Your Previous Output` after attempt 3** if that's not already the case. Per-artifact checklist (now rich with nested paths via E1) carries the missing-fields signal more concisely than full dump.

5. **Compress `## Resources`**: those are workflow-author-written reference blobs. Some are 5kB+ of conventions text. Move long ones to artifact_store and inject by reference, not inline.

6. **RAG block size**: top-k semantic hits — verify k is reasonable (k=3-5, not k=10).

7. **Profile**: add cumulative section-size logging in `_build_context` so future diagnostics show `task: 0.8k, context: 1.2k, prior_artifacts: 47k, ...`. One-line summary per call.

---

## What NOT to do

- DON'T strip the schema/example block — it's load-bearing for structured output (0856cd5 just made it richer).
- DON'T drop the `_schema_error` retry hint or per-artifact checklist — those are how the model learns from failure (f08c9c0, render_checklist).
- DON'T mass-rewrite all input_artifacts at once — surface the worst offenders, fix surgically.
- DON'T call `call_model()` directly — `LLMDispatcher.request()` only.
- DON'T `taskkill llama-server`. Use `/restart` via Telegram.
- DON'T `pytest` without timeout: `timeout 60 pytest tests/...`

---

## Mission 57 state (canary)

- 290+ completed, 96 pending, 1 processing, 9 skipped, 2 ungraded, 6 DLQ'd then retried
- 4423 (4.5b openapi_spec) — completed at attempt 5 yesterday
- 4441 (5.4b forms_and_states) — completed at attempt 3
- 4409 (3.5 integration_requirements) — was ungraded; today's tools-not-supported groq error stopped it
- **Today's blocker**: groq compound model picked for tool-using researcher → `tool calling not supported` error. User reverted my proposed fix (capability flag in adapter); that whole avenue is OFF the table for this session.

---

## Commits this arc (chronological)

```
0856cd5  feat(schema): E1 nested schema dialect
3139a1c  fix(schema): 6 i2p_v3 schema fixes + prose nudge
5b46e3f  fix(beckman): admission cache (pick spam fix)
389e6c0  fix(vector_store): chroma compaction self-heal + asyncio.to_thread
```

Branch ahead of origin/main by 261+ commits.

---

## Memory entries written this session

- `project_session_20260427_arc.md` — yesterday's session
- (no new memory file written today; this handoff covers it)

Update memory after next session if architectural conclusions change.

---

## Where to start (fresh session)

1. **Pull the prompt-size offender**: grep `User context (` from kutai.jsonl, identify the 90k+ call's task_id and step.
2. **Read its actual prompt** via `[Task #N] User context` log entry — see which section ate the budget.
3. **Pick the largest section** (likely `Results from Previous Steps`).
4. **Audit and fix surgically** — switch to `_summary` artifact, cap per-artifact size, OR move a Resources blob to artifact_store.
5. **Add per-section size logging** in `_build_context` so future regressions are visible.
6. **Restart KutAI** to load base.py changes; workflow JSON edits propagate without restart.

**User's preference**: surface options before destructive change. Don't unilaterally compress/drop blocks. Show what's bloated and propose specific cuts.

**End state goal**: typical step prompt ≤ 30k tokens. Worst case (4.5b openapi_spec with all upstream design docs) ≤ 60k.
