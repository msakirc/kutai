# P2 Prompt Simplification — Validation Procedure

Branch: `prompt-simplification-p2` (HEAD `ee62b3b`)
Anchor: 2026-04-30
Author setup: caveman terse + telemetry shipped on main (`fa760ba`)

---

## What P2 changes

1. **`## Additional Context` deny-list** — drops 17 meta/plumbing keys
   (workflow_step_id, step_name, workflow_phase, mission_id, chat_id,
   is_workflow_step, may_need_clarification, triggers_clarification,
   difficulty, needs_thinking, prefer_quality, condition, step_type,
   trigger, skip_when, skip_when_expr, failed_models, classification,
   tools_hint). Free-form step keys still pass through.
2. **System-prompt INPUT ARTIFACTS warning removed** — user-prompt deps
   header carries the same instruction whenever deps exist.

Smoke test on representative i2p step: 60% reduction on Additional
Context block (665 → 265 chars), ~280 chars saved on system prompt
per workflow step. Combined ~680 chars per call.

## Risks under test

| Risk | Signal that fired |
|------|-------------------|
| Agent re-fetches injected input_artifacts | `read_file` / `read_pdf` calls on artifact names already in `## Results from Previous Steps` |
| Free-form step config silently broken | Workflow author notices missing field; grader cites missing context |
| Retry rate climbs | More tasks hit `_schema_error` retry; `worker_attempts >= 2` count up |
| Grader pass rate drops | First-attempt grader-FAIL count climbs |
| Hidden meta-key dependency | Some agent code path was reading mission_id/etc. from prompt instead of `task[...]` |

---

## Baseline (collect on main)

Pre-requisite: P1 (commit `6bec6dd`) live on main, telemetry from
`fa760ba` emitting `context sections (total=Xc): ...` lines.

Collect over 24h or 1 full i2p mission, whichever lands first.

### Command kit

```bash
# Per-section size distribution for workflow steps
grep -oP 'context sections \(total=\d+c\):.*' logs/kutai.jsonl \
  | grep -oP 'Additional Context=\d+c' \
  | grep -oP '\d+' \
  | sort -n | awk 'BEGIN{c=0;s=0} {a[c++]=$1; s+=$1} END{
    print "n="c, "mean="s/c, "p50="a[int(c*0.5)],
    "p90="a[int(c*0.9)], "p99="a[int(c*0.99)], "max="a[c-1]
  }'

# Retry rate
sqlite3 kutai.db "SELECT
  SUM(CASE WHEN worker_attempts >= 2 THEN 1 ELSE 0 END) AS retried,
  COUNT(*) AS total,
  100.0 * SUM(CASE WHEN worker_attempts >= 2 THEN 1 ELSE 0 END) / COUNT(*) AS pct
  FROM tasks
  WHERE created_at > datetime('now', '-24 hours')
    AND json_extract(context, '\$.is_workflow_step') = 1;"

# Grader first-pass rate
sqlite3 kutai.db "SELECT
  COUNT(*) AS graded,
  SUM(CASE WHEN status='completed' AND worker_attempts=1 THEN 1 ELSE 0 END) AS first_pass,
  SUM(CASE WHEN error_category='quality' THEN 1 ELSE 0 END) AS quality_fail
  FROM tasks
  WHERE created_at > datetime('now', '-24 hours')
    AND json_extract(context, '\$.is_workflow_step') = 1;"

# Re-fetch detection — read_file / read_pdf calls on declared input_artifacts
grep -E "tool_call.*\"(read_file|read_pdf|read_docx|read_blackboard)\"" logs/kutai.jsonl \
  | wc -l
```

Save numbers as `BASELINE_*`.

---

## Test run (on P2 branch)

```bash
git checkout prompt-simplification-p2
# /restart via Telegram (loads new code; workflow JSON edits propagate without restart)
```

Then run an i2p mission of comparable shape to baseline (same workflow
version, similar idea complexity). Or wait 24h of organic mission
activity.

Collect the same metrics as baseline.

---

## Decision criteria

Merge P2 to main if ALL of:

| Metric | Threshold |
|--------|-----------|
| `Additional Context` p50 | ≥ 40% smaller than baseline |
| Retry rate | within ±15% of baseline (not more than 1.15× baseline) |
| Grader first-pass rate | within ±5pp of baseline |
| `read_*` calls on injected input_artifacts | ≤ 1.10× baseline |

Hold or revert if ANY of:

- Retry rate > 1.20× baseline
- Grader first-pass rate drops > 5pp absolute
- Re-fetch attempts > 1.25× baseline
- Workflow author flags a missing step-level config key (audit
  `_drop_meta` for an over-aggressive entry)

---

## Rollback path

```bash
git checkout main
# /restart via Telegram
```

`prompt-simplification-p2` branch stays for follow-up tuning. If only
one of the two changes regressed (e.g. system-prompt warning removal
hurt but deny-list was fine), cherry-pick the safe half to main.

---

## Notes

- Telemetry from `fa760ba` is the validation input. It must be live on
  both branches to compare. If main loses the telemetry commit (e.g.
  another rebase) re-cherry-pick before baseline.
- Tool-call telemetry uses raw kutai.jsonl grep; not yet a dedicated
  metric. Consider adding a counter in dispatcher post-hook if this
  becomes a routine check.
- `_drop_meta` is conservative — keeps any unknown key. New step-level
  config keys (e.g. `per_site_n`, `requires_grading`) survive without
  code changes. If a workflow author wants to FORCE-drop a key, add to
  `_drop_meta` directly.
