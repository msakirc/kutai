# synthetic_check_lighthouse v1

Runs `lighthouse` CLI against `TARGET_URL` and emits JSON metrics. The
`synthetic_check` executor parses the output, persists a row in
`perf_baselines`, and compares against the last green baseline.

If any metric regresses by more than `REGRESSION_THRESHOLD_PCT` vs the last
green baseline, the task is marked `regression_detected` so the on-call agent
can decide whether to roll back.

Requires `lighthouse` on PATH. Skipped with `skipped=true` when absent.
