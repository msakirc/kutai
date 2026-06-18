"""One-shot: clear context-bloat poison from the B-table rollup.

Background (mission 86 / step 1.4a, 2026-06-18): ``compute_layer_budgets``
scaled the context-layer pool to ``model_ctx * 0.40`` with no ceiling, so a
gemini-class 1M-token window produced a 400k pool. The deps + board layers
filled it with the legacy completed-results dump and the ~102k-token mission
blackboard → ~190k prompt tokens. The rollup
(``model_call_tokens`` → ``step_token_stats``, 14-day p90) learned that, the
estimator forced a ~226k ``ctx_needed``, every model was filtered (window +
free-tier TPM), and the task DLQ'd — self-reinforcing.

The cause is fixed by ``context_policy.CONTEXT_ABS_CAP`` (32k pool) and the
rollup now drops samples above ``SANE_MAX_PROMPT_TOKENS`` (64k) so bloat can
never re-enter the estimate. This script clears the *already-learned* poison
row(s) in ``step_token_stats`` so the estimate recovers immediately instead of
waiting for the next hourly rollup.

It does NOT delete ``model_call_tokens`` rows — those are the cost/usage ledger
(``cost_by_iteration`` etc.) and the rollup's new sanity filter already excludes
the bloated samples from the estimate. Only the derived ``step_token_stats`` row
is removed; the next rollup regenerates it cleanly from the surviving (≤64k)
samples, or the step falls back to the agent default until fresh samples land.

Safe to run on the live DB (single short transaction, busy_timeout), but the
cap fix must be live first (restart) so cleared rows can't be re-poisoned.

Usage:
    python scripts/prune_btable_context_poison.py <db_path>            # dry-run
    python scripts/prune_btable_context_poison.py <db_path> --apply    # execute
"""
from __future__ import annotations

import os
import sqlite3
import sys


def _sane_max() -> int:
    try:
        return int(os.environ["KUTAI_BTABLE_SANE_MAX_PROMPT"])
    except (KeyError, ValueError, TypeError):
        return 65536


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    db_path = sys.argv[1]
    apply = "--apply" in sys.argv[2:]
    ceil = _sane_max()

    con = sqlite3.connect(db_path)
    con.execute("PRAGMA busy_timeout=5000")
    try:
        rows = con.execute(
            "SELECT workflow_step_id, agent_type, in_p90, samples_n "
            "FROM step_token_stats WHERE in_p90 > ? ORDER BY in_p90 DESC", (ceil,)
        ).fetchall()
        print(f"sane_max prompt_tokens = {ceil}")
        print(f"poisoned step_token_stats rows (in_p90 > sane_max): {len(rows)}")
        for step, agent, in_p90, n in rows:
            print(f"  {step} ({agent}) in_p90={in_p90} samples={n}")
        print("(model_call_tokens cost ledger is left intact by design.)")

        if not apply:
            print("\nDRY-RUN — re-run with --apply to delete the rollup row(s).")
            return 0

        con.execute("BEGIN")
        cur = con.execute("DELETE FROM step_token_stats WHERE in_p90 > ?", (ceil,))
        con.commit()
        print(f"\nDELETED {cur.rowcount} rollup row(s). Next rollup regenerates "
              "clean estimates from surviving samples.")
        return 0
    except sqlite3.OperationalError as exc:
        con.rollback()
        print(f"ERROR (DB busy? retry when idle): {exc}", file=sys.stderr)
        return 1
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
