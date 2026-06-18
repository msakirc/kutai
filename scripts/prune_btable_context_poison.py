"""One-shot: prune context-budget-bloat poison from the B-table.

Background (mission 86 / step 1.4a, 2026-06-18): ``compute_layer_budgets``
scaled the context-layer pool to ``model_ctx * 0.40`` with no ceiling, so a
gemini-class 1M-token window produced a 400k pool. The deps + board layers
filled it with the legacy completed-results dump and the ~102k-token mission
blackboard → ~190k prompt tokens. The B-table rollup
(``model_call_tokens`` → ``step_token_stats``, 14-day p90) learned that, the
estimator forced a 226k ``ctx_needed``, every model was filtered (window +
free-tier TPM), and the task DLQ'd — self-reinforcing.

``context_policy.CONTEXT_ABS_CAP`` (64k) fixes the *cause*. This script clears
the already-learned poison so the estimate recovers immediately instead of
waiting ~14 days for the bad rows to age out of the rollup window.

RUN ONLY AFTER the cap fix is live (restart), so cleaned rows can't be
re-poisoned by a fresh oversized run.

Usage:
    python scripts/prune_btable_context_poison.py <db_path>            # dry-run
    python scripts/prune_btable_context_poison.py <db_path> --apply    # execute

Deletes:
  * ``model_call_tokens`` rows with ``prompt_tokens > CAP`` (the bloated
    source samples — provably impossible post-cap).
  * ``step_token_stats`` rows with ``in_p90 > CAP`` (the poisoned rollups).
The next scheduled rollup regenerates ``step_token_stats`` for the affected
steps from the surviving (sub-cap) samples; steps with no surviving sample
fall back to the ``AGENT_REQUIREMENTS`` default until fresh samples land.
"""
from __future__ import annotations

import os
import sqlite3
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    db_path = sys.argv[1]
    apply = "--apply" in sys.argv[2:]
    cap = int(os.getenv("KUTAI_CONTEXT_ABS_CAP", "65536"))

    con = sqlite3.connect(db_path)
    try:
        n_raw = con.execute(
            "SELECT COUNT(*) FROM model_call_tokens WHERE prompt_tokens > ?", (cap,)
        ).fetchone()[0]
        n_roll = con.execute(
            "SELECT COUNT(*) FROM step_token_stats WHERE in_p90 > ?", (cap,)
        ).fetchone()[0]

        print(f"cap = {cap}")
        print(f"poisoned model_call_tokens rows (prompt_tokens > cap): {n_raw}")
        print(f"poisoned step_token_stats rows   (in_p90 > cap):       {n_roll}")
        # Show the affected steps for the operator's confidence.
        rows = con.execute(
            "SELECT workflow_step_id, agent_type, in_p90, samples_n "
            "FROM step_token_stats WHERE in_p90 > ? ORDER BY in_p90 DESC", (cap,)
        ).fetchall()
        for step, agent, in_p90, n in rows:
            print(f"  rollup: {step} ({agent}) in_p90={in_p90} samples={n}")

        if not apply:
            print("\nDRY-RUN — re-run with --apply to delete.")
            return 0

        con.execute("DELETE FROM model_call_tokens WHERE prompt_tokens > ?", (cap,))
        con.execute("DELETE FROM step_token_stats WHERE in_p90 > ?", (cap,))
        con.commit()
        print(f"\nDELETED {n_raw} source rows + {n_roll} rollup rows. "
              "Next rollup regenerates clean estimates.")
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
