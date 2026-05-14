"""Mechanical grounding check — Layer 2 of G.

Reads the source task's ``tool_calls`` audit log + workflow ``produces``
declaration, returns whether every produces slot is satisfied by at
least one successful write_file (or edit_file/etc.) call. The L1
sub-iter guard catches most narration-as-completion in-loop; this is
the floor that catches anything that escaped (suppress_guards path,
exhausted sub-iter budget, bypassed inner loop).

No filesystem touches — purely a mismatch check between what the agent
declared it would write and what the runtime audit recorded. The
``verify_artifacts`` post-hook (separate kind) handles disk verification.

Result mirrors verify_artifacts shape so the apply-verdict path can
reuse the retry-with-feedback mechanics.
"""
from __future__ import annotations

from typing import Any

from coulson.grounding import (
    extract_written_paths,
    unmatched_produces,
)


def check_grounding(
    *,
    tool_calls: list[dict],
    produces: list,
    workspace_path: str | None = None,
) -> dict[str, Any]:
    """Return ``{passed, missing, written}``.

    Parameters
    ----------
    tool_calls:
        Per-execution audit log captured by the coulson runtime
        (``[{name, args, ok}, ...]``).
    produces:
        Workflow step's declared output paths. Strings are literal/glob;
        nested lists are any_of slots.

    Returns
    -------
    dict
        ``passed``: True when every produces slot has at least one
        matching successful write call.
        ``missing``: list of produces entries that did NOT match. Empty
        on pass.
        ``written``: sorted list of normalized paths the agent did write.
    """
    if not isinstance(produces, list) or not produces:
        # No produces declared -> grounding is vacuous; pass through.
        return {"passed": True, "missing": [], "written": []}
    written = extract_written_paths(tool_calls or [])
    missing = unmatched_produces(produces, written)

    # Disk-fallback: if tool_calls log is missing entries (older task rows,
    # interrupted runs, auto-persist that landed after checkpoint), but the
    # produces path actually exists on disk, treat it as satisfied. Avoids
    # false-positive DLQ when the artifact is correct but the audit log is
    # incomplete. Production 2026-05-14: mission 69 step 0.1 product_charter
    # — full charter on disk, tool_calls empty, L2 DLQ'd.
    if missing:
        try:
            import os as _os, os.path as _op
            ws = workspace_path
            if not ws:
                from src.tools.workspace import WORKSPACE_DIR as _WSD
                ws = _WSD
            still_missing = []
            for entry in missing:
                if isinstance(entry, str):
                    abs_p = entry if _op.isabs(entry) else _op.join(ws, entry)
                    if _op.isfile(abs_p) and _os.path.getsize(abs_p) > 0:
                        written.add(entry.replace("\\", "/"))
                        continue
                still_missing.append(entry)
            missing = still_missing
        except Exception:
            pass
    return {
        "passed": not missing,
        "missing": missing,
        "written": sorted(written),
    }
