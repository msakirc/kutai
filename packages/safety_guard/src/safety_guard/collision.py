"""Collision and blocklist detectors. Pure functions, no I/O."""
from __future__ import annotations

import os
import re
import shlex


SHARED_BRANCHES = {"main", "master", "develop"}
SHARED_BRANCH_PATTERNS = [re.compile(r"^release/")]


def _is_shared_branch(branch: str) -> bool:
    if branch in SHARED_BRANCHES:
        return True
    return any(p.match(branch) for p in SHARED_BRANCH_PATTERNS)


_FORCE_PUSH_RE = re.compile(
    r"\bgit\s+push\b.*?(?:\s-f\b|\s--force(?:-with-lease)?\b)"
)


def detect_force_push(cmd: str) -> bool:
    return bool(_FORCE_PUSH_RE.search(cmd))


_REBASE_RE = re.compile(r"\bgit\s+rebase\b")
_RESET_HARD_RE = re.compile(r"\bgit\s+reset\b.*?\s--hard\b")


def detect_shared_history_rewrite(cmd: str, current_branch: str) -> bool:
    if not _is_shared_branch(current_branch):
        return False
    return bool(_REBASE_RE.search(cmd) or _RESET_HARD_RE.search(cmd))


def detect_shell_outside_workspace(cmd: str, workspace_root: str) -> bool:
    workspace_norm = os.path.normpath(os.path.abspath(workspace_root)).rstrip(os.sep)
    workspace_sep = workspace_norm + os.sep
    try:
        tokens = shlex.split(cmd)
    except ValueError:
        return True
    for tok in tokens:
        # Check if token looks like an absolute path (Unix or Windows style)
        if tok.startswith("/") or (len(tok) > 1 and tok[1:3] == ":\\"):
            abs_tok = os.path.normpath(os.path.abspath(tok))
            # Check if path is inside workspace
            if not (abs_tok == workspace_norm or abs_tok.startswith(workspace_sep)):
                return True
    return False


_DESTRUCTIVE_DB_RE = re.compile(
    r"^\s*(DROP\s+TABLE|TRUNCATE(?:\s+TABLE)?)\s+(\w+)", re.IGNORECASE
)


def detect_destructive_shared_db(query: str) -> bool:
    m = _DESTRUCTIVE_DB_RE.match(query)
    if not m:
        return False
    table = m.group(2)
    if re.match(r"^mission_\d+(_|$)", table):
        return False
    return True


_BLOCKLIST_PATTERNS = [
    re.compile(r"\bgit\s+push\b.*?(?:-f|--force(?:-with-lease)?)\b.*\b(main|master)\b"),
    re.compile(r"\bstripe\.charges\.create\b"),
    re.compile(r"\bvercel\s+deploy\s+--prod\b"),
    re.compile(r"\baws\s+s3\s+rm\b"),
    re.compile(r"\brm\s+-rf\s+/(?!\w*tmp)"),
]


def detect_blocklist(cmd: str) -> bool:
    return any(p.search(cmd) for p in _BLOCKLIST_PATTERNS)
