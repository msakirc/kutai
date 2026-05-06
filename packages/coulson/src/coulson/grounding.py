"""Tool-call grounding helpers.

Pure functions, no side effects. Shared by the L1 sub-iter grounding guard
(``guards.check_grounding_sub_iter``) and the L2 post-hook grounding verdict
(beckman ``apply._apply_grounding_verdict``).

Compares the per-execution ``tool_calls`` audit log captured by the runtime
against the workflow step's declared ``produces`` paths. The agent claims a
build step is done; this module asks "did you actually call write_file for
the path you said you'd produce?"

Public API
----------
WRITE_TOOLS                — frozenset of tools that count as a "write" for
                             grounding purposes (write_file, edit_file, etc.)
extract_written_paths()    — pull successful write paths out of tool_calls
match_produces_entry()     — single produces slot vs written-paths set
unmatched_produces()       — return [entries] with no matching write
build_grounding_message()  — didactic feedback string for retry
"""
from __future__ import annotations

import fnmatch
from typing import Iterable


# Tools that satisfy the "the agent wrote a file" requirement. Each has
# ``path`` / ``filepath`` / ``file`` in its args. Intentionally narrow —
# tool families that only modify in-memory state (read_file, query_codebase,
# shell-without-side-effect) don't count.
WRITE_TOOLS: frozenset[str] = frozenset({
    "write_file",
    "edit_file",
    "patch_file",
    "apply_diff",
    "create_file",
})


_PATH_KEYS: tuple[str, ...] = ("path", "filepath", "file", "target", "dest")


def _normalize(p: str) -> str:
    """Normalize a path for comparison.

    Windows ↔ POSIX slashes, leading ``./`` stripped. Trailing slashes
    kept (a write to ``foo/`` is suspicious anyway). No realpath/abspath
    resolution — produces is workspace-relative and so is the args we
    capture; absolute paths are explicit failures elsewhere.
    """
    if not isinstance(p, str):
        return ""
    s = p.replace("\\", "/").strip()
    while s.startswith("./"):
        s = s[2:]
    return s


def _is_glob(p: str) -> bool:
    return any(c in p for c in "*?[")


def _extract_path_from_args(args: dict) -> str | None:
    """Return the first path-shaped arg value. None if none present."""
    if not isinstance(args, dict):
        return None
    for key in _PATH_KEYS:
        v = args.get(key)
        if isinstance(v, str) and v.strip():
            return v
    return None


def extract_written_paths(tool_calls: Iterable[dict]) -> set[str]:
    """Pull normalized paths from successful write-tool calls.

    Failed calls (``ok: False``) are excluded — a write that errored
    didn't actually land on disk. Non-write tools are skipped. Calls
    missing a path arg are skipped (e.g. shell, web_search).
    """
    out: set[str] = set()
    for call in tool_calls or []:
        if not isinstance(call, dict):
            continue
        if not call.get("ok"):
            continue
        if call.get("name") not in WRITE_TOOLS:
            continue
        path = _extract_path_from_args(call.get("args") or {})
        if not path:
            continue
        out.add(_normalize(path))
    return out


def _match_single(pattern: str, written: set[str]) -> bool:
    """One path/glob pattern vs the set of written paths."""
    if not isinstance(pattern, str):
        return False
    norm = _normalize(pattern)
    if not norm:
        return False
    if _is_glob(norm):
        return any(fnmatch.fnmatch(p, norm) for p in written)
    return norm in written


def match_produces_entry(entry, written: set[str]) -> bool:
    """A produces entry is satisfied if:

    - string literal/glob: matches at least one written path
    - any_of nested list: at least one alternative matches
    """
    if isinstance(entry, list):
        return any(_match_single(alt, written) for alt in entry)
    return _match_single(entry, written)


def unmatched_produces(produces: list, written: set[str]) -> list:
    """Return the produces entries that have NO matching write call.

    Empty list = fully grounded. Non-empty = grounding failed; caller
    builds retry feedback or returns a fail verdict.
    """
    if not isinstance(produces, list):
        return []
    return [e for e in produces if not match_produces_entry(e, written)]


def build_grounding_message(
    *, missing: list, written: set[str], task_title: str = "",
) -> str:
    """Render the didactic retry message for the L1 sub-iter guard.

    Spells out which paths were declared, what was actually written,
    and steers the agent toward a concrete write_file call. Same
    anti-token-flipping framing as the boolean.equals rule.
    """
    declared_lines = []
    for entry in missing:
        if isinstance(entry, list):
            declared_lines.append("  - any of: " + ", ".join(entry))
        else:
            declared_lines.append(f"  - {entry}")
    declared_block = "\n".join(declared_lines) or "  (none)"

    if written:
        written_block = "\n".join(f"  - {p}" for p in sorted(written))
    else:
        written_block = "  (no write_file / edit_file calls succeeded)"

    title = f"Task: {task_title}\n\n" if task_title else ""

    pick = "the path"
    if missing:
        first = missing[0]
        pick = first[0] if isinstance(first, list) and first else (
            first if isinstance(first, str) else "the path"
        )

    return (
        "STOP. You declared this task done but did NOT call write_file "
        "(or edit_file/patch_file) for any of the paths this step is "
        "supposed to produce.\n\n"
        f"{title}"
        f"Declared output paths (each must be written to):\n{declared_block}\n\n"
        f"Successful write-tool calls observed:\n{written_block}\n\n"
        "Call write_file FIRST, then return final_answer. Example:\n"
        "```json\n"
        '{"action": "tool_call", "tool": "write_file", '
        f'"args": {{"filepath": "{pick}", "content": "<the actual file contents>"}}}}'
        "\n```\n\n"
        "Do NOT just re-emit final_answer with the same narration — "
        "that wastes a retry. Do the write, then finish."
    )
