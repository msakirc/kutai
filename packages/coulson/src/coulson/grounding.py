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
import re
from typing import Callable, Iterable


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
    # A trailing-slash pattern is a DIRECTORY slot (5.20a `.screens/`, 5.30a
    # `.web/` author an unknown number of files): satisfied when at least one
    # written path lives under it. The kept trailing slash makes this a
    # boundary-respecting prefix — `.screens/` is not matched by `.screens2/…`.
    if norm.endswith("/"):
        return any(w.startswith(norm) for w in written)
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


# Matches a fenced code block, capturing the body (without the fence lines).
# Tolerates an optional language tag (```yaml / ```md / ```json / ```).
_FENCE_RE = re.compile(r"```[^\n`]*\n(.*?)\n```", re.DOTALL)


def _looks_like_artifact(body: str) -> bool:
    """A fence body that looks like a document/data artifact, not a shell
    snippet — starts with YAML front-matter, a markdown header, or JSON."""
    s = body.strip()
    return bool(s) and (
        s.startswith("---") or s.startswith("#")
        or s.startswith("{") or s.startswith("[")
    )


def unwrap_fenced_artifact(result) -> str | None:
    """Extract a buried artifact from a narration-wrapped ``result``.

    Cloud LLMs frequently answer with a chat-style report
    (``## Analysis / ### Summary / ### Corrected Artifact Content``) that
    embeds the *real* document inside a ```` ```yaml ```` / ```` ```md ````
    fence. Persisting the raw narration to the produces path makes the
    verify gate read a report instead of the artifact (mission 81, 1.4a).

    Returns the inner body of the most-substantial artifact-looking fenced
    block (front-matter / header / JSON preserved, fence markers + narration
    stripped). Returns ``None`` when ``result`` is not a string or has no
    fenced block — the caller then falls back to the raw ``result``.
    """
    if not isinstance(result, str):
        return None
    blocks = _FENCE_RE.findall(result)
    if not blocks:
        return None
    artifactish = [b for b in blocks if _looks_like_artifact(b)] or blocks
    best = max(artifactish, key=lambda b: len(b.strip())).strip()
    return best or None


import json as _json

# Leading YAML front-matter: ``---\n...\n---`` at the very start of the file.
_FRONT_MATTER_RE = re.compile(r"^---\n(.*?)\n---\n?", re.DOTALL)


def stamp_front_matter(content: str, mission_id: int, kind: str) -> str:
    """Idempotently stamp ``mission_id`` into an artifact's metadata.

    ``md``  : ensure a leading ``---`` front-matter block carrying
              ``mission_id``. Inject the key if the block exists without it;
              prepend a minimal block if absent; no-op if already present.
              Never produces a second ``---`` block (handoff Q(c)).
    ``json``: ensure a top-level ``mission_id`` key. No-op if present or the
              content does not parse (best-effort — never corrupt a file).
    """
    if not isinstance(content, str):
        return content
    if kind == "json":
        try:
            obj = _json.loads(content)
        except (ValueError, TypeError):
            return content
        if isinstance(obj, dict) and "mission_id" not in obj:
            obj = {"mission_id": mission_id, **obj}
            return _json.dumps(obj, ensure_ascii=False, indent=2)
        return content

    # markdown
    m = _FRONT_MATTER_RE.match(content)
    if m:
        body = m.group(1)
        if re.search(r"^\s*mission_id\s*:", body, re.MULTILINE):
            return content  # already stamped — idempotent
        new_block = f"---\n{body}\nmission_id: {mission_id}\n---\n"
        return new_block + content[m.end():]
    return f"---\nmission_id: {mission_id}\n---\n\n{content}"


def select_canonical(candidates, schema_ok: Callable[[str], bool]):
    """Pick the best artifact form from competing candidates, by priority.

    ``candidates`` is an ordered list of raw strings by PRIORITY, highest
    first (the materializer passes ``[disk_content, output_value]`` — the
    agent's committed on-disk file outranks the final_answer body so a rich
    valid artifact is never clobbered by a thinner result, intake #73). For
    each source the fence-unwrapped form is preferred over the raw form when
    it also passes ``schema_ok`` (strips a narration wrapper, mission 81).

    Returns the first priority source's passing form (unwrapped if it passes,
    else raw); if no source passes, the most-substantial form overall so a
    file always exists. ``None`` only when there is no usable candidate.
    """
    all_forms: list[str] = []
    for c in candidates:
        if not isinstance(c, str) or not c.strip():
            continue
        group: list[str] = []
        u = unwrap_fenced_artifact(c)
        if isinstance(u, str) and u.strip() and u.strip() != c.strip():
            group.append(u)   # unwrapped preferred within this source
        group.append(c)
        all_forms.extend(group)
        for form in group:
            if schema_ok(form):
                return form
    if not all_forms:
        return None
    return max(all_forms, key=lambda f: len(f.strip()))


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
