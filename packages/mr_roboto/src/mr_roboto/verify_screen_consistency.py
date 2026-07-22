"""Cross-screen consistency verifier — Z1 Tier 3 (C18 + B8).

Mechanical sibling step that asserts a chunk's screens declare the same
``inherits_shell`` (or that any screen overriding it includes an explicit
``<!-- inherits_shell_override: <reason> -->`` comment in the body).

Per B8 the chunk size is capped at 3-5 screens — this verifier is run once
per chunk, after both ``5.1`` and ``5.2`` chunked sub-steps emit their
artifacts, to catch shell drift before it cascades into the HTML
prototypes.

Inputs (caller wires via payload):

- ``screen_plan_paths``: list of paths to ``screen_plan.md`` files in the
  chunk
- ``shared_shell_components`` (optional): list of canonical shell
  component names (from ``shared_shell.md``); when present, every plan's
  ``inherits_shell`` must be a subset of this list

Returns
-------
dict
    ``ok`` (bool), ``shells_per_screen`` (dict), ``mismatches`` (list),
    ``override_comments`` (dict), ``problems`` (list of strings).
"""
from __future__ import annotations

import re
from typing import Any

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_INHERITS_RE = re.compile(
    r'^inherits_shell\s*:\s*\[(.*?)\]\s*$', re.MULTILINE
)
_SCREEN_ID_RE = re.compile(
    r'^screen_id\s*:\s*["\']?([^"\'\n]+)["\']?\s*$', re.MULTILINE
)
_OVERRIDE_RE = re.compile(
    # Allow hyphens inside the reason (e.g. "full-screen onboarding"); HTML
    # comment terminator `-->` is what we anchor on. We accept any chars
    # except a literal `-->` sequence by using a lazy `.+?` and anchoring
    # to `-->`. DOTALL keeps multiline reasons supported.
    r"<!--\s*inherits_shell_override\s*:\s*(.+?)\s*-->",
    re.IGNORECASE | re.DOTALL,
)


def _parse_inherits_shell(md: str) -> tuple[str | None, list[str], str | None]:
    """Return ``(screen_id, inherits_shell_list, override_reason)``."""
    m = _FRONTMATTER_RE.match(md)
    if not m:
        return None, [], None
    block = m.group(1)
    sid_m = _SCREEN_ID_RE.search(block)
    sid = sid_m.group(1).strip() if sid_m else None
    inh_m = _INHERITS_RE.search(block)
    inherits: list[str] = []
    if inh_m:
        inner = inh_m.group(1).strip()
        if inner:
            # Split on `,` and strip surrounding quotes/whitespace.
            inherits = [
                tok.strip().strip('"').strip("'")
                for tok in inner.split(",")
                if tok.strip()
            ]
    body = md[m.end():]
    ov_m = _OVERRIDE_RE.search(body)
    override = ov_m.group(1).strip() if ov_m else None
    return sid, inherits, override


def verify_screen_consistency(
    *,
    screen_plan_paths: list[str] | None = None,
    screen_plans: dict[str, str] | None = None,
    shared_shell_components: list[str] | None = None,
) -> dict[str, Any]:
    """Cross-screen `inherits_shell` consistency check.

    Either ``screen_plan_paths`` (read from disk) or ``screen_plans``
    (in-memory ``{screen_id: markdown}``) must be supplied. The chunk is
    consistent when every plan declares the SAME ``inherits_shell`` list
    (set-equality), unless a plan carries an explicit
    ``<!-- inherits_shell_override: <reason> -->`` comment in the body.
    """
    plans: dict[str, tuple[list[str], str | None]] = {}
    problems: list[str] = []

    sources: list[tuple[str, str]] = []  # (label, markdown)
    if screen_plan_paths:
        import os
        import glob as _glob
        # Per-screen plans are written under a runtime dir
        # (mission_<id>/.screens/) whose individual filenames are unknown at
        # workflow-author time, so a `checks` entry points at the DIRECTORY.
        # Expand any directory entry to its contained .md files (sorted for
        # determinism); plain file entries are read directly as before.
        # RECURSE — production nests one subdir per screen
        # (`.screens/<slug>/screen_plan.md`); a flat `*.md` glob missed them.
        expanded: list[str] = []
        for p in screen_plan_paths:
            if os.path.isdir(p):
                expanded.extend(
                    sorted(_glob.glob(os.path.join(p, "**", "*.md"), recursive=True))
                )
            else:
                expanded.append(p)
        for p in expanded:
            try:
                with open(p, encoding="utf-8") as fh:
                    sources.append((p, fh.read()))
            except OSError as e:
                problems.append(f"could not read {p}: {e}")
    if screen_plans:
        sources.extend(screen_plans.items())

    if not sources:
        return {
            "ok": False,
            "error": "no screen_plan_paths or screen_plans supplied",
            "shells_per_screen": {},
            "mismatches": [],
            "override_comments": {},
            "problems": problems or ["no input"],
        }

    for label, md in sources:
        sid, inherits, override = _parse_inherits_shell(md)
        key = sid or label
        plans[key] = (inherits, override)

    # Pick the modal (most common) inherits_shell across the chunk as the
    # canonical reference. Plans that carry an explicit override comment
    # are excluded from canonical-determination — they self-declare as
    # exceptions, so they shouldn't drag the canonical with them.
    by_shell: dict[tuple[str, ...], int] = {}
    for sid, (inherits, override) in plans.items():
        if override:
            continue
        key = tuple(sorted(inherits))
        by_shell[key] = by_shell.get(key, 0) + 1
    if by_shell:
        canonical_key = max(by_shell.items(), key=lambda kv: kv[1])[0]
    else:
        # Edge case: every plan has an override comment. Fall back to
        # the first plan's shell.
        first_inherits = next(iter(plans.values()))[0] if plans else []
        canonical_key = tuple(sorted(first_inherits))
    canonical = list(canonical_key)

    mismatches: list[dict[str, Any]] = []
    override_comments: dict[str, str] = {}
    for sid, (inherits, override) in plans.items():
        if tuple(sorted(inherits)) != canonical_key:
            if override:
                override_comments[sid] = override
            else:
                mismatches.append({
                    "screen_id": sid,
                    "found": inherits,
                    "expected": canonical,
                })

    # Shared-shell allow-list check (if supplied).
    out_of_set: list[dict[str, Any]] = []
    if shared_shell_components is not None:
        allowed = set(shared_shell_components)
        for sid, (inherits, _override) in plans.items():
            unknown = [c for c in inherits if c not in allowed]
            if unknown:
                out_of_set.append({"screen_id": sid, "unknown_components": unknown})

    if mismatches:
        problems.append(
            f"{len(mismatches)} screen(s) declare inherits_shell that "
            f"differs from the chunk canonical without an override comment"
        )
    if out_of_set:
        problems.append(
            f"{len(out_of_set)} screen(s) reference shell components not "
            f"in shared_shell_components"
        )

    ok = not mismatches and not out_of_set

    return {
        "ok": ok,
        "canonical_inherits_shell": canonical,
        "shells_per_screen": {sid: inh for sid, (inh, _o) in plans.items()},
        "mismatches": mismatches,
        "override_comments": override_comments,
        "out_of_set": out_of_set,
        "problems": problems,
    }
