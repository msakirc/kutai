"""Screen-plan ⇄ inventory correspondence gate.

Mechanical post-hook asserting that the per-screen plans a chunk step produced
are EXACTLY the screens declared for that chunk in ``screen_inventory.md`` —
no invented screens, none dropped. m90 5.20a/5.20b invented their own screen
set (Dashboard/Habit Editor/Habit Tracker/…) and ignored
``screen_inventory.chunks[0]`` (Landing/Sign Up/Login/Forgot Password); the
shape gate validated FORM but never CORRESPONDENCE, so the drift passed
silently and 14/19 inventory screens were never planned.

Keys on ROUTE, not screen_id: models rename screen_ids freely (Dashboard →
``dashboard``, invented "Habit Tracker") but the route (``/signup``,
``/habits/:id``) is the stable contract that downstream HTML / code depend on.

Pure function — no I/O. The caller (mr_roboto dispatch) reads the plan files and
the inventory off the mission workspace and passes their text. An unparseable /
chunk-less inventory yields ``empty`` (a vacuous safe pass — false-blocking a
producer is worse than a missed drift the reviewer still backstops).
"""
from __future__ import annotations

import re
from typing import Any

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_ROUTE_IN_BACKTICKS_RE = re.compile(r"`([^`]+)`")
_ROUTE_KEY_RE = re.compile(r'^route\s*:\s*(.+?)\s*$', re.MULTILINE)


def _norm_route(r: str) -> str:
    """Normalize a route for comparison: strip quotes/whitespace, drop a
    trailing slash (except the root ``/``)."""
    if not isinstance(r, str):
        return ""
    s = r.strip().strip('"').strip("'").strip()
    if len(s) > 1 and s.endswith("/"):
        s = s.rstrip("/")
    return s


def _inventory_chunks(inventory_text: str) -> list[list[str]] | None:
    """Return the ``chunks`` list-of-lists of routes, or None if unparseable.

    Each chunk entry is a label like ``Landing Page (`/`)``; the route is the
    backtick-quoted token. ``yaml.safe_load`` is the codebase's frontmatter
    arbiter (handles block + flow sequences)."""
    m = _FRONTMATTER_RE.match(inventory_text or "")
    if not m:
        return None
    try:
        import yaml
        data = yaml.safe_load(m.group(1))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    chunks = data.get("chunks")
    if not isinstance(chunks, list):
        return None
    out: list[list[str]] = []
    for chunk in chunks:
        if not isinstance(chunk, list):
            continue
        routes: list[str] = []
        for entry in chunk:
            for rm in _ROUTE_IN_BACKTICKS_RE.findall(str(entry)):
                routes.append(_norm_route(rm))
        out.append([r for r in routes if r])
    return out


def _plan_route(plan_text: str) -> str | None:
    m = _FRONTMATTER_RE.match(plan_text or "")
    block = m.group(1) if m else (plan_text or "")
    rm = _ROUTE_KEY_RE.search(block)
    return _norm_route(rm.group(1)) if rm else None


def verify_screen_plans_match_inventory(
    *,
    plan_texts: list[str] | None = None,
    inventory_text: str = "",
    chunk_index: int = 0,
    cumulative: bool = True,
) -> dict[str, Any]:
    """Assert the produced plans' routes == the inventory chunk's routes.

    Parameters
    ----------
    plan_texts
        Frontmatter-bearing text of every produced ``screen_plan.md``.
    inventory_text
        Full text of ``screen_inventory.md``.
    chunk_index
        Which chunk this step produces (0 = chunk a, 1 = chunk b, …).
    cumulative
        When True (chunk steps write into ONE shared ``.screens/`` dir, so by
        chunk N the dir must hold chunks 0..N), the expected set is the union of
        chunks ``0..chunk_index``. When False, only ``chunk_index``.

    Returns ``{ok, expected, produced, missing, extra, empty}`` (routes).
    """
    plan_texts = plan_texts or []
    chunks = _inventory_chunks(inventory_text)

    expected: set[str] = set()
    if chunks:
        lo = 0 if cumulative else chunk_index
        for i in range(lo, chunk_index + 1):
            if 0 <= i < len(chunks):
                expected |= set(chunks[i])

    produced = {r for r in (_plan_route(t) for t in plan_texts) if r}

    empty = not expected
    missing = sorted(expected - produced)
    extra = sorted(produced - expected)
    ok = empty or (not missing and not extra)
    return {
        "ok": ok,
        "expected": sorted(expected),
        "produced": sorted(produced),
        "missing": missing,
        "extra": extra,
        "empty": empty,
    }
