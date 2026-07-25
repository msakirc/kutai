"""Scaffold-then-fill for per-screen plans — mechanical frontmatter, model body.

The deepest root for the m90 5.20a/5.20b drift (invented / renamed / dropped /
routeless screens): a screen's MECHANICAL fields (screen_id / route /
mission_id / surface) are DECLARED in ``screen_inventory.md`` — they are not a
model choice. So the engine materializes EXACTLY one file per inventory-chunk
screen with authoritative frontmatter, grafting in the model's authored BODY
where it produced a matching file (matched by slug or route). Files the model
invented (slug/route not in the chunk) are dropped. This makes drift +
routeless / renamed frontmatter structurally impossible; the model authors only
the semantic body, and the shape gate still checks that body is complete.

Pure functions — no I/O. The mechanical executor (mr_roboto dispatch) reads the
inventory + the model's files off the workspace and applies the returned plan.
"""
from __future__ import annotations

import re
from typing import Any

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_ROUTE_IN_BACKTICKS_RE = re.compile(r"`([^`]+)`")
_NAME_RE = re.compile(r"^\s*(.*?)\s*\(`[^`]+`\)\s*$")
_ROUTE_KEY_RE = re.compile(r'^route\s*:\s*(.+?)\s*$', re.MULTILINE)


def _slugify(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", (name or "").strip().lower())
    return s.strip("-")


def _norm_route(r: str) -> str:
    if not isinstance(r, str):
        return ""
    s = r.strip().strip('"').strip("'").strip()
    if len(s) > 1 and s.endswith("/"):
        s = s.rstrip("/")
    return s


def _chunk_screens(
    inventory_text: str, chunk_index: int, cumulative: bool = True
) -> list[dict[str, str]]:
    """Return ``[{name, route, slug}]`` for the inventory chunk(s).

    ``cumulative`` (default) returns chunks ``0..chunk_index`` — the chunk steps
    write into ONE shared ``.screens/`` dir, so by chunk N the dir must hold
    chunks 0..N (materializing chunk 1 must KEEP chunk 0's plans, not drop them)."""
    m = _FRONTMATTER_RE.match(inventory_text or "")
    if not m:
        return []
    try:
        import yaml
        data = yaml.safe_load(m.group(1))
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    chunks = data.get("chunks")
    if not isinstance(chunks, list) or not (0 <= chunk_index < len(chunks)):
        return []
    lo = 0 if cumulative else chunk_index
    screens: list[dict[str, str]] = []
    seen: set[str] = set()
    for i in range(lo, chunk_index + 1):
        for entry in chunks[i]:
            s = str(entry)
            rm = _ROUTE_IN_BACKTICKS_RE.search(s)
            if not rm:
                continue
            route = _norm_route(rm.group(1))
            if route in seen:
                continue
            seen.add(route)
            nm = _NAME_RE.match(s)
            name = (nm.group(1) if nm else s).strip()
            screens.append({"name": name, "route": route, "slug": _slugify(name)})
    return screens


def _frontmatter(slug: str, route: str, mission_id: str, surface: str) -> str:
    return (
        "---\n"
        '_schema_version: "1"\n'
        f"mission_id: {mission_id}\n"
        f"screen_id: {slug}\n"
        f'route: "{route}"\n'
        f"surface: {surface}\n"
        "inherits_shell: []\n"
        "---\n"
    )


def _template_body(name: str) -> str:
    return (
        f"\n# {name}\n\n"
        f"The {name} screen. Author its purpose and content in this step.\n\n"
        "## Overview\n"
        f"- Primary content and actions for {name}.\n\n"
        "## States\n\n"
        "### Default\n- Populated content.\n\n"
        "### Empty\n- Empty-state guidance with a call to action.\n\n"
        "### Loading\n- Skeleton loaders while data is fetched.\n\n"
        "### Error\n- Error message with a retry action.\n"
    )


def _extract_body(model_text: str) -> str:
    """Return the content after the model file's frontmatter block (the body).

    A model file with no frontmatter is treated as all-body. Leading blank lines
    are trimmed; an empty body yields ``""`` (caller falls back to the template).
    """
    if not isinstance(model_text, str):
        return ""
    m = _FRONTMATTER_RE.match(model_text)
    body = model_text[m.end():] if m else model_text
    return body.strip("\n")


def _model_index(model_files) -> tuple[dict[str, str], dict[str, str]]:
    """Index model files by slug (path dir) and by normalized route (frontmatter)."""
    by_slug: dict[str, str] = {}
    by_route: dict[str, str] = {}
    for mf in model_files or []:
        if not isinstance(mf, dict):
            continue
        path = str(mf.get("path") or "").replace("\\", "/")
        text = mf.get("text") or ""
        # slug = the directory name holding screen_plan.md
        parts = [p for p in path.split("/") if p]
        slug = parts[-2] if len(parts) >= 2 and parts[-1].endswith(".md") else ""
        if slug:
            by_slug.setdefault(slug, text)
        fm = _FRONTMATTER_RE.match(text)
        if fm:
            rk = _ROUTE_KEY_RE.search(fm.group(1))
            if rk:
                by_route.setdefault(_norm_route(rk.group(1)), text)
    return by_slug, by_route


def build_screen_plan_files(
    *,
    inventory_text: str,
    chunk_index: int,
    mission_id: str,
    surface: str = "web",
    model_files: list[dict[str, Any]] | None = None,
    cumulative: bool = True,
) -> dict[str, Any]:
    """Materialize the chunk's plan files from the inventory + model bodies.

    Returns ``{targets: [{slug, route, path, content}], invented: [path]}`` —
    ``targets`` is EXACTLY the inventory chunks ``0..chunk_index`` (cumulative,
    matching the correspondence gate; authoritative frontmatter, model body where
    matched else a template body); ``invented`` lists model files whose
    slug/route is in NO covered chunk (removed by the executor).
    """
    model_files = model_files or []
    screens = _chunk_screens(inventory_text, chunk_index, cumulative=cumulative)
    by_slug, by_route = _model_index(model_files)

    targets: list[dict[str, Any]] = []
    matched_paths: set[str] = set()
    chunk_slugs = {s["slug"] for s in screens}

    for s in screens:
        slug, route, name = s["slug"], s["route"], s["name"]
        model_text = by_slug.get(slug) or by_route.get(route)
        body = _extract_body(model_text) if model_text else ""
        if not body:
            body = _template_body(name)
        elif not body.startswith("\n"):
            body = "\n" + body
        content = _frontmatter(slug, route, mission_id, surface) + body + "\n"
        targets.append({
            "slug": slug,
            "route": route,
            "path": f"mission_{mission_id}/.screens/{slug}/screen_plan.md",
            "content": content,
        })

    invented: list[str] = []
    for mf in model_files:
        if not isinstance(mf, dict):
            continue
        path = str(mf.get("path") or "").replace("\\", "/")
        parts = [p for p in path.split("/") if p]
        slug = parts[-2] if len(parts) >= 2 and parts[-1].endswith(".md") else ""
        # Any model file NOT at a canonical chunk slug is removable — a fully
        # invented screen OR a leftover DUPLICATE slug for a valid route (m90:
        # `habits`/`errands` vs canonical `habit-list`/`errands-list`). Its body
        # is already grafted into the canonical target by route match; keeping
        # the stale dir only pollutes the recursive shape glob. Requiring the
        # route to ALSO be foreign (old predicate) let duplicate-slug leftovers
        # survive across retries → shape DLQ on files missing _schema_version.
        if slug and slug not in chunk_slugs:
            invented.append(str(mf.get("path")))

    return {"targets": targets, "invented": invented}
