"""Asset → spec propagation primitive — Z1 Tier 4 (T4B / B2).

Paraflow's actual differentiator: founder points at an asset and says
"this card is too busy"; the system finds every dependent artifact and
proposes a coordinated patch. Without this, every founder edit becomes a
manual "remember to update the style guide too" treadmill.

Wiring
------
Mechanical action invoked from a Telegram inline button on every
artifact-emit notification. Walks the produces/consumes graph (loaded
from i2p_v3.json) and surfaces the dependents. The actual regeneration
is T4A's territory (``regen_artifact``) — this action only finds WHAT
needs to change.

Output: ``propagation_proposal.md`` listing affected artifacts +
suggested per-artifact patches. Founder reviews via Telegram clarify
shape (accept / reject per item).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable


def _normalize_path(p: str, mission_id: str | int) -> str:
    """Strip mission prefix + leading slashes so paths from different
    sources compare cleanly. Reduces ``mission_1/.style/x`` and
    ``.style/x`` to the same suffix when matching against
    ``produces`` declarations that carry ``mission_{mission_id}/...``.
    """
    s = (p or "").replace("\\", "/").lstrip("./")
    prefix = f"mission_{mission_id}/"
    if s.startswith(prefix):
        s = s[len(prefix):]
    return s


def _resolve_produces(step: dict, mission_id: str | int) -> list[str]:
    """Return all literal/glob produces entries flattened (any_of
    nested lists collapsed). Substitutes ``{mission_id}``."""
    out: list[str] = []
    for entry in step.get("produces") or []:
        if isinstance(entry, str):
            out.append(entry.format(mission_id=mission_id))
        elif isinstance(entry, list):
            for alt in entry:
                if isinstance(alt, str):
                    out.append(alt.format(mission_id=mission_id))
    return out


def _matches_produces(asset_path: str, produces: list[str]) -> bool:
    """Match by directory prefix or exact filename.

    A produces entry like ``mission_1/.style/`` matches any asset under
    that directory; ``mission_1/.style/design_tokens.json`` matches
    only that exact file.
    """
    a = (asset_path or "").replace("\\", "/")
    for p in produces:
        ps = p.replace("\\", "/")
        if ps.endswith("/"):
            if a.startswith(ps):
                return True
        elif a == ps:
            return True
        elif ps.endswith("/*") and a.startswith(ps[:-2] + "/"):
            return True
    return False


def _suggest_patch(
    step: dict,
    change_description: str,
    asset_path: str,
) -> str:
    """Format a one-line suggested patch for a dependent step.

    Cheap heuristic — surfaces the step name + the change verbatim so
    the founder reviewer (and downstream LLM regen) has a consistent
    shape to work from.
    """
    name = step.get("name") or step.get("id") or "unknown"
    return (
        f"Re-emit `{name}` to reflect change in `{asset_path}`: "
        f"{change_description}"
    )


def _build_dep_graph(
    steps: Iterable[dict],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Return (produces_to_step, consumes_from_artifact) maps.

    - ``produces_to_step``: artifact_name -> {step_id, ...} (steps
      whose ``output_artifacts`` list this name)
    - ``consumes_from_artifact``: artifact_name -> {step_id, ...}
      (steps whose ``input_artifacts`` list this name)
    """
    prod: dict[str, set[str]] = {}
    cons: dict[str, set[str]] = {}
    for s in steps:
        sid = s.get("id") or ""
        for a in s.get("output_artifacts", []) or []:
            prod.setdefault(a, set()).add(sid)
        for a in s.get("input_artifacts", []) or []:
            cons.setdefault(a, set()).add(sid)
    return prod, cons


def _render_proposal_markdown(
    asset_path: str,
    change_description: str,
    origin_step_id: str,
    dependents: list[dict],
    upstream_candidates: list[dict],
) -> str:
    lines: list[str] = []
    lines.append("# Propagation Proposal")
    lines.append("")
    lines.append(f"- **Asset**: `{asset_path}`")
    lines.append(f"- **Origin step**: `{origin_step_id}`")
    lines.append(f"- **Change**: {change_description}")
    lines.append("")
    lines.append("## Downstream dependents (will likely need regen)")
    lines.append("")
    if not dependents:
        lines.append("_None — this asset is a leaf._")
    for d in dependents:
        lines.append(f"- **`{d['step_id']}`** ({d.get('step_name', '')})")
        lines.append(f"  - Suggested patch: {d['suggested_patch']}")
    lines.append("")
    lines.append("## Upstream candidates (consider patching the source)")
    lines.append("")
    if not upstream_candidates:
        lines.append("_None._")
    for u in upstream_candidates:
        lines.append(f"- **`{u['step_id']}`** ({u.get('step_name', '')})")
        lines.append(f"  - Suggested patch: {u['suggested_patch']}")
    lines.append("")
    lines.append(
        "_Founder reviews via Telegram clarify (accept/reject per item). "
        "Accepted items are dispatched to `regen_artifact` (T4A)._"
    )
    return "\n".join(lines) + "\n"


def _load_workflow_steps(workflow_path: str | None) -> list[dict]:
    if not workflow_path:
        return []
    try:
        data = json.loads(Path(workflow_path).read_text(encoding="utf-8"))
    except Exception:
        return []
    return list(data.get("steps") or [])


def propagate_asset_change(
    *,
    asset_path: str,
    change_description: str,
    steps: list[dict] | None = None,
    workflow_path: str | None = None,
    mission_id: str | int = "0",
    out_path: str | None = None,
) -> dict[str, Any]:
    """Walk produces/consumes graph for ``asset_path`` and propose
    coordinated patches for all dependents.

    Parameters
    ----------
    asset_path:
        Mission-relative path the founder pointed at, e.g.
        ``"mission_1/.style/design_tokens.json"``.
    change_description:
        Short founder-supplied description of the desired change.
    steps:
        Pre-loaded workflow step list. If absent, ``workflow_path``
        is consulted.
    workflow_path:
        Path to ``i2p_v3.json``. Ignored when ``steps`` is given.
    mission_id:
        Mission ID — substituted into ``produces`` placeholders.
    out_path:
        If supplied, the rendered ``propagation_proposal.md`` is
        written here.

    Returns
    -------
    dict
        ``ok``, ``origin_step_id``, ``dependents`` (list), ``upstream_candidates``
        (list), ``proposal_path`` (when written), ``error`` (when ``ok=False``).
    """
    if not steps:
        steps = _load_workflow_steps(workflow_path)
    if not steps:
        return {
            "ok": False,
            "error": "no workflow steps supplied (steps= or workflow_path=)",
        }

    # Step 1: find producing step
    origin_step_id: str | None = None
    origin_artifact_names: list[str] = []
    for s in steps:
        prods = _resolve_produces(s, mission_id)
        if _matches_produces(asset_path, prods):
            origin_step_id = s.get("id")
            origin_artifact_names = list(s.get("output_artifacts") or [])
            origin_step = s
            break
    else:
        return {
            "ok": False,
            "error": f"no producing step matches asset_path={asset_path!r}",
        }

    # Step 2: find downstream consumers (steps whose input_artifacts
    # list any of the producer's output_artifacts).
    _prod_map, cons_map = _build_dep_graph(steps)
    dependent_ids: set[str] = set()
    for art in origin_artifact_names:
        dependent_ids |= cons_map.get(art, set())
    dependent_ids.discard(origin_step_id)

    step_by_id = {s.get("id"): s for s in steps}
    dependents: list[dict[str, Any]] = []
    for sid in sorted(dependent_ids, key=lambda x: (x or "")):
        s = step_by_id.get(sid)
        if not s:
            continue
        dependents.append({
            "step_id": sid,
            "step_name": s.get("name", ""),
            "input_artifacts": list(s.get("input_artifacts", []) or []),
            "produces": _resolve_produces(s, mission_id),
            "suggested_patch": _suggest_patch(s, change_description, asset_path),
        })

    # Step 3: upstream candidates — when origin is a leaf (no
    # downstream consumers), surface the origin's input-producing
    # steps. Founder may want to patch the source instead of the leaf.
    upstream_candidates: list[dict[str, Any]] = []
    if not dependents:
        prod_map, _ = _build_dep_graph(steps)
        upstream_ids: set[str] = set()
        for art in origin_step.get("input_artifacts", []) or []:
            upstream_ids |= prod_map.get(art, set())
        upstream_ids.discard(origin_step_id)
        for sid in sorted(upstream_ids, key=lambda x: (x or "")):
            s = step_by_id.get(sid)
            if not s:
                continue
            upstream_candidates.append({
                "step_id": sid,
                "step_name": s.get("name", ""),
                "output_artifacts": list(s.get("output_artifacts", []) or []),
                "suggested_patch": _suggest_patch(s, change_description, asset_path),
            })

    proposal_md = _render_proposal_markdown(
        asset_path=asset_path,
        change_description=change_description,
        origin_step_id=origin_step_id or "?",
        dependents=dependents,
        upstream_candidates=upstream_candidates,
    )

    proposal_path: str | None = None
    if out_path:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
            Path(out_path).write_text(proposal_md, encoding="utf-8")
            proposal_path = out_path
        except Exception as e:
            return {"ok": False, "error": f"write proposal failed: {e}"}

    return {
        "ok": True,
        "origin_step_id": origin_step_id,
        "origin_artifacts": origin_artifact_names,
        "dependents": dependents,
        "upstream_candidates": upstream_candidates,
        "proposal_md": proposal_md,
        "proposal_path": proposal_path,
    }
