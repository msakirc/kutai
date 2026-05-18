"""Audit report — markdown emitter + hot-spot ranking.

Hot-spot formula (intentionally simple, tunable):

    score = log1p(LOC) * age_factor * inverse_test_factor

- ``age_factor``: 1.0 + months_since_last_touched / 12 (clamped 1..4)
- ``inverse_test_factor``: 1.0 / (1 + test_count) — heavily punishes 0 tests

Layers with no last-touched info default to age_factor=2.0 (mid-stale).
The formula intentionally favours LOC × age × untested-ness, matching the
"stay simple" prior: big old code with no tests is the prime candidate
for the bash-replaces-it audit.
"""
from __future__ import annotations

import math
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path

from sade_kalsin.inventory import LayerReport
from sade_kalsin.audit_questions import audit_questions_for, AUDIT_QUESTIONS


# ---- quarter helpers -------------------------------------------------------


def quarter_for_date(d: date) -> str:
    q = (d.month - 1) // 3 + 1
    return f"{d.year}-Q{q}"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


# ---- ranking ---------------------------------------------------------------


def _months_since(iso: str | None) -> float:
    if not iso:
        return 6.0  # mid-stale default
    try:
        # Tolerate trailing Z or offset-aware ISO
        s = iso.replace("Z", "+00:00") if iso.endswith("Z") else iso
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return 6.0
    delta = _now_utc() - dt
    return max(0.0, delta.days / 30.0)


def _hot_spot_score(layer: LayerReport) -> float:
    age_f = 1.0 + _months_since(layer.last_touched_iso) / 12.0
    age_f = max(1.0, min(age_f, 4.0))
    inv_tests = 1.0 / (1.0 + max(0, layer.test_count))
    return math.log1p(max(0, layer.loc)) * age_f * inv_tests


def rank_hot_spots(layers: list[LayerReport]) -> list[LayerReport]:
    return sorted(layers, key=_hot_spot_score, reverse=True)


# ---- markdown rendering ----------------------------------------------------


def _fmt_loc(n: int) -> str:
    return f"{n:,}"


def render_report(layers: list[LayerReport], quarter: str) -> str:
    lines: list[str] = []
    lines.append(f"# Bash audit — {quarter}")
    lines.append("")
    lines.append(
        "_Quarterly check: what does each scaffolding layer do that bash + "
        "Claude can't? Mini-SWE-agent showed 65% SWE-bench in 100 LOC + "
        "bash — every layer below is on trial._"
    )
    lines.append("")

    # Per-layer table
    lines.append("## Per-layer inventory")
    lines.append("")
    lines.append(
        "| Layer | Kind | LOC | Pub-syms | Tests | Deps | Last touched | Rationale |"
    )
    lines.append(
        "|---|---|---:|---:|---:|---:|---|---|"
    )
    for layer in sorted(layers, key=lambda x: x.name):
        rationale = (layer.rationale or "—").replace("|", "\\|")
        if len(rationale) > 80:
            rationale = rationale[:77] + "..."
        last = layer.last_touched_iso or "—"
        lines.append(
            f"| `{layer.name}` | {layer.kind} | {_fmt_loc(layer.loc)} | "
            f"{layer.public_symbols} | {layer.test_count} | "
            f"{layer.dependency_count} | {last[:10]} | {rationale} |"
        )
    lines.append("")

    # Aggregate by category (kind)
    lines.append("## Aggregate LOC by category")
    lines.append("")
    by_kind: dict[str, list[LayerReport]] = {}
    for layer in layers:
        by_kind.setdefault(layer.kind, []).append(layer)
    lines.append("| Category | Layers | LOC | Tests |")
    lines.append("|---|---:|---:|---:|")
    for kind, group in sorted(by_kind.items()):
        loc_sum = sum(layer.loc for layer in group)
        test_sum = sum(layer.test_count for layer in group)
        lines.append(
            f"| {kind} | {len(group)} | {_fmt_loc(loc_sum)} | {test_sum} |"
        )
    lines.append("")

    # Hot-spots ranked
    lines.append("## Hot-spots (LOC x age x inverse-tests)")
    lines.append("")
    lines.append("Top candidates for the four-question interrogation below.")
    lines.append("")
    lines.append("| Rank | Layer | Score | LOC | Tests | Last touched |")
    lines.append("|---:|---|---:|---:|---:|---|")
    ranked = rank_hot_spots(layers)
    for i, layer in enumerate(ranked[: min(15, len(ranked))], 1):
        score = _hot_spot_score(layer)
        last = (layer.last_touched_iso or "—")[:10]
        lines.append(
            f"| {i} | `{layer.name}` | {score:.2f} | {_fmt_loc(layer.loc)} "
            f"| {layer.test_count} | {last} |"
        )
    lines.append("")

    # Standing audit questions (canonical, layer-agnostic header)
    lines.append("## The four audit questions")
    lines.append("")
    # Render with a placeholder-free generic phrasing for the canonical list;
    # per-hot-spot blocks below substitute the layer name into Q2.
    _generic = [q.replace("{layer}", "this layer") for q in AUDIT_QUESTIONS]
    for i, q in enumerate(_generic, 1):
        lines.append(f"{i}. {q}")
    lines.append("")

    # Per-layer interrogation block (top 5 hot-spots get verbatim Q2)
    lines.append("## Per-hot-spot interrogation")
    lines.append("")
    for layer in ranked[: min(5, len(ranked))]:
        lines.append(f"### `{layer.name}`")
        lines.append("")
        lines.append(f"- **Path:** `{layer.path}`")
        lines.append(f"- **LOC:** {_fmt_loc(layer.loc)}  •  **Tests:** {layer.test_count}  •  **Last touched:** {(layer.last_touched_iso or '—')[:10]}")
        lines.append(f"- **Rationale:** {layer.rationale or '—'}")
        lines.append("")
        for i, q in enumerate(audit_questions_for(layer), 1):
            lines.append(f"  {i}. {q}")
            lines.append("     - _answer:_")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_report(
    layers: list[LayerReport],
    quarter: str,
    out_path: Path,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_report(layers, quarter), encoding="utf-8")
    return out_path


def run_audit(
    root: Path,
    quarter: str | None = None,
    out_dir: Path | None = None,
    layer_filter: str | None = None,
) -> dict:
    """Inventory + render + write. Returns a tiny summary."""
    from sade_kalsin.inventory import walk_layers

    root = Path(root)
    layers = walk_layers(root)
    if layer_filter:
        layers = [layer for layer in layers if layer.name == layer_filter or layer.name.endswith("/" + layer_filter)]
    quarter = quarter or quarter_for_date(date.today())
    out_dir = Path(out_dir) if out_dir else (root / "docs" / "audits")
    out_path = out_dir / f"{quarter}-bash-audit.md"
    write_report(layers, quarter, out_path)
    return {
        "ok": True,
        "quarter": quarter,
        "report_path": str(out_path),
        "layer_count": len(layers),
        "total_loc": sum(layer.loc for layer in layers),
        "layers": [asdict(layer) for layer in layers],
    }
