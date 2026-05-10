"""Tests for sade_kalsin.audit_report — markdown emission + hot-spot ranking."""
from __future__ import annotations

from pathlib import Path

from sade_kalsin.inventory import LayerReport
from sade_kalsin.audit_questions import audit_questions_for
from sade_kalsin.audit_report import (
    rank_hot_spots,
    render_report,
    write_report,
    quarter_for_date,
)


def _layer(name: str, **kw) -> LayerReport:
    base = dict(
        name=name,
        kind="package",
        path=f"packages/{name}",
        loc=100,
        public_symbols=5,
        test_count=1,
        dependency_count=0,
        rationale=f"{name} — does the thing",
        last_touched_iso="2026-01-01T00:00:00Z",
    )
    base.update(kw)
    return LayerReport(**base)


def test_audit_questions_has_four_items() -> None:
    qs = audit_questions_for(_layer("foo"))
    assert len(qs) == 4
    assert all(isinstance(q, str) and len(q) > 10 for q in qs)
    # Expected substrings from the B12 spec:
    joined = " ".join(qs).lower()
    assert "bash" in joined
    assert "delete" in joined or "deleted" in joined


def test_rank_hot_spots_orders_by_loc_age_inverse_tests() -> None:
    # high-LOC + old + zero-tests should rank above small + young + many-tests
    big_old_untested = _layer(
        "big_old", loc=5000, test_count=0, last_touched_iso="2024-01-01T00:00:00Z"
    )
    small_young_tested = _layer(
        "small", loc=50, test_count=20, last_touched_iso="2026-05-01T00:00:00Z"
    )
    ranked = rank_hot_spots([small_young_tested, big_old_untested])
    assert ranked[0].name == "big_old"
    assert ranked[-1].name == "small"


def test_render_report_emits_required_sections() -> None:
    layers = [_layer("alpha", loc=200), _layer("beta", loc=50, kind="src_module", path="src/beta")]
    md = render_report(layers, quarter="2026-Q2")
    assert "# Bash audit — 2026-Q2" in md
    # per-layer table
    assert "| Layer |" in md
    assert "alpha" in md and "beta" in md
    # aggregate by category
    assert "## Aggregate LOC by category" in md
    # hot-spot ranking
    assert "## Hot-spots" in md
    # 4 audit questions referenced at least once
    assert "What does this layer do that bash" in md


def test_write_report_creates_file(tmp_path: Path) -> None:
    layers = [_layer("alpha")]
    out = tmp_path / "audits" / "2026-Q2-bash-audit.md"
    path = write_report(layers, quarter="2026-Q2", out_path=out)
    assert path.exists()
    assert "2026-Q2" in path.read_text(encoding="utf-8")


def test_quarter_for_date_jan_is_q1() -> None:
    from datetime import date
    assert quarter_for_date(date(2026, 1, 15)) == "2026-Q1"
    assert quarter_for_date(date(2026, 4, 1)) == "2026-Q2"
    assert quarter_for_date(date(2026, 7, 31)) == "2026-Q3"
    assert quarter_for_date(date(2026, 10, 5)) == "2026-Q4"
    assert quarter_for_date(date(2026, 12, 31)) == "2026-Q4"
