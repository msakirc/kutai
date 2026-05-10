"""Tests for sade_kalsin.inventory — package/layer walker."""
from __future__ import annotations

from pathlib import Path

from sade_kalsin.inventory import LayerReport, walk_layers, _count_loc, _public_symbols


def _mk(p: Path, body: str = "") -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")


def test_walk_layers_finds_packages_and_src(tmp_path: Path) -> None:
    # Fake repo layout: 1 package + 1 src module
    _mk(
        tmp_path / "packages" / "fake_pkg" / "src" / "fake_pkg" / "__init__.py",
        '"""fake_pkg — does the thing nothing else does."""\n\ndef hello():\n    return 1\n',
    )
    _mk(
        tmp_path / "packages" / "fake_pkg" / "tests" / "test_smoke.py",
        "def test_x():\n    assert True\n",
    )
    _mk(
        tmp_path / "src" / "core" / "thing.py",
        '"""thing module."""\n\ndef do_it():\n    return 2\n',
    )

    layers = walk_layers(tmp_path)
    names = {layer.name for layer in layers}
    assert "fake_pkg" in names
    assert "src/core" in names

    pkg = next(l for l in layers if l.name == "fake_pkg")
    assert pkg.kind == "package"
    assert pkg.loc > 0
    assert pkg.test_count == 1
    assert "fake_pkg" in pkg.rationale  # docstring sniffed
    assert pkg.public_symbols >= 1


def test_count_loc_excludes_blank_and_comments(tmp_path: Path) -> None:
    f = tmp_path / "x.py"
    f.write_text(
        "# comment\n"
        "\n"
        "def foo():\n"
        "    return 1\n"
        "    # inline\n",
        encoding="utf-8",
    )
    # 2 real source lines: `def foo():`, `return 1`
    assert _count_loc(f) == 2


def test_public_symbols_counts_top_level_defs(tmp_path: Path) -> None:
    f = tmp_path / "y.py"
    f.write_text(
        "def public_one():\n    pass\n\n"
        "def _private():\n    pass\n\n"
        "class PublicCls:\n    pass\n\n"
        "class _Hidden:\n    pass\n",
        encoding="utf-8",
    )
    assert _public_symbols(f) == 2


def test_layer_report_shape() -> None:
    r = LayerReport(
        name="x",
        kind="package",
        path="packages/x",
        loc=100,
        public_symbols=5,
        test_count=2,
        dependency_count=0,
        rationale="x — does y",
        last_touched_iso=None,
    )
    d = r.to_dict()
    assert d["name"] == "x"
    assert d["loc"] == 100
