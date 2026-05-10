"""Smoke test for `python -m sade_kalsin audit` CLI."""
from __future__ import annotations

from pathlib import Path

from sade_kalsin.__main__ import main


def _seed(tmp_path: Path) -> None:
    p = tmp_path / "packages" / "demo" / "src" / "demo" / "__init__.py"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('"""demo — example."""\n\ndef go():\n    return 1\n', encoding="utf-8")
    t = tmp_path / "packages" / "demo" / "tests" / "test_demo.py"
    t.parent.mkdir(parents=True, exist_ok=True)
    t.write_text("def test_go():\n    assert True\n", encoding="utf-8")


def test_cli_audit_writes_report(tmp_path: Path) -> None:
    _seed(tmp_path)
    out_dir = tmp_path / "docs" / "audits"
    rc = main(
        [
            "audit",
            "--quarter",
            "2026-Q2",
            "--root",
            str(tmp_path),
            "--out-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    report = out_dir / "2026-Q2-bash-audit.md"
    assert report.exists()
    body = report.read_text(encoding="utf-8")
    assert "demo" in body


def test_cli_layer_filter(tmp_path: Path) -> None:
    _seed(tmp_path)
    # add a second package that should be filtered out
    p2 = tmp_path / "packages" / "other" / "src" / "other" / "__init__.py"
    p2.parent.mkdir(parents=True, exist_ok=True)
    p2.write_text('"""other."""\n', encoding="utf-8")

    out_dir = tmp_path / "docs" / "audits"
    rc = main(
        [
            "audit",
            "--quarter",
            "2026-Q2",
            "--root",
            str(tmp_path),
            "--out-dir",
            str(out_dir),
            "--layer",
            "demo",
        ]
    )
    assert rc == 0
    body = (out_dir / "2026-Q2-bash-audit.md").read_text(encoding="utf-8")
    assert "demo" in body
    assert "other" not in body
