"""Tests for mr_roboto's `run_bash_audit` mechanical action wrapper."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest


def _seed_fake_repo(root: Path) -> None:
    p = root / "packages" / "demo_pkg" / "src" / "demo_pkg" / "__init__.py"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('"""demo_pkg — example."""\n\ndef go():\n    return 1\n', encoding="utf-8")


@pytest.mark.asyncio
async def test_run_invokes_audit_and_returns_summary(tmp_path: Path) -> None:
    _seed_fake_repo(tmp_path)
    from mr_roboto.run_bash_audit import run as bash_audit_run

    out_dir = tmp_path / "docs" / "audits"
    task = {
        "id": 1,
        "payload": {
            "action": "run_bash_audit",
            "root": str(tmp_path),
            "out_dir": str(out_dir),
            "quarter": "2026-Q2",
            "notify": False,
        },
    }
    res = await bash_audit_run(task)
    assert res["ok"] is True
    assert res["quarter"] == "2026-Q2"
    assert "demo_pkg" in Path(res["report_path"]).read_text(encoding="utf-8")
    assert res["layer_count"] >= 1
    assert res["notified"] is False


@pytest.mark.asyncio
async def test_dispatcher_routes_run_bash_audit(tmp_path: Path) -> None:
    """mr_roboto.run() must route action='run_bash_audit' to the wrapper."""
    from mr_roboto import run as mr_roboto_run

    fake_result = {
        "ok": True,
        "quarter": "2026-Q2",
        "report_path": "docs/audits/2026-Q2-bash-audit.md",
        "layer_count": 5,
        "total_loc": 1234,
        "notified": False,
    }
    with patch(
        "mr_roboto.run_bash_audit.run",
        new=AsyncMock(return_value=fake_result),
    ) as m:
        action = await mr_roboto_run(
            {"id": 1, "payload": {"action": "run_bash_audit", "notify": False}}
        )
    m.assert_awaited_once()
    assert action.status == "completed"
    assert action.result == fake_result


@pytest.mark.asyncio
async def test_run_layer_filter(tmp_path: Path) -> None:
    _seed_fake_repo(tmp_path)
    p2 = tmp_path / "packages" / "other_pkg" / "src" / "other_pkg" / "__init__.py"
    p2.parent.mkdir(parents=True, exist_ok=True)
    p2.write_text('"""other_pkg."""\n', encoding="utf-8")

    from mr_roboto.run_bash_audit import run as bash_audit_run

    out_dir = tmp_path / "docs" / "audits"
    task = {
        "id": 1,
        "payload": {
            "action": "run_bash_audit",
            "root": str(tmp_path),
            "out_dir": str(out_dir),
            "quarter": "2026-Q2",
            "layer": "demo_pkg",
            "notify": False,
        },
    }
    res = await bash_audit_run(task)
    body = Path(res["report_path"]).read_text(encoding="utf-8")
    assert "demo_pkg" in body
    assert "other_pkg" not in body


@pytest.mark.asyncio
async def test_run_notify_failure_does_not_break(tmp_path: Path) -> None:
    _seed_fake_repo(tmp_path)
    from mr_roboto.run_bash_audit import run as bash_audit_run

    async def _boom(*a, **kw):
        raise RuntimeError("telegram down")

    with patch("mr_roboto.run_bash_audit._notify_founder", new=_boom):
        out_dir = tmp_path / "docs" / "audits"
        task = {
            "id": 1,
            "payload": {
                "action": "run_bash_audit",
                "root": str(tmp_path),
                "out_dir": str(out_dir),
                "quarter": "2026-Q2",
                "notify": True,
            },
        }
        res = await bash_audit_run(task)
    # Notify failure must NOT fail the task.
    assert res["ok"] is True
    assert res["notified"] is False
