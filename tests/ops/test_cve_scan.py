"""Z8 T5C — cve_scan + secret_scan executor tests."""
from __future__ import annotations

import json
import shutil
import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mr_roboto.executors.cve_scan import run as cve_run
from mr_roboto.executors.secret_scan import run as secret_run


# ─────────────────────── cve_scan tests ──────────────────────────────────


@pytest.mark.asyncio
async def test_cve_scan_empty_packages_returns_ok():
    res = await cve_run({"payload": {"ecosystem": "PyPI", "packages": []}})
    assert res["ok"] is True
    assert res["queried"] == 0
    assert "no packages supplied" in (res["reason"] or "")


class _FakeRespCtx:
    def __init__(self, status: int, json_data):
        self.status = status
        self._json = json_data

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def json(self):
        return self._json


class _FakeSessionCtx:
    def __init__(self, responses):
        self._responses = list(responses)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    def post(self, url, json=None):
        return self._responses.pop(0)


@pytest.mark.asyncio
async def test_cve_scan_clean_package(monkeypatch):
    fake_session = _FakeSessionCtx([_FakeRespCtx(200, {"vulns": []})])
    import aiohttp
    monkeypatch.setattr(aiohttp, "ClientSession", lambda **kw: fake_session)

    res = await cve_run({"payload": {
        "ecosystem": "PyPI",
        "packages": [{"name": "safe-pkg", "version": "1.0"}],
    }})
    assert res["ok"] is True
    assert res["queried"] == 1
    assert res["vulnerabilities"] == []


@pytest.mark.asyncio
async def test_cve_scan_flags_vulns(monkeypatch):
    fake_session = _FakeSessionCtx([_FakeRespCtx(200, {
        "vulns": [{
            "id": "GHSA-xyz",
            "summary": "Bad bug in pkg",
            "severity": [{"type": "CVSS_V3", "score": "7.5"}],
        }],
    })])
    import aiohttp
    monkeypatch.setattr(aiohttp, "ClientSession", lambda **kw: fake_session)

    res = await cve_run({"payload": {
        "ecosystem": "PyPI",
        "packages": [{"name": "evil", "version": "0.1"}],
    }})
    assert res["ok"] is False
    assert len(res["vulnerabilities"]) == 1
    v = res["vulnerabilities"][0]
    assert v["package"] == "evil"
    assert v["id"] == "GHSA-xyz"
    assert v["severity"] == "7.5"


@pytest.mark.asyncio
async def test_cve_scan_handles_osv_500(monkeypatch):
    fake_session = _FakeSessionCtx([_FakeRespCtx(500, {})])
    import aiohttp
    monkeypatch.setattr(aiohttp, "ClientSession", lambda **kw: fake_session)

    res = await cve_run({"payload": {
        "ecosystem": "PyPI",
        "packages": [{"name": "anything", "version": "1.0"}],
    }})
    # Non-200 is silently skipped — vuln list stays empty so ok=True.
    assert res["queried"] == 1
    assert res["vulnerabilities"] == []


# ─────────────────────── secret_scan tests ───────────────────────────────


@pytest.mark.asyncio
async def test_secret_scan_skipped_when_gitleaks_missing(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _cmd: None)
    res = await secret_run({"payload": {"workspace_path": "."}})
    assert res["skipped"] is True
    assert res["ok"] is True
    assert "gitleaks not installed" in (res["reason"] or "")


@pytest.mark.asyncio
async def test_secret_scan_no_findings(monkeypatch, tmp_path):
    monkeypatch.setattr(shutil, "which", lambda _cmd: "/usr/bin/gitleaks")

    def fake_run(cmd, **kw):
        # gitleaks writes report to --report-path; empty list when no findings.
        idx = cmd.index("--report-path")
        report = cmd[idx + 1]
        from pathlib import Path
        Path(report).write_text("[]", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    res = await secret_run({"payload": {"workspace_path": str(tmp_path)}})
    assert res["ok"] is True
    assert res["findings"] == []


@pytest.mark.asyncio
async def test_secret_scan_with_findings(monkeypatch, tmp_path):
    monkeypatch.setattr(shutil, "which", lambda _cmd: "/usr/bin/gitleaks")

    def fake_run(cmd, **kw):
        idx = cmd.index("--report-path")
        report = cmd[idx + 1]
        from pathlib import Path
        Path(report).write_text(json.dumps([{
            "RuleID": "aws-access-key",
            "File": "config.py",
            "StartLine": 42,
            "Commit": "abc123",
        }]), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 1, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    res = await secret_run({"payload": {"workspace_path": str(tmp_path)}})
    assert res["ok"] is False
    assert len(res["findings"]) == 1
    f = res["findings"][0]
    assert f["rule"] == "aws-access-key"
    assert f["file"] == "config.py"
    assert f["line"] == 42


@pytest.mark.asyncio
async def test_secret_scan_timeout(monkeypatch, tmp_path):
    monkeypatch.setattr(shutil, "which", lambda _cmd: "/usr/bin/gitleaks")

    def fake_run(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 1)

    monkeypatch.setattr(subprocess, "run", fake_run)
    res = await secret_run({"payload": {"workspace_path": str(tmp_path)}})
    assert res["ok"] is False
    assert "timed out" in (res["reason"] or "")
