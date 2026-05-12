"""Z8 T5B — dependency_scan executor tests."""
from __future__ import annotations

import json
import shutil
import subprocess
from unittest.mock import patch

import pytest

from mr_roboto.executors.dependency_scan import run as dep_run


@pytest.mark.asyncio
async def test_python_skipped_when_pip_audit_missing(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _cmd: None)
    res = await dep_run({"payload": {"ecosystem": "python"}})
    assert res["skipped"] is True
    assert res["ok"] is True
    assert "pip-audit not installed" in (res["reason"] or "")


@pytest.mark.asyncio
async def test_python_no_vulns(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _cmd: "/usr/bin/pip-audit")
    fake = subprocess.CompletedProcess(
        ["pip-audit"], 0,
        stdout=json.dumps({"dependencies": [
            {"name": "ok-pkg", "version": "1.0", "vulns": []},
        ]}).encode(),
        stderr=b"",
    )
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: fake)
    res = await dep_run({"payload": {"ecosystem": "python"}})
    assert res["ok"] is True
    assert res["vulnerabilities"] == []


@pytest.mark.asyncio
async def test_python_flags_vulnerable_pkg(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _cmd: "/usr/bin/pip-audit")
    fake = subprocess.CompletedProcess(
        ["pip-audit"], 1,
        stdout=json.dumps({"dependencies": [{
            "name": "requests",
            "version": "2.20.0",
            "vulns": [{
                "id": "PYSEC-2023-001",
                "fix_versions": ["2.31.0"],
                "description": "header injection",
            }],
        }]}).encode(),
        stderr=b"",
    )
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: fake)
    res = await dep_run({"payload": {"ecosystem": "python"}})
    assert res["ok"] is False
    assert len(res["vulnerabilities"]) == 1
    v = res["vulnerabilities"][0]
    assert v["package"] == "requests"
    assert v["id"] == "PYSEC-2023-001"
    assert v["fix_versions"] == ["2.31.0"]


@pytest.mark.asyncio
async def test_python_handles_bad_json(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _cmd: "/usr/bin/pip-audit")
    fake = subprocess.CompletedProcess(
        ["pip-audit"], 0, stdout=b"not json", stderr=b"",
    )
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: fake)
    res = await dep_run({"payload": {"ecosystem": "python"}})
    assert res["ok"] is False
    assert "JSON parse" in (res["reason"] or "")


@pytest.mark.asyncio
async def test_node_skipped_when_npm_missing(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _cmd: None)
    res = await dep_run({"payload": {"ecosystem": "node"}})
    assert res["skipped"] is True
    assert res["ok"] is True


@pytest.mark.asyncio
async def test_node_flags_vulnerable_pkg(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _cmd: "/usr/bin/npm")
    fake = subprocess.CompletedProcess(
        ["npm", "audit"], 1,
        stdout=json.dumps({"vulnerabilities": {
            "lodash": {"severity": "high", "via": ["CVE-2021-0001"], "fixAvailable": True},
        }}).encode(),
        stderr=b"",
    )
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: fake)
    res = await dep_run({"payload": {"ecosystem": "node"}})
    assert res["ok"] is False
    assert res["vulnerabilities"][0]["package"] == "lodash"
    assert res["vulnerabilities"][0]["severity"] == "high"


@pytest.mark.asyncio
async def test_unsupported_ecosystem():
    res = await dep_run({"payload": {"ecosystem": "rust"}})
    assert res["ok"] is False
    assert "unsupported ecosystem" in (res["reason"] or "")
