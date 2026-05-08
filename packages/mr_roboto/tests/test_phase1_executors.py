"""Phase-1 real-tools executors: social_preview_check + staging_smoke_check.

These executors stitch the lower-level verbs (parse_og_tags, http_check) to
the workflow artifact store. We monkeypatch `_load_artifact` and the
underlying verb to keep the suite hermetic — no network, no DB.
"""
from __future__ import annotations

import importlib

import pytest

import mr_roboto


social = importlib.import_module("mr_roboto.executors.social_preview_check")
staging = importlib.import_module("mr_roboto.executors.staging_smoke_check")


# ---------- social_preview_check --------------------------------------------


@pytest.mark.asyncio
async def test_social_preview_check_no_mission_id():
    res = await social.run({"payload": {"action": "social_preview_check"}})
    assert res["ok"] is False
    assert "mission_id" in res["error"]


@pytest.mark.asyncio
async def test_social_preview_check_artifact_missing(monkeypatch):
    async def _none(_m, _n):
        return None
    monkeypatch.setattr(social, "_load_artifact", _none)

    res = await social.run({"mission_id": 1, "payload": {}})
    assert res["ok"] is False
    assert "missing" in res["error"]


@pytest.mark.asyncio
async def test_social_preview_check_no_urls(monkeypatch):
    async def _empty(_m, _n):
        return {"some_unrelated_field": "x"}
    monkeypatch.setattr(social, "_load_artifact", _empty)

    res = await social.run({"mission_id": 1, "payload": {}})
    assert res["ok"] is False
    assert "no URLs" in res["error"]


@pytest.mark.asyncio
async def test_social_preview_check_all_pass(monkeypatch):
    async def _art(_m, _n):
        return {"social_preview_urls": ["https://a.test", "https://b.test"]}
    monkeypatch.setattr(social, "_load_artifact", _art)

    async def _fake_parse(url, **kw):
        return {"ok": True, "url": url, "tags": {}, "missing": []}
    monkeypatch.setattr(social, "parse_og_tags", _fake_parse)

    res = await social.run({"mission_id": 1, "payload": {}})
    assert res["ok"] is True
    assert res["pass_fail"] == "pass"
    assert res["passed"] == 2
    assert res["total"] == 2


@pytest.mark.asyncio
async def test_social_preview_check_partial_fail(monkeypatch):
    async def _art(_m, _n):
        return {"urls": ["https://a.test", "https://b.test"]}
    monkeypatch.setattr(social, "_load_artifact", _art)

    async def _fake_parse(url, **kw):
        return {"ok": "a.test" in url, "url": url}
    monkeypatch.setattr(social, "parse_og_tags", _fake_parse)

    res = await social.run({"mission_id": 1, "payload": {}})
    assert res["ok"] is False
    assert res["pass_fail"] == "fail"
    assert res["passed"] == 1
    assert res["total"] == 2


@pytest.mark.asyncio
async def test_social_preview_check_via_dispatcher(monkeypatch):
    async def _art(_m, _n):
        return {"urls": ["https://x.test"]}
    monkeypatch.setattr(social, "_load_artifact", _art)

    async def _fake_parse(url, **kw):
        return {"ok": True, "url": url}
    monkeypatch.setattr(social, "parse_og_tags", _fake_parse)

    action = await mr_roboto.run({
        "mission_id": 7,
        "payload": {"action": "social_preview_check"},
    })
    assert action.status == "completed"
    assert action.result["ok"] is True


# ---------- staging_smoke_check ---------------------------------------------


@pytest.mark.asyncio
async def test_staging_smoke_no_mission_id():
    res = await staging.run({"payload": {"action": "staging_smoke_check"}})
    assert res["ok"] is False
    assert res["smoke_tests_passed"] is False


@pytest.mark.asyncio
async def test_staging_smoke_no_input_artifact():
    task = {"mission_id": 1, "payload": {}, "context": {"input_artifacts": []}}
    res = await staging.run(task)
    assert res["ok"] is False
    assert "no input artifact" in res["error"]


@pytest.mark.asyncio
async def test_staging_smoke_artifact_missing(monkeypatch):
    async def _none(_m, _n):
        return None
    monkeypatch.setattr(staging, "_load_artifact", _none)
    task = {
        "mission_id": 1,
        "payload": {"artifact": "auth__staging_deployment_result"},
    }
    res = await staging.run(task)
    assert res["ok"] is False
    assert "missing" in res["error"]


@pytest.mark.asyncio
async def test_staging_smoke_no_url_field(monkeypatch):
    async def _art(_m, _n):
        return {"deployed": True}
    monkeypatch.setattr(staging, "_load_artifact", _art)
    task = {
        "mission_id": 1,
        "payload": {"artifact": "auth__staging_deployment_result"},
    }
    res = await staging.run(task)
    assert res["ok"] is False
    assert "no url field" in res["error"]


@pytest.mark.asyncio
async def test_staging_smoke_happy(monkeypatch):
    async def _art(_m, _n):
        return {"deployed": True, "url": "https://staging.test/auth"}
    monkeypatch.setattr(staging, "_load_artifact", _art)

    async def _fake_check(url, **kw):
        return {"ok": True, "final_status": 200, "attempts": 1, "elapsed_s": 0.1, "final_error": None}
    monkeypatch.setattr(staging, "http_check", _fake_check)

    task = {
        "mission_id": 1,
        "payload": {"artifact": "auth__staging_deployment_result"},
    }
    res = await staging.run(task)
    assert res["ok"] is True
    assert res["smoke_tests_passed"] is True
    assert res["url"] == "https://staging.test/auth"


@pytest.mark.asyncio
async def test_staging_smoke_resolves_from_input_artifacts(monkeypatch):
    """When payload doesn't set artifact, we fall back to scanning
    input_artifacts for the canonical suffix. This is the path the template
    expander produces (per-feature prefixed names)."""
    captured = {}

    async def _art(mid, name):
        captured["name"] = name
        return {"url": "https://staging.test/x"}
    monkeypatch.setattr(staging, "_load_artifact", _art)

    async def _fake_check(url, **kw):
        return {"ok": True, "final_status": 200, "attempts": 1, "elapsed_s": 0.0, "final_error": None}
    monkeypatch.setattr(staging, "http_check", _fake_check)

    task = {
        "mission_id": 1,
        "payload": {},
        "context": {
            "input_artifacts": [
                "unrelated_input",
                "F-001__staging_deployment_result",
            ],
        },
    }
    res = await staging.run(task)
    assert res["ok"] is True
    assert captured["name"] == "F-001__staging_deployment_result"


@pytest.mark.asyncio
async def test_staging_smoke_via_dispatcher(monkeypatch):
    async def _art(_m, _n):
        return {"url": "https://staging.test/y"}
    monkeypatch.setattr(staging, "_load_artifact", _art)

    async def _fake_check(url, **kw):
        return {"ok": True, "final_status": 200, "attempts": 1, "elapsed_s": 0.0, "final_error": None}
    monkeypatch.setattr(staging, "http_check", _fake_check)

    action = await mr_roboto.run({
        "mission_id": 1,
        "payload": {"action": "staging_smoke_check", "artifact": "x"},
    })
    assert action.status == "completed"
    assert action.result["smoke_tests_passed"] is True


@pytest.mark.asyncio
async def test_staging_smoke_dispatcher_failure_path(monkeypatch):
    async def _art(_m, _n):
        return {"url": "https://staging.test/z"}
    monkeypatch.setattr(staging, "_load_artifact", _art)

    async def _fake_check(url, **kw):
        return {"ok": False, "final_status": 503, "attempts": 5, "elapsed_s": 1.0, "final_error": None}
    monkeypatch.setattr(staging, "http_check", _fake_check)

    action = await mr_roboto.run({
        "mission_id": 1,
        "payload": {"action": "staging_smoke_check", "artifact": "x"},
    })
    assert action.status == "failed"
