# packages/mr_roboto/tests/test_deploy_staging.py
import pytest
from mr_roboto.executors import deploy_staging as ds

@pytest.mark.asyncio
async def test_rejects_non_render_backend_arch():
    task = {"payload": {"action": "deploy_staging", "backend_arch": "serverless_workers",
                        "repo": "https://github.com/kutay/habithub.git", "workspace": "/tmp/x"}}
    res = await ds.run(task)
    assert res["ok"] is False and res["reason"] == "serverless_not_yet_supported"

@pytest.mark.asyncio
async def test_missing_params_fail_fast():
    res = await ds.run({"payload": {"action": "deploy_staging"}})
    assert res["ok"] is False and "missing" in res["reason"].lower()

def test_is_mocked_detects_tag():
    assert ds._is_mocked({"status": "ok", "data": {}, "mocked": True}) is True
    assert ds._is_mocked({"status": "ok", "data": {}}) is False

@pytest.mark.asyncio
async def test_provision_captures_conn_and_redis(monkeypatch):
    async def fake_call(service, action, params):
        if service == "neon" and action == "create_project":
            return {"status": "ok", "data": {"connection_uris": [{"connection_uri": "postgresql://u:p@h/db"}]}}
        if service == "upstash" and action == "create_redis":
            return {"status": "ok", "data": {"endpoint": "r.upstash.io", "port": 6379, "password": "pw", "rest_token": "t"}}
        if action.startswith("list"):
            return {"status": "ok", "data": []}
        return {"status": "ok", "data": {}}
    monkeypatch.setattr(ds, "_call", fake_call)
    out = await ds._provision(mission_id=90)
    assert out["ok"]
    assert out["env"]["DATABASE_URL"].startswith("postgresql://")
    assert out["env"]["REDIS_URL"] and out["services"]["db"] and out["services"]["cache"]
