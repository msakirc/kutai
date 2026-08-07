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

@pytest.mark.asyncio
async def test_deploy_backend_sets_env_and_polls(monkeypatch):
    seen = {}
    async def fake_call(service, action, params):
        seen[(service, action)] = params
        if action == "create_service":
            return {"status": "ok", "data": {"service": {"id": "srv1", "serviceDetails": {"url": "https://b.onrender.com"}}}}
        if action == "get_deploy":
            return {"status": "ok", "data": {"status": "live"}}
        return {"status": "ok", "data": {}}
    monkeypatch.setattr(ds, "_call", fake_call)
    out = await ds._deploy_backend(repo="https://github.com/k/h.git",
                                   env={"DATABASE_URL": "postgresql://x", "REDIS_URL": "rediss://y"})
    assert out["ok"] and out["url"] == "https://b.onrender.com"
    # env vars must be present in the create call (set before first boot)
    create_params = seen[("render", "create_service")]
    assert "DATABASE_URL" in str(create_params)

@pytest.mark.asyncio
async def test_migrate_runs_prisma_deploy(monkeypatch, tmp_path):
    (tmp_path / "backend").mkdir()
    calls = {}
    async def fake_shell(cmd, cwd, env):
        calls["cmd"] = cmd; calls["env"] = env
        return {"returncode": 0, "stdout": "migrations applied", "stderr": ""}
    monkeypatch.setattr(ds, "_shell", fake_shell)
    out = await ds._migrate(workspace=str(tmp_path), database_url="postgresql://u:p@h/db")
    assert out["ok"]
    assert "prisma" in " ".join(calls["cmd"]) and "migrate" in " ".join(calls["cmd"])
    assert calls["env"]["DATABASE_URL"] == "postgresql://u:p@h/db"
