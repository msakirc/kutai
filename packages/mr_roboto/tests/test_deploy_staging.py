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
async def test_provision_through_real_registry_mock_mode():
    """C3 — structural drift guard: exercise _provision through the REAL adapter.execute()
    (mock mode), NOT a stubbed _call. This is the only test that would catch a caller↔config
    required_params mismatch like C1 (upstash create_redis needs 'primary_region', not 'region').

    Installs a mock-on registry AS the module singleton so deploy_staging._call resolves the
    real HttpIntegration adapters. FAILS before the C1 fix (config guard returns
    'Missing required params: [primary_region]' → upstash_provision_failed), passes after —
    proving the config↔caller params align.
    """
    import src.integrations.registry as reg_mod
    from src.integrations.registry import IntegrationRegistry
    orig = reg_mod._registry
    reg_mod._registry = IntegrationRegistry(auto_discover=True, mock_mode=True)
    try:
        out = await ds._provision(mission_id=90)
        assert out["ok"] is True, out
        assert out["env"]["DATABASE_URL"].startswith("postgresql://")
        assert out["env"]["REDIS_URL"].startswith("rediss://")
        assert out["mocked"] is True  # mock-mode adapters tag responses
    finally:
        reg_mod._registry = orig


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

@pytest.mark.asyncio
async def test_deploy_frontend_passes_backend_url_and_polls(monkeypatch):
    seen = {}
    async def fake_call(service, action, params):
        seen[(service, action)] = params
        if action == "deploy":
            return {"status": "ok", "data": {"id": "dpl1", "url": "f.vercel.app", "readyState": "QUEUED"}}
        if action == "get_deployment":
            return {"status": "ok", "data": {"id": "dpl1", "url": "f.vercel.app", "readyState": "READY"}}
        return {"status": "ok", "data": {}}
    monkeypatch.setattr(ds, "_call", fake_call)
    out = await ds._deploy_frontend(repo="https://github.com/k/h.git", backend_url="https://b.onrender.com")
    assert out["ok"] and out["url"].endswith("vercel.app")
    assert "b.onrender.com" in str(seen[("vercel", "deploy")])

@pytest.mark.asyncio
async def test_health_check_retries_cold_start_then_ok(monkeypatch):
    seq = iter([502, 502, 200])  # cold-start 502s then healthy
    async def fake_get(url): return {"status_code": next(seq)}
    monkeypatch.setattr(ds, "_http_get", fake_get)
    out = await ds._health_check("https://b.onrender.com", attempts=5, delay_s=0)
    assert out["ok"] and out["passed"] is True

@pytest.mark.asyncio
async def test_health_check_fails_after_attempts(monkeypatch):
    async def fake_get(url): return {"status_code": 500}
    monkeypatch.setattr(ds, "_http_get", fake_get)
    out = await ds._health_check("https://b.onrender.com", attempts=2, delay_s=0)
    assert out["passed"] is False

@pytest.mark.asyncio
async def test_full_mock_chain_forces_health_false(monkeypatch, tmp_path):
    (tmp_path / "backend").mkdir()
    # every adapter call returns a mocked:true envelope
    async def fake_call(service, action, params):
        base = {"status": "ok", "mocked": True}
        data = {
            ("neon", "create_project"): {"connection_uris": [{"connection_uri": "postgresql://u:p@h/db"}]},
            ("upstash", "create_redis"): {"endpoint": "r.io", "port": 6379, "password": "pw", "rest_token": "t"},
            ("render", "create_service"): {"service": {"id": "srv1", "serviceDetails": {"url": "https://b.onrender.com"}}},
            ("render", "get_deploy"): {"status": "live"},
            ("vercel", "deploy"): {"id": "dpl1", "url": "f.vercel.app", "readyState": "QUEUED"},
            ("vercel", "get_deployment"): {"id": "dpl1", "url": "f.vercel.app", "readyState": "READY"},
        }.get((service, action), {})
        return {**base, "data": data}
    monkeypatch.setattr(ds, "_call", fake_call)
    async def ok_migrate(**k): return {"ok": True}
    monkeypatch.setattr(ds, "_migrate", ok_migrate)
    # hermetic: never touch the network — the guard skips the real health check on a mock run.
    async def fake_get(url): return {"status_code": 200}
    monkeypatch.setattr(ds, "_http_get", fake_get)

    task = {"payload": {"action": "deploy_staging", "backend_arch": "nestjs_render",
                        "repo": "https://github.com/k/h.git", "workspace": str(tmp_path)},
            "context": {"mission_id": 90}}
    res = await ds.run(task)
    # DAG completes but the guard MUST refuse to certify a mocked deploy
    arts = res["artifacts"]
    assert arts["staging_deployment_verified"]["health_check_passed"] is False
    assert arts["staging_deployment_verified"]["reason"] == "mock_mode_active"
    assert res["ok"] is False
    assert arts["staging_environment"]["url"]  # env artifact still populated

@pytest.mark.asyncio
async def test_deploy_staging_dispatches(monkeypatch, tmp_path):
    import mr_roboto
    from mr_roboto.executors import deploy_staging as _ds
    # hermetic: deploy_staging is irreversible; if a founder exported
    # KUTAI_CONFIRM_POLICY=irreversible_only, run() would auto-arm confirmation and park
    # (needs_clarification) → status != completed (Opus review). Unset it for the test.
    monkeypatch.delenv("KUTAI_CONFIRM_POLICY", raising=False)
    async def ok_run(_t): return {"ok": True, "artifacts": {"staging_environment": {"url": "x"}}}
    monkeypatch.setattr(_ds, "run", ok_run)
    act = await mr_roboto.run({"payload": {"action": "deploy_staging"}, "context": {}})
    assert act.status == "completed"

def test_deploy_staging_reversibility_registered():
    from mr_roboto.reversibility import get_reversibility
    assert get_reversibility("deploy_staging") == "irreversible"
