# Mechanical Deploy Orchestrator (`deploy_staging`) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A deterministic mechanical mr_roboto executor `deploy_staging` that chains the Plan-1 adapters into a real $0 staging deploy (provision DB+cache → deploy backend → migrate → deploy frontend → health-check) and writes m90 7.13's two artifacts — with an anti-fake guard that mock responses can never satisfy.

**Architecture:** A bespoke Python executor (precedent: `stripe_provision_products.py`) with an explicit ordering DAG, a bounded `poll_until` helper, secret plumbing (provisioned creds → backend env before boot), SSRF/cold-start-aware health check, and idempotent abort-on-partial. Reaches adapters via **direct `adapter.execute()`** (NOT the mr_roboto vendor_call wrapper — that strips the `mocked` flag the guard needs). Registered in `_run_dispatch`.

**Tech Stack:** Python 3.10, pytest/pytest-asyncio, `IntegrationRegistry`, `mr_roboto` dispatch + `Action`, `reversibility`.

**Spec:** `docs/superpowers/specs/2026-08-04-deploy-orchestrator-spec.md`
**Depends on:** Plan 1 (`2026-08-07-deploy-adapters-plan.md`) — render/neon/upstash/vercel adapters + `git_prepare_repo`.

---

## File Structure

- Create: `packages/mr_roboto/src/mr_roboto/deploy_util.py` — `poll_until` helper (reusable).
- Create: `packages/mr_roboto/src/mr_roboto/executors/deploy_staging.py` — the orchestrator.
- Modify: `packages/mr_roboto/src/mr_roboto/__init__.py` — dispatch branch (~line 2897).
- Modify: `packages/mr_roboto/src/mr_roboto/reversibility.py` — `deploy_staging: "irreversible"`.
- Create: `packages/mr_roboto/tests/test_deploy_util.py`, `test_deploy_staging.py`.
- Modify: `src/workflows/i2p/i2p_v3.json` — 7.13 `agent → mechanical` + `payload.action`.

---

## Task 1: `poll_until` bounded-poll helper

**Files:**
- Create: `packages/mr_roboto/src/mr_roboto/deploy_util.py`
- Test: `packages/mr_roboto/tests/test_deploy_util.py`

- [ ] **Step 1: Write the failing tests**

```python
# packages/mr_roboto/tests/test_deploy_util.py
import pytest
from mr_roboto.deploy_util import poll_until

@pytest.mark.asyncio
async def test_poll_ready_on_third_call():
    seq = iter(["BUILDING", "BUILDING", "READY"])
    async def fetch(): return {"state": next(seq)}
    res = await poll_until(fetch, ready=lambda r: r["state"] == "READY",
                           fail=lambda r: r["state"] == "ERROR",
                           max_wait_s=5, base_delay_s=0)
    assert res["ok"] and res["result"]["state"] == "READY"

@pytest.mark.asyncio
async def test_poll_terminal_fail_aborts_early():
    async def fetch(): return {"state": "ERROR"}
    res = await poll_until(fetch, ready=lambda r: r["state"] == "READY",
                           fail=lambda r: r["state"] == "ERROR",
                           max_wait_s=5, base_delay_s=0)
    assert res["ok"] is False and res["reason"] == "terminal_fail"

@pytest.mark.asyncio
async def test_poll_timeout():
    async def fetch(): return {"state": "BUILDING"}
    res = await poll_until(fetch, ready=lambda r: r["state"] == "READY",
                           fail=lambda r: False, max_wait_s=0.1, base_delay_s=0.05)
    assert res["ok"] is False and res["reason"] == "timeout"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest packages/mr_roboto/tests/test_deploy_util.py -v`
Expected: FAIL (module missing)

- [ ] **Step 3: Implement `deploy_util.py`**

```python
"""Bounded poll-until-ready helper for deploy orchestration."""
from __future__ import annotations
import asyncio, time
from typing import Any, Awaitable, Callable

async def poll_until(
    fetch: Callable[[], Awaitable[dict]],
    ready: Callable[[dict], bool],
    fail: Callable[[dict], bool],
    *, max_wait_s: float = 600, base_delay_s: float = 5, cap_delay_s: float = 30,
) -> dict[str, Any]:
    """Poll ``fetch`` until ``ready`` (→ ok), ``fail`` (→ terminal_fail), or timeout.

    Uses monotonic-clock deadline + exponential backoff capped at ``cap_delay_s``.
    Returns {ok, result?|reason}.
    """
    deadline = time.monotonic() + max_wait_s
    delay = base_delay_s
    last = None
    while time.monotonic() < deadline:
        last = await fetch()
        if fail(last):
            return {"ok": False, "reason": "terminal_fail", "result": last}
        if ready(last):
            return {"ok": True, "result": last}
        await asyncio.sleep(delay)
        delay = min(delay * 2, cap_delay_s) if delay else base_delay_s
    return {"ok": False, "reason": "timeout", "result": last}
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest packages/mr_roboto/tests/test_deploy_util.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
rtk git add packages/mr_roboto/src/mr_roboto/deploy_util.py packages/mr_roboto/tests/test_deploy_util.py
rtk git commit -m "feat(mr_roboto): poll_until bounded-poll helper for deploy"
```

---

## Task 2: `deploy_staging` skeleton — param validation, backend_arch assert, direct-execute + mock guard

**Files:**
- Create: `packages/mr_roboto/src/mr_roboto/executors/deploy_staging.py`
- Test: `packages/mr_roboto/tests/test_deploy_staging.py`

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest packages/mr_roboto/tests/test_deploy_staging.py -v`
Expected: FAIL (module missing)

- [ ] **Step 3: Implement the skeleton**

```python
"""Mechanical deploy_staging orchestrator — stands up a $0 staging env, writes 7.13 artifacts.

Reaches adapters via DIRECT adapter.execute() (NOT mr_roboto vendor_call — that wrapper
strips the ``mocked`` flag the anti-fake guard depends on). A mock response can never
certify health_check_passed:true.
"""
from __future__ import annotations
from typing import Any

def _is_mocked(envelope: dict) -> bool:
    """True when a registry response carries the mock tag."""
    return bool(isinstance(envelope, dict) and envelope.get("mocked") is True)

async def _call(service: str, action: str, params: dict) -> dict:
    """Direct adapter.execute(); preserves the mocked flag."""
    from src.integrations.registry import get_integration_registry
    adapter = get_integration_registry().get(service)
    if adapter is None:
        return {"status": "error", "error": f"adapter {service} not registered"}
    return await adapter.execute(action, params)

def _fail(reason: str, **extra) -> dict:
    return {"ok": False, "reason": reason, **extra}

async def run(task: dict) -> dict[str, Any]:
    payload = (task.get("payload") or (task.get("context") or {}).get("payload") or {})
    repo = payload.get("repo")
    workspace = payload.get("workspace")
    backend_arch = payload.get("backend_arch", "nestjs_render")
    if backend_arch != "nestjs_render":
        return _fail("serverless_not_yet_supported")
    if not repo or not workspace:
        return _fail("missing repo or workspace")
    ctx = task.get("context") or {}
    mission_id = ctx.get("mission_id") or payload.get("mission_id")

    state = {"mocked_any": False, "services": {}, "provisioned": []}
    # DAG steps 2-9 are added in later tasks; skeleton returns not-implemented for now.
    return _fail("dag_not_implemented", state=state)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest packages/mr_roboto/tests/test_deploy_staging.py -v`
Expected: PASS (the two rejection tests + `_is_mocked`)

- [ ] **Step 5: Commit**

```bash
rtk git add packages/mr_roboto/src/mr_roboto/executors/deploy_staging.py packages/mr_roboto/tests/test_deploy_staging.py
rtk git commit -m "feat(mr_roboto): deploy_staging skeleton (validation + mock-tag detection)"
```

---

## Task 3: Provision DB (Neon) + cache (Upstash), idempotent

**Files:**
- Modify: `packages/mr_roboto/src/mr_roboto/executors/deploy_staging.py`
- Test: `packages/mr_roboto/tests/test_deploy_staging.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest packages/mr_roboto/tests/test_deploy_staging.py::test_provision_captures_conn_and_redis -v`
Expected: FAIL (no `_provision`)

- [ ] **Step 3: Implement `_provision`**

```python
async def _provision(mission_id) -> dict:
    """Provision Neon Postgres + Upstash Redis; return env vars + service descriptors.

    Idempotent: list-before-create keyed on a mission tag would go here (list_* returns []
    in mock; real impl reuses a resource named kutay_mission_{id}). Returns
    {ok, env:{DATABASE_URL,REDIS_URL}, services:{db,cache}, mocked}.
    """
    name = f"kutay_mission_{mission_id}"
    db = await _call("neon", "create_project", {"project": {"name": name}})
    if db.get("status") != "ok":
        return _fail("neon_provision_failed", detail=db.get("error"))
    conn = (db.get("data", {}).get("connection_uris") or [{}])[0].get("connection_uri")
    if not conn:
        return _fail("neon_no_connection_uri")

    cache = await _call("upstash", "create_redis", {"database_name": name, "region": "us-east-1"})
    if cache.get("status") != "ok":
        return _fail("upstash_provision_failed", detail=cache.get("error"))
    cd = cache.get("data", {})
    redis_url = f"rediss://default:{cd.get('password')}@{cd.get('endpoint')}:{cd.get('port')}"

    return {
        "ok": True,
        "env": {"DATABASE_URL": conn, "REDIS_URL": redis_url},
        "services": {"db": {"provider": "neon"}, "cache": {"provider": "upstash"}},
        "mocked": _is_mocked(db) or _is_mocked(cache),
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest packages/mr_roboto/tests/test_deploy_staging.py::test_provision_captures_conn_and_redis -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
rtk git add packages/mr_roboto/src/mr_roboto/executors/deploy_staging.py packages/mr_roboto/tests/test_deploy_staging.py
rtk git commit -m "feat(mr_roboto): deploy_staging provision neon+upstash"
```

---

## Task 4: Deploy backend (Render) with env-at-create + poll to live

**Files:**
- Modify: `packages/mr_roboto/src/mr_roboto/executors/deploy_staging.py`
- Test: `packages/mr_roboto/tests/test_deploy_staging.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest packages/mr_roboto/tests/test_deploy_staging.py::test_deploy_backend_sets_env_and_polls -v`
Expected: FAIL

- [ ] **Step 3: Implement `_deploy_backend`**

```python
async def _deploy_backend(repo: str, env: dict, *, owner_id: str = "", max_wait_s: float = 900) -> dict:
    """Create the Render web service with env vars at create time (auto-deploys once), then
    poll get_deploy until 'live'. Render create auto-initiates the first deploy — no separate
    trigger_deploy for first boot; env-var updates do NOT auto-deploy (handled only on change).
    """
    from mr_roboto.deploy_util import poll_until
    env_vars = [{"key": k, "value": v} for k, v in env.items()]
    create = await _call("render", "create_service", {
        "ownerId": owner_id, "type": "web_service", "name": "kutay-backend", "repo": repo,
        "serviceDetails": {"env": "node", "envVars": env_vars},  # free instance type per live docs
    })
    if create.get("status") != "ok":
        return _fail("render_create_failed", detail=create.get("error"))
    svc = create.get("data", {}).get("service", {})
    sid = svc.get("id")
    url = (svc.get("serviceDetails") or {}).get("url")

    poll = await poll_until(
        lambda: _latest_deploy_state(sid),
        ready=lambda r: r.get("status") == "live",
        fail=lambda r: r.get("status") in ("build_failed", "canceled", "deactivated"),
        max_wait_s=max_wait_s,
    )
    if not poll["ok"]:
        return _fail(f"backend_deploy_{poll['reason']}", service_id=sid)
    return {"ok": True, "url": url, "service_id": sid,
            "mocked": _is_mocked(create), "services": {"backend": {"provider": "render", "url": url}}}

async def _latest_deploy_state(service_id: str) -> dict:
    """Fetch the latest deploy's state for polling. Uses get_deploy on the newest deploy id."""
    lst = await _call("render", "get_service", {"id": service_id})
    # In mock mode the get_deploy mock returns {status:'live'} directly:
    d = await _call("render", "get_deploy", {"id": service_id, "deployId": "latest"})
    st = d.get("data", {})
    # preserve mocked so the caller's guard can see it
    if _is_mocked(d):
        st = {**st, "mocked": True}
    return st
```

> Note: the exact "latest deploy id" retrieval depends on the live Render API (list deploys →
> take `[0].id`). Reconcile `_latest_deploy_state` with the live doc during implementation; the
> mock path returns `{status:"live"}` directly so the test passes without that detail.

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest packages/mr_roboto/tests/test_deploy_staging.py::test_deploy_backend_sets_env_and_polls -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
rtk git add packages/mr_roboto/src/mr_roboto/executors/deploy_staging.py packages/mr_roboto/tests/test_deploy_staging.py
rtk git commit -m "feat(mr_roboto): deploy_staging render backend deploy + poll"
```

---

## Task 5: Run Prisma migrations (out-of-band, against provisioned DATABASE_URL)

**Files:**
- Modify: `packages/mr_roboto/src/mr_roboto/executors/deploy_staging.py`
- Test: `packages/mr_roboto/tests/test_deploy_staging.py`

- [ ] **Step 1: Write the failing test (subprocess stubbed)**

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest packages/mr_roboto/tests/test_deploy_staging.py::test_migrate_runs_prisma_deploy -v`
Expected: FAIL

- [ ] **Step 3: Implement `_migrate` + `_shell`**

```python
import asyncio, os

async def _shell(cmd: list[str], cwd: str, env: dict) -> dict:
    proc = await asyncio.create_subprocess_exec(
        *cmd, cwd=cwd, env={**os.environ, **env},
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    out, err = await proc.communicate()
    return {"returncode": proc.returncode,
            "stdout": out.decode(errors="replace"), "stderr": err.decode(errors="replace")}

async def _migrate(workspace: str, database_url: str) -> dict:
    """Run `npx prisma migrate deploy` in the backend dir against the provisioned DATABASE_URL.

    (Alternative: run as a Render release command — chosen here as an explicit executor step so
    the migration outcome is observable. Requires Node/npm + backend deps on the host.)
    """
    backend = os.path.join(workspace, "backend")
    if not os.path.isdir(backend):
        return _fail("backend_dir_missing", path=backend)
    res = await _shell(["npx", "prisma", "migrate", "deploy"], cwd=backend,
                       env={"DATABASE_URL": database_url})
    if res["returncode"] != 0:
        return _fail("migration_failed", detail=res["stderr"][:300])
    return {"ok": True, "output": res["stdout"][:300]}
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest packages/mr_roboto/tests/test_deploy_staging.py::test_migrate_runs_prisma_deploy -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
rtk git add packages/mr_roboto/src/mr_roboto/executors/deploy_staging.py packages/mr_roboto/tests/test_deploy_staging.py
rtk git commit -m "feat(mr_roboto): deploy_staging prisma migrate deploy step"
```

---

## Task 6: Deploy frontend (Vercel) with backend URL + poll

**Files:**
- Modify: `packages/mr_roboto/src/mr_roboto/executors/deploy_staging.py`
- Test: `packages/mr_roboto/tests/test_deploy_staging.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest packages/mr_roboto/tests/test_deploy_staging.py::test_deploy_frontend_passes_backend_url_and_polls -v`
Expected: FAIL

- [ ] **Step 3: Implement `_deploy_frontend`**

```python
async def _deploy_frontend(repo: str, backend_url: str, *, max_wait_s: float = 600) -> dict:
    from mr_roboto.deploy_util import poll_until
    dep = await _call("vercel", "deploy", {
        "name": "kutay-frontend",
        "gitSource": {"type": "github", "repo": repo},
        "env": {"NEXT_PUBLIC_API_URL": backend_url},
    })
    if dep.get("status") != "ok":
        return _fail("vercel_deploy_failed", detail=dep.get("error"))
    dep_id = dep.get("data", {}).get("id")

    async def fetch():
        r = await _call("vercel", "get_deployment", {"id": dep_id})
        d = r.get("data", {})
        if _is_mocked(r):
            d = {**d, "mocked": True}
        return d
    poll = await poll_until(fetch, ready=lambda r: r.get("readyState") == "READY",
                            fail=lambda r: r.get("readyState") in ("ERROR", "CANCELED"),
                            max_wait_s=max_wait_s)
    if not poll["ok"]:
        return _fail(f"frontend_deploy_{poll['reason']}", deployment_id=dep_id)
    url = poll["result"].get("url") or dep.get("data", {}).get("url")
    full = url if str(url).startswith("http") else f"https://{url}"
    return {"ok": True, "url": full, "mocked": _is_mocked(dep),
            "services": {"frontend": {"provider": "vercel", "url": full}}}
```

- [ ] **Step 4: Run to verify it passes** — Expected: PASS
- [ ] **Step 5: Commit**

```bash
rtk git add packages/mr_roboto/src/mr_roboto/executors/deploy_staging.py packages/mr_roboto/tests/test_deploy_staging.py
rtk git commit -m "feat(mr_roboto): deploy_staging vercel frontend deploy + poll"
```

---

## Task 7: Health check with cold-start + SSRF-aware retry

**Files:**
- Modify: `packages/mr_roboto/src/mr_roboto/executors/deploy_staging.py`
- Test: `packages/mr_roboto/tests/test_deploy_staging.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest packages/mr_roboto/tests/test_deploy_staging.py -k health_check -v`
Expected: FAIL

- [ ] **Step 3: Implement `_health_check` + `_http_get`**

```python
async def _http_get(url: str) -> dict:
    import httpx
    async with httpx.AsyncClient(timeout=15.0) as c:
        r = await c.get(url)
        return {"status_code": r.status_code}

async def _health_check(url: str, *, attempts: int = 6, delay_s: float = 10) -> dict:
    """GET the public URL; treat 502/503/timeout (cold-start-during-wake) as retryable.
    Render free services spin down on idle — the first request after deploy can cold-start."""
    import asyncio
    last = None
    for i in range(attempts):
        try:
            last = await _http_get(url)
            code = last.get("status_code")
            if 200 <= code < 400:
                return {"ok": True, "passed": True, "status_code": code}
            if code not in (502, 503, 504):   # non-retryable server/client error
                return {"ok": True, "passed": False, "status_code": code}
        except Exception as e:                  # incl. SSRF ValueError / timeout → retry as "not yet public"
            last = {"error": str(e)}
        if i < attempts - 1:
            await asyncio.sleep(delay_s)
    return {"ok": True, "passed": False, "last": last}
```

- [ ] **Step 4: Run to verify it passes** — Expected: PASS
- [ ] **Step 5: Commit**

```bash
rtk git add packages/mr_roboto/src/mr_roboto/executors/deploy_staging.py packages/mr_roboto/tests/test_deploy_staging.py
rtk git commit -m "feat(mr_roboto): deploy_staging health check with cold-start retry"
```

---

## Task 8: Wire the DAG in `run()` + anti-fake guard + artifact write

**Files:**
- Modify: `packages/mr_roboto/src/mr_roboto/executors/deploy_staging.py`
- Test: `packages/mr_roboto/tests/test_deploy_staging.py`

- [ ] **Step 1: Write the failing tests (the primary CI test — mock-chain proves guard fires)**

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest packages/mr_roboto/tests/test_deploy_staging.py::test_full_mock_chain_forces_health_false -v`
Expected: FAIL (`run` still returns `dag_not_implemented`)

- [ ] **Step 3: Replace the skeleton's DAG tail in `run()`**

Replace the `# DAG steps 2-9 …` return with:

```python
    prov = await _provision(mission_id)
    if not prov["ok"]:
        return prov
    state["mocked_any"] |= prov.get("mocked", False)
    state["services"].update(prov["services"])

    be = await _deploy_backend(repo=repo, env=prov["env"], owner_id=payload.get("owner_id", ""))
    if not be["ok"]:
        return {**be, "state": state}
    state["mocked_any"] |= be.get("mocked", False)
    state["services"].update(be["services"])

    mig = await _migrate(workspace=workspace, database_url=prov["env"]["DATABASE_URL"])
    if not mig["ok"]:
        return {**mig, "state": state}

    fe = await _deploy_frontend(repo=repo, backend_url=be["url"])
    if not fe["ok"]:
        return {**fe, "state": state}
    state["mocked_any"] |= fe.get("mocked", False)
    state["services"].update(fe["services"])

    hc = await _health_check(be["url"])

    # Anti-fake guard: a mocked run can NEVER certify a live deploy.
    if state["mocked_any"]:
        health_passed, reason = False, "mock_mode_active"
    else:
        health_passed, reason = bool(hc.get("passed")), (None if hc.get("passed") else "health_check_failed")

    artifacts = {
        "staging_environment": {"url": fe["url"], "services": state["services"]},
        "staging_deployment_verified": {"deployed": True, "health_check_passed": health_passed,
                                        "reason": reason},
    }
    ok = health_passed
    return {"ok": ok, "artifacts": artifacts, "state": state,
            "reason": None if ok else reason}
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest packages/mr_roboto/tests/test_deploy_staging.py -v`
Expected: PASS (all deploy_staging tests, including the full-chain guard test)

- [ ] **Step 5: Commit**

```bash
rtk git add packages/mr_roboto/src/mr_roboto/executors/deploy_staging.py packages/mr_roboto/tests/test_deploy_staging.py
rtk git commit -m "feat(mr_roboto): deploy_staging DAG + anti-fake guard + artifact write"
```

---

## Task 9: Dispatch registration + reversibility + confirm policy

**Files:**
- Modify: `packages/mr_roboto/src/mr_roboto/__init__.py` (near `git_prepare_repo`/`stripe_provision_products`)
- Modify: `packages/mr_roboto/src/mr_roboto/reversibility.py`
- Test: `packages/mr_roboto/tests/test_deploy_staging.py`, `tests/*/test_reversibility_registry.py`

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.asyncio
async def test_deploy_staging_dispatches(monkeypatch, tmp_path):
    import mr_roboto
    from mr_roboto.executors import deploy_staging as _ds
    async def ok_run(_t): return {"ok": True, "artifacts": {"staging_environment": {"url": "x"}}}
    monkeypatch.setattr(_ds, "run", ok_run)
    act = await mr_roboto.run({"payload": {"action": "deploy_staging"}, "context": {}})
    assert act.status == "completed"

def test_deploy_staging_reversibility_registered():
    from mr_roboto.reversibility import get_reversibility
    assert get_reversibility("deploy_staging") == "irreversible"
```

- [ ] **Step 2: Run to verify it fails** — Expected: FAIL

- [ ] **Step 3: Add the dispatch branch** in `__init__.py` (after the `git_prepare_repo` branch):

```python
    if action == "deploy_staging":
        from mr_roboto.executors.deploy_staging import run as _deploy_run
        try:
            res = await _deploy_run(task)
            return Action(status="completed" if res.get("ok") else "failed",
                          error=None if res.get("ok") else res.get("reason"), result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))
```

Add to `reversibility.py`: `"deploy_staging": "irreversible"` in the verb map.

- [ ] **Step 4: Run to verify it passes** (incl. `test_reversibility_registry.py` staying green) — Expected: PASS

- [ ] **Step 5: Commit**

```bash
rtk git add packages/mr_roboto/src/mr_roboto/__init__.py packages/mr_roboto/src/mr_roboto/reversibility.py packages/mr_roboto/tests/test_deploy_staging.py
rtk git commit -m "feat(mr_roboto): register deploy_staging dispatch + irreversible reversibility"
```

---

## Task 10: Rewire 7.13 to mechanical (i2p_v3.json) + document in-flight reconcile

**Files:**
- Modify: `src/workflows/i2p/i2p_v3.json` (the `staging_environment` step)
- Test: `tests/workflows/` (config-guard)

- [ ] **Step 1: Write the failing test**

```python
def test_staging_env_is_mechanical_deploy_staging():
    import json
    with open("src/workflows/i2p/i2p_v3.json") as f:
        wf = json.load(f)
    step = next(s for s in wf["steps"] if s.get("name") == "staging_environment")
    assert step["agent"] == "mechanical"
    assert step["payload"]["action"] == "deploy_staging"
    assert step["payload"]["backend_arch"] == "nestjs_render"
```

- [ ] **Step 2: Run to verify it fails** — Expected: FAIL (`agent` is `executor`)

- [ ] **Step 3: Edit the step in `i2p_v3.json`** — set `"agent": "mechanical"` and add:

```json
"payload": {
  "action": "deploy_staging",
  "backend_arch": "nestjs_render",
  "repo": "{git_repo_url}",
  "workspace": "mission_{mission_id}/",
  "confirm_policy": "irreversible_only"
}
```

(`{git_repo_url}` and `{mission_id}` are substituted by `_substitute_payload` at expansion.)

- [ ] **Step 4: Run to verify it passes** — Expected: PASS

- [ ] **Step 5: Commit**

```bash
rtk git add src/workflows/i2p/i2p_v3.json tests/workflows/
rtk git commit -m "fix(i2p): 7.13 runs mechanical deploy_staging (was LLM executor)"
```

> **In-flight m90 note (do NOT rely on refreshers):** the already-expanded 7.13 row (`567489`)
> is frozen `agent_type="executor"`; `refresh_workflow_agent_type` refuses executor→mechanical
> (`task_refresh.py:71`). To flip the LIVE m90 row, a direct DB reconcile is required (set
> `agent_type` + `context.executor="mechanical"` + `context.payload`) OR re-expand the step —
> this is a founder-run operational step at deploy time, not part of this code plan.

---

## Verification (whole plan)

- [ ] `python -m pytest packages/mr_roboto/tests/test_deploy_util.py packages/mr_roboto/tests/test_deploy_staging.py -v` — all green.
- [ ] The full-mock-chain test proves the anti-fake guard forces `health_check_passed:false` + `mock_mode_active` (the single most important assertion).
- [ ] `test_reversibility_registry.py` green (deploy_staging + git_prepare_repo entries present).
- [ ] No live network in CI.

## Real-deploy runbook (founder-run, after both plans land + `/restart`)
1. `/credential add github` (PAT), `render` (api_key), `neon` (api_key), `upstash` (basic_auth_b64 = base64("email:api_key")).
2. Set `KUTAI_VENDOR_LIVE=1` (and `KUTAI_ENV=prod` or ensure the live-flag path) so mock mode is OFF.
3. Run `git_prepare_repo` for mission 90 (or let the workflow step do it) → repo pushed.
4. Reconcile the live 7.13 row to mechanical (direct DB edit) OR re-pend on a fresh expansion.
5. Re-pend 7.13 → `deploy_staging` runs the real DAG → writes a genuine `health_check_passed:true`.
   The `irreversible_only` confirm policy parks for your ack first.

## Notes for the executor
- Run only the targeted test files (cold full-suite import hangs in this env).
- All `.py` changes are restart-gated — founder `/restart` to load.
- `_latest_deploy_state` (Task 4) and the Render/Vercel deploy bodies must be reconciled against
  live API docs during implementation — the mock paths make tests pass, but a real run needs the
  exact request shapes (Plan 1's live-doc gate applies here too).
