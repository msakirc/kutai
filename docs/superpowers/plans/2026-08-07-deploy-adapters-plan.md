# Free-tier Deploy Adapters + Git-Prereq Chain — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the atomic building blocks for a $0/no-card staging deploy — Render/Neon/Upstash adapter configs + Vercel poll enrichment + credential schemas + the git-host prerequisite chain — all additive and mock-testable.

**Architecture:** New declarative `HttpIntegration` JSON configs auto-discovered from `src/integrations/configs/`, auth via `credential_store`, reachable via the registry. No engine changes (Upstash auth uses the existing `auth_type:"header"` + pre-encoded-Basic pattern, per `twilio.json`). The git-prereq chain (PAT → create repo → push scaffold → reachability preflight) is a precondition for any deploy.

**Tech Stack:** Python 3.10, pytest, `HttpIntegration` (`src/integrations/http_integration.py`), `credential_store` (`src/security/credential_store.py`), git via `shell`.

**Spec:** `docs/superpowers/specs/2026-08-04-deploy-adapters-spec.md`

> ⚠️ **Live-doc gate (from the Opus review):** every EXTERNAL API action table below is INDICATIVE. Each adapter task begins with a **verify-against-live-docs** step — do NOT author the config JSON from this plan's tables alone; the review already caught wrong Render/Upstash shapes. Use WebFetch on the provider's API reference, confirm endpoint/method/auth/field-names, THEN author.

---

## File Structure

- Create: `src/integrations/configs/render.json`, `neon.json`, `upstash.json` — declarative adapters.
- Modify: `src/integrations/configs/vercel.json` — add `get_deployment` poll action + `mock_responses`.
- Create: `credential_schemas/render.json`, `neon.json`, `upstash.json` — credential contracts.
- Create: `tests/integrations/test_deploy_adapter_configs.py` — config-guard tests (load, action shape, mock tagging).
- Create: `packages/mr_roboto/src/mr_roboto/executors/git_prepare_repo.py` — mechanical create-repo + push-scaffold.
- Modify: `packages/mr_roboto/src/mr_roboto/__init__.py` — dispatch `git_prepare_repo`.
- Create: `packages/mr_roboto/tests/test_git_prepare_repo.py`.
- Modify: `src/workflows/i2p/i2p_v3.json` — 7.13 `real_tool_kind` → `vercel|render`.

---

## Task 1: Vercel `get_deployment` poll action + mock_responses

**Files:**
- Modify: `src/integrations/configs/vercel.json`
- Test: `tests/integrations/test_deploy_adapter_configs.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/integrations/test_deploy_adapter_configs.py
import json, os
CONFIGS = os.path.join(os.path.dirname(__file__), "..", "..", "src", "integrations", "configs")

def _load(name):
    with open(os.path.join(CONFIGS, f"{name}.json")) as f:
        return json.load(f)

def test_vercel_has_get_deployment_poll_action():
    cfg = _load("vercel")
    assert "get_deployment" in cfg["actions"]
    act = cfg["actions"]["get_deployment"]
    assert act["method"] == "GET"
    assert "{id}" in act["path"]
    assert act["required_params"] == ["id"]

def test_vercel_deploy_actions_have_mock_responses():
    cfg = _load("vercel")
    mocks = cfg.get("mock_responses", {})
    assert "deploy" in mocks and "get_deployment" in mocks
    # get_deployment mock must model a READY terminal state for the poll loop
    assert mocks["get_deployment"].get("readyState") == "READY"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/integrations/test_deploy_adapter_configs.py::test_vercel_has_get_deployment_poll_action tests/integrations/test_deploy_adapter_configs.py::test_vercel_deploy_actions_have_mock_responses -v`
Expected: FAIL (`KeyError: 'get_deployment'` / mock_responses missing)

- [ ] **Step 3: Verify the Vercel deployment-get endpoint against live docs**

Use WebFetch on `https://vercel.com/docs/rest-api/reference/endpoints/deployments/get-a-deployment` — confirm `GET /v13/deployments/{id}` and the `readyState` field values (`QUEUED|BUILDING|READY|ERROR|CANCELED`). Adjust the path/field below if the live doc differs.

- [ ] **Step 4: Add the action + mock to `vercel.json`**

Add to `actions`:
```json
"get_deployment": { "method": "GET", "path": "/v13/deployments/{id}", "required_params": ["id"] }
```
Add a top-level `mock_responses` block (payloads are wrapped by the registry as `{status:"ok", data:<payload>, mocked:true}`):
```json
"mock_responses": {
  "deploy": { "id": "dpl_mock123", "readyState": "QUEUED", "url": "mock-app.vercel.app" },
  "get_deployment": { "id": "dpl_mock123", "readyState": "READY", "url": "mock-app.vercel.app" }
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/integrations/test_deploy_adapter_configs.py -k vercel -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
rtk git add src/integrations/configs/vercel.json tests/integrations/test_deploy_adapter_configs.py
rtk git commit -m "feat(integrations): vercel get_deployment poll action + mock_responses"
```

---

## Task 2: Credential schemas for render / neon / upstash

**Files:**
- Create: `credential_schemas/render.json`, `credential_schemas/neon.json`, `credential_schemas/upstash.json`
- Test: `tests/integrations/test_deploy_adapter_configs.py`

- [ ] **Step 1: Write the failing test**

```python
import json, os
CRED = os.path.join(os.path.dirname(__file__), "..", "..", "credential_schemas")

def _load_cred(name):
    with open(os.path.join(CRED, f"{name}.json")) as f:
        return json.load(f)

def test_new_credential_schemas_exist_and_shaped():
    for name, required in [("render", ["api_key"]), ("neon", ["api_key"]),
                           ("upstash", ["basic_auth_b64"])]:
        s = _load_cred(name)
        assert s["service_name"] == name
        for field in required:
            assert field in s["required_fields"], f"{name} missing {field}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/integrations/test_deploy_adapter_configs.py::test_new_credential_schemas_exist_and_shaped -v`
Expected: FAIL (FileNotFoundError)

- [ ] **Step 3: Author the three schemas (mirror `credential_schemas/vercel.json` shape)**

`credential_schemas/render.json`:
```json
{ "service_name": "render", "required_fields": ["api_key"], "optional_fields": ["owner_id"],
  "scopes": ["read_write"], "default_scope": "read_write", "rotation_recommended_days": 90,
  "test_endpoint": { "action": "get_service", "expect_status": 200 },
  "docs_url": "https://render.com/docs/api" }
```
`credential_schemas/neon.json`:
```json
{ "service_name": "neon", "required_fields": ["api_key"], "optional_fields": [],
  "scopes": ["read_write"], "default_scope": "read_write", "rotation_recommended_days": 90,
  "test_endpoint": { "action": "list_projects", "expect_status": 200 },
  "docs_url": "https://api-docs.neon.tech" }
```
`credential_schemas/upstash.json` (pre-encoded Basic auth — the credential is `base64(email:api_key)`):
```json
{ "service_name": "upstash", "required_fields": ["basic_auth_b64"],
  "optional_fields": ["email"], "scopes": ["read_write"], "default_scope": "read_write",
  "rotation_recommended_days": 90, "test_endpoint": null,
  "docs_url": "https://upstash.com/docs/devops/developer-api" }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/integrations/test_deploy_adapter_configs.py::test_new_credential_schemas_exist_and_shaped -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
rtk git add credential_schemas/render.json credential_schemas/neon.json credential_schemas/upstash.json tests/integrations/test_deploy_adapter_configs.py
rtk git commit -m "feat(credentials): render/neon/upstash credential schemas"
```

---

## Task 3: Render adapter config + mock

**Files:**
- Create: `src/integrations/configs/render.json`
- Test: `tests/integrations/test_deploy_adapter_configs.py`

- [ ] **Step 1: Write the failing test**

```python
def test_render_config_actions_and_mock():
    cfg = _load("render")
    assert cfg["service_name"] == "render"
    assert cfg["auth_type"] == "bearer"
    for a in ("create_service", "get_service", "trigger_deploy", "get_deploy", "update_env_vars"):
        assert a in cfg["actions"], f"missing action {a}"
    # poll target must model a terminal 'live' state
    assert cfg["mock_responses"]["get_deploy"]["status"] == "live"
    # create mock returns a service id downstream needs
    assert "id" in cfg["mock_responses"]["create_service"].get("service", {})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/integrations/test_deploy_adapter_configs.py::test_render_config_actions_and_mock -v`
Expected: FAIL (FileNotFoundError)

- [ ] **Step 3: Verify Render API against live docs (REQUIRED — review caught wrong shapes)**

WebFetch `https://api-docs.render.com/reference/create-service`, `.../get-deploy`, `.../update-env-vars-for-service`. Confirm: base `https://api.render.com/v1`; `create_service` body is **nested** (`ownerId`, `type`, `name`, `repo`, `serviceDetails{runtime, envSpecificDetails, envVars, plan/instanceType}`) and **NO `plan:"free"`** (free is an instance type); create **auto-initiates the first deploy**; `update_env_vars` does **NOT** auto-deploy. Capture the exact field names before authoring.

- [ ] **Step 4: Author `render.json` (fill from Step 3; template below)**

```json
{
  "service_name": "render",
  "base_url": "https://api.render.com/v1",
  "auth_type": "bearer",
  "auth_header": "Authorization",
  "auth_token_field": "api_key",
  "actions": {
    "create_service":  { "method": "POST", "path": "/services", "required_params": ["ownerId", "type", "name", "repo", "serviceDetails"] },
    "get_service":     { "method": "GET",  "path": "/services/{id}", "required_params": ["id"] },
    "trigger_deploy":  { "method": "POST", "path": "/services/{id}/deploys", "required_params": ["id"] },
    "get_deploy":      { "method": "GET",  "path": "/services/{id}/deploys/{deployId}", "required_params": ["id", "deployId"] },
    "update_env_vars": { "method": "PUT",  "path": "/services/{id}/env-vars", "required_params": ["id", "envVars"] }
  },
  "mock_responses": {
    "create_service": { "service": { "id": "srv_mock123", "serviceDetails": { "url": "https://mock-backend.onrender.com" } } },
    "get_service":    { "id": "srv_mock123", "serviceDetails": { "url": "https://mock-backend.onrender.com" } },
    "trigger_deploy": { "id": "dep_mock123", "status": "created" },
    "get_deploy":     { "id": "dep_mock123", "status": "live" },
    "update_env_vars": { "ok": true }
  }
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/integrations/test_deploy_adapter_configs.py::test_render_config_actions_and_mock -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
rtk git add src/integrations/configs/render.json tests/integrations/test_deploy_adapter_configs.py
rtk git commit -m "feat(integrations): render adapter config + mock (backend host)"
```

---

## Task 4: Neon adapter config + mock

**Files:**
- Create: `src/integrations/configs/neon.json`
- Test: `tests/integrations/test_deploy_adapter_configs.py`

- [ ] **Step 1: Write the failing test**

```python
def test_neon_config_actions_and_mock():
    cfg = _load("neon")
    assert cfg["service_name"] == "neon"
    for a in ("create_project", "get_project", "list_projects"):
        assert a in cfg["actions"]
    # create must surface the connection string downstream needs
    conn = cfg["mock_responses"]["create_project"].get("connection_uris")
    assert conn and conn[0].get("connection_uri", "").startswith("postgresql://")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/integrations/test_deploy_adapter_configs.py::test_neon_config_actions_and_mock -v`
Expected: FAIL

- [ ] **Step 3: Verify Neon API against live docs**

WebFetch `https://api-docs.neon.tech/reference/createproject` and `listprojects`. Confirm base `https://console.neon.tech/api/v2`, bearer auth, `POST /projects` returns `connection_uris[].connection_uri`. Note: **no run-SQL REST endpoint** — migrations run out-of-band (Plan 2). Adjust below if fields differ.

- [ ] **Step 4: Author `neon.json`**

```json
{
  "service_name": "neon",
  "base_url": "https://console.neon.tech/api/v2",
  "auth_type": "bearer",
  "auth_header": "Authorization",
  "auth_token_field": "api_key",
  "actions": {
    "create_project": { "method": "POST", "path": "/projects", "required_params": ["project"] },
    "get_project":    { "method": "GET",  "path": "/projects/{project_id}", "required_params": ["project_id"] },
    "list_projects":  { "method": "GET",  "path": "/projects", "required_params": [] }
  },
  "mock_responses": {
    "create_project": { "project": { "id": "proj_mock123" }, "connection_uris": [ { "connection_uri": "postgresql://mockuser:mockpw@ep-mock.neon.tech/neondb" } ] },
    "get_project":    { "project": { "id": "proj_mock123" } },
    "list_projects":  { "projects": [] }
  }
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/integrations/test_deploy_adapter_configs.py::test_neon_config_actions_and_mock -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
rtk git add src/integrations/configs/neon.json tests/integrations/test_deploy_adapter_configs.py
rtk git commit -m "feat(integrations): neon adapter config + mock (postgres)"
```

---

## Task 5: Upstash adapter config + mock (pre-encoded Basic auth)

**Files:**
- Create: `src/integrations/configs/upstash.json`
- Test: `tests/integrations/test_deploy_adapter_configs.py`

- [ ] **Step 1: Write the failing test**

```python
def test_upstash_config_uses_header_auth_and_has_mock():
    cfg = _load("upstash")
    assert cfg["service_name"] == "upstash"
    # header auth carrying a pre-encoded "Basic <b64>" token — no engine change (twilio pattern)
    assert cfg["auth_type"] == "header"
    assert cfg["auth_header"] == "Authorization"
    assert cfg["auth_token_field"] == "basic_auth_b64"
    for a in ("create_redis", "get_redis", "list_redis"):
        assert a in cfg["actions"]
    m = cfg["mock_responses"]["create_redis"]
    assert m.get("endpoint") and m.get("password")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/integrations/test_deploy_adapter_configs.py::test_upstash_config_uses_header_auth_and_has_mock -v`
Expected: FAIL

- [ ] **Step 3: Verify Upstash Developer API against live docs**

WebFetch `https://upstash.com/docs/devops/developer-api/redis/create_database`. Confirm base `https://api.upstash.com/v2`, `POST /redis/database`, HTTP Basic (email:api_key), and the **body field names** (`database_name`, `region`/`primary_region`, `plan`) and response fields (`endpoint`, `port`, `password`, `rest_token`).

> The credential stores `basic_auth_b64 = base64("email:api_key")`. With `auth_type:"header"` + `auth_token_field:"basic_auth_b64"`, `HttpIntegration` sets `Authorization: <token>` verbatim (`http_integration.py:396-397`), so the stored value MUST be the full `Basic <b64>` string. Document this in the credential-add flow.

- [ ] **Step 4: Author `upstash.json`**

```json
{
  "service_name": "upstash",
  "base_url": "https://api.upstash.com/v2",
  "auth_type": "header",
  "auth_header": "Authorization",
  "auth_token_field": "basic_auth_b64",
  "actions": {
    "create_redis": { "method": "POST", "path": "/redis/database", "required_params": ["database_name", "region"] },
    "get_redis":    { "method": "GET",  "path": "/redis/database/{id}", "required_params": ["id"] },
    "list_redis":   { "method": "GET",  "path": "/redis/databases", "required_params": [] }
  },
  "mock_responses": {
    "create_redis": { "database_id": "db_mock123", "endpoint": "mock.upstash.io", "port": 6379, "password": "mockpw", "rest_token": "mocktoken" },
    "get_redis":    { "database_id": "db_mock123", "endpoint": "mock.upstash.io" },
    "list_redis":   []
  }
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/integrations/test_deploy_adapter_configs.py::test_upstash_config_uses_header_auth_and_has_mock -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
rtk git add src/integrations/configs/upstash.json tests/integrations/test_deploy_adapter_configs.py
rtk git commit -m "feat(integrations): upstash adapter config + mock (redis, header-basic auth)"
```

---

## Task 6: Registry loads all new adapters + mock-mode returns tagged responses

**Files:**
- Test: `tests/integrations/test_deploy_adapter_configs.py`

- [ ] **Step 1: Write the failing test**

> ⚠️ **CORRECTED per Opus plan-review (empirically reproduced):** `HttpIntegration._maybe_mock`
> (`http_integration.py:249`) consults the module **singleton** `get_integration_registry()`, NOT a
> fresh `IntegrationRegistry(...)` instance, and `mock_mode` is resolved once at first singleton
> creation and cached module-globally — `monkeypatch.setenv` can't change it after. So a naive
> fresh-instance test is flaky (passes clean-process, FAILS if any earlier test froze the singleton
> with `KUTAI_VENDOR_LIVE=1`). Install a mock-on registry AS the singleton (the canonical pattern in
> `tests/test_integration_mock_mode.py:174-199`) and restore it:

```python
import pytest

@pytest.mark.asyncio
async def test_registry_discovers_new_adapters_and_mocks_are_tagged():
    import src.integrations.registry as reg_mod
    from src.integrations.registry import IntegrationRegistry
    orig = reg_mod._registry
    reg_mod._registry = IntegrationRegistry(auto_discover=True, mock_mode=True)
    try:
        reg = reg_mod._registry
        for svc in ("render", "neon", "upstash", "vercel"):
            assert reg.get(svc) is not None, f"{svc} not discovered"
        # a mocked deploy/provision response must carry mocked:true (anti-fake guard depends on it)
        render = reg.get("render")
        res = await render.execute("get_deploy", {"id": "srv_mock123", "deployId": "dep_mock123"})
        assert res.get("mocked") is True
        assert res["data"]["status"] == "live"
    finally:
        reg_mod._registry = orig
```

- [ ] **Step 2: Run test to verify it fails, then passes**

Run: `python -m pytest tests/integrations/test_deploy_adapter_configs.py::test_registry_discovers_new_adapters_and_mocks_are_tagged -v`
Expected: FAIL before Tasks 3-5 land; PASS after (this task adds only the test — it validates prior tasks end-to-end). If `mocked` is absent, the mock block is missing on that action — fix the config, not the test.

- [ ] **Step 3: Commit**

```bash
rtk git add tests/integrations/test_deploy_adapter_configs.py
rtk git commit -m "test(integrations): registry discovers deploy adapters + mock tagging"
```

---

## Task 7: 7.13 real_tool_kind housekeeping

**Files:**
- Modify: `src/workflows/i2p/i2p_v3.json` (the `staging_environment` step, `real_tool_kind`)
- Test: `tests/workflows/test_z6_w1_real_tool_kind.py` (existing test file for this field)

- [ ] **Step 1: Locate the step + the EXISTING test that pins the old value**

Run: `python -m pytest tests/workflows/test_z6_w1_real_tool_kind.py -v` (baseline green). Note:
`tests/workflows/test_z6_w1_real_tool_kind.py:56` **hard-asserts** `by_id["7.13"]["real_tool_kind"]
== "vercel|railway|fly"` — Task 7 MUST update that line too, or it breaks (Opus review).

- [ ] **Step 2: Write the new test (UTF-8 encoding is MANDATORY on Windows)**

```python
def test_staging_env_real_tool_kind_targets_vercel_render():
    import json
    from pathlib import Path
    p = Path(__file__).resolve().parents[2] / "src" / "workflows" / "i2p" / "i2p_v3.json"
    with open(p, encoding="utf-8") as f:   # i2p_v3.json has non-cp1252 bytes → utf-8 required
        wf = json.load(f)
    steps = wf.get("steps") or wf.get("workflow", {}).get("steps") or []
    step = next(s for s in steps if s.get("id") == "7.13"
                or s.get("name") == "staging_environment")
    assert step["real_tool_kind"] == "vercel|render"
```

- [ ] **Step 3: Run to verify it fails** — Expected: FAIL (still `vercel|railway|fly`).

- [ ] **Step 4: Edit `i2p_v3.json`** — change the 7.13 step's `real_tool_kind` to `"vercel|render"`.

- [ ] **Step 5: Update the existing pin** — in `tests/workflows/test_z6_w1_real_tool_kind.py:56`
change the assertion to `assert by_id["7.13"]["real_tool_kind"] == "vercel|render"`.

- [ ] **Step 6: Run to verify BOTH pass** — `python -m pytest tests/workflows/test_z6_w1_real_tool_kind.py -v` and the new test.

- [ ] **Step 6: Commit**

```bash
rtk git add src/workflows/i2p/i2p_v3.json tests/workflows/test_z6_w1_real_tool_kind.py
rtk git commit -m "fix(i2p): 7.13 real_tool_kind vercel|render (drop railway/fly, not free)"
```

---

## Task 8: Git-prereq — `git_prepare_repo` mechanical executor (create repo + push scaffold)

**Files:**
- Create: `packages/mr_roboto/src/mr_roboto/executors/git_prepare_repo.py`
- Modify: `packages/mr_roboto/src/mr_roboto/__init__.py` (dispatch branch, near `stripe_provision_products` at ~line 2882)
- Test: `packages/mr_roboto/tests/test_git_prepare_repo.py`

> **Purpose (review H2):** deploys build from a connected git repo; the app is not pushed
> (`github_init_status.md = pending:gh_unauthenticated`). This executor creates the GitHub repo
> and pushes the mission's `backend/`+`frontend/` trees using a PAT-authenticated `git push` via
> the `shell` path. Idempotent (skip create if repo exists). Returns `{ok, repo_url, pushed}`.
>
> **NOTE (Opus review):** the repo already has `packages/mr_roboto/src/mr_roboto/init_mission_github_repo.py`
> which does create+push via the **`gh` CLI**. We deliberately add a PAT-push variant here because
> `gh` is unauthenticated on this host (`github_init_status.md`). Persist the resulting repo URL to
> `missions.github_repo_url` (Plan 2 reads it) — mirror `init_mission_github_repo._persist_repo_url`.

- [ ] **Step 1: Write the failing test (git ops stubbed — no network)**

```python
# packages/mr_roboto/tests/test_git_prepare_repo.py
import pytest

@pytest.mark.asyncio
async def test_git_prepare_repo_creates_and_pushes(monkeypatch, tmp_path):
    from mr_roboto.executors import git_prepare_repo as gpr
    calls = {"created": False, "pushed": False}

    async def fake_create_repo(name, token):
        calls["created"] = True
        return {"ok": True, "repo_url": f"https://github.com/kutay/{name}.git", "existed": False}

    async def fake_git_push(workspace, repo_url, token):
        calls["pushed"] = True
        return {"ok": True}

    monkeypatch.setattr(gpr, "_create_repo", fake_create_repo)
    monkeypatch.setattr(gpr, "_git_push_scaffold", fake_git_push)

    task = {"payload": {"action": "git_prepare_repo", "repo_name": "habithub",
                        "workspace": str(tmp_path)}, "context": {"mission_id": 90}}
    res = await gpr.run(task)
    assert res["ok"] and res["pushed"] and calls["created"] and calls["pushed"]
    assert res["repo_url"].endswith("habithub.git")

@pytest.mark.asyncio
async def test_git_prepare_repo_requires_token(monkeypatch, tmp_path):
    from mr_roboto.executors import git_prepare_repo as gpr
    async def no_cred(_service): return None
    monkeypatch.setattr(gpr, "_get_github_token", no_cred)
    task = {"payload": {"action": "git_prepare_repo", "repo_name": "x", "workspace": str(tmp_path)}}
    res = await gpr.run(task)
    assert res["ok"] is False and "credential" in (res.get("reason") or "").lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest packages/mr_roboto/tests/test_git_prepare_repo.py -v`
Expected: FAIL (module missing)

- [ ] **Step 3: Implement `git_prepare_repo.py`**

```python
"""Mechanical: create a GitHub repo + push the mission scaffold (git-prereq for deploy)."""
from __future__ import annotations
import asyncio, os
from typing import Any

async def _get_github_token(service: str = "github") -> str | None:
    from src.security.credential_store import get_credential
    cred = await get_credential(service)
    if not cred:
        return None
    return cred.get("token") or cred.get("api_key")

async def _gh_login(token: str) -> str | None:
    """Resolve the authenticated user's login (owner) for building an owner/name clone URL."""
    import httpx
    async with httpx.AsyncClient(timeout=15.0) as c:
        r = await c.get("https://api.github.com/user",
                        headers={"Authorization": f"Bearer {token}",
                                 "Accept": "application/vnd.github+json"})
        return r.json().get("login") if r.status_code == 200 else None

async def _create_repo(name: str, token: str) -> dict[str, Any]:
    """Create repo via the GitHub adapter; treat 'already exists' as success (idempotent)."""
    from src.integrations.registry import get_integration_registry
    gh = get_integration_registry().get("github")
    res = await gh.execute("create_repo", {"name": name})
    if res.get("status") == "ok":
        data = res.get("data", {})
        return {"ok": True, "repo_url": data.get("clone_url") or data.get("html_url"), "existed": False}
    # 422 = name already exists → idempotent: resolve the REAL owner/name clone URL
    # (do NOT return a bare-name URL — https://github.com/{name}.git is unpushable). Opus review.
    if res.get("status_code") == 422:
        login = await _gh_login(token)
        if not login:
            return {"ok": False, "reason": "repo_exists_but_owner_unresolved"}
        return {"ok": True, "repo_url": f"https://github.com/{login}/{name}.git", "existed": True}
    return {"ok": False, "reason": res.get("error", "create_repo failed")}

async def _git_push_scaffold(workspace: str, repo_url: str, token: str) -> dict[str, Any]:
    """Init + commit + push the workspace tree using a PAT-authenticated remote."""
    authed = repo_url.replace("https://", f"https://x-access-token:{token}@")
    cmds = [
        ["git", "init", "-q"],
        ["git", "add", "-A"],
        ["git", "-c", "user.email=kutai@local", "-c", "user.name=KutAI",
         "commit", "-q", "-m", "chore: initial scaffold"],
        ["git", "branch", "-M", "main"],
        ["git", "remote", "add", "origin", authed],
        ["git", "push", "-q", "-u", "origin", "main"],
    ]
    for cmd in cmds:
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=workspace,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        _, err = await proc.communicate()
        if proc.returncode != 0 and b"nothing to commit" not in (err or b""):
            return {"ok": False, "reason": f"{cmd[1] if len(cmd)>1 else cmd[0]}: {err.decode(errors='replace')[:200]}"}
    return {"ok": True}

async def run(task: dict) -> dict[str, Any]:
    payload = (task.get("payload") or (task.get("context") or {}).get("payload") or {})
    repo_name = payload.get("repo_name")
    workspace = payload.get("workspace")
    if not repo_name or not workspace:
        return {"ok": False, "reason": "missing repo_name or workspace"}
    if not os.path.isdir(workspace):
        return {"ok": False, "reason": f"workspace not found: {workspace}"}
    token = await _get_github_token()
    if not token:
        return {"ok": False, "reason": "no github credential (add a PAT via /credential add github)"}
    created = await _create_repo(repo_name, token)
    if not created.get("ok"):
        return created
    pushed = await _git_push_scaffold(workspace, created["repo_url"], token)
    if not pushed.get("ok"):
        return {"ok": False, "reason": pushed.get("reason"), "repo_url": created["repo_url"]}
    return {"ok": True, "repo_url": created["repo_url"], "pushed": True, "existed": created.get("existed", False)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest packages/mr_roboto/tests/test_git_prepare_repo.py -v`
Expected: PASS

- [ ] **Step 5: Register dispatch in `mr_roboto/__init__.py`** (after the `stripe_provision_products` branch, ~line 2897)

```python
    if action == "git_prepare_repo":
        from mr_roboto.executors.git_prepare_repo import run as _gpr_run
        try:
            res = await _gpr_run(task)
            return Action(status="completed" if res.get("ok") else "failed",
                          error=None if res.get("ok") else res.get("reason"), result=res)
        except Exception as e:
            return Action(status="failed", error=str(e))
```

Add `VERB_REVERSIBILITY["git_prepare_repo"] = "partial"` (creating a repo + push is partially reversible) in `packages/mr_roboto/src/mr_roboto/reversibility.py`, and confirm `test_reversibility_registry.py` stays green.

- [ ] **Step 6: Write + run a dispatch test**

```python
@pytest.mark.asyncio
async def test_git_prepare_repo_dispatches(monkeypatch, tmp_path):
    import mr_roboto
    from mr_roboto.executors import git_prepare_repo as gpr
    async def ok_run(_t): return {"ok": True, "repo_url": "https://github.com/kutay/x.git", "pushed": True}
    monkeypatch.setattr(gpr, "run", ok_run)
    act = await mr_roboto.run({"payload": {"action": "git_prepare_repo", "repo_name": "x", "workspace": str(tmp_path)}})
    assert act.status == "completed"
```

Run: `python -m pytest packages/mr_roboto/tests/test_git_prepare_repo.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
rtk git add packages/mr_roboto/src/mr_roboto/executors/git_prepare_repo.py packages/mr_roboto/src/mr_roboto/__init__.py packages/mr_roboto/src/mr_roboto/reversibility.py packages/mr_roboto/tests/test_git_prepare_repo.py
rtk git commit -m "feat(mr_roboto): git_prepare_repo executor (create repo + push scaffold)"
```

---

## Task 9: Confirm `github.json` has `create_repo` (dependency of Task 8)

**Files:**
- Read/Modify: `src/integrations/configs/github.json`

- [ ] **Step 1:** Read `github.json`. Per Opus review, `create_repo` **already exists**
(`POST /user/repos`, `required_params:["name"]`) — so the action is a no-op verify. But it has
**no `mock_responses`**, so the git-prereq chain can't be mock-dry-run. Add a `mock_responses`
block for `create_repo` (tagged like the others) so mock-mode runs return a clone URL:

```json
"mock_responses": { "create_repo": { "clone_url": "https://github.com/kutay/mock-repo.git", "html_url": "https://github.com/kutay/mock-repo" } }
```
Add a config-guard assertion:

```python
def test_github_has_create_repo_and_mock():
    cfg = _load("github")
    assert "create_repo" in cfg["actions"]
    assert "create_repo" in cfg.get("mock_responses", {})
```

- [ ] **Step 2:** Run `python -m pytest tests/integrations/test_deploy_adapter_configs.py::test_github_has_create_repo_and_mock -v`; make it pass (add the mock block; add the action only if somehow missing).

- [ ] **Step 3: Commit** (only if changed)

```bash
rtk git add src/integrations/configs/github.json tests/integrations/test_deploy_adapter_configs.py
rtk git commit -m "feat(integrations): ensure github create_repo action for deploy prereq"
```

---

## Verification (whole plan)

- [ ] `python -m pytest tests/integrations/test_deploy_adapter_configs.py packages/mr_roboto/tests/test_git_prepare_repo.py -v` — all green.
- [ ] Registry loads render/neon/upstash/vercel with no discovery warnings (check logs).
- [ ] No live network calls in any test (all mock-mode / stubbed).
- [ ] Every external API table was reconciled against live docs before the config was authored.

## Notes for the executor
- Do NOT run the full pytest suite cold in this env (chromadb/sentence-transformers import hangs) — run the targeted files above only.
- These configs are additive; nothing here performs a real deploy. Real calls need `KUTAI_VENDOR_LIVE=1` + stored creds (Plan 2).
- Restart-gated changes: `git_prepare_repo.py` + the `__init__.py` dispatch branch are `.py` → the bot needs a founder `/restart` to load them. The JSON configs are picked up on next registry init.
