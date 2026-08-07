"""Config-guard tests for the free-tier deploy adapters.

Covers vercel poll enrichment, render/neon/upstash declarative adapter
configs, their credential schemas, registry discovery + mock tagging, and
the github create_repo prereq. All mock-mode / stubbed — no live network.
"""
import json
import os

import pytest

CONFIGS = os.path.join(
    os.path.dirname(__file__), "..", "..", "src", "integrations", "configs"
)
CRED = os.path.join(os.path.dirname(__file__), "..", "..", "credential_schemas")


def _load(name):
    with open(os.path.join(CONFIGS, f"{name}.json")) as f:
        return json.load(f)


def _load_cred(name):
    with open(os.path.join(CRED, f"{name}.json")) as f:
        return json.load(f)


# --------------------------------------------------------------------------
# Task 1 — vercel get_deployment poll action + mock_responses
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# Task 2 — credential schemas for render / neon / upstash
# --------------------------------------------------------------------------
def test_new_credential_schemas_exist_and_shaped():
    for name, required in [("render", ["api_key"]), ("neon", ["api_key"]),
                           ("upstash", ["basic_auth_b64"])]:
        s = _load_cred(name)
        assert s["service_name"] == name
        for field in required:
            assert field in s["required_fields"], f"{name} missing {field}"


# --------------------------------------------------------------------------
# Task 3 — render adapter config + mock (backend host)
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# Task 4 — neon adapter config + mock (postgres)
# --------------------------------------------------------------------------
def test_neon_config_actions_and_mock():
    cfg = _load("neon")
    assert cfg["service_name"] == "neon"
    for a in ("create_project", "get_project", "list_projects"):
        assert a in cfg["actions"]
    # create must surface the connection string downstream needs
    conn = cfg["mock_responses"]["create_project"].get("connection_uris")
    assert conn and conn[0].get("connection_uri", "").startswith("postgresql://")


# --------------------------------------------------------------------------
# Task 5 — upstash adapter config + mock (redis, header-basic auth)
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# Task 6 — registry discovers new adapters + mock-mode returns tagged responses
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# Task 9 — github create_repo action + mock (deploy git-prereq)
# --------------------------------------------------------------------------
def test_github_has_create_repo_and_mock():
    cfg = _load("github")
    assert "create_repo" in cfg["actions"]
    assert "create_repo" in cfg.get("mock_responses", {})
