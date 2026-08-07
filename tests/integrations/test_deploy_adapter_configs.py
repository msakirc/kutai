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
