"""Z6 W4 — mobile workflow step tagged with apple_appstore|google_play.

Audit found that 0 steps referenced the mobile adapters added by T6C.
Reality check: i2p_v3.json DOES have a mobile step (14.8
app_store_submission), so the right fix is to tag it rather than
deferring to a future Z5 mobile track. This test pins the tag in place
and verifies the adapters parse and round-trip through the resolver.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_WF_JSON = _REPO / "src" / "workflows" / "i2p" / "i2p_v3.json"
_CFG_DIR = _REPO / "src" / "integrations" / "configs"
_CRED_SCHEMAS = _REPO / "credential_schemas"


def test_step_14_8_is_tagged_for_mobile_adapters():
    with _WF_JSON.open("r", encoding="utf-8") as f:
        wf = json.load(f)
    step = next(s for s in wf["steps"] if s.get("id") == "14.8")
    assert step["needs_real_tools"] is True
    assert step["real_tool_kind"] == "apple_appstore|google_play"
    # Still gated on the no_mobile_app skip_when so non-mobile missions
    # don't get the founder_action card pop.
    assert "no_mobile_app" in step.get("skip_when", [])


def test_apple_appstore_config_parses_and_has_submit_action():
    with (_CFG_DIR / "apple_appstore.json").open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg["service_name"] == "apple_appstore"
    assert cfg["auth_type"] == "jwt_p8"
    assert "submit_for_review" in cfg["actions"]


def test_google_play_config_parses_and_has_submission_action():
    with (_CFG_DIR / "google_play.json").open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg["service_name"] == "google_play"
    # Google Play uses OAuth2 service account.
    assert cfg["auth_type"] in (
        "oauth2_sa", "google_sa", "service_account", "oauth_service_account",
    )
    # At least one action that drives a real submission.
    assert any(
        k in cfg["actions"] for k in (
            "create_edit", "commit_edit", "upload_apk", "upload_bundle",
            "list_apps", "edits_insert", "edits_commit",
        )
    ), f"google_play config missing a submission action: {list(cfg['actions'])}"


def test_apple_credential_schema_valid():
    with (_CRED_SCHEMAS / "apple_appstore.json").open("r", encoding="utf-8") as f:
        sch = json.load(f)
    assert sch["service_name"] == "apple_appstore"
    required = set(sch["required_fields"])
    assert {"team_id", "key_id", "private_key_pem"}.issubset(required)


def test_google_credential_schema_valid():
    with (_CRED_SCHEMAS / "google_play.json").open("r", encoding="utf-8") as f:
        sch = json.load(f)
    assert sch["service_name"] == "google_play"
    # Service-account-style schema: needs at least private_key/key_json.
    required = set(sch["required_fields"])
    assert required, "google_play credential schema must declare required fields"


@pytest.mark.asyncio
async def test_resolver_picks_apple_when_only_apple_registered(monkeypatch):
    from src.integrations.resolver import resolve_real_tool

    class _Reg:
        def __init__(self, names):
            self._names = set(names)

        def get(self, name):
            return object() if name in self._names else None

    import src.integrations.registry as reg_mod
    import src.security.credential_store as cs_mod
    monkeypatch.setattr(
        reg_mod, "get_integration_registry", lambda: _Reg({"apple_appstore"}),
    )

    async def _has_apple(svc):
        return {"team_id": "T", "key_id": "K", "private_key_pem": "P"} if svc == "apple_appstore" else None
    monkeypatch.setattr(cs_mod, "get_credential", _has_apple)

    picked = await resolve_real_tool("apple_appstore|google_play")
    assert picked == "apple_appstore"


@pytest.mark.asyncio
async def test_resolver_picks_google_when_only_google_registered(monkeypatch):
    from src.integrations.resolver import resolve_real_tool

    class _Reg:
        def __init__(self, names):
            self._names = set(names)

        def get(self, name):
            return object() if name in self._names else None

    import src.integrations.registry as reg_mod
    import src.security.credential_store as cs_mod
    monkeypatch.setattr(
        reg_mod, "get_integration_registry", lambda: _Reg({"google_play"}),
    )

    async def _has_google(svc):
        return {"key_json": "{}"} if svc == "google_play" else None
    monkeypatch.setattr(cs_mod, "get_credential", _has_google)

    picked = await resolve_real_tool("apple_appstore|google_play")
    assert picked == "google_play"


def test_expander_propagates_mobile_real_tool_kind_to_task_context():
    from src.workflows.engine.expander import expand_steps_to_tasks

    with _WF_JSON.open("r", encoding="utf-8") as f:
        wf = json.load(f)
    step = next(s for s in wf["steps"] if s.get("id") == "14.8")

    tasks = expand_steps_to_tasks([step], mission_id="1")
    assert tasks
    ctx = tasks[0]["context"]
    if isinstance(ctx, str):
        ctx = json.loads(ctx)
    assert ctx.get("real_tool_kind") == "apple_appstore|google_play"
    assert ctx.get("needs_real_tools") is True
