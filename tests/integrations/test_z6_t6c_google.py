"""Z6 T6C — Google Play Console adapter."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest


_REPO = Path(__file__).resolve().parents[2]
_CONFIG = _REPO / "src" / "integrations" / "configs" / "google_play.json"
_CRED_SCHEMA = _REPO / "credential_schemas" / "google_play.json"


def _make_rsa_pem() -> str:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()


def _make_service_account() -> dict:
    return {
        "type": "service_account",
        "project_id": "test-proj",
        "private_key_id": "PKID",
        "private_key": _make_rsa_pem(),
        "client_email": "test@test-proj.iam.gserviceaccount.com",
        "client_id": "12345",
        "token_uri": "https://oauth2.googleapis.com/token",
    }


def _b64url_decode(seg: str) -> bytes:
    seg += "=" * (-len(seg) % 4)
    return base64.urlsafe_b64decode(seg.encode())


def test_config_parses():
    with _CONFIG.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg["service_name"] == "google_play"
    assert cfg["auth_type"] == "oauth_service_account"
    for action in ("list_apps", "list_release_tracks",
                   "upload_apk_metadata", "list_review_status"):
        assert action in cfg["actions"], f"missing action {action}"


def test_credential_schema():
    with _CRED_SCHEMA.open("r", encoding="utf-8") as f:
        sch = json.load(f)
    assert sch["service_name"] == "google_play"
    assert "service_account_json" in sch["required_fields"]


def test_http_integration_registers_google_play():
    from src.integrations.http_integration import HttpIntegration
    integ = HttpIntegration.from_service_name("google_play")
    assert integ.service_name == "google_play"
    assert "list_apps" in integ.capabilities()


def test_validates_required_sa_fields():
    from src.integrations.adapters.google_sa import _mint_assertion
    sa = _make_service_account()
    # Should produce a 3-part JWT.
    jwt = _mint_assertion(sa, ["https://www.googleapis.com/auth/androidpublisher"])
    parts = jwt.split(".")
    assert len(parts) == 3
    header = json.loads(_b64url_decode(parts[0]))
    payload = json.loads(_b64url_decode(parts[1]))
    assert header["alg"] == "RS256"
    assert header["kid"] == "PKID"
    assert payload["iss"] == sa["client_email"]
    assert payload["aud"] == sa["token_uri"]
    assert payload["scope"] == "https://www.googleapis.com/auth/androidpublisher"


@pytest.mark.asyncio
async def test_mint_google_oauth_token_caches(monkeypatch):
    from src.integrations.adapters import google_sa
    google_sa._clear_cache_for_tests()
    sa = _make_service_account()

    calls = {"n": 0}

    async def _fake_exchange(assertion, token_uri):
        calls["n"] += 1
        return ("ya29.fake-token", 3600)

    monkeypatch.setattr(google_sa, "_exchange_assertion", _fake_exchange)

    t1 = await google_sa.mint_google_oauth_token(sa)
    t2 = await google_sa.mint_google_oauth_token(sa)
    assert t1 == t2 == "ya29.fake-token"
    assert calls["n"] == 1  # second call hit the cache.


@pytest.mark.asyncio
async def test_mint_validates_inputs():
    from src.integrations.adapters import google_sa
    google_sa._clear_cache_for_tests()
    with pytest.raises(ValueError):
        await google_sa.mint_google_oauth_token({})
    with pytest.raises(ValueError):
        await google_sa.mint_google_oauth_token(
            {"client_email": "x", "private_key": ""}
        )


@pytest.mark.asyncio
async def test_http_integration_injects_oauth_bearer(monkeypatch):
    from src.integrations import adapters
    from src.integrations.adapters import google_sa
    from src.integrations.http_integration import HttpIntegration

    google_sa._clear_cache_for_tests()
    integ = HttpIntegration.from_service_name("google_play")
    cred = {"service_account_json": _make_service_account()}

    async def _fake_get_cred(service):
        return cred if service == "google_play" else None

    async def _fake_exchange(assertion, token_uri):
        return ("ya29.injected-token", 3600)

    monkeypatch.setattr(google_sa, "_exchange_assertion", _fake_exchange)
    monkeypatch.setattr(
        "src.security.credential_store.get_credential", _fake_get_cred,
    )

    captured = {}

    async def _fake_http(method, url, headers, json_body=None, params=None):
        captured["headers"] = headers
        captured["url"] = url
        return {"status_code": 200, "body": "{}", "headers": {}}

    monkeypatch.setattr(
        "src.integrations.http_integration._get_http_func",
        lambda: _fake_http,
    )

    result = await integ.execute(
        "list_apps", {"package_name": "com.example.app"},
    )
    assert result["status"] == "ok"
    assert captured["headers"]["Authorization"] == "Bearer ya29.injected-token"
    # Path parameter substitution.
    assert "com.example.app" in captured["url"]


@pytest.mark.asyncio
async def test_admission_emits_google_play_vendor_enroll(tmp_path, monkeypatch):
    db_path = tmp_path / "z6_t6c_g.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    await db_mod.init_db()
    import src.founder_actions as fa
    fa._reset_lifecycle_cache()

    from general_beckman.z6_admission import check_z6_admission
    monkeypatch.setattr(
        "general_beckman.z6_admission._resolve_adapter",
        lambda kinds: "google_play" if "google_play" in kinds else None,
    )

    async def _no_cred(_svc):
        return None
    monkeypatch.setattr(
        "src.security.credential_store.get_credential", _no_cred,
    )

    mid = await db_mod.add_mission("m", "")
    task = {
        "id": 1, "mission_id": mid, "needs_real_tools": 1,
        "context": (
            '{"workflow_step_id": "13.1", '
            '"real_tool_kind": "google_play"}'
        ),
    }
    res = await check_z6_admission(task, mid)
    assert res.admit is False
    rows = await fa.list_by_mission(mid)
    assert rows
    assert rows[0].kind == "vendor_enroll"
    assert "Google Play" in rows[0].title or "Google Play" in " ".join(rows[0].instructions)
