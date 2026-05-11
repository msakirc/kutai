"""Z6 T6C — Apple App Store Connect adapter.

Covers:
  • Config parses and registers as an HttpIntegration.
  • Credential schema has the right fields.
  • mint_apple_jwt produces a 3-part JWT with the right header/claims
    (verified by decoding without signature check).
  • HttpIntegration.execute injects ``Authorization: Bearer <jwt>``
    when ``auth_type='jwt_p8'``.
  • Admission emits the rich vendor_enroll card when credentials missing.
"""
from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import pytest


_REPO = Path(__file__).resolve().parents[2]
_CONFIG = _REPO / "src" / "integrations" / "configs" / "apple_appstore.json"
_CRED_SCHEMA = _REPO / "credential_schemas" / "apple_appstore.json"


# Pre-generated P-256 private key for tests (deterministic; not used in
# production). Generated once via cryptography ec.generate_private_key.
def _make_p256_pem() -> str:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    key = ec.generate_private_key(ec.SECP256R1())
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateKeyFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return pem.decode()


def _b64url_decode(seg: str) -> bytes:
    seg += "=" * (-len(seg) % 4)
    return base64.urlsafe_b64decode(seg.encode())


def test_config_parses():
    with _CONFIG.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg["service_name"] == "apple_appstore"
    assert cfg["base_url"].startswith("https://api.appstoreconnect.apple.com")
    assert cfg["auth_type"] == "jwt_p8"
    for action in ("list_apps", "list_builds", "list_app_store_versions",
                   "submit_for_review", "list_review_states"):
        assert action in cfg["actions"], f"missing action {action}"


def test_credential_schema():
    with _CRED_SCHEMA.open("r", encoding="utf-8") as f:
        sch = json.load(f)
    assert sch["service_name"] == "apple_appstore"
    assert set(sch["required_fields"]) == {"team_id", "key_id", "private_key_pem"}


def test_http_integration_registers_apple():
    from src.integrations.http_integration import HttpIntegration
    integ = HttpIntegration.from_service_name("apple_appstore")
    assert integ.service_name == "apple_appstore"
    assert "list_apps" in integ.capabilities()


def test_mint_apple_jwt_structure():
    from src.integrations.adapters.apple_jwt import mint_apple_jwt
    pem = _make_p256_pem()
    tok = mint_apple_jwt("TEAM123", "KID456", pem, expires_in_seconds=600)
    parts = tok.split(".")
    assert len(parts) == 3
    header = json.loads(_b64url_decode(parts[0]))
    payload = json.loads(_b64url_decode(parts[1]))
    assert header["alg"] == "ES256"
    assert header["typ"] == "JWT"
    assert header["kid"] == "KID456"
    assert payload["iss"] == "TEAM123"
    assert payload["aud"] == "appstoreconnect-v1"
    assert payload["exp"] - payload["iat"] == 600
    assert payload["iat"] <= int(time.time()) + 5


def test_mint_apple_jwt_validates_required_fields():
    from src.integrations.adapters.apple_jwt import mint_apple_jwt
    with pytest.raises(ValueError):
        mint_apple_jwt("", "k", "x")
    with pytest.raises(ValueError):
        mint_apple_jwt("t", "", "x")
    with pytest.raises(ValueError):
        mint_apple_jwt("t", "k", "")


@pytest.mark.asyncio
async def test_http_integration_injects_jwt_bearer(monkeypatch):
    """The execute path should mint a JWT and pass it as Authorization."""
    from src.integrations.http_integration import HttpIntegration

    integ = HttpIntegration.from_service_name("apple_appstore")
    pem = _make_p256_pem()
    cred = {
        "team_id": "TEAM999",
        "key_id": "KID999",
        "private_key_pem": pem,
    }

    async def _fake_get_cred(service):
        return cred if service == "apple_appstore" else None

    monkeypatch.setattr(
        "src.security.credential_store.get_credential", _fake_get_cred,
    )

    captured: dict = {}

    async def _fake_http(method, url, headers, json_body=None, params=None):
        captured["method"] = method
        captured["url"] = url
        captured["headers"] = headers
        return {"status_code": 200, "body": "{}", "headers": {}}

    monkeypatch.setattr(
        "src.integrations.http_integration._get_http_func",
        lambda: _fake_http,
    )

    result = await integ.execute("list_apps", {})
    assert result["status"] == "ok"
    auth = captured["headers"].get("Authorization", "")
    assert auth.startswith("Bearer ")
    jwt_tok = auth[len("Bearer "):]
    parts = jwt_tok.split(".")
    assert len(parts) == 3
    payload = json.loads(_b64url_decode(parts[1]))
    assert payload["iss"] == "TEAM999"
    assert payload["aud"] == "appstoreconnect-v1"


@pytest.mark.asyncio
async def test_admission_emits_apple_vendor_enroll(tmp_path, monkeypatch):
    """Missing credentials on apple_appstore should yield a rich
    vendor_enroll founder_action, not the generic credential_paste."""
    db_path = tmp_path / "z6_t6c.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    await db_mod.init_db()
    import src.founder_actions as fa
    fa._reset_lifecycle_cache()

    from general_beckman.z6_admission import check_z6_admission
    monkeypatch.setattr(
        "general_beckman.z6_admission._resolve_adapter",
        lambda kinds: "apple_appstore" if "apple_appstore" in kinds else None,
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
            '"real_tool_kind": "apple_appstore"}'
        ),
    }
    res = await check_z6_admission(task, mid)
    assert res.admit is False
    rows = await fa.list_by_mission(mid)
    assert rows
    assert rows[0].kind == "vendor_enroll"
    assert "Apple" in rows[0].title
    joined = " ".join(rows[0].instructions)
    assert "developer.apple.com" in joined or "App Store Connect" in joined
