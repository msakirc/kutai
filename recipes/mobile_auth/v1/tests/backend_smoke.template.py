"""mobile_auth recipe — backend smoke tests.

NOTE — execution context:
    This template runs POST-instantiation. After the recipe is instantiated
    via mr_roboto ``instantiate_recipe`` into a target mission workspace,
    ``backend.template.py`` becomes the project's real auth module and these
    smoke tests are copied alongside. Imports below reference the instantiated
    layout, NOT the recipe source tree.

    Running this file directly from ``recipes/mobile_auth/v1/tests/`` will
    fail — the recipe source intentionally has no importable module (the
    ``.template`` suffix marks scaffolds). Use the structural Z5 recipe tests
    to validate the template files; use this file in the instantiated
    workspace.

What is covered
---------------
* email/password register → login → /me round-trip
* refresh-token rotation (old token revoked, new pair issued)
* Apple ID-token verification + JWT issue — the provider JWKS fetch and
  ``jwt.decode`` are monkeypatched so the test needs no network and no real
  Apple key.
* Google ID-token audience check.

The provider verification is tested by patching ``_verify_provider_id_token``
so the test asserts the *router wiring* (claims → user lookup → session
issue), not jose's RS256 maths.

RECIPE_PARAM:BCRYPT_COST=4  (fast cost for tests — override in conftest)

Run (post-instantiation):
    pytest tests/mobile_auth/ -x -q
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest
import httpx
from fastapi import FastAPI

# JWT_SECRET + provider client ids must exist before importing the module.
os.environ.setdefault("MOBILE_AUTH_JWT_SECRET", "test-secret-do-not-use-in-prod")  # noqa: S105
os.environ.setdefault("APPLE_BUNDLE_ID", "com.example.testapp")
os.environ.setdefault("GOOGLE_OAUTH_CLIENT_ID", "test-google-client-id.apps.googleusercontent.com")

# During instantiation, backend.template.py becomes mobile_auth/routes.py.
from recipes.mobile_auth.v1 import backend_template as auth  # type: ignore[import]  # noqa: E402

# ---------------------------------------------------------------------------
# In-memory DB stubs
# ---------------------------------------------------------------------------

_USERS: dict = {}      # id -> row
_SESSIONS: dict = {}   # token_hash -> {id, user_id, expires_at, revoked_at}
_NEXT_ID = [1]


def _next_id() -> int:
    uid = _NEXT_ID[0]
    _NEXT_ID[0] += 1
    return uid


def _reset_state():
    _USERS.clear()
    _SESSIONS.clear()
    _NEXT_ID[0] = 1


async def _db_get_user_by_email(email: str):
    return next((u for u in _USERS.values() if u.get("email") == email), None)


async def _db_get_user_by_provider(provider: str, subject: str):
    return next(
        (u for u in _USERS.values()
         if u.get("provider") == provider and u.get("provider_subject") == subject),
        None,
    )


async def _db_create_user(email, password_hash, provider, provider_subject, display_name) -> int:
    uid = _next_id()
    _USERS[uid] = {
        "id": uid,
        "email": email,
        "password_hash": password_hash,
        "provider": provider,
        "provider_subject": provider_subject,
        "display_name": display_name,
    }
    return uid


async def _db_create_session(user_id: int, refresh_token_hash: str, expires_at) -> int:
    sid = _next_id()
    _SESSIONS[refresh_token_hash] = {
        "id": sid, "user_id": user_id, "expires_at": expires_at, "revoked_at": None,
    }
    return sid


async def _db_get_session_by_hash(token_hash: str):
    return _SESSIONS.get(token_hash)


async def _db_revoke_session_by_token_hash(token_hash: str) -> None:
    if token_hash in _SESSIONS:
        _SESSIONS[token_hash]["revoked_at"] = datetime.now(timezone.utc).isoformat()


# Patch the module-level DB stubs.
auth._db_get_user_by_email = _db_get_user_by_email          # type: ignore[attr-defined]
auth._db_get_user_by_provider = _db_get_user_by_provider    # type: ignore[attr-defined]
auth._db_create_user = _db_create_user                      # type: ignore[attr-defined]
auth._db_create_session = _db_create_session                # type: ignore[attr-defined]
auth._db_get_session_by_hash = _db_get_session_by_hash      # type: ignore[attr-defined]
auth._db_revoke_session_by_token_hash = _db_revoke_session_by_token_hash  # type: ignore[attr-defined]


@pytest.fixture(autouse=True)
def reset_state():
    _reset_state()
    yield
    _reset_state()


@pytest.fixture
def app() -> FastAPI:
    app = FastAPI()
    app.include_router(auth.router)
    return app


@pytest.fixture
def client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(app=app, base_url="http://test")


# ---------------------------------------------------------------------------
# email/password
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_register_login_me_roundtrip(client: httpx.AsyncClient):
    """register → login → access token decodes → /me accepts it."""
    reg = await client.post(
        "/auth/register", json={"email": "user@example.com", "password": "securepass1"}
    )
    assert reg.status_code == 201
    body = reg.json()
    assert "access_token" in body and "refresh_token" in body
    assert body["expires_in"] > 0

    login = await client.post(
        "/auth/login", json={"email": "user@example.com", "password": "securepass1"}
    )
    assert login.status_code == 200
    access = login.json()["access_token"]
    payload = auth.decode_token(access)
    assert payload["type"] == "access"

    me = await client.get("/auth/me", headers={"Authorization": f"Bearer {access}"})
    assert me.status_code == 200


@pytest.mark.asyncio
async def test_login_rejects_bad_password(client: httpx.AsyncClient):
    await client.post("/auth/register", json={"email": "u@example.com", "password": "correctpass"})
    resp = await client.post("/auth/login", json={"email": "u@example.com", "password": "wrongpass"})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_register_rejects_duplicate_email(client: httpx.AsyncClient):
    await client.post("/auth/register", json={"email": "dup@example.com", "password": "pass1234"})
    resp = await client.post("/auth/register", json={"email": "dup@example.com", "password": "pass5678"})
    assert resp.status_code == 409


# ---------------------------------------------------------------------------
# refresh-token rotation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_refresh_rotates_and_revokes_old(client: httpx.AsyncClient):
    """A successful refresh issues a new pair and revokes the presented token."""
    reg = await client.post(
        "/auth/register", json={"email": "rot@example.com", "password": "securepass1"}
    )
    old_refresh = reg.json()["refresh_token"]

    refreshed = await client.post("/auth/refresh", json={"refresh_token": old_refresh})
    assert refreshed.status_code == 200
    new_refresh = refreshed.json()["refresh_token"]
    assert new_refresh != old_refresh

    # The old refresh token is now revoked → reuse is rejected.
    reused = await client.post("/auth/refresh", json={"refresh_token": old_refresh})
    assert reused.status_code == 401


# ---------------------------------------------------------------------------
# Apple — provider verification patched
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_apple_login_issues_session(client: httpx.AsyncClient, monkeypatch):
    """A verified Apple token mints an app session and creates the user once."""
    async def fake_verify(id_token, **kwargs):
        assert kwargs["audience"] == auth.APPLE_CLIENT_ID
        assert kwargs["expected_nonce"] == "raw-nonce-123"
        return {"sub": "apple-subject-001", "email": "apple-user@example.com"}

    monkeypatch.setattr(auth, "_verify_provider_id_token", fake_verify)

    first = await client.post(
        "/auth/apple",
        json={"identity_token": "stub", "nonce": "raw-nonce-123", "full_name": "Test User"},
    )
    assert first.status_code == 200
    uid = first.json()["user_id"]

    # Second sign-in with the same subject reuses the user, no duplicate row.
    second = await client.post(
        "/auth/apple", json={"identity_token": "stub", "nonce": "raw-nonce-123"}
    )
    assert second.status_code == 200
    assert second.json()["user_id"] == uid
    assert len([u for u in _USERS.values() if u["provider"] == "apple"]) == 1


# ---------------------------------------------------------------------------
# Google — provider verification patched
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_google_login_issues_session(client: httpx.AsyncClient, monkeypatch):
    async def fake_verify(id_token, **kwargs):
        assert kwargs["audience"] == auth.GOOGLE_CLIENT_ID
        return {"sub": "google-subject-009", "email": "g@example.com",
                "email_verified": True, "name": "G User"}

    monkeypatch.setattr(auth, "_verify_provider_id_token", fake_verify)

    resp = await client.post("/auth/google", json={"id_token": "stub", "nonce": "n"})
    assert resp.status_code == 200
    assert resp.json()["user_id"] > 0


@pytest.mark.asyncio
async def test_google_login_rejects_unverified_email(client: httpx.AsyncClient, monkeypatch):
    async def fake_verify(id_token, **kwargs):
        return {"sub": "g-bad", "email": "bad@example.com", "email_verified": False}

    monkeypatch.setattr(auth, "_verify_provider_id_token", fake_verify)

    resp = await client.post("/auth/google", json={"id_token": "stub"})
    assert resp.status_code == 401
