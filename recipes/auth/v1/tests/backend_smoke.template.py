"""Auth recipe — backend smoke tests.

Pytest scaffold using httpx.AsyncClient against an in-process FastAPI app.
All tests are self-contained: a fresh SQLite DB is created per test session
via the `app` fixture.

RECIPE_PARAM:BCRYPT_COST=4  (fast cost for tests — override in conftest)

Run:
    pytest tests/auth/ -x -q
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from typing import AsyncGenerator

import pytest
import pytest_asyncio
import httpx
from fastapi import FastAPI

# ---------------------------------------------------------------------------
# Test app setup
# ---------------------------------------------------------------------------
# The instantiated recipe replaces these stubs with the real DB layer.
# For smoke tests we use an in-memory dict to avoid aiosqlite/asyncpg deps.

_USERS: dict = {}        # email -> {id, email, password_hash, email_verified}
_RESET_TOKENS: dict = {}  # token_hash -> {user_id, expires_at, used}
_VERIFY_TOKENS: dict = {}  # token_hash -> {user_id, expires_at, used}
_SESSIONS: dict = {}     # token_hash -> {user_id, expires_at, revoked}
_EMAIL_LOG: list = []    # for asserting email was sent
_NEXT_ID = [1]


def _next_id() -> int:
    uid = _NEXT_ID[0]
    _NEXT_ID[0] += 1
    return uid


def _reset_state():
    _USERS.clear()
    _RESET_TOKENS.clear()
    _VERIFY_TOKENS.clear()
    _SESSIONS.clear()
    _EMAIL_LOG.clear()
    _NEXT_ID[0] = 1


# ---------------------------------------------------------------------------
# Import the template module and patch its DB stubs
# ---------------------------------------------------------------------------

# During instantiation, backend.template.py becomes auth/routes.py.
# In tests we import it directly. JWT_SECRET must be set before import.
os.environ.setdefault("AUTH_JWT_SECRET", "test-secret-do-not-use-in-prod")  # noqa: S105

from recipes.auth.v1 import backend_template as auth  # type: ignore[import]  # noqa: E402


async def _db_create_user(email: str, password_hash: str) -> int:
    if email in {u["email"] for u in _USERS.values()}:
        raise ValueError("duplicate email")
    uid = _next_id()
    _USERS[uid] = {"id": uid, "email": email, "password_hash": password_hash, "email_verified": 0}
    return uid


async def _db_get_user_by_email(email: str):
    return next((u for u in _USERS.values() if u["email"] == email), None)


async def _db_get_user_by_id(user_id: int):
    return _USERS.get(user_id)


async def _db_create_session(user_id: int, refresh_token_hash: str, expires_at) -> int:
    sid = _next_id()
    _SESSIONS[refresh_token_hash] = {"id": sid, "user_id": user_id, "expires_at": expires_at, "revoked": False}
    return sid


async def _db_revoke_session_by_token_hash(token_hash: str) -> None:
    if token_hash in _SESSIONS:
        _SESSIONS[token_hash]["revoked"] = True


async def _db_create_reset_token(user_id: int, token_hash: str, expires_at) -> None:
    _RESET_TOKENS[token_hash] = {"user_id": user_id, "expires_at": expires_at, "used": False}


async def _db_consume_reset_token(token_hash: str):
    entry = _RESET_TOKENS.get(token_hash)
    if not entry or entry["used"]:
        return None
    if entry["expires_at"] < datetime.now(timezone.utc):
        return None
    entry["used"] = True
    return entry["user_id"]


async def _db_update_password(user_id: int, password_hash: str) -> None:
    if user_id in _USERS:
        _USERS[user_id]["password_hash"] = password_hash


async def _db_create_verify_token(user_id: int, token_hash: str, expires_at) -> None:
    _VERIFY_TOKENS[token_hash] = {"user_id": user_id, "expires_at": expires_at, "used": False}


async def _db_consume_verify_token(token_hash: str):
    entry = _VERIFY_TOKENS.get(token_hash)
    if not entry or entry["used"]:
        return None
    if entry["expires_at"] < datetime.now(timezone.utc):
        return None
    entry["used"] = True
    return entry["user_id"]


async def _db_set_email_verified(user_id: int) -> None:
    if user_id in _USERS:
        _USERS[user_id]["email_verified"] = 1


async def _send_email(to: str, subject: str, body: str) -> None:
    _EMAIL_LOG.append({"to": to, "subject": subject, "body": body})


# Patch the module-level stubs
auth._db_create_user = _db_create_user  # type: ignore[attr-defined]
auth._db_get_user_by_email = _db_get_user_by_email  # type: ignore[attr-defined]
auth._db_get_user_by_id = _db_get_user_by_id  # type: ignore[attr-defined]
auth._db_create_session = _db_create_session  # type: ignore[attr-defined]
auth._db_revoke_session_by_token_hash = _db_revoke_session_by_token_hash  # type: ignore[attr-defined]
auth._db_create_reset_token = _db_create_reset_token  # type: ignore[attr-defined]
auth._db_consume_reset_token = _db_consume_reset_token  # type: ignore[attr-defined]
auth._db_update_password = _db_update_password  # type: ignore[attr-defined]
auth._db_create_verify_token = _db_create_verify_token  # type: ignore[attr-defined]
auth._db_consume_verify_token = _db_consume_verify_token  # type: ignore[attr-defined]
auth._db_set_email_verified = _db_set_email_verified  # type: ignore[attr-defined]
auth._send_email = _send_email  # type: ignore[attr-defined]


@pytest.fixture(autouse=True)
def reset_state():
    """Reset all in-memory state before each test."""
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
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_register_and_login_roundtrip(client: httpx.AsyncClient):
    """POST /register then POST /login → access token returned."""
    reg = await client.post("/auth/register", json={"email": "user@example.com", "password": "securepass1"})
    assert reg.status_code == 201
    assert reg.json()["email"] == "user@example.com"

    login = await client.post("/auth/login", json={"email": "user@example.com", "password": "securepass1"})
    assert login.status_code == 200
    data = login.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    # Verify token is decodable
    payload = auth.decode_token(data["access_token"])
    assert payload["type"] == "access"
    assert int(payload["sub"]) == data["user_id"]


@pytest.mark.asyncio
async def test_login_rejects_bad_password(client: httpx.AsyncClient):
    """POST /login with wrong password → 401."""
    await client.post("/auth/register", json={"email": "user@example.com", "password": "correctpass"})
    resp = await client.post("/auth/login", json={"email": "user@example.com", "password": "wrongpass"})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_register_rejects_duplicate_email(client: httpx.AsyncClient):
    """Second registration with same email → 409."""
    await client.post("/auth/register", json={"email": "dup@example.com", "password": "pass1234"})
    resp = await client.post("/auth/register", json={"email": "dup@example.com", "password": "pass5678"})
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_password_reset_flow(client: httpx.AsyncClient):
    """Request reset → token in email log → confirm → login with new password."""
    await client.post("/auth/register", json={"email": "reset@example.com", "password": "oldpass1"})

    req = await client.post("/auth/password/reset/request", json={"email": "reset@example.com"})
    assert req.status_code == 200
    # Email was "sent"
    assert len(_EMAIL_LOG) >= 1
    email_body = _EMAIL_LOG[-1]["body"]
    # Extract raw token from email body (format: ...token=<token>)
    raw_token = email_body.split("token=")[-1].strip()

    confirm = await client.post(
        "/auth/password/reset/confirm",
        json={"token": raw_token, "new_password": "newpass99"},
    )
    assert confirm.status_code == 200

    # Old password should now fail
    old_login = await client.post("/auth/login", json={"email": "reset@example.com", "password": "oldpass1"})
    assert old_login.status_code == 401

    # New password should work
    new_login = await client.post("/auth/login", json={"email": "reset@example.com", "password": "newpass99"})
    assert new_login.status_code == 200


@pytest.mark.asyncio
async def test_email_verify_flow(client: httpx.AsyncClient):
    """Register → email log contains verify token → verify → email_verified=True."""
    await client.post("/auth/register", json={"email": "verify@example.com", "password": "pass1234"})

    assert len(_EMAIL_LOG) >= 1
    verify_body = _EMAIL_LOG[0]["body"]
    raw_token = verify_body.split("token=")[-1].strip()

    resp = await client.post("/auth/email/verify", json={"token": raw_token})
    assert resp.status_code == 200

    user = await _db_get_user_by_email("verify@example.com")
    assert user is not None
    assert user["email_verified"] == 1


@pytest.mark.asyncio
async def test_logout_clears_session(client: httpx.AsyncClient):
    """Login → get access token → logout → session marked revoked."""
    await client.post("/auth/register", json={"email": "logout@example.com", "password": "pass1234"})
    login = await client.post("/auth/login", json={"email": "logout@example.com", "password": "pass1234"})
    access_token = login.json()["access_token"]

    logout = await client.post(
        "/auth/logout",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert logout.status_code == 200
    # All sessions for that user should now be revoked
    active = [s for s in _SESSIONS.values() if not s["revoked"]]
    assert len(active) == 0


@pytest.mark.asyncio
async def test_password_reset_token_single_use(client: httpx.AsyncClient):
    """Consuming a reset token twice → second attempt returns 400."""
    await client.post("/auth/register", json={"email": "once@example.com", "password": "oldpass1"})
    await client.post("/auth/password/reset/request", json={"email": "once@example.com"})
    raw_token = _EMAIL_LOG[-1]["body"].split("token=")[-1].strip()

    first = await client.post(
        "/auth/password/reset/confirm",
        json={"token": raw_token, "new_password": "newpass99"},
    )
    assert first.status_code == 200

    second = await client.post(
        "/auth/password/reset/confirm",
        json={"token": raw_token, "new_password": "anotherpass"},
    )
    assert second.status_code == 400
