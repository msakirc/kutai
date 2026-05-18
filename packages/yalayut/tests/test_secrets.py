"""yalayut.secrets — fernet round-trip + env_status."""
import os

import pytest

from yalayut import secrets as ysec


@pytest.fixture(autouse=True)
def _fernet_key(monkeypatch):
    # A valid 32-byte url-safe base64 fernet key for tests.
    from cryptography.fernet import Fernet
    monkeypatch.setenv("YALAYUT_SECRET_KEY", Fernet.generate_key().decode())


@pytest.mark.asyncio
async def test_set_and_get_secret_round_trip(tmp_path, monkeypatch):
    rows = {}

    async def fake_upsert(key, blob):
        rows[key] = blob

    async def fake_fetch(key):
        return rows.get(key)

    monkeypatch.setattr(ysec, "_db_upsert_secret", fake_upsert)
    monkeypatch.setattr(ysec, "_db_fetch_secret", fake_fetch)

    await ysec.set_secret("OPENAQ_API_KEY", "super-secret-value")
    got = await ysec.get_secret("OPENAQ_API_KEY")
    assert got == "super-secret-value"
    # Stored blob is NOT plaintext.
    assert b"super-secret-value" not in rows["OPENAQ_API_KEY"]


@pytest.mark.asyncio
async def test_get_missing_secret_returns_none(monkeypatch):
    async def fake_fetch(key):
        return None
    monkeypatch.setattr(ysec, "_db_fetch_secret", fake_fetch)
    assert await ysec.get_secret("NOPE_KEY") is None


@pytest.mark.asyncio
async def test_env_status_ready_when_env_present(monkeypatch):
    monkeypatch.setenv("PRESENT_KEY", "x")
    monkeypatch.setattr(ysec, "get_secret", _async_none)
    status = await ysec.compute_env_status(["PRESENT_KEY"])
    assert status == "ready"


@pytest.mark.asyncio
async def test_env_status_ready_when_secret_present(monkeypatch):
    monkeypatch.delenv("VAULT_KEY", raising=False)

    async def fake_get_secret(key):
        return "from-vault" if key == "VAULT_KEY" else None

    monkeypatch.setattr(ysec, "get_secret", fake_get_secret)
    status = await ysec.compute_env_status(["VAULT_KEY"])
    assert status == "ready"


@pytest.mark.asyncio
async def test_env_status_missing(monkeypatch):
    monkeypatch.delenv("ABSENT_KEY", raising=False)
    monkeypatch.setattr(ysec, "get_secret", _async_none)
    status = await ysec.compute_env_status(["ABSENT_KEY"])
    assert status == "missing_ABSENT_KEY"


@pytest.mark.asyncio
async def test_env_status_no_keys_is_ready(monkeypatch):
    assert await ysec.compute_env_status([]) == "ready"
    assert await ysec.compute_env_status(None) == "ready"


async def _async_none(key):
    return None
