"""Z8 T3B — per-vendor signature verification tests."""
from __future__ import annotations

import base64
import hashlib
import hmac
import time

import pytest

from src.app.webhook_signing import verify_signature

SECRET = "test-secret"
RAW = b'{"event_id":"abc","data":{}}'


def _sentry_sig(raw: bytes, secret: str) -> str:
    return hmac.new(secret.encode(), raw, hashlib.sha256).hexdigest()


def _betterstack_sig(raw: bytes, secret: str) -> str:
    return hmac.new(secret.encode(), raw, hashlib.sha256).hexdigest()


def _github_sig(raw: bytes, secret: str) -> str:
    return "sha256=" + hmac.new(
        secret.encode(), raw, hashlib.sha256
    ).hexdigest()


def _stripe_sig(raw: bytes, secret: str, ts: int | None = None) -> str:
    if ts is None:
        ts = int(time.time())
    signed = (str(ts) + ".").encode() + raw
    v1 = hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()
    return f"t={ts},v1={v1}"


def _twilio_sig(raw: bytes, secret: str, url: str) -> str:
    digest = hmac.new(
        secret.encode(), (url + raw.decode()).encode(), hashlib.sha1
    ).digest()
    return base64.b64encode(digest).decode()


@pytest.mark.asyncio
async def test_sentry_valid_and_tampered():
    valid = _sentry_sig(RAW, SECRET)
    assert await verify_signature(
        "sentry", RAW, {"sentry-hook-signature": valid}, secret=SECRET,
    ) is True
    assert await verify_signature(
        "sentry", RAW, {"sentry-hook-signature": "deadbeef"}, secret=SECRET,
    ) is False
    # Missing header rejects.
    assert await verify_signature(
        "sentry", RAW, {}, secret=SECRET,
    ) is False


@pytest.mark.asyncio
async def test_github_valid_invalid_and_missing_prefix():
    valid = _github_sig(RAW, SECRET)
    assert await verify_signature(
        "github", RAW, {"x-hub-signature-256": valid}, secret=SECRET,
    ) is True
    # No sha256= prefix.
    assert await verify_signature(
        "github", RAW, {"x-hub-signature-256": valid[7:]}, secret=SECRET,
    ) is False
    # Tampered.
    assert await verify_signature(
        "github", RAW, {"x-hub-signature-256": "sha256=00"}, secret=SECRET,
    ) is False


@pytest.mark.asyncio
async def test_stripe_valid_and_stale():
    fresh = _stripe_sig(RAW, SECRET)
    assert await verify_signature(
        "stripe", RAW, {"stripe-signature": fresh}, secret=SECRET,
    ) is True
    stale = _stripe_sig(RAW, SECRET, ts=int(time.time()) - 1000)
    assert await verify_signature(
        "stripe", RAW, {"stripe-signature": stale}, secret=SECRET,
    ) is False
    # Wrong v1.
    bad = "t=" + str(int(time.time())) + ",v1=ffff"
    assert await verify_signature(
        "stripe", RAW, {"stripe-signature": bad}, secret=SECRET,
    ) is False


@pytest.mark.asyncio
async def test_betterstack_valid_and_tampered():
    valid = _betterstack_sig(RAW, SECRET)
    assert await verify_signature(
        "betterstack", RAW, {"x-betterstack-signature": valid}, secret=SECRET,
    ) is True
    assert await verify_signature(
        "betterstack", RAW, {"x-betterstack-signature": "xx"}, secret=SECRET,
    ) is False


@pytest.mark.asyncio
async def test_twilio_valid_and_missing_url():
    url = "https://example.com/webhook/twilio"
    valid = _twilio_sig(RAW, SECRET, url)
    assert await verify_signature(
        "twilio", RAW,
        {"x-twilio-signature": valid, "x-twilio-url": url},
        secret=SECRET,
    ) is True
    # Missing URL header (listener must inject it).
    assert await verify_signature(
        "twilio", RAW, {"x-twilio-signature": valid}, secret=SECRET,
    ) is False


@pytest.mark.asyncio
async def test_unknown_integration_rejects():
    assert await verify_signature(
        "definitely-not-a-vendor", RAW, {"x-sig": "anything"}, secret=SECRET,
    ) is False


@pytest.mark.asyncio
async def test_missing_secret_rejects(monkeypatch):
    # secret=None and registry returns no integration → reject.
    from src.app import webhook_signing as ws

    async def _none(_id):
        return None

    monkeypatch.setattr(ws, "_load_webhook_secret", _none)
    assert await verify_signature(
        "sentry", RAW, {"sentry-hook-signature": "x"},
    ) is False
