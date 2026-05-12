"""Z8 T3C — webhook_secret env-ref resolution via IntegrationRegistry."""
from __future__ import annotations

import pytest

from src.app.webhook_signing import _load_webhook_secret, _resolve_env_ref
from src.integrations.registry import IntegrationRegistry


def test_resolve_env_ref_plain_string():
    assert _resolve_env_ref("plain") == "plain"


def test_resolve_env_ref_missing_env(monkeypatch):
    monkeypatch.delenv("ZZ_DOES_NOT_EXIST", raising=False)
    assert _resolve_env_ref("${ZZ_DOES_NOT_EXIST}") is None


def test_resolve_env_ref_present(monkeypatch):
    monkeypatch.setenv("ZZ_PRESENT", "hello")
    assert _resolve_env_ref("${ZZ_PRESENT}") == "hello"


@pytest.mark.asyncio
async def test_sentry_secret_resolves_from_env(monkeypatch):
    monkeypatch.setenv("SENTRY_WEBHOOK_SECRET", "shh-sentry")
    # Force a fresh registry so auto_discover re-reads configs.
    secret = await _load_webhook_secret("sentry")
    assert secret == "shh-sentry"


@pytest.mark.asyncio
async def test_stripe_secret_resolves_from_env(monkeypatch):
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "shh-stripe")
    secret = await _load_webhook_secret("stripe")
    assert secret == "shh-stripe"


@pytest.mark.asyncio
async def test_github_secret_resolves_from_env(monkeypatch):
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "shh-github")
    secret = await _load_webhook_secret("github")
    assert secret == "shh-github"


@pytest.mark.asyncio
async def test_betterstack_secret_resolves_from_env(monkeypatch):
    monkeypatch.setenv("BETTERSTACK_WEBHOOK_SECRET", "shh-bs")
    secret = await _load_webhook_secret("betterstack")
    assert secret == "shh-bs"


@pytest.mark.asyncio
async def test_twilio_secret_resolves_from_env(monkeypatch):
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "shh-twilio")
    secret = await _load_webhook_secret("twilio")
    assert secret == "shh-twilio"


@pytest.mark.asyncio
async def test_unknown_integration_returns_none():
    secret = await _load_webhook_secret("definitely-not-a-vendor")
    assert secret is None


def test_betterstack_config_loaded():
    """Sanity: registry auto-discovery picked up the new betterstack.json."""
    reg = IntegrationRegistry()
    integ = reg.get("betterstack")
    assert integ is not None
    assert integ._config.get("webhook_secret") == "${BETTERSTACK_WEBHOOK_SECRET}"


def test_twilio_config_loaded():
    reg = IntegrationRegistry()
    integ = reg.get("twilio")
    assert integ is not None
    assert integ._config.get("webhook_secret") == "${TWILIO_AUTH_TOKEN}"
