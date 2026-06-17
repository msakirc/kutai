"""Tests for the single fail-loud llama-server endpoint resolver.

The 2026-06-14 wrong-port orphan incident was caused by six independent
`os.environ.get("LLAMA_SERVER_PORT", "8080")` sites silently defaulting to
8080 when a process was spawned without the env var. The resolver removes
the silent default: it reads .env, and raises if the port is still
unresolved rather than guessing.

2026-06-17: a SECOND failure mode — a long-lived wrapper launched before a
`.env` edit kept a STALE `LLAMA_SERVER_PORT` in its process env; orchestrator
restarts inherited it, and the resolver (env-wins, .env only consulted when
unset) never picked up the corrected `.env` value, so llama-server kept
binding the old port (Expo:8081 collision). The resolver now treats `.env` as
the deployment source of truth: on a mismatch between the inherited env and
`.env`, it prefers `.env` and warns loudly.
"""
import pytest

from src.infra import llama_endpoint


def test_resolve_port_reads_env(monkeypatch):
    monkeypatch.setenv("LLAMA_SERVER_PORT", "8081")
    monkeypatch.setattr(llama_endpoint, "_dotenv_port", lambda: None)
    assert llama_endpoint.resolve_llama_port() == 8081


def test_resolve_port_raises_when_unset(monkeypatch):
    monkeypatch.delenv("LLAMA_SERVER_PORT", raising=False)
    monkeypatch.setattr(llama_endpoint, "_dotenv_port", lambda: None)
    with pytest.raises(RuntimeError, match="LLAMA_SERVER_PORT"):
        llama_endpoint.resolve_llama_port()


def test_resolve_port_uses_dotenv_when_env_unset(monkeypatch):
    """If the env is missing, the resolver reads .env before giving up."""
    monkeypatch.delenv("LLAMA_SERVER_PORT", raising=False)
    monkeypatch.setattr(llama_endpoint, "_dotenv_port", lambda: "8099")
    assert llama_endpoint.resolve_llama_port() == 8099


def test_resolve_port_rejects_garbage(monkeypatch):
    monkeypatch.setenv("LLAMA_SERVER_PORT", "not-a-port")
    monkeypatch.setattr(llama_endpoint, "_dotenv_port", lambda: None)
    with pytest.raises(RuntimeError, match="LLAMA_SERVER_PORT"):
        llama_endpoint.resolve_llama_port()


def test_dotenv_wins_over_stale_inherited_env(monkeypatch):
    """The 2026-06-17 trap: a stale inherited env disagrees with .env. The
    resolver must prefer .env (deployment truth), not the stale value."""
    monkeypatch.setenv("LLAMA_SERVER_PORT", "8081")          # stale inherited
    monkeypatch.setattr(llama_endpoint, "_dotenv_port", lambda: "8090")  # .env truth
    assert llama_endpoint.resolve_llama_port() == 8090


def test_env_used_when_no_dotenv_value(monkeypatch):
    """No .env value present → fall back to the process env."""
    monkeypatch.setenv("LLAMA_SERVER_PORT", "8081")
    monkeypatch.setattr(llama_endpoint, "_dotenv_port", lambda: None)
    assert llama_endpoint.resolve_llama_port() == 8081


def test_agreement_returns_value(monkeypatch):
    monkeypatch.setenv("LLAMA_SERVER_PORT", "8090")
    monkeypatch.setattr(llama_endpoint, "_dotenv_port", lambda: "8090")
    assert llama_endpoint.resolve_llama_port() == 8090


def test_resolve_url_builds_from_port(monkeypatch):
    monkeypatch.setenv("LLAMA_SERVER_PORT", "8081")
    monkeypatch.setattr(llama_endpoint, "_dotenv_port", lambda: None)
    assert llama_endpoint.resolve_llama_url() == "http://127.0.0.1:8081"


def test_resolve_url_appends_path(monkeypatch):
    monkeypatch.setenv("LLAMA_SERVER_PORT", "8081")
    monkeypatch.setattr(llama_endpoint, "_dotenv_port", lambda: None)
    assert llama_endpoint.resolve_llama_url("/health") == "http://127.0.0.1:8081/health"
