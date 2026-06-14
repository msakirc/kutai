"""Tests for the single fail-loud llama-server endpoint resolver.

The 2026-06-14 wrong-port orphan incident was caused by six independent
`os.environ.get("LLAMA_SERVER_PORT", "8080")` sites silently defaulting to
8080 when a process was spawned without the env var. The resolver removes
the silent default: it loads .env on demand, and raises if the port is
still unresolved rather than guessing.
"""
import pytest

from src.infra import llama_endpoint


def test_resolve_port_reads_env(monkeypatch):
    monkeypatch.setenv("LLAMA_SERVER_PORT", "8081")
    assert llama_endpoint.resolve_llama_port() == 8081


def test_resolve_port_raises_when_unset(monkeypatch):
    monkeypatch.delenv("LLAMA_SERVER_PORT", raising=False)
    # Stub .env discovery so the test is deterministic regardless of cwd.
    monkeypatch.setattr(llama_endpoint, "_ensure_dotenv_loaded", lambda: None)
    with pytest.raises(RuntimeError, match="LLAMA_SERVER_PORT"):
        llama_endpoint.resolve_llama_port()


def test_resolve_port_loads_dotenv_when_unset(monkeypatch):
    """If the env is missing, the resolver loads .env before giving up."""
    monkeypatch.delenv("LLAMA_SERVER_PORT", raising=False)

    def _fake_load():
        import os
        os.environ["LLAMA_SERVER_PORT"] = "8099"

    monkeypatch.setattr(llama_endpoint, "_ensure_dotenv_loaded", _fake_load)
    assert llama_endpoint.resolve_llama_port() == 8099


def test_resolve_port_rejects_garbage(monkeypatch):
    monkeypatch.setenv("LLAMA_SERVER_PORT", "not-a-port")
    monkeypatch.setattr(llama_endpoint, "_ensure_dotenv_loaded", lambda: None)
    with pytest.raises(RuntimeError, match="LLAMA_SERVER_PORT"):
        llama_endpoint.resolve_llama_port()


def test_resolve_url_builds_from_port(monkeypatch):
    monkeypatch.setenv("LLAMA_SERVER_PORT", "8081")
    assert llama_endpoint.resolve_llama_url() == "http://127.0.0.1:8081"


def test_resolve_url_appends_path(monkeypatch):
    monkeypatch.setenv("LLAMA_SERVER_PORT", "8081")
    assert llama_endpoint.resolve_llama_url("/health") == "http://127.0.0.1:8081/health"
