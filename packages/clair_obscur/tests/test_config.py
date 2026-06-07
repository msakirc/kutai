# packages/clair_obscur/tests/test_config.py
import pytest
from clair_obscur.config import ClairObscurConfig, load_config


def test_default_backend_is_comfyui(monkeypatch):
    monkeypatch.delenv("CLAIR_OBSCUR_BACKEND", raising=False)
    monkeypatch.delenv("CLAIR_OBSCUR_URL", raising=False)
    cfg = load_config()
    assert cfg.backend == "comfyui"
    assert cfg.port == 8188
    assert cfg.base_url == "http://127.0.0.1:8188"


def test_env_selects_a1111(monkeypatch):
    monkeypatch.setenv("CLAIR_OBSCUR_BACKEND", "a1111")
    monkeypatch.setenv("CLAIR_OBSCUR_PORT", "7860")
    monkeypatch.delenv("CLAIR_OBSCUR_URL", raising=False)
    cfg = load_config()
    assert cfg.backend == "a1111"
    assert cfg.port == 7860
    assert cfg.base_url == "http://127.0.0.1:7860"


def test_unknown_backend_rejected(monkeypatch):
    monkeypatch.setenv("CLAIR_OBSCUR_BACKEND", "midjourney")
    with pytest.raises(ValueError):
        load_config()


def test_explicit_url_overrides_host_port(monkeypatch):
    monkeypatch.setenv("CLAIR_OBSCUR_URL", "http://192.168.1.7:9000")
    cfg = load_config()
    assert cfg.base_url == "http://192.168.1.7:9000"
