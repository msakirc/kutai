"""DaLLaMa.reconcile_strays() must clear wrong-port orphans on demand.

Used by the orchestrator/wrapper at boot so a stray llama-server on the
wrong port is killed even when no local model is loaded this session
(the lazy `kill_orphans()` inside `start()` never ran in the 2026-06-14
incident).
"""
from dallama import DaLLaMa, DaLLaMaConfig


def test_reconcile_strays_uses_config_port(monkeypatch):
    d = DaLLaMa(DaLLaMaConfig(port=8081))
    seen = {}

    def _fake(port):
        seen["port"] = port
        return 2

    monkeypatch.setattr(d._platform, "kill_stray_servers", _fake)
    assert d.reconcile_strays() == 2
    assert seen["port"] == 8081
