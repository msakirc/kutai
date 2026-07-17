"""Prove the KutAI registry block reproduces the hardcoded kutai_wrapper.py
GuardConfig 1:1 (config-equivalence only; runtime behavior is gated by the
characterization suite)."""
import os
from pathlib import Path
from yasar_usta.registry import load_registry

ROOT = Path(__file__).resolve().parents[3]  # repo root


def test_kutai_registry_matches_legacy_config():
    os.environ.setdefault("YASAR_USTA_BOT_TOKEN", "tok")
    os.environ.setdefault("TELEGRAM_ADMIN_CHAT_ID", "42")
    hub, projects = load_registry(ROOT / "registry.yaml", project_root=str(ROOT))
    kutai = next(p for p in projects if p.id == "kutai")
    t = kutai.targets[0]
    assert t.command[-1].endswith("run.py")
    # Path compare — loader normalizes separators (Windows-safe, finding #7)
    assert Path(t.cwd) == ROOT
    assert t.restart_exit_code == 42
    assert Path(t.heartbeat_file) == ROOT / "logs" / "orchestrator.heartbeat"
    assert Path(t.log_file) == ROOT / "logs" / "orchestrator.jsonl"
    assert Path(t.env["NERD_HERD_PROJECT_ROOT"]) == ROOT
    names = {sc.name for sc in t.sidecars}
    assert names == {"yazbunu", "nerd_herd"}
    assert kutai.hook_module == "yasar_usta.projects.kutai.hooks"
