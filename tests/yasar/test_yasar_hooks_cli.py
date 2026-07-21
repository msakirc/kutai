import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HOOK = ROOT / "yasar_hooks.py"


def test_on_exit_clean_restart_is_noop_rc0():
    # exit_code=42 => clean restart => _kill_orphan_processes returns early (no kill)
    ctx = json.dumps({"project_id": "kutai", "script_paths": [], "exit_code": 42})
    r = subprocess.run([sys.executable, str(HOOK), "on_exit", "--context", ctx],
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 0


def test_pre_boot_parses_context_and_runs():
    # FAKE non-matching path (cannot match any live process) + LLAMA_SERVER_PORT stripped
    ctx = json.dumps({"project_id": "kutai", "script_paths": ["C:/__yasar_fake_test__/app/run.py"]})
    env = {k: v for k, v in os.environ.items() if k != "LLAMA_SERVER_PORT"}
    r = subprocess.run([sys.executable, str(HOOK), "pre_boot", "--context", ctx],
                       capture_output=True, text=True, timeout=30, env=env)
    assert r.returncode == 0
