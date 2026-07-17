import pytest
from pathlib import Path
from yasar_usta.registry import load_registry


def _write(tmp_path, body):
    p = tmp_path / "registry.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def test_load_registry_parses_hub_and_projects(tmp_path):
    reg = _write(tmp_path, """
hub:
  telegram_token_env: MY_TOKEN
  telegram_chat_id_env: MY_CHAT
  log_dir: "${project_root}/logs"
projects:
  demo:
    name: Demo
    targets:
      - id: web
        command: ["python", "-m", "http.server"]
        cwd: "${project_root}"
        log_dir: "${project_root}/logs"
        heartbeat_file: "${project_root}/logs/web.heartbeat"
""")
    import os
    os.environ["MY_TOKEN"] = "tok"
    os.environ["MY_CHAT"] = "99"
    hub, projects = load_registry(reg, project_root=str(tmp_path))
    assert hub.telegram_token == "tok"
    assert hub.telegram_chat_id == "99"
    assert len(projects) == 1
    p = projects[0]
    assert p.id == "demo"
    assert p.name == "Demo"
    assert len(p.targets) == 1
    t = p.targets[0]
    assert t.command == ["python", "-m", "http.server"]
    # Compare as Path (loader normalizes separators; Windows-safe)
    assert Path(t.cwd) == tmp_path
    assert Path(t.log_dir) == tmp_path / "logs"
    assert Path(t.heartbeat_file) == tmp_path / "logs" / "web.heartbeat"


def test_load_registry_fails_fast_on_missing_targets(tmp_path):
    reg = _write(tmp_path, """
hub: {telegram_token_env: T, telegram_chat_id_env: C, log_dir: "l"}
projects:
  bad: {name: Bad}
""")
    with pytest.raises(ValueError, match="targets"):
        load_registry(reg, project_root=str(tmp_path))


def test_sidecar_unknown_key_ignored(tmp_path):
    reg = _write(tmp_path, """
hub: {telegram_token_env: T, telegram_chat_id_env: C, log_dir: "l"}
projects:
  demo:
    name: Demo
    targets:
      - id: web
        command: ["python"]
        sidecars:
          - name: sc1
            command: ["python", "-m", "x"]
            bogus_key: 123
""")
    hub, projects = load_registry(reg, project_root=str(tmp_path))
    sc = projects[0].targets[0].sidecars[0]
    assert sc.name == "sc1"
    assert sc.command == ["python", "-m", "x"]
