def test_heartbeat_paths_from_state_dir(monkeypatch):
    monkeypatch.setenv("YASAR_USTA_STATE_DIR", r"C:\state\kutai")
    from src.app.hb_paths import heartbeat_paths
    paths = heartbeat_paths()
    assert paths[0].replace("\\", "/").endswith("state/kutai/orchestrator.heartbeat")
    assert paths[1].replace("\\", "/").endswith("state/kutai/heartbeat")


def test_heartbeat_paths_fallback_when_env_absent(monkeypatch):
    monkeypatch.delenv("YASAR_USTA_STATE_DIR", raising=False)
    from src.app.hb_paths import heartbeat_paths
    paths = heartbeat_paths()
    assert paths[0].replace("\\", "/").endswith("logs/orchestrator.heartbeat")


def test_state_snapshot_path_from_state_dir(monkeypatch):
    monkeypatch.setenv("YASAR_USTA_STATE_DIR", r"C:\state\kutai")
    from src.app.hb_paths import state_snapshot_path
    assert state_snapshot_path().replace("\\", "/").endswith("state/kutai/orchestrator.state.json")


def test_state_snapshot_path_fallback(monkeypatch):
    monkeypatch.delenv("YASAR_USTA_STATE_DIR", raising=False)
    from src.app.hb_paths import state_snapshot_path
    assert state_snapshot_path().replace("\\", "/").endswith("logs/orchestrator.state.json")
