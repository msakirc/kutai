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


def test_writer_path_equals_state_dir_join_exact(monkeypatch):
    """COUPLING GUARD (split-brain regression). The orchestrator WRITES its
    heartbeat here; the Yaşar Usta hub READS it from the registry's
    ``${state_dir}/orchestrator.heartbeat``. These MUST be the identical path
    or the hub false-kills a healthy orchestrator. If this filename changes,
    the hub counterpart in
    yasar_usta/tests/test_registry_statedir.py::test_reader_writer_filenames_match
    must change in lockstep."""
    import os
    sd = r"C:\some\state\kutai"
    monkeypatch.setenv("YASAR_USTA_STATE_DIR", sd)
    from src.app.hb_paths import heartbeat_paths, state_snapshot_path
    assert heartbeat_paths()[0] == os.path.join(sd, "orchestrator.heartbeat")
    assert state_snapshot_path() == os.path.join(sd, "orchestrator.state.json")


def test_call_sites_use_helper_not_hardcoded_literal():
    """REGRESSION GUARD. run.py + orchestrator.py must derive heartbeat/state
    paths from hb_paths (env-aware), never a hardcoded 'logs/...' literal — a
    revert would silently re-introduce the split-brain false-kill while every
    other test stayed green (the exact gap the reviewer flagged)."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[2]  # kutay repo root
    for rel in ("src/app/run.py", "src/core/orchestrator.py"):
        text = (root / rel).read_text(encoding="utf-8")
        assert "logs/orchestrator.heartbeat" not in text, \
            f"{rel} hardcodes the heartbeat path — must use hb_paths.heartbeat_paths()"
        assert "logs/orchestrator.state.json" not in text, \
            f"{rel} hardcodes the state path — must use hb_paths.state_snapshot_path()"
        assert "heartbeat_paths" in text, \
            f"{rel} must derive the heartbeat path via hb_paths.heartbeat_paths()"
