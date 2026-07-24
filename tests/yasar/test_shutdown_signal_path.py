"""R3 — shutdown.signal read-path migration (Dropbox logs/ → state_dir).

The Yaşar Usta hub WRITES the shutdown signal to ``${log_dir}/shutdown.signal``
(supervisor.py:_write_shutdown_signal); the migration flips the target's
registry ``log_dir`` from ``${project_root}/logs`` to ``${state_dir}/logs`` so
the file leaves Dropbox. The orchestrator READS it. Reader + writer must point
at the identical absolute path or ``/shutdown_hub`` silently stops working.

These tests pin the env-aware dual-read contract so the KutAI read-side can land
BEFORE the hub flips its registry (legacy CWD path kept as a transition
fallback → no ordering hazard)."""

import os


def test_shutdown_signal_paths_from_state_dir(monkeypatch):
    monkeypatch.setenv("YASAR_USTA_STATE_DIR", r"C:\state\kutai")
    from src.app.hb_paths import shutdown_signal_paths
    paths = shutdown_signal_paths()
    # new, authoritative path first: the hub's log_dir is ${state_dir}/logs, so
    # the signal lands one 'logs' subdir below state_dir (NOT the state_dir root
    # where the heartbeat sits).
    assert paths[0].replace("\\", "/").endswith("state/kutai/logs/shutdown.signal")
    # legacy CWD-relative fallback for the transition window.
    assert paths[-1].replace("\\", "/").endswith("logs/shutdown.signal")
    assert "state" not in paths[-1].replace("\\", "/")


def test_shutdown_signal_paths_fallback_when_env_absent(monkeypatch):
    monkeypatch.delenv("YASAR_USTA_STATE_DIR", raising=False)
    from src.app.hb_paths import shutdown_signal_paths
    paths = shutdown_signal_paths()
    assert len(paths) == 1
    assert paths[0].replace("\\", "/").endswith("logs/shutdown.signal")


def test_shutdown_signal_writer_reader_coupling_exact(monkeypatch):
    """COUPLING GUARD (split-brain regression). The hub writes the signal at
    ``Path(cfg.log_dir) / 'shutdown.signal'`` with ``log_dir == ${state_dir}/logs``.
    The orchestrator's first (authoritative) read candidate MUST be the identical
    absolute path. If this join changes, the hub-side write path
    (yasar_usta supervisor.py:152 + registry.yaml target ``log_dir``) must change
    in lockstep."""
    sd = r"C:\some\state\kutai"
    monkeypatch.setenv("YASAR_USTA_STATE_DIR", sd)
    from src.app.hb_paths import shutdown_signal_paths
    assert shutdown_signal_paths()[0] == os.path.join(sd, "logs", "shutdown.signal")


def test_orchestrator_reads_signal_via_helper_not_hardcoded_literal():
    """REGRESSION GUARD. orchestrator.py must derive the shutdown-signal path
    from hb_paths (env-aware), never the hardcoded CWD-relative literal — a
    revert would silently re-couple the signal to Dropbox while every other test
    stayed green."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[2]  # kutay repo root
    text = (root / "src/core/orchestrator.py").read_text(encoding="utf-8")
    assert "shutdown_signal_paths" in text, \
        "orchestrator.py must read the signal via hb_paths.shutdown_signal_paths()"
    assert 'Path("logs") / "shutdown.signal"' not in text, \
        "orchestrator.py hardcodes the CWD-relative shutdown.signal path"
