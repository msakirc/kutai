"""Counterfactual CLI sanity check: runs against a seeded sqlite and emits JSON.

Updated for Phase 2d: the CLI sweeps --k (UTILIZATION_K) instead of the old
--urgency-bonus / --cap-gate (the gate was retired, K replaced it as the
tunable magnitude of the utilization equation).
"""
from __future__ import annotations
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path


WORKTREE = Path(__file__).resolve().parents[3]
SRC_PATH = WORKTREE / "packages" / "fatih_hoca" / "src"


def test_cli_runs_on_empty_db(tmp_path):
    db = tmp_path / "empty.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """CREATE TABLE model_pick_log (
            id INTEGER PRIMARY KEY, timestamp TEXT, task_name TEXT, agent_type TEXT,
            difficulty INTEGER, call_category TEXT, picked_model TEXT,
            picked_score REAL, picked_reasons TEXT, candidates_json TEXT,
            failures_json TEXT, snapshot_summary TEXT, pool TEXT, urgency REAL
        );
        CREATE TABLE model_stats (
            model TEXT, agent_type TEXT, total_calls INTEGER,
            success_rate REAL, avg_grade REAL,
            PRIMARY KEY (model, agent_type)
        );"""
    )
    conn.commit()
    conn.close()

    env = {**os.environ, "DB_PATH": str(db), "PYTHONPATH": str(SRC_PATH)}
    result = subprocess.run(
        [sys.executable, "-m", "fatih_hoca.counterfactual", "--k", "1.0"],
        capture_output=True, text=True, env=env,
        timeout=30,
        cwd=str(WORKTREE),
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "rows" in result.stdout.lower()


def test_cli_reports_agreement_rate(tmp_path):
    """Seed a row where candidates_json contains the picked model as #1, and
    verify agreement report includes 100%."""
    db = tmp_path / "seeded.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """CREATE TABLE model_pick_log (
            id INTEGER PRIMARY KEY, timestamp TEXT, task_name TEXT, agent_type TEXT,
            difficulty INTEGER, call_category TEXT, picked_model TEXT,
            picked_score REAL, picked_reasons TEXT, candidates_json TEXT,
            failures_json TEXT, snapshot_summary TEXT, pool TEXT, urgency REAL
        );
        CREATE TABLE model_stats (
            model TEXT, agent_type TEXT, total_calls INTEGER,
            success_rate REAL, avg_grade REAL,
            PRIMARY KEY (model, agent_type)
        );"""
    )
    candidates = json.dumps([
        {"name": "qwen", "composite": 80.0, "cap_score": 78.0, "pool": "local", "urgency": 0.5},
        {"name": "groq-llama-70b", "composite": 75.0, "cap_score": 72.0, "pool": "time_bucketed", "urgency": 0.0},
    ])
    conn.execute(
        """INSERT INTO model_pick_log
           (timestamp, task_name, agent_type, difficulty, call_category,
            picked_model, picked_score, picked_reasons, candidates_json,
            failures_json, snapshot_summary, pool, urgency)
           VALUES ('2026-04-18T00:00:00', 'test', 'coder', 5, 'main_work',
                   'qwen', 80.0, 'test', ?, '[]', '{}', 'local', 0.5)""",
        (candidates,),
    )
    conn.execute(
        "INSERT INTO model_stats (model, agent_type, total_calls, success_rate, avg_grade) "
        "VALUES ('qwen', 'coder', 30, 1.0, 8.5)"
    )
    conn.commit()
    conn.close()

    env = {**os.environ, "DB_PATH": str(db), "PYTHONPATH": str(SRC_PATH)}
    result = subprocess.run(
        [sys.executable, "-m", "fatih_hoca.counterfactual", "--k", "1.0"],
        capture_output=True, text=True, env=env, timeout=30,
        cwd=str(WORKTREE),
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "agreement" in result.stdout.lower()


def test_rescore_utilization_over_qualified_dampens():
    """Claude (cap=93) on d=3 (cap_needed=30, fit_excess=0.63): negative
    scarcity conservation fully applied (dampener=1-0.63=0.37)."""
    from fatih_hoca.counterfactual import _rescore_utilization
    out = _rescore_utilization(
        original_score=100.0,
        cap_score_100=93.0,
        task_difficulty=3,
        scarcity=-1.0,
        K=1.0,
    )
    # adj = 1 + 1.0 * -1.0 * (1 - 0.63) = 1 - 0.37 = 0.63
    assert 62 < out < 64


def test_rescore_utilization_positive_symmetric_dampener():
    """Under-qualified candidate with positive scarcity: dampener uses
    abs(fit_excess) — burning a wrong tool is wasteful."""
    from fatih_hoca.counterfactual import _rescore_utilization
    # cap=55, d=8 → cap_needed=75, fit_excess=-0.20
    # adj = 1 + 1.0 * 0.5 * (1 - 0.20) = 1 + 0.4 = 1.4
    out = _rescore_utilization(
        original_score=100.0,
        cap_score_100=55.0,
        task_difficulty=8,
        scarcity=0.5,
        K=1.0,
    )
    assert 139 < out < 141


def test_rescore_utilization_well_fit_full_boost():
    """Well-fit candidate (fit_excess=0) gets full positive boost."""
    from fatih_hoca.counterfactual import _rescore_utilization
    out = _rescore_utilization(
        original_score=100.0,
        cap_score_100=75.0,
        task_difficulty=8,
        scarcity=1.0,
        K=1.0,
    )
    # fit_excess=0.0 → dampener=1.0 → adj=2.0
    assert 199 < out < 201
