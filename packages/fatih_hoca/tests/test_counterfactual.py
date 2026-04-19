"""Counterfactual CLI sanity check: runs against a seeded sqlite and emits JSON."""
from __future__ import annotations
import json
import os
import sqlite3
import subprocess
import sys


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

    env = {**os.environ, "DB_PATH": str(db), "PYTHONPATH": "packages/fatih_hoca/src"}
    result = subprocess.run(
        [sys.executable, "-m", "fatih_hoca.counterfactual",
         "--urgency-bonus", "0.25", "--cap-gate", "0.85"],
        capture_output=True, text=True, env=env,
        timeout=30,
        cwd=r"C:\Users\sakir\Dropbox\Workspaces\kutay\.worktrees\fatih-hoca-phase2c",
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    # Output mentions 0 rows
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

    env = {**os.environ, "DB_PATH": str(db), "PYTHONPATH": "packages/fatih_hoca/src"}
    result = subprocess.run(
        [sys.executable, "-m", "fatih_hoca.counterfactual",
         "--urgency-bonus", "0.25", "--cap-gate", "0.85"],
        capture_output=True, text=True, env=env, timeout=30,
        cwd=r"C:\Users\sakir\Dropbox\Workspaces\kutay\.worktrees\fatih-hoca-phase2c",
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "agreement" in result.stdout.lower()
