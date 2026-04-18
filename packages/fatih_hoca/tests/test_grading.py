"""grading_perf_score blends model_stats into Performance History."""
from __future__ import annotations
import sqlite3
import pytest
from fatih_hoca.grading import grading_perf_score, GRADING_MIN_SAMPLES


@pytest.fixture
def stats_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """CREATE TABLE model_stats (
            model TEXT,
            agent_type TEXT,
            total_calls INTEGER DEFAULT 0,
            success_rate REAL DEFAULT 1.0,
            avg_grade REAL DEFAULT 0.0,
            PRIMARY KEY (model, agent_type)
        )"""
    )
    conn.commit()
    monkeypatch.setenv("DB_PATH", str(db_path))
    yield conn
    conn.close()


def test_returns_none_when_no_stats(stats_db):
    assert grading_perf_score("nonexistent-model") is None


def test_returns_none_when_below_min_samples(stats_db):
    stats_db.execute(
        "INSERT INTO model_stats (model, agent_type, total_calls, success_rate) "
        "VALUES (?, ?, ?, ?)",
        ("qwen", "coder", GRADING_MIN_SAMPLES - 1, 0.9),
    )
    stats_db.commit()
    assert grading_perf_score("qwen") is None


def test_returns_blended_score_above_threshold(stats_db):
    stats_db.execute(
        "INSERT INTO model_stats (model, agent_type, total_calls, success_rate) "
        "VALUES (?, ?, ?, ?)",
        ("qwen", "coder", 50, 1.0),
    )
    stats_db.commit()
    score = grading_perf_score("qwen")
    assert score is not None
    assert 90.0 <= score <= 95.0   # 100% success → top of scale


def test_zero_success_maps_to_floor(stats_db):
    stats_db.execute(
        "INSERT INTO model_stats (model, agent_type, total_calls, success_rate) "
        "VALUES (?, ?, ?, ?)",
        ("brokenmodel", "coder", 50, 0.0),
    )
    stats_db.commit()
    score = grading_perf_score("brokenmodel")
    assert score is not None
    assert score == 20.0  # floor


def test_aggregates_across_agent_types(stats_db):
    """A model used across agents aggregates weighted by total_calls."""
    stats_db.executemany(
        "INSERT INTO model_stats (model, agent_type, total_calls, success_rate) VALUES (?,?,?,?)",
        [
            ("qwen", "coder", 30, 1.0),
            ("qwen", "planner", 30, 0.5),
        ],
    )
    stats_db.commit()
    score = grading_perf_score("qwen")
    # weighted success ≈ 0.75 → maps to ~20 + 0.75*(95-20) = 76.25
    assert 73.0 <= score <= 80.0
