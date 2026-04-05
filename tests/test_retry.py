import pytest
from src.core.retry import (
    compute_retry_timing, RetryDecision,
    update_exclusions_on_failure, get_model_constraints,
)


class TestComputeRetryTiming:
    def test_quality_immediate_first_attempts(self):
        for i in range(3):
            d = compute_retry_timing("quality", attempts=i, max_attempts=6)
            assert d.action == "immediate"

    def test_quality_delayed_after_3(self):
        d = compute_retry_timing("quality", attempts=3, max_attempts=6)
        assert d.action == "delayed"
        assert d.delay_seconds == 600

    def test_quality_terminal_at_max(self):
        d = compute_retry_timing("quality", attempts=6, max_attempts=6)
        assert d.action == "terminal"

    def test_availability_first_failure_60s(self):
        d = compute_retry_timing("availability", last_avail_delay=0)
        assert d.action == "delayed"
        assert d.delay_seconds == 60

    def test_availability_doubles(self):
        d = compute_retry_timing("availability", last_avail_delay=60)
        assert d.delay_seconds == 120

    def test_availability_caps_at_7200(self):
        d = compute_retry_timing("availability", last_avail_delay=5000)
        assert d.delay_seconds == 7200

    def test_availability_terminal_after_cap(self):
        d = compute_retry_timing("availability", last_avail_delay=7200)
        assert d.action == "terminal"


class TestModelExclusions:
    def test_update_exclusions_adds_model(self):
        ctx = {}
        update_exclusions_on_failure(ctx, "model_a", 1)
        assert ctx["failed_models"] == ["model_a"]

    def test_update_exclusions_no_duplicates(self):
        ctx = {"failed_models": ["model_a"]}
        update_exclusions_on_failure(ctx, "model_a", 2)
        assert ctx["failed_models"] == ["model_a"]

    def test_constraints_no_exclusion_before_3(self):
        ctx = {"failed_models": ["model_a", "model_b"]}
        excluded, bump = get_model_constraints(ctx, attempts=2)
        assert excluded == []
        assert bump == 0

    def test_constraints_exclude_at_3(self):
        ctx = {"failed_models": ["model_a"]}
        excluded, bump = get_model_constraints(ctx, attempts=3)
        assert excluded == ["model_a"]
        assert bump == 0

    def test_constraints_difficulty_bump_at_4(self):
        ctx = {"failed_models": ["model_a"]}
        excluded, bump = get_model_constraints(ctx, attempts=4)
        assert excluded == ["model_a"]
        assert bump == 2

    def test_constraints_difficulty_bump_scales(self):
        ctx = {"failed_models": []}
        _, bump = get_model_constraints(ctx, attempts=5)
        assert bump == 4
