"""Pushes on begin/end_call — Task 9."""
from unittest.mock import MagicMock

from kuleden_donen_var.in_flight import InFlightTracker
from nerd_herd.types import (
    CloudModelState,
    CloudProviderState,
    RateLimit,
    RateLimitMatrix,
)


def _make_state(provider="anthropic", model="claude-sonnet-4-6"):
    return CloudProviderState(
        provider=provider,
        models={
            model: CloudModelState(
                model_id=model,
                limits=RateLimitMatrix(rpd=RateLimit(limit=1000, remaining=900)),
            ),
        },
        limits=RateLimitMatrix(rpd=RateLimit(limit=1000, remaining=900)),
    )


def test_begin_call_pushes_in_flight_one():
    nh = MagicMock()
    state = _make_state()
    t = InFlightTracker(nerd_herd=nh, state_getter=lambda p: state)

    t.begin_call("anthropic", "claude-sonnet-4-6")

    assert nh.push_cloud_state.called
    pushed = nh.push_cloud_state.call_args[0][0]
    assert pushed.provider == "anthropic"
    model_state = pushed.models["claude-sonnet-4-6"]
    assert model_state.limits.rpd.in_flight == 1


def test_end_call_pushes_in_flight_zero():
    nh = MagicMock()
    state = _make_state()
    t = InFlightTracker(nerd_herd=nh, state_getter=lambda p: state)

    h = t.begin_call("anthropic", "claude-sonnet-4-6")
    nh.reset_mock()
    t.end_call(h)

    assert nh.push_cloud_state.called
    pushed = nh.push_cloud_state.call_args[0][0]
    assert pushed.models["claude-sonnet-4-6"].limits.rpd.in_flight == 0


def test_push_noop_when_nerd_herd_missing():
    # No nerd_herd provided — tracker still works, no crash.
    t = InFlightTracker()
    h = t.begin_call("anthropic", "claude-sonnet-4-6")
    t.end_call(h)
    assert t.count("anthropic", "claude-sonnet-4-6") == 0


def test_push_noop_when_state_getter_returns_none():
    nh = MagicMock()
    t = InFlightTracker(nerd_herd=nh, state_getter=lambda p: None)
    t.begin_call("anthropic", "claude-sonnet-4-6")
    # Getter returned None -> nothing to push.
    assert not nh.push_cloud_state.called


def test_begin_call_pushes_rpm_in_flight():
    """RPM cell must carry in_flight count too — pressure signals
    (S1/S5/S7/S9) read it for burst-protection projection. Without this,
    rpm.in_flight stays 0 and concurrent dispatch is invisible to the
    selector until response headers land. RPD-only was incomplete."""
    nh = MagicMock()
    state = CloudProviderState(
        provider="cerebras",
        models={
            "cerebras/llama3.1-8b": CloudModelState(
                model_id="cerebras/llama3.1-8b",
                limits=RateLimitMatrix(
                    rpm=RateLimit(limit=30, remaining=25),
                    rpd=RateLimit(limit=14400, remaining=14000),
                ),
            ),
        },
        limits=RateLimitMatrix(
            rpm=RateLimit(limit=30, remaining=25),
            rpd=RateLimit(limit=14400, remaining=14000),
        ),
    )
    t = InFlightTracker(nerd_herd=nh, state_getter=lambda p: state)
    t.begin_call("cerebras", "cerebras/llama3.1-8b")

    pushed = nh.push_cloud_state.call_args[0][0]
    model_state = pushed.models["cerebras/llama3.1-8b"]
    assert model_state.limits.rpm.in_flight == 1
    assert model_state.limits.rpd.in_flight == 1
    assert pushed.limits.rpm.in_flight == 1
    assert pushed.limits.rpd.in_flight == 1


def test_concurrent_dispatch_rpm_in_flight_grows():
    """Three concurrent begin_calls should leave rpm.in_flight=3 until
    end_call fires. Mirrors the fan-out scenario this signal protects."""
    nh = MagicMock()
    state = CloudProviderState(
        provider="cerebras",
        models={
            "cerebras/qwen": CloudModelState(
                model_id="cerebras/qwen",
                limits=RateLimitMatrix(rpm=RateLimit(limit=30, remaining=27)),
            ),
        },
        limits=RateLimitMatrix(rpm=RateLimit(limit=30, remaining=27)),
    )
    t = InFlightTracker(nerd_herd=nh, state_getter=lambda p: state)
    h1 = t.begin_call("cerebras", "cerebras/qwen")
    h2 = t.begin_call("cerebras", "cerebras/qwen")
    h3 = t.begin_call("cerebras", "cerebras/qwen")

    pushed = nh.push_cloud_state.call_args[0][0]
    assert pushed.models["cerebras/qwen"].limits.rpm.in_flight == 3
    assert pushed.limits.rpm.in_flight == 3

    t.end_call(h2)
    pushed = nh.push_cloud_state.call_args[0][0]
    assert pushed.models["cerebras/qwen"].limits.rpm.in_flight == 2


def test_provider_level_in_flight_sums_models():
    nh = MagicMock()
    state = CloudProviderState(
        provider="anthropic",
        models={
            "sonnet": CloudModelState(
                model_id="sonnet",
                limits=RateLimitMatrix(rpd=RateLimit(limit=1000, remaining=900)),
            ),
            "opus": CloudModelState(
                model_id="opus",
                limits=RateLimitMatrix(rpd=RateLimit(limit=500, remaining=400)),
            ),
        },
        limits=RateLimitMatrix(rpd=RateLimit(limit=1500, remaining=1300)),
    )
    t = InFlightTracker(nerd_herd=nh, state_getter=lambda p: state)
    t.begin_call("anthropic", "sonnet")
    t.begin_call("anthropic", "opus")
    t.begin_call("anthropic", "opus")

    pushed = nh.push_cloud_state.call_args[0][0]
    assert pushed.models["sonnet"].limits.rpd.in_flight == 1
    assert pushed.models["opus"].limits.rpd.in_flight == 2
    assert pushed.limits.rpd.in_flight == 3
