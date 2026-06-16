"""Tests for NerdHerdClient — HTTP proxy for NerdHerd."""
from __future__ import annotations

import pytest
import pytest_asyncio
from nerd_herd import NerdHerd
from nerd_herd.client import NerdHerdClient, GPUStateProxy

TEST_PORT = 19882
UNREACHABLE_PORT = 19899


@pytest_asyncio.fixture
async def server():
    """Start a real NerdHerd server on TEST_PORT, yield, then stop."""
    nh = NerdHerd(metrics_port=TEST_PORT, llama_server_url=None)
    await nh.start()
    yield nh
    await nh.stop()


@pytest_asyncio.fixture
async def client(server):
    """NerdHerdClient pointing at the live server."""
    c = NerdHerdClient(port=TEST_PORT)
    yield c
    await c.close()


@pytest_asyncio.fixture
async def dead_client():
    """NerdHerdClient pointing at an unused port — all calls should degrade safely."""
    c = NerdHerdClient(port=UNREACHABLE_PORT, timeout=0.5)
    yield c
    await c.close()


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_load_mode(client, server):
    mode = await client.get_load_mode()
    assert mode == "full"


@pytest.mark.asyncio
async def test_set_load_mode(client, server):
    result = await client.set_load_mode("shared", source="test")
    assert isinstance(result, str)
    # Server state should have changed
    mode = await client.get_load_mode()
    assert mode == "shared"
    # Restore
    await client.set_load_mode("full", source="test")


@pytest.mark.asyncio
async def test_enable_auto_management(client, server):
    # First set a manual mode to disable auto
    await client.set_load_mode("shared", source="user")
    # Re-enable auto
    await client.enable_auto_management()
    is_auto = await client.is_auto_managed()
    assert is_auto is True
    # Restore
    await client.set_load_mode("full", source="test")


@pytest.mark.asyncio
async def test_is_auto_managed(client, server):
    result = await client.is_auto_managed()
    assert isinstance(result, bool)


@pytest.mark.asyncio
async def test_is_local_inference_allowed(client, server):
    result = await client.is_local_inference_allowed()
    assert result is True  # default "full" mode allows inference


@pytest.mark.asyncio
async def test_get_vram_budget_fraction(client, server):
    fraction = await client.get_vram_budget_fraction()
    assert isinstance(fraction, float)
    assert 0.0 <= fraction <= 1.0


@pytest.mark.asyncio
async def test_get_vram_budget_mb(client, server):
    mb = await client.get_vram_budget_mb()
    assert isinstance(mb, int)
    assert mb >= 0


@pytest.mark.asyncio
async def test_gpu_state(client, server):
    state = await client.gpu_state()
    assert isinstance(state, GPUStateProxy)
    assert isinstance(state.vram_total_mb, int)
    assert isinstance(state.vram_free_mb, int)
    assert isinstance(state.vram_used_mb, int)
    assert isinstance(state.gpu_name, str)
    assert isinstance(state.gpu_util_pct, (int, float))


@pytest.mark.asyncio
async def test_mark_degraded(client, server):
    # Should not raise
    await client.mark_degraded("test_capability")


@pytest.mark.asyncio
async def test_prometheus_lines(client, server):
    text = await client.prometheus_lines()
    assert isinstance(text, str)
    assert len(text) > 0


# ---------------------------------------------------------------------------
# Graceful degradation tests (unreachable server)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dead_get_load_mode(dead_client):
    mode = await dead_client.get_load_mode()
    assert mode == "full"


@pytest.mark.asyncio
async def test_dead_set_load_mode(dead_client):
    result = await dead_client.set_load_mode("minimal")
    assert isinstance(result, str)  # safe default, no exception


@pytest.mark.asyncio
async def test_dead_enable_auto_management(dead_client):
    # Should not raise
    await dead_client.enable_auto_management()


@pytest.mark.asyncio
async def test_dead_is_auto_managed(dead_client):
    result = await dead_client.is_auto_managed()
    assert result is True  # safe default


@pytest.mark.asyncio
async def test_dead_is_local_inference_allowed(dead_client):
    result = await dead_client.is_local_inference_allowed()
    assert result is True  # safe default


@pytest.mark.asyncio
async def test_dead_get_vram_budget_fraction(dead_client):
    result = await dead_client.get_vram_budget_fraction()
    assert result == 1.0  # safe default


@pytest.mark.asyncio
async def test_dead_get_vram_budget_mb(dead_client):
    result = await dead_client.get_vram_budget_mb()
    assert result == 0  # safe default


@pytest.mark.asyncio
async def test_dead_gpu_state(dead_client):
    state = await dead_client.gpu_state()
    assert isinstance(state, GPUStateProxy)
    assert state.vram_total_mb == 0
    assert state.gpu_name == ""


@pytest.mark.asyncio
async def test_dead_mark_degraded(dead_client):
    # Should not raise
    await dead_client.mark_degraded("some_cap")


@pytest.mark.asyncio
async def test_dead_prometheus_lines(dead_client):
    text = await dead_client.prometheus_lines()
    assert isinstance(text, str)  # empty string is acceptable


# ---------------------------------------------------------------------------
# Snapshot roundtrip — per-model state must survive HTTP serialization
# ---------------------------------------------------------------------------

def test_parse_snapshot_preserves_per_model_state():
    """Regression: pre-2026-05-04, _parse_snapshot stripped the `models`
    dict and `circuit_breaker_open` flag from cloud provider state.
    Selector eligibility checks (daily_exhausted, rpm_cooldown,
    circuit_breaker) silently no-op'd, leading to repeated admission of
    exhausted gemini ids (294 violations in 2h before the fix)."""
    from nerd_herd.client import NerdHerdClient
    payload = {
        "vram_available_mb": 8000,
        "local": {},
        "cloud": {
            "gemini": {
                "provider": "gemini",
                "utilization_pct": 75.0,
                "consecutive_failures": 2,
                "circuit_breaker_open": True,
                "models": {
                    "gemini/gemini-2.5-flash": {
                        "model_id": "gemini/gemini-2.5-flash",
                        "utilization_pct": 90.0,
                        "recent_success_rate": 0.45,
                        "recent_samples_n": 12,
                        "provider_prior_rate": 0.50,
                        "daily_exhausted": True,
                        "rpm_cooldown": False,
                    },
                    "gemini/gemini-2.5-flash-lite": {
                        "daily_exhausted": False,
                        "rpm_cooldown": True,
                    },
                },
            },
        },
        "in_flight_calls": [],
        "recent_swap_count": 0,
    }
    client = NerdHerdClient.__new__(NerdHerdClient)
    snap = client._parse_snapshot(payload)
    prov = snap.cloud["gemini"]
    assert prov.circuit_breaker_open is True
    assert prov.consecutive_failures == 2
    flash = prov.models["gemini/gemini-2.5-flash"]
    assert flash.daily_exhausted is True
    assert flash.recent_success_rate == 0.45
    assert flash.recent_samples_n == 12
    assert flash.provider_prior_rate == 0.50
    lite = prov.models["gemini/gemini-2.5-flash-lite"]
    assert lite.rpm_cooldown is True
    assert lite.daily_exhausted is False
    # Defaults when fields missing
    assert lite.recent_success_rate == 1.0
    assert lite.provider_prior_rate is None


def test_parse_snapshot_handles_empty_cloud():
    from nerd_herd.client import NerdHerdClient
    client = NerdHerdClient.__new__(NerdHerdClient)
    snap = client._parse_snapshot({"vram_available_mb": 0, "local": {},
                                    "cloud": {}, "in_flight_calls": []})
    assert snap.cloud == {}


def test_parse_snapshot_preserves_desktop_fields():
    """Regression: _parse_snapshot dropped the 6 desktop-signal fields,
    so they silently defaulted to away/full (load_mode='full',
    user_idle_s=1e9, etc.). Prod reads snapshots via this HTTP client,
    so the entire desktop-signals feature was dead in production even
    though the sidecar emits the fields via dataclasses.asdict."""
    from nerd_herd.client import NerdHerdClient
    client = NerdHerdClient.__new__(NerdHerdClient)  # no network
    data = {
        "vram_available_mb": 8000, "local": {}, "cloud": {},
        "load_mode": "shared", "user_idle_s": 5.0, "foreground_fullscreen": True,
        "ram_available_mb": 4000, "ram_total_mb": 32000, "external_gpu_fraction": 0.7,
    }
    snap = client._parse_snapshot(data)
    assert snap.load_mode == "shared"
    assert snap.user_idle_s == 5.0
    assert snap.foreground_fullscreen is True
    assert snap.ram_available_mb == 4000
    assert snap.ram_total_mb == 32000
    assert snap.external_gpu_fraction == 0.7


def test_parse_snapshot_preserves_image_server_fields():
    """Regression (2nd recurrence of the dropped-field bug class):
    _parse_snapshot dropped image_server_resident/image_server_vram_mb, so
    prod (which reads snapshots through this HTTP client) always saw the
    image server as non-resident — eviction cost/warm bonus dead."""
    from nerd_herd.client import NerdHerdClient
    client = NerdHerdClient.__new__(NerdHerdClient)  # no network
    data = {
        "vram_available_mb": 8000, "local": {}, "cloud": {},
        "image_server_resident": True, "image_server_vram_mb": 4500,
    }
    snap = client._parse_snapshot(data)
    assert snap.image_server_resident is True
    assert snap.image_server_vram_mb == 4500


def test_parse_snapshot_field_completeness_guard():
    """Field-completeness guard: every top-level SystemSnapshot field must
    survive sidecar asdict → JSON wire → _parse_snapshot.

    This bug class (client silently dropping fields newly added to
    SystemSnapshot) recurred twice: desktop signals (2026-06-09) and
    image_server_* (2026-06-11). Adding a field to SystemSnapshot without
    teaching _parse_snapshot — or without giving it a non-default value
    here — now fails this test.

    Nested `limits` matrices stay dataclass-default on purpose: the client
    intentionally rebuilds them (selector consumes the plumbed booleans
    daily_exhausted / rpm_cooldown / circuit_breaker_open instead).
    """
    import dataclasses
    import json

    from nerd_herd.client import NerdHerdClient
    from nerd_herd.types import (
        CloudModelState, CloudProviderState, InFlightCall, LocalModelState,
        QueueProfile, SystemSnapshot,
    )

    snap_in = SystemSnapshot(
        vram_available_mb=1234,
        local=LocalModelState(
            model_name="qwen2.5", thinking_enabled=True, vision_enabled=True,
            measured_tps=12.5, pp_tps=300.0, context_length=8192,
            is_swapping=True, kv_cache_ratio=0.5, idle_seconds=3.5,
            requests_processing=2,
        ),
        cloud={
            "gemini": CloudProviderState(
                provider="gemini", utilization_pct=75.0,
                consecutive_failures=2, last_failure_at=1700000000,
                circuit_breaker_open=True,
                models={
                    "gemini/gemini-2.5-flash": CloudModelState(
                        model_id="gemini/gemini-2.5-flash",
                        utilization_pct=90.0, recent_success_rate=0.45,
                        recent_samples_n=12, provider_prior_rate=0.5,
                        daily_exhausted=True, rpm_cooldown=True,
                    ),
                },
            ),
        },
        queue_profile=QueueProfile(
            hard_tasks_count=3, total_ready_count=7,
            by_difficulty={7: 2, 5: 1}, by_capability={"code": 4},
            projected_tokens=120000, projected_calls=9,
        ),
        in_flight_calls=[InFlightCall(
            call_id="c1", task_id=42, category="main_work", model="m1",
            provider="p1", is_local=True, started_at=123.0, est_tokens=500,
        )],
        recent_swap_count=2,
        image_server_resident=True,
        image_server_vram_mb=4500,
        load_mode="shared",
        user_idle_s=5.0,
        foreground_fullscreen=True,
        ram_available_mb=4000,
        ram_total_mb=32000,
        external_gpu_fraction=0.7,
        local_inference_down=True,
    )

    # Every top-level field must hold a non-default value, otherwise the
    # guard cannot detect that field being dropped. A future field added
    # to SystemSnapshot but not set above fails HERE first.
    defaults = SystemSnapshot()
    for f in dataclasses.fields(SystemSnapshot):
        assert getattr(snap_in, f.name) != getattr(defaults, f.name), (
            f"guard input must set a non-default value for {f.name!r} "
            "(new SystemSnapshot field? populate it above)"
        )

    # Simulate the actual wire: asdict (exposition._handle_snapshot) → JSON.
    wire = json.loads(json.dumps(dataclasses.asdict(snap_in)))
    client = NerdHerdClient.__new__(NerdHerdClient)  # no network
    snap_out = client._parse_snapshot(wire)

    in_d = dataclasses.asdict(snap_in)
    out_d = dataclasses.asdict(snap_out)
    for f in dataclasses.fields(SystemSnapshot):
        assert out_d[f.name] == in_d[f.name], (
            f"SystemSnapshot.{f.name} dropped/mutated by _parse_snapshot"
        )


def test_snapshot_overlay_passthrough_field_completeness():
    """Companion to the field-completeness guard for the 2026-06-12 local
    overlays (_overlay_local): with NO locally-written state (no swaps
    recorded, no queue_profile pushed, no image-server state written),
    client.snapshot() must return the parsed cached snapshot unmodified —
    the overlay must not perturb a single field of genuine sidecar data."""
    import dataclasses

    from nerd_herd.client import NerdHerdClient
    from nerd_herd.types import (
        CloudModelState, CloudProviderState, InFlightCall, LocalModelState,
        QueueProfile, SystemSnapshot,
    )

    snap_in = SystemSnapshot(
        vram_available_mb=1234,
        local=LocalModelState(model_name="qwen2.5", measured_tps=12.5),
        cloud={"gemini": CloudProviderState(
            provider="gemini", utilization_pct=75.0,
            models={"m": CloudModelState(model_id="m", daily_exhausted=True)},
        )},
        queue_profile=QueueProfile(hard_tasks_count=3, total_ready_count=7),
        in_flight_calls=[InFlightCall(
            call_id="c1", task_id=42, category="main_work", model="m1",
            provider="p1", is_local=True, started_at=123.0, est_tokens=500,
        )],
        recent_swap_count=2,
        image_server_resident=True,
        image_server_vram_mb=4500,
        load_mode="shared",
        user_idle_s=5.0,
        foreground_fullscreen=True,
        ram_available_mb=4000,
        ram_total_mb=32000,
        external_gpu_fraction=0.7,
        local_inference_down=True,
    )

    client = NerdHerdClient(port=UNREACHABLE_PORT, timeout=0.5)
    client._cached_snapshot = snap_in
    snap_out = client.snapshot()
    assert dataclasses.asdict(snap_out) == dataclasses.asdict(snap_in), (
        "overlay with no local state must be a pure pass-through"
    )


def test_parse_snapshot_skips_non_dict_model_entries():
    from nerd_herd.client import NerdHerdClient
    client = NerdHerdClient.__new__(NerdHerdClient)
    payload = {
        "local": {}, "in_flight_calls": [],
        "cloud": {"groq": {"models": {"x": "not_a_dict", "y": {"daily_exhausted": True}}}},
    }
    snap = client._parse_snapshot(payload)
    assert "x" not in snap.cloud["groq"].models
    assert snap.cloud["groq"].models["y"].daily_exhausted is True
