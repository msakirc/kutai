"""FIX 5.2 — module-level push_queue_profile mirrors to the default client.

Pre-fix, queue_profile was pushed ONLY in-process (beckman's
queue_profile_push → module-level push_queue_profile → singleton), while
the text selector's snapshot comes from the NerdHerdClient (sidecar
cache). The sidecar never receives queue_profile, so the text path read
queue_profile=None forever — queue-pressure scarcity arms were dead.

The fix: push_queue_profile also stores the profile on the default
client, which overlays it onto parsed snapshots (local wins when set;
parsed value kept when local unset).
"""
from __future__ import annotations

import dataclasses
import json

import pytest

import nerd_herd
from nerd_herd.client import NerdHerdClient, set_default
from nerd_herd.types import QueueProfile, SystemSnapshot

UNREACHABLE_PORT = 19899


@pytest.fixture
def fresh_singleton(monkeypatch):
    monkeypatch.setattr(nerd_herd, "_singleton", None)
    yield


@pytest.fixture
def default_client(fresh_singleton):
    client = NerdHerdClient(port=UNREACHABLE_PORT, timeout=0.5)
    set_default(client)
    yield client
    set_default(None)


def _profile() -> QueueProfile:
    return QueueProfile(
        hard_tasks_count=3,
        total_ready_count=7,
        by_difficulty={7: 2, 5: 1},
        by_capability={"code": 4},
        projected_tokens=120000,
        projected_calls=9,
    )


def test_module_push_mirrors_to_client_snapshot(default_client):
    """Beckman-style push → the client snapshot (what the text selector
    reads) carries the profile, even though the sidecar never saw it."""
    assert default_client.snapshot().queue_profile is None
    profile = _profile()
    nerd_herd.push_queue_profile(profile)
    snap = default_client.snapshot()
    assert snap.queue_profile is not None
    assert snap.queue_profile.hard_tasks_count == 3
    assert snap.queue_profile.by_difficulty == {7: 2, 5: 1}
    # Singleton still gets the write (image path reads it).
    assert nerd_herd._get_singleton().snapshot().queue_profile is not None


def test_local_profile_wins_over_parsed_none_roundtrip(default_client):
    """Round-trip a json-serialized sidecar snapshot (queue_profile=None on
    the wire today) — the locally-pushed profile overlays it."""
    wire = json.loads(json.dumps(dataclasses.asdict(
        SystemSnapshot(vram_available_mb=8000))))
    default_client._cached_snapshot = default_client._parse_snapshot(wire)
    nerd_herd.push_queue_profile(_profile())
    snap = default_client.snapshot()
    assert snap.queue_profile == _profile()
    # int keys restored / preserved despite JSON stringification concerns
    assert list(snap.queue_profile.by_difficulty) == [7, 5]


def test_parsed_profile_kept_when_local_unset(default_client):
    """A genuine sidecar-delivered profile (future transport) passes through
    untouched when nothing was pushed locally this process."""
    wire = json.loads(json.dumps(dataclasses.asdict(SystemSnapshot(
        queue_profile=QueueProfile(hard_tasks_count=9, total_ready_count=11,
                                   by_difficulty={8: 9})))))
    default_client._cached_snapshot = default_client._parse_snapshot(wire)
    snap = default_client.snapshot()
    assert snap.queue_profile is not None
    assert snap.queue_profile.hard_tasks_count == 9
    assert snap.queue_profile.by_difficulty == {8: 9}


def test_module_push_without_default_client(fresh_singleton):
    set_default(None)
    nerd_herd.push_queue_profile(_profile())  # must not crash
    assert nerd_herd._get_singleton().snapshot().queue_profile is not None


def test_overlay_does_not_mutate_cached_snapshot(default_client):
    cached = SystemSnapshot()
    default_client._cached_snapshot = cached
    nerd_herd.push_queue_profile(_profile())
    assert default_client.snapshot().queue_profile is not None
    assert cached.queue_profile is None
