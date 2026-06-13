"""FIX 5.3 — module-level record_image_server_state mirrors to the client.

Pre-fix, clair_obscur wrote image-server residency to the in-process
singleton only; client._parse_snapshot parses image_server_resident /
image_server_vram_mb but no transport delivers them to the sidecar, so
client-snapshot consumers read permanently False/0.

The fix: record_image_server_state also stores on the default client and
overlays onto parsed snapshots. A "written" flag distinguishes "locally
set to False/0" (overlay wins) from "never written this process" (parsed
values pass through), so a future sidecar transport isn't clobbered.

Note: fatih_hoca image_select._effective_snapshot reads the singleton
directly for these fields and remains correct; this mirror makes the
client snapshot truthful for OTHER consumers.
"""
from __future__ import annotations

import dataclasses
import json

import pytest

import nerd_herd
from nerd_herd.client import NerdHerdClient, set_default
from nerd_herd.types import SystemSnapshot

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


def test_module_record_mirrors_to_client_snapshot(default_client):
    """clair_obscur-style write → client snapshot carries residency."""
    assert default_client.snapshot().image_server_resident is False
    nerd_herd.record_image_server_state(resident=True, vram_mb=4500)
    snap = default_client.snapshot()
    assert snap.image_server_resident is True
    assert snap.image_server_vram_mb == 4500
    # Singleton still gets the write (image path's _effective_snapshot).
    sing = nerd_herd._get_singleton().snapshot()
    assert sing.image_server_resident is True


def test_roundtrip_overlay_local_wins_even_when_falsy(default_client):
    """Once written this process, the local value wins — including a
    falsy stop-state (resident=False, vram=0) over stale parsed True."""
    wire = json.loads(json.dumps(dataclasses.asdict(SystemSnapshot(
        image_server_resident=True, image_server_vram_mb=9999))))
    default_client._cached_snapshot = default_client._parse_snapshot(wire)
    nerd_herd.record_image_server_state(resident=False, vram_mb=0)
    snap = default_client.snapshot()
    assert snap.image_server_resident is False
    assert snap.image_server_vram_mb == 0


def test_parsed_values_pass_through_when_never_written(default_client):
    """Never written this process → genuine sidecar data (some future
    transport) passes through untouched; defaults stay defaults."""
    wire = json.loads(json.dumps(dataclasses.asdict(SystemSnapshot(
        image_server_resident=True, image_server_vram_mb=4500))))
    default_client._cached_snapshot = default_client._parse_snapshot(wire)
    snap = default_client.snapshot()
    assert snap.image_server_resident is True
    assert snap.image_server_vram_mb == 4500
    # And unset everywhere → defaults
    default_client._cached_snapshot = SystemSnapshot()
    snap = default_client.snapshot()
    assert snap.image_server_resident is False
    assert snap.image_server_vram_mb == 0


def test_module_record_without_default_client(fresh_singleton):
    set_default(None)
    nerd_herd.record_image_server_state(resident=True, vram_mb=4500)  # no crash
    assert nerd_herd._get_singleton().snapshot().image_server_resident is True


def test_overlay_does_not_mutate_cached_snapshot(default_client):
    cached = SystemSnapshot()
    default_client._cached_snapshot = cached
    nerd_herd.record_image_server_state(resident=True, vram_mb=4500)
    assert default_client.snapshot().image_server_resident is True
    assert cached.image_server_resident is False
