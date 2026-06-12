"""FIX 5.1 — module-level record_swap mirrors to the default client.

Pre-fix, THREE disjoint SwapBudget instances existed in prod:
  (1) the in-process singleton's — where both prod writers write
      (dispatcher + husam via module-level nerd_herd.record_swap),
  (2) the NerdHerdClient's local _swap_budget — what the text selector's
      HARD GATE reads (selector.py self._nerd_herd.recent_swap_count();
      client.record_swap had ZERO prod callers),
  (3) the sidecar's — what ranking stickiness reads via the client's
      cached snapshot (sidecar has NO swap write path at all).
Net: the 3-per-5min swap gate AND the anti-flap stickiness ramp read 0
forever in prod.

The fix: module-level record_swap fans out to BOTH the singleton and the
default client's local budget; client.snapshot() overlays
recent_swap_count = max(parsed_sidecar_value, local_budget) so ranking
stickiness sees local swaps too.
"""
from __future__ import annotations

import pytest

import nerd_herd
from nerd_herd.client import NerdHerdClient, set_default
from nerd_herd.types import SystemSnapshot

UNREACHABLE_PORT = 19899


@pytest.fixture
def fresh_singleton(monkeypatch):
    """Isolate the module-level NerdHerd singleton per test."""
    monkeypatch.setattr(nerd_herd, "_singleton", None)
    yield


@pytest.fixture
def default_client(fresh_singleton):
    """Install a default NerdHerdClient (never contacted) and tear it down."""
    client = NerdHerdClient(port=UNREACHABLE_PORT, timeout=0.5)
    set_default(client)
    yield client
    set_default(None)


def test_module_record_swap_mirrors_to_client_budget(default_client):
    """The selector hard gate reads client.recent_swap_count() — module
    record_swap must feed it, not just the singleton."""
    assert default_client.recent_swap_count() == 0
    nerd_herd.record_swap("model_a")
    nerd_herd.record_swap("model_b")
    assert default_client.recent_swap_count() == 2
    # The singleton still gets the write too (image path reads it).
    assert nerd_herd._get_singleton().recent_swap_count() == 2


def test_client_snapshot_overlays_local_swap_count(default_client):
    """Ranking stickiness reads snapshot.recent_swap_count — the client must
    overlay its local budget onto the (sidecar-)parsed value with max()."""
    default_client._cached_snapshot = SystemSnapshot(recent_swap_count=1)
    nerd_herd.record_swap("m1")
    nerd_herd.record_swap("m2")
    nerd_herd.record_swap("m3")
    assert default_client.snapshot().recent_swap_count == 3


def test_client_snapshot_overlay_takes_max_not_sum(default_client):
    """A genuine (future-transport) sidecar value higher than the local
    budget must win — overlay is max(), not replacement or sum."""
    default_client._cached_snapshot = SystemSnapshot(recent_swap_count=5)
    nerd_herd.record_swap("m1")
    assert default_client.snapshot().recent_swap_count == 5


def test_module_record_swap_without_default_client(fresh_singleton):
    """No default client wired → singleton-only write, no crash."""
    set_default(None)
    nerd_herd.record_swap("model_a")
    assert nerd_herd._get_singleton().recent_swap_count() == 1


def test_overlay_does_not_mutate_cached_snapshot(default_client):
    """The overlay must not write into the client's cache — the cache stays
    pure-parsed so a later refresh/read can't double-apply local state."""
    cached = SystemSnapshot(recent_swap_count=0)
    default_client._cached_snapshot = cached
    nerd_herd.record_swap("m1")
    snap = default_client.snapshot()
    assert snap.recent_swap_count == 1
    assert cached.recent_swap_count == 0
