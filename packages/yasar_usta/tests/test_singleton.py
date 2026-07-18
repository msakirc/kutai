"""Tests for yasar_usta.singleton — the named-mutex single-instance authority.

The mutex is the CWD/session-independent authority for "one hub, never two"
(design: docs/superpowers/specs/2026-07-17-yasar-usta-always-live-singleton-design.md).
Win32 CreateMutexW is injected so the decision tree + circuit-breaker are pure
and testable off-Windows; a real-mutex integration test covers the seam.
"""

import sys
import tempfile
from pathlib import Path

import pytest

from yasar_usta.singleton import (
    OWNED, ALREADY_RUNNING, ERROR,
    ERROR_ALREADY_EXISTS, ERROR_ACCESS_DENIED,
    decide_singleton, record_fault, enforce_singleton,
)


class FakeMutex:
    """Injectable CreateMutexW. Returns queued (handle, err) per call and
    records the qualified names it was asked for (to assert Global-then-Local)."""

    def __init__(self, results):
        self._results = list(results)
        self.calls = []

    def __call__(self, qualified_name):
        self.calls.append(qualified_name)
        return self._results.pop(0)


class TestDecideSingleton:
    def test_global_owned(self):
        fake = FakeMutex([(0xAB, 0)])
        outcome, handle, ns = decide_singleton("YasarUstaHub", create_mutex=fake)
        assert outcome == OWNED
        assert handle == 0xAB
        assert ns == "Global\\"
        assert fake.calls == ["Global\\YasarUstaHub"]

    def test_global_already_running(self):
        # CreateMutexW returns a valid handle to the EXISTING mutex + err 183.
        fake = FakeMutex([(0xAB, ERROR_ALREADY_EXISTS)])
        outcome, handle, ns = decide_singleton("YasarUstaHub", create_mutex=fake)
        assert outcome == ALREADY_RUNNING
        assert fake.calls == ["Global\\YasarUstaHub"]

    def test_global_access_denied_falls_back_to_local_owned(self):
        fake = FakeMutex([(None, ERROR_ACCESS_DENIED), (0xCD, 0)])
        outcome, handle, ns = decide_singleton("YasarUstaHub", create_mutex=fake)
        assert outcome == OWNED
        assert handle == 0xCD
        assert ns == "Local\\"
        assert fake.calls == ["Global\\YasarUstaHub", "Local\\YasarUstaHub"]

    def test_global_access_denied_local_already_running(self):
        fake = FakeMutex([(None, ERROR_ACCESS_DENIED), (0xCD, ERROR_ALREADY_EXISTS)])
        outcome, handle, ns = decide_singleton("YasarUstaHub", create_mutex=fake)
        assert outcome == ALREADY_RUNNING

    def test_ambiguous_error_never_falls_back_and_is_error(self):
        # A non-ACCESS_DENIED failure (NULL handle) must NOT be read as "free".
        fake = FakeMutex([(None, 1450)])  # ERROR_NO_SYSTEM_RESOURCES
        outcome, handle, ns = decide_singleton("YasarUstaHub", create_mutex=fake)
        assert outcome == ERROR
        assert fake.calls == ["Global\\YasarUstaHub"]  # did not try Local

    def test_access_denied_on_both_namespaces_is_error(self):
        fake = FakeMutex([(None, ERROR_ACCESS_DENIED), (None, ERROR_ACCESS_DENIED)])
        outcome, handle, ns = decide_singleton("YasarUstaHub", create_mutex=fake)
        assert outcome == ERROR


class TestRecordFault:
    def _marker(self, tmp):
        return Path(tmp) / "mutex_fault.json"

    def test_first_fault_alerts(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = record_fault(self._marker(tmp), "access_denied", now=1000.0)
            assert d.should_alert is True
            assert d.count == 1
            assert d.give_up is False

    def test_same_fault_within_dedup_window_suppresses_alert(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = self._marker(tmp)
            record_fault(m, "access_denied", now=1000.0)
            d = record_fault(m, "access_denied", now=1060.0)  # +60s, < 1h
            assert d.should_alert is False
            assert d.count == 2

    def test_same_fault_after_dedup_window_realerts(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = self._marker(tmp)
            record_fault(m, "access_denied", now=1000.0)
            d = record_fault(m, "access_denied", now=1000.0 + 3601)
            assert d.should_alert is True

    def test_give_up_after_k_consecutive(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = self._marker(tmp)
            d = None
            for i in range(5):
                d = record_fault(m, "access_denied", now=1000.0 + i)
            assert d.count == 5
            assert d.give_up is True

    def test_different_signature_resets_counter(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = self._marker(tmp)
            record_fault(m, "access_denied", now=1000.0)
            record_fault(m, "access_denied", now=1001.0)
            d = record_fault(m, "path_unwritable", now=1002.0)
            assert d.count == 1
            assert d.should_alert is True

    def test_corrupt_marker_is_treated_as_first_fault(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = self._marker(tmp)
            m.write_text("{ not json")
            d = record_fault(m, "access_denied", now=1000.0)
            assert d.count == 1
            assert d.should_alert is True


class TestEnforceSingleton:
    """Acting glue: OWNED proceeds, ALREADY_RUNNING exits 0, ERROR fails closed
    with the circuit-breaker (never fail-open into a possible duplicate)."""

    def test_owned_proceeds_without_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = []
            enforce_singleton(
                "H", state_dir=tmp, create_mutex=FakeMutex([(1, 0)]),
                exit_fn=lambda c: calls.append(c), alert=lambda m: None,
                now_fn=lambda: 1000.0)
            assert calls == []  # proceeds — never exits

    def test_already_running_exits_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = []
            enforce_singleton(
                "H", state_dir=tmp, create_mutex=FakeMutex([(1, ERROR_ALREADY_EXISTS)]),
                exit_fn=lambda c: calls.append(c), alert=lambda m: None,
                now_fn=lambda: 1000.0)
            assert calls == [0]

    def test_error_alerts_once_and_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls, alerts = [], []
            enforce_singleton(
                "H", state_dir=tmp, create_mutex=FakeMutex([(None, 1450)]),
                exit_fn=lambda c: calls.append(c), alert=lambda m: alerts.append(m),
                now_fn=lambda: 1000.0)
            assert calls == [3]           # nonzero → Layer 0 retries
            assert len(alerts) == 1       # loud, once

    def test_error_gives_up_after_k_and_exits_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            last_calls = None
            for i in range(5):
                last_calls = []
                enforce_singleton(
                    "H", state_dir=tmp, create_mutex=FakeMutex([(None, 1450)]),
                    exit_fn=lambda c: last_calls.append(c), alert=lambda m: None,
                    now_fn=lambda: 1000.0 + i)
            assert last_calls == [0]       # circuit-breaker stops the retry hammer


@pytest.mark.skipif(sys.platform != "win32", reason="Win32 named mutex")
class TestRealMutexSeam:
    def test_second_acquire_detects_first(self):
        from yasar_usta.singleton import _win32_create_mutex
        name = f"YasarUstaTest_{__import__('os').getpid()}"
        o1, h1, _ = decide_singleton(name, create_mutex=_win32_create_mutex)
        assert o1 == OWNED
        # Second create for the same name (handle kept open) → ALREADY_EXISTS.
        o2, h2, _ = decide_singleton(name, create_mutex=_win32_create_mutex)
        assert o2 == ALREADY_RUNNING
