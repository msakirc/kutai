"""Per-test cleanup for the in-flight registry.

Beckman's admission tick now overlays the local in_flight registry from
src.core.in_flight onto the snapshot before pressure_for runs. Tests
that don't explicitly populate that registry MUST start with it empty —
otherwise leftovers from a prior test (especially test_admission_local_
inflight which adds dispatcher slots) leak into the next test's
admission decision and produce unexpected REJECTs.

Autouse fixture clears both _task_slots and _call_entries before each
test. Cleanup-only — never adds entries.

Also clears general_beckman.apply._source_verdict_locks between tests.
SP3b FIX 2 intentionally does NOT evict lock objects from the dict
(doing so reopens the lost-update race under 3+ concurrent appliers —
see apply.py _source_verdict_guard). Without this test-only cleanup,
asyncio.Lock objects created in test N's event loop would persist into
test N+1's event loop and raise "bound to a different event loop" on
the first lock acquisition in any concurrency test.
"""
import pytest


@pytest.fixture(autouse=True)
def _clear_in_flight_registry():
    try:
        import src.core.in_flight as in_flight_mod
        in_flight_mod._task_slots.clear()
        in_flight_mod._call_entries.clear()
    except Exception:
        pass
    # Clear per-source verdict locks so asyncio.Lock objects bound to the
    # previous test's event loop do not bleed into the next test.
    try:
        import general_beckman.apply as apply_mod
        apply_mod._source_verdict_locks.clear()
    except Exception:
        pass
    yield
    try:
        import src.core.in_flight as in_flight_mod
        in_flight_mod._task_slots.clear()
        in_flight_mod._call_entries.clear()
    except Exception:
        pass
    try:
        import general_beckman.apply as apply_mod
        apply_mod._source_verdict_locks.clear()
    except Exception:
        pass
