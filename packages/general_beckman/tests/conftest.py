"""Per-test cleanup for the in-flight registry.

Beckman's admission tick now overlays the local in_flight registry from
src.core.in_flight onto the snapshot before pressure_for runs. Tests
that don't explicitly populate that registry MUST start with it empty —
otherwise leftovers from a prior test (especially test_admission_local_
inflight which adds dispatcher slots) leak into the next test's
admission decision and produce unexpected REJECTs.

Autouse fixture clears both _task_slots and _call_entries before each
test. Cleanup-only — never adds entries.
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
    yield
    try:
        import src.core.in_flight as in_flight_mod
        in_flight_mod._task_slots.clear()
        in_flight_mod._call_entries.clear()
    except Exception:
        pass
