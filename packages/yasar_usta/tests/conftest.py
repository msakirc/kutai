"""Package-wide test isolation for yasar_usta.

The hub's single-instance authority is a MACHINE-GLOBAL named mutex
(``Global\\YasarUstaHub``). If a test invokes the real ``Hub.run()`` /
``_acquire_singleton`` it would CreateMutexW on that exact prod name — which
(a) makes the pytest process HOLD the prod mutex for its lifetime, so the live
hub can't start (a hung/zombie pytest = a dead hub), (b) aborts via
``sys.exit(0)`` once the live hub already holds it, and (c) collides across
parallel pytest workers. So neutralize the real Win32 seam for every test:
Hub construction gets a fake "we own it" mutex by default.

Tests that exercise the gate logic override ``hub._create_mutex`` explicitly
(with their own fake), so they are unaffected. The real-mutex seam test in
test_singleton.py imports ``_win32_create_mutex`` from ``yasar_usta.singleton``
directly (not via hub) and uses a PID-unique name, so it too is unaffected.
"""

import pytest


@pytest.fixture(autouse=True)
def _no_real_global_mutex(monkeypatch):
    # Fake "owned" (non-null handle, err 0) → decide_singleton returns OWNED
    # without ever touching CreateMutexW or the prod mutex name.
    monkeypatch.setattr(
        "yasar_usta.hub._win32_create_mutex",
        lambda qualified_name: (0xF00D, 0),
        raising=False,
    )
