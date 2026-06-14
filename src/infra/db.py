# db.py — back-compat shim.
#
# The DB engine moved into the standalone `dabidabi` package
# (packages/db/src/dabidabi/). This module is now an ALIAS: the sys.modules
# rebind below makes `src.infra.db` and `dabidabi` the SAME module object, so
# every existing `import src.infra.db` / `from src.infra.db import X` keeps
# working unchanged — including test fixtures that
# `monkeypatch.setattr(<db module>, "DB_PATH", ...)`. A star re-export would
# NOT preserve that (it creates separate name bindings, so a monkeypatch on the
# shim wouldn't reach the connection code that reads DB_PATH → test isolation
# breaks → leaked writes to the prod DB, the exact 2026-05-04 bug).
#
# Packages should import `dabidabi` directly (this shim exists only so the
# in-app `src.*` callers don't all have to change at once).
import sys

import dabidabi

sys.modules[__name__] = dabidabi
