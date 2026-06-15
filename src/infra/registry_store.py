# registry_store.py — back-compat shim.
#
# The sync provider/model registry moved into the `fatih_hoca` package
# (packages/fatih_hoca/src/fatih_hoca/registry_store.py), which is now the
# sync read/write owner of the providers/models/registry_events tables. This
# module is an ALIAS: the sys.modules rebind below makes `src.infra.registry_store`
# and `fatih_hoca.registry_store` the SAME module object, so every existing
# `import src.infra.registry_store` / `from src.infra import registry_store`
# keeps working unchanged.
#
# Why an alias and not `from fatih_hoca.registry_store import *`: tests freeze
# time via `monkeypatch.setattr(rs, "_now_iso", ...)` + `monkeypatch.setattr(
# rs.time, "time", ...)`, and the internal mark_dead/is_dead/_auto_revive code
# reads the module-global `_now_iso`. A star re-export creates separate name
# bindings on the shim, so the patch wouldn't reach the real code and the
# TTL/auto-revive tests would break. The alias makes the patch land on the one
# real module — same reasoning as the Phase A src/infra/db.py shim.
#
# New code should import `fatih_hoca.registry_store` directly; this shim exists
# only so existing `src.*` callers don't all have to change at once.
import sys

import fatih_hoca.registry_store as _registry_store

sys.modules[__name__] = _registry_store
