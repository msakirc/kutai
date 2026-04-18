"""Backward-compat shim. Real module lives in general_beckman.watchdog.

Aliases the package module into sys.modules so ``patch('src.core.watchdog.X')``
propagates to the real symbol. Existing tests rely on this.
"""
import sys as _sys

import general_beckman.watchdog as _pkg

_sys.modules[__name__] = _pkg
