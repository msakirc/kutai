"""Backward-compat shim. Real module lives in general_beckman.result_guards.

Aliases the package module into sys.modules so ``patch('src.core.result_guards.X')``
propagates to the real symbol. Existing tests rely on this.
"""
import sys as _sys

import general_beckman.result_guards as _pkg

_sys.modules[__name__] = _pkg
