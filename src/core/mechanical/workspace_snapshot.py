"""Shim — real implementation lives in `salako.workspace_snapshot`.

Aliases the salako module into sys.modules under this path so existing imports
and `unittest.mock.patch` calls targeting `src.core.mechanical.workspace_snapshot.*`
continue to work transparently.
"""

import sys as _sys
from salako import workspace_snapshot as _real

_sys.modules[__name__] = _real
