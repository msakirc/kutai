"""Shim — real implementation lives in `mr_roboto.workspace_snapshot`.

Aliases the mr_roboto module into sys.modules under this path so existing imports
and `unittest.mock.patch` calls targeting `src.core.mechanical.workspace_snapshot.*`
continue to work transparently.
"""

import sys as _sys
from mr_roboto import workspace_snapshot as _real

_sys.modules[__name__] = _real
