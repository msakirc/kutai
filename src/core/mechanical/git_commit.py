"""Shim — real implementation lives in `mr_roboto.git_commit`.

Aliases the mr_roboto module into sys.modules under this path so existing imports
and `unittest.mock.patch` calls targeting `src.core.mechanical.git_commit.*`
continue to work transparently.
"""

import sys as _sys
from mr_roboto import git_commit as _real

_sys.modules[__name__] = _real
