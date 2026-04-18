"""Backward-compat shim. Real module lives in general_beckman.scheduled_jobs."""
import sys as _sys

import general_beckman.scheduled_jobs as _pkg

_sys.modules[__name__] = _pkg
