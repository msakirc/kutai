"""Shim — re-exports from coulson.parsing. See packages/coulson/."""
from coulson.parsing import *  # noqa: F401,F403
from coulson import parsing as _src  # noqa: F401

def __getattr__(name):
    return getattr(_src, name)
