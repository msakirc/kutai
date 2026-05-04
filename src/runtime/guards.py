"""Shim — re-exports from coulson.guards. See packages/coulson/."""
from coulson.guards import *  # noqa: F401,F403
from coulson import guards as _src  # noqa: F401

def __getattr__(name):
    return getattr(_src, name)
