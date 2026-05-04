"""Shim — re-exports from coulson.react. See packages/coulson/."""
from coulson.react import *  # noqa: F401,F403
from coulson import react as _src  # noqa: F401

def __getattr__(name):
    return getattr(_src, name)
