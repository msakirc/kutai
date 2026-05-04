"""Shim — re-exports from coulson.reflection. See packages/coulson/."""
from coulson.reflection import *  # noqa: F401,F403
from coulson import reflection as _src  # noqa: F401

def __getattr__(name):
    return getattr(_src, name)
