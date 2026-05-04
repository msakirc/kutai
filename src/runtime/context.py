"""Shim — re-exports from coulson.context. See packages/coulson/."""
from coulson.context import *  # noqa: F401,F403
from coulson import context as _src  # noqa: F401

def __getattr__(name):
    return getattr(_src, name)
