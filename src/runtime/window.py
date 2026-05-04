"""Shim — re-exports from coulson.window. See packages/coulson/."""
from coulson.window import *  # noqa: F401,F403
from coulson import window as _src  # noqa: F401

def __getattr__(name):
    return getattr(_src, name)
