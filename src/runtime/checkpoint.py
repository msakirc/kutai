"""Shim — re-exports from coulson.checkpoint. See packages/coulson/."""
from coulson.checkpoint import *  # noqa: F401,F403
from coulson import checkpoint as _src  # noqa: F401

def __getattr__(name):
    return getattr(_src, name)
