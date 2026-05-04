"""Shim — re-exports from coulson.single_shot. See packages/coulson/."""
from coulson.single_shot import *  # noqa: F401,F403
from coulson import single_shot as _src  # noqa: F401

def __getattr__(name):
    return getattr(_src, name)
