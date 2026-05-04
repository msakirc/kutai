"""Shim — re-exports from coulson.validation. See packages/coulson/."""
from coulson.validation import *  # noqa: F401,F403
from coulson import validation as _src  # noqa: F401

def __getattr__(name):
    return getattr(_src, name)
