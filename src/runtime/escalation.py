"""Shim — re-exports from coulson.escalation. See packages/coulson/."""
from coulson.escalation import *  # noqa: F401,F403
from coulson import escalation as _src  # noqa: F401

def __getattr__(name):
    return getattr(_src, name)
