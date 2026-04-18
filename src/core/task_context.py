"""Backward-compat shim. Real module lives in general_beckman.task_context."""
from general_beckman.task_context import parse_context, set_context  # noqa: F401

__all__ = ["parse_context", "set_context"]
