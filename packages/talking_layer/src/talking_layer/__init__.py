"""Talking Layer — LLM call execution hub."""
from .types import CallResult, CallError
from .caller import call
__all__ = ["call", "CallResult", "CallError"]
