"""HaLLederiz Kadir — LLM call execution hub."""
import os as _os

# GOOGLE_API_KEY shadow guard — pop BEFORE any litellm import.
#
# litellm's `gemini/<model>` auto-router inspects GOOGLE_API_KEY at
# module-load time and prefers the vertex_ai_beta backend when set,
# regardless of subsequent custom_llm_provider="gemini" overrides on
# individual calls. d3b13c7 added the per-call override but production
# 2026-05-02 11:34 UTC kept seeing `Vertex_ai_betaException` — the env
# var leaked from an unrelated Google CLI tool in the user's shell
# (verified: GOOGLE_API_KEY was set, GEMINI_API_KEY was also set).
#
# Save the value to GOOGLE_API_KEY_SAVED in case any non-litellm code
# in the process needs it later, then remove from os.environ so
# litellm's router doesn't see it. GEMINI_API_KEY is what we route
# gemini/* through anyway.
if "GOOGLE_API_KEY" in _os.environ:
    _os.environ["GOOGLE_API_KEY_SAVED"] = _os.environ.pop("GOOGLE_API_KEY")

from .types import CallResult, CallError
from .caller import call
__all__ = ["call", "CallResult", "CallError"]
