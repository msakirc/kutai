"""CallResult -> legacy response dict mapping.

Ported VERBATIM from ``LLMDispatcher._result_to_dict`` (SP3b Task 2). The
dispatcher is now a dumb pipe that returns a raw ``hallederiz_kadir.CallResult``;
husam owns the mapping into the legacy dict shape every caller expects.
"""
from __future__ import annotations

from typing import Any


def result_to_dict(result: Any, model: Any) -> dict:
    """Convert a hallederiz_kadir CallResult to the legacy response dict format."""
    return {
        "content": result.content,
        "model": result.model,
        "model_name": result.model_name,
        "cost": result.cost,
        "usage": result.usage,
        "tool_calls": result.tool_calls,
        "latency": result.latency,
        "thinking": result.thinking,
        "is_local": result.is_local,
        "ran_on": "local" if result.is_local else result.provider,
        "provider": result.provider,
        "task": result.task,
        "capability_score": 0.0,
        "difficulty": 5,
    }
