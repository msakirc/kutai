"""Core types: Task, AgentResult, Lane."""
from __future__ import annotations

from enum import Enum
from typing import Any

Task = dict[str, Any]
AgentResult = dict[str, Any]


class Lane(str, Enum):
    LOCAL_LLM = "local_llm"
    CLOUD_LLM = "cloud_llm"
    MECHANICAL = "mechanical"
