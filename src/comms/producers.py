"""SP4b Plan 3 — crisis/incident/press_kit CPS producers (LLM out of mr_roboto).

Each function builds the verb-specific prompt + a raw_dispatch OVERHEAD spec and
enqueues it as an admitted Beckman task with a durable continuation
(on_complete -> mechanical sink). NO await_inline. Prompts live HERE.
"""
from __future__ import annotations

import time
import uuid

from general_beckman import enqueue  # module-level for test patching
from general_beckman.lanes import LANE_ONESHOT
from src.infra.logging_config import get_logger

logger = get_logger("comms.producers")


def _suffix() -> str:
    return f"{time.monotonic_ns() % 1_000_000:06d}-{uuid.uuid4().hex[:6]}"


def _overhead_spec(title: str, description: str, prompt: str,
                   in_tok: int, out_tok: int) -> dict:
    return {
        "title": title,
        "description": description,
        "agent_type": "reviewer",
        "kind": "overhead",
        "priority": 2,
        "context": {"llm_call": {
            "raw_dispatch": True,
            "call_category": "overhead",
            "task": "reviewer",
            "agent_type": "reviewer",
            "difficulty": 3,
            "messages": [{"role": "user", "content": prompt}],
            "failures": [],
            "estimated_input_tokens": in_tok,
            "estimated_output_tokens": out_tok,
        }},
    }

# enqueue_* functions added in Tasks 2-4.
