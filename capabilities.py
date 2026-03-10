# capabilities.py
"""
Capability dimensions, task profiles, and model-task scoring.

Each model gets a 14-dimension capability vector (0.0–10.0).
Each task/role is a weight vector over the same 14 dimensions.
Model selection = weighted dot product + hard constraint filtering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ─── The 14 Capability Dimensions ────────────────────────────────────────────

class Cap(str, Enum):
    """
    Orthogonal capability dimensions.
    Each measures something distinct and benchmarkable.
    """
    # ── Reasoning & Planning ──
    REASONING           = "reasoning"            # Multi-step logic, CoT, math, formal reasoning
    PLANNING            = "planning"             # Goal decomposition, step ordering, foresight
    ANALYSIS            = "analysis"             # Pattern recognition, root-cause, comparison

    # ── Code ──
    CODE_GENERATION     = "code_generation"      # Writing correct idiomatic code from specs
    CODE_REASONING      = "code_reasoning"       # Reading/tracing code, debugging, defect ID
    SYSTEM_DESIGN       = "system_design"        # Architecture, API design, scalability tradeoffs

    # ── Communication ──
    PROSE_QUALITY       = "prose_quality"         # Clear writing, docs, tone, structure
    INSTRUCTION_ADHERENCE = "instruction_adherence"  # Following complex constraints, format compliance

    # ── Knowledge & Context ──
    DOMAIN_KNOWLEDGE    = "domain_knowledge"     # Breadth/depth across domains, API awareness
    CONTEXT_UTILIZATION = "context_utilization"   # Long-input handling, retrieval, coherent long output

    # ── Agent / Integration ──
    STRUCTURED_OUTPUT   = "structured_output"    # JSON/XML/schema reliability, parseable first try
    TOOL_USE            = "tool_use"             # Function calling, param extraction, multi-tool

    # ── Multimodal ──
    VISION              = "vision"               # Image understanding, screenshots, diagrams, OCR

    # ── Interaction ──
    CONVERSATION        = "conversation"         # Multi-turn coherence, persona, memory, empathy


ALL_CAPABILITIES = [c.value for c in Cap]

# Dimensions where total params matter more than active params (MoE)
KNOWLEDGE_DIMENSIONS = {
    Cap.DOMAIN_KNOWLEDGE, Cap.PROSE_QUALITY, Cap.VISION,
}

# Dimensions where active params dominate (speed of inference doesn't help
# quality directly, but smaller active-param models genuinely reason worse)
REASONING_DIMENSIONS = {
    Cap.REASONING, Cap.PLANNING, Cap.ANALYSIS,
    Cap.CODE_REASONING, Cap.SYSTEM_DESIGN,
}

# Everything else uses active params directly
EXECUTION_DIMENSIONS = {
    Cap.CODE_GENERATION, Cap.INSTRUCTION_ADHERENCE,
    Cap.CONTEXT_UTILIZATION, Cap.STRUCTURED_OUTPUT,
    Cap.TOOL_USE, Cap.CONVERSATION,
}


# ─── Task Profiles ──────────────────────────────────────────────────────────
# Each task is a weight vector over the 14 capabilities.
# Weights are 0.0–1.0 indicating importance of that dimension for the task.

TASK_PROFILES: dict[str, dict[str, float]] = {
    "planner": {
        Cap.REASONING:             0.9,
        Cap.PLANNING:              1.0,
        Cap.ANALYSIS:              0.6,
        Cap.CODE_GENERATION:       0.1,
        Cap.CODE_REASONING:        0.3,
        Cap.SYSTEM_DESIGN:         0.7,
        Cap.PROSE_QUALITY:         0.3,
        Cap.INSTRUCTION_ADHERENCE: 0.8,
        Cap.DOMAIN_KNOWLEDGE:      0.5,
        Cap.CONTEXT_UTILIZATION:   0.6,
        Cap.STRUCTURED_OUTPUT:     0.8,
        Cap.TOOL_USE:              0.1,
        Cap.VISION:                0.0,
        Cap.CONVERSATION:          0.1,
    },
    "architect": {
        Cap.REASONING:             0.8,
        Cap.PLANNING:              0.8,
        Cap.ANALYSIS:              0.7,
        Cap.CODE_GENERATION:       0.3,
        Cap.CODE_REASONING:        0.6,
        Cap.SYSTEM_DESIGN:         1.0,
        Cap.PROSE_QUALITY:         0.5,
        Cap.INSTRUCTION_ADHERENCE: 0.5,
        Cap.DOMAIN_KNOWLEDGE:      0.9,
        Cap.CONTEXT_UTILIZATION:   0.7,
        Cap.STRUCTURED_OUTPUT:     0.5,
        Cap.TOOL_USE:              0.1,
        Cap.VISION:                0.2,
        Cap.CONVERSATION:          0.1,
    },
    "coder": {
        Cap.REASONING:             0.5,
        Cap.PLANNING:              0.3,
        Cap.ANALYSIS:              0.3,
        Cap.CODE_GENERATION:       1.0,
        Cap.CODE_REASONING:        0.5,
        Cap.SYSTEM_DESIGN:         0.3,
        Cap.PROSE_QUALITY:         0.2,
        Cap.INSTRUCTION_ADHERENCE: 0.9,
        Cap.DOMAIN_KNOWLEDGE:      0.7,
        Cap.CONTEXT_UTILIZATION:   0.5,
        Cap.STRUCTURED_OUTPUT:     0.4,
        Cap.TOOL_USE:              0.3,
        Cap.VISION:                0.1,
        Cap.CONVERSATION:          0.0,
    },
    "implementer": {
        Cap.REASONING:             0.4,
        Cap.PLANNING:              0.3,
        Cap.ANALYSIS:              0.3,
        Cap.CODE_GENERATION:       0.9,
        Cap.CODE_REASONING:        0.5,
        Cap.SYSTEM_DESIGN:         0.2,
        Cap.PROSE_QUALITY:         0.1,
        Cap.INSTRUCTION_ADHERENCE: 1.0,
        Cap.DOMAIN_KNOWLEDGE:      0.6,
        Cap.CONTEXT_UTILIZATION:   0.8,
        Cap.STRUCTURED_OUTPUT:     0.5,
        Cap.TOOL_USE:              0.4,
        Cap.VISION:                0.0,
        Cap.CONVERSATION:          0.0,
    },
    "fixer": {
        Cap.REASONING:             0.8,
        Cap.PLANNING:              0.3,
        Cap.ANALYSIS:              0.9,
        Cap.CODE_GENERATION:       0.6,
        Cap.CODE_REASONING:        1.0,
        Cap.SYSTEM_DESIGN:         0.3,
        Cap.PROSE_QUALITY:         0.2,
        Cap.INSTRUCTION_ADHERENCE: 0.6,
        Cap.DOMAIN_KNOWLEDGE:      0.6,
        Cap.CONTEXT_UTILIZATION:   0.8,
        Cap.STRUCTURED_OUTPUT:     0.3,
        Cap.TOOL_USE:              0.3,
        Cap.VISION:                0.2,
        Cap.CONVERSATION:          0.0,
    },
    "test_generator": {
        Cap.REASONING:             0.6,
        Cap.PLANNING:              0.4,
        Cap.ANALYSIS:              0.8,
        Cap.CODE_GENERATION:       0.9,
        Cap.CODE_REASONING:        0.9,
        Cap.SYSTEM_DESIGN:         0.2,
        Cap.PROSE_QUALITY:         0.2,
        Cap.INSTRUCTION_ADHERENCE: 0.7,
        Cap.DOMAIN_KNOWLEDGE:      0.6,
        Cap.CONTEXT_UTILIZATION:   0.7,
        Cap.STRUCTURED_OUTPUT:     0.4,
        Cap.TOOL_USE:              0.2,
        Cap.VISION:                0.0,
        Cap.CONVERSATION:          0.0,
    },
    "reviewer": {
        Cap.REASONING:             0.7,
        Cap.PLANNING:              0.3,
        Cap.ANALYSIS:              1.0,
        Cap.CODE_GENERATION:       0.2,
        Cap.CODE_REASONING:        0.9,
        Cap.SYSTEM_DESIGN:         0.6,
        Cap.PROSE_QUALITY:         0.7,
        Cap.INSTRUCTION_ADHERENCE: 0.5,
        Cap.DOMAIN_KNOWLEDGE:      0.8,
        Cap.CONTEXT_UTILIZATION:   0.8,
        Cap.STRUCTURED_OUTPUT:     0.5,
        Cap.TOOL_USE:              0.1,
        Cap.VISION:                0.3,
        Cap.CONVERSATION:          0.0,
    },
    "researcher": {
        Cap.REASONING:             0.7,
        Cap.PLANNING:              0.4,
        Cap.ANALYSIS:              0.9,
        Cap.CODE_GENERATION:       0.1,
        Cap.CODE_REASONING:        0.2,
        Cap.SYSTEM_DESIGN:         0.2,
        Cap.PROSE_QUALITY:         0.7,
        Cap.INSTRUCTION_ADHERENCE: 0.5,
        Cap.DOMAIN_KNOWLEDGE:      1.0,
        Cap.CONTEXT_UTILIZATION:   0.9,
        Cap.STRUCTURED_OUTPUT:     0.3,
        Cap.TOOL_USE:              0.6,
        Cap.VISION:                0.3,
        Cap.CONVERSATION:          0.1,
    },
    "writer": {
        Cap.REASONING:             0.3,
        Cap.PLANNING:              0.4,
        Cap.ANALYSIS:              0.3,
        Cap.CODE_GENERATION:       0.1,
        Cap.CODE_REASONING:        0.1,
        Cap.SYSTEM_DESIGN:         0.1,
        Cap.PROSE_QUALITY:         1.0,
        Cap.INSTRUCTION_ADHERENCE: 0.8,
        Cap.DOMAIN_KNOWLEDGE:      0.5,
        Cap.CONTEXT_UTILIZATION:   0.6,
        Cap.STRUCTURED_OUTPUT:     0.4,
        Cap.TOOL_USE:              0.1,
        Cap.VISION:                0.1,
        Cap.CONVERSATION:          0.3,
    },
    "executor": {
        Cap.REASONING:             0.4,
        Cap.PLANNING:              0.3,
        Cap.ANALYSIS:              0.2,
        Cap.CODE_GENERATION:       0.3,
        Cap.CODE_REASONING:        0.3,
        Cap.SYSTEM_DESIGN:         0.0,
        Cap.PROSE_QUALITY:         0.1,
        Cap.INSTRUCTION_ADHERENCE: 0.9,
        Cap.DOMAIN_KNOWLEDGE:      0.3,
        Cap.CONTEXT_UTILIZATION:   0.4,
        Cap.STRUCTURED_OUTPUT:     1.0,
        Cap.TOOL_USE:              1.0,
        Cap.VISION:                0.0,
        Cap.CONVERSATION:          0.0,
    },
    "router": {
        Cap.REASONING:             0.5,
        Cap.PLANNING:              0.3,
        Cap.ANALYSIS:              0.6,
        Cap.CODE_GENERATION:       0.0,
        Cap.CODE_REASONING:        0.0,
        Cap.SYSTEM_DESIGN:         0.0,
        Cap.PROSE_QUALITY:         0.1,
        Cap.INSTRUCTION_ADHERENCE: 0.9,
        Cap.DOMAIN_KNOWLEDGE:      0.3,
        Cap.CONTEXT_UTILIZATION:   0.3,
        Cap.STRUCTURED_OUTPUT:     1.0,
        Cap.TOOL_USE:              0.2,
        Cap.VISION:                0.0,
        Cap.CONVERSATION:          0.0,
    },
    "visual_reviewer": {
        Cap.REASONING:             0.5,
        Cap.PLANNING:              0.2,
        Cap.ANALYSIS:              0.8,
        Cap.CODE_GENERATION:       0.1,
        Cap.CODE_REASONING:        0.3,
        Cap.SYSTEM_DESIGN:         0.4,
        Cap.PROSE_QUALITY:         0.6,
        Cap.INSTRUCTION_ADHERENCE: 0.5,
        Cap.DOMAIN_KNOWLEDGE:      0.5,
        Cap.CONTEXT_UTILIZATION:   0.4,
        Cap.STRUCTURED_OUTPUT:     0.4,
        Cap.TOOL_USE:              0.1,
        Cap.VISION:                1.0,
        Cap.CONVERSATION:          0.1,
    },
    "assistant": {
        Cap.REASONING:             0.6,
        Cap.PLANNING:              0.5,
        Cap.ANALYSIS:              0.5,
        Cap.CODE_GENERATION:       0.3,
        Cap.CODE_REASONING:        0.2,
        Cap.SYSTEM_DESIGN:         0.2,
        Cap.PROSE_QUALITY:         0.8,
        Cap.INSTRUCTION_ADHERENCE: 0.7,
        Cap.DOMAIN_KNOWLEDGE:      0.7,
        Cap.CONTEXT_UTILIZATION:   0.6,
        Cap.STRUCTURED_OUTPUT:     0.4,
        Cap.TOOL_USE:              0.5,
        Cap.VISION:                0.3,
        Cap.CONVERSATION:          1.0,
    },
    "summarizer": {
        Cap.REASONING:             0.4,
        Cap.PLANNING:              0.2,
        Cap.ANALYSIS:              0.8,
        Cap.CODE_GENERATION:       0.0,
        Cap.CODE_REASONING:        0.2,
        Cap.SYSTEM_DESIGN:         0.0,
        Cap.PROSE_QUALITY:         0.9,
        Cap.INSTRUCTION_ADHERENCE: 0.8,
        Cap.DOMAIN_KNOWLEDGE:      0.5,
        Cap.CONTEXT_UTILIZATION:   1.0,
        Cap.STRUCTURED_OUTPUT:     0.3,
        Cap.TOOL_USE:              0.0,
        Cap.VISION:                0.0,
        Cap.CONVERSATION:          0.0,
    },
}


# ─── Scoring ─────────────────────────────────────────────────────────────────

@dataclass
class TaskRequirements:
    """Hard constraints that must be met for a model to be eligible."""
    task_name: str
    min_context: int = 0
    needs_function_calling: bool = False
    needs_json_mode: bool = False
    needs_vision: bool = False
    needs_thinking: bool = False
    max_cost_per_1k_output: float = float("inf")
    prefer_local: bool = False
    prefer_fast: bool = False
    latency_sensitive: bool = False
    min_capability: Optional[dict[str, float]] = None  # e.g. {"code_generation": 7}


def score_model_for_task(
    model_capabilities: dict[str, float],
    model_operational: dict,
    requirements: TaskRequirements,
) -> float:
    """
    Score a model for a given task.
    Returns -1 if hard constraints fail, otherwise 0-10 score.
    """
    ops = model_operational

    # ── Hard constraint filtering ──
    if requirements.needs_function_calling and not ops.get("supports_function_calling", False):
        return -1.0
    if requirements.needs_json_mode and not ops.get("supports_json_mode", False):
        return -1.0
    if requirements.needs_vision and model_capabilities.get(Cap.VISION, 0) < 1.0:
        return -1.0
    if requirements.needs_thinking and not ops.get("thinking_model", False):
        return -1.0
    if requirements.min_context and ops.get("context_length", 0) < requirements.min_context:
        return -1.0
    if ops.get("cost_per_1k_output", 0) > requirements.max_cost_per_1k_output:
        return -1.0
    if requirements.min_capability:
        for cap_name, min_val in requirements.min_capability.items():
            if model_capabilities.get(cap_name, 0) < min_val:
                return -1.0

    # ── Get task profile ──
    task_profile = TASK_PROFILES.get(requirements.task_name)
    if not task_profile:
        # Unknown task — use flat weights
        task_profile = {c: 0.5 for c in ALL_CAPABILITIES}

    # ── Weighted dot product ──
    weighted_sum = 0.0
    weight_total = 0.0
    for dim, weight in task_profile.items():
        dim_key = dim.value if isinstance(dim, Cap) else dim
        if weight > 0:
            cap_score = model_capabilities.get(dim_key, 0.0)
            weighted_sum += cap_score * weight
            weight_total += weight

    if weight_total == 0:
        return 0.0

    base_score = weighted_sum / weight_total  # 0-10 scale

    # ── Soft preference modifiers (±10% max) ──
    modifier = 1.0

    if requirements.prefer_local and ops.get("location") == "local":
        modifier += 0.05
    elif requirements.prefer_local and ops.get("location") == "cloud":
        modifier -= 0.03

    if requirements.prefer_fast:
        tps = ops.get("tokens_per_second", 0)
        if tps > 100:
            modifier += 0.05
        elif tps > 50:
            modifier += 0.02

    if requirements.latency_sensitive:
        cost = ops.get("cost_per_1k_output", 0)
        if cost == 0:  # local = no cost
            modifier += 0.03
        elif cost > 0.01:
            modifier -= 0.05

    # Cost efficiency bonus for equivalent quality
    cost_out = ops.get("cost_per_1k_output", 0)
    if cost_out == 0:
        modifier += 0.02  # free is good
    elif cost_out > 0.05:
        modifier -= 0.02  # expensive penalty (mild)

    return round(min(10.0, base_score * modifier), 2)


def rank_models_for_task(
    models: dict[str, tuple[dict[str, float], dict]],  # name → (capabilities, operational)
    requirements: TaskRequirements,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """
    Rank all models for a task. Returns [(model_name, score), ...] descending.
    """
    scored = []
    for name, (caps, ops) in models.items():
        s = score_model_for_task(caps, ops, requirements)
        if s >= 0:
            scored.append((name, s))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
