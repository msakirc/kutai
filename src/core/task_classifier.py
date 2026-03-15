# task_classifier.py
"""
Unified Task Classification — LLM-based with keyword fallback.

Returns a TaskClassification dataclass used by the orchestrator to
set agent_type, difficulty, and feature flags before execution.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass

import litellm

from .router import ModelRequirements, select_model
from ..models.rate_limiter import get_rate_limit_manager

logger = logging.getLogger(__name__)


@dataclass
class TaskClassification:
    agent_type: str = "executor"
    difficulty: int = 5
    needs_tools: bool = False
    needs_vision: bool = False
    needs_thinking: bool = False
    local_only: bool = False
    priority: int = 5
    confidence: float = 0.5
    method: str = "keyword"


# ─── LLM Classification Prompt ────────────────────────────────────────────

CLASSIFIER_PROMPT = """You are a task classifier for an AI agent system. Classify this task.
Respond ONLY with valid JSON, no markdown.

Available agent types:
- "planner": goal decomposition, project planning, step ordering
- "architect": system design, API design, technology decisions
- "coder": writing new code from specs
- "implementer": following detailed implementation plans exactly
- "fixer": debugging, fixing errors, root cause analysis
- "test_generator": writing tests, edge case identification
- "reviewer": code review, quality analysis, critique
- "researcher": finding information, comparisons, documentation lookup
- "writer": prose, documentation, emails, reports
- "executor": running tools, file operations, simple transformations
- "visual_reviewer": analyzing screenshots, UI review, diagram understanding
- "assistant": general conversation, Q&A, personal assistance
- "summarizer": condensing long content, extracting key points

Determine:
- agent_type: best matching type from above
- difficulty (1-10): how capable the model needs to be.
  1-3: trivial (definitions, formatting, classification)
  4-6: moderate (standard code, summaries, Q&A)
  7-8: complex (multi-file refactoring, architecture, deep analysis)
  9-10: critical (production decisions, novel algorithms, security audits)
- needs_tools: does this need to execute actions (files, shell, search)?
- needs_vision: does this need to look at images/screenshots?
- needs_thinking: does this need deep multi-step reasoning?
- local_only: personal/sensitive data that shouldn't go to cloud?
- priority: "critical" | "high" | "normal" | "low" | "background"

BIAS: Most tasks need difficulty 4-6. Only use 8+ for genuinely complex work.
Default to needs_tools=false unless the task clearly requires execution.
Default to local_only=false unless personal data is explicitly mentioned.

Task: {task_description}

Respond as: {{"agent_type": "coder", "difficulty": 6, "needs_tools": true, "needs_vision": false, "needs_thinking": false, "local_only": false, "priority": "normal"}}"""


PRIORITY_MAP = {
    "critical": 10, "high": 8, "normal": 5, "low": 3, "background": 1,
}


# ─── Public API ────────────────────────────────────────────────────────────

async def classify_task(title: str, description: str) -> TaskClassification:
    """
    Classify a task. Tries LLM first, falls back to keywords.
    """
    try:
        return await _classify_with_llm(title, description)
    except Exception as e:
        logger.warning(f"LLM classification failed ({e}), using keyword fallback")
        return _classify_by_keywords(title, description)


# ─── LLM-Based Classification ─────────────────────────────────────────────

async def _classify_with_llm(title: str, description: str) -> TaskClassification:
    """Classify using a cheap/fast model via select_model."""
    reqs = ModelRequirements(
        task="router",
        difficulty=3,
        prefer_local=True,
        prefer_speed=True,
        estimated_input_tokens=500,
        estimated_output_tokens=200,
    )

    candidates = select_model(reqs)
    if not candidates:
        raise RuntimeError("No models available for classification")

    model = candidates[0].model

    # Prefer cloud if local model isn't loaded (avoid swap for classification)
    if model.is_local and not model.is_loaded:
        cloud = [c for c in candidates if not c.model.is_local]
        if cloud:
            model = cloud[0].model

    # Rate limiting for cloud models
    if not model.is_local:
        rl_manager = get_rate_limit_manager()
        await rl_manager.wait_and_acquire(
            litellm_name=model.litellm_name,
            provider=model.provider,
            estimated_tokens=700,
        )

    completion_kwargs = dict(
        model=model.litellm_name,
        messages=[{
            "role": "user",
            "content": CLASSIFIER_PROMPT.format(
                task_description=f"{title}: {description[:500]}"
            ),
        }],
        max_tokens=200,
        temperature=0,
    )
    if model.api_base:
        completion_kwargs["api_base"] = model.api_base

    response = await asyncio.wait_for(
        litellm.acompletion(**completion_kwargs),
        timeout=30,
    )

    # Record token usage for rate limiting
    if not model.is_local and response.usage:
        total_tokens = (
            (response.usage.prompt_tokens or 0)
            + (response.usage.completion_tokens or 0)
        )
        get_rate_limit_manager().record_tokens(
            model.litellm_name,
            model.provider,
            total_tokens,
        )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        raw = raw.rsplit("```", 1)[0]

    result = json.loads(raw)

    cls = TaskClassification(
        agent_type=result.get("agent_type", "executor"),
        difficulty=max(1, min(10, int(result.get("difficulty", 5)))),
        needs_tools=result.get("needs_tools", False),
        needs_vision=result.get("needs_vision", False),
        needs_thinking=result.get("needs_thinking", False),
        local_only=result.get("local_only", False),
        priority=PRIORITY_MAP.get(result.get("priority", "normal"), 5),
        confidence=0.85,
        method="llm",
    )

    logger.info(
        f"Classified: agent={cls.agent_type}, diff={cls.difficulty}, "
        f"tools={cls.needs_tools}, vision={cls.needs_vision}, "
        f"thinking={cls.needs_thinking}, local={cls.local_only}, "
        f"priority={cls.priority}"
    )

    # Record 429s for adaptive rate limiting
    return cls


# ─── Keyword Fallback ─────────────────────────────────────────────────────

_KEYWORD_RULES: list[tuple[str, int, list[str]]] = [
    # (agent_type, difficulty, keywords)
    ("fixer",          6, ["fix", "bug", "error", "debug", "traceback", "crash"]),
    ("coder",          7, ["implement", "create", "build", "write code", "refactor"]),
    ("test_generator", 5, ["test", "spec", "coverage", "unit test"]),
    ("reviewer",       6, ["review", "analyze", "audit", "critique"]),
    ("planner",        7, ["plan", "design", "architect", "roadmap", "schema"]),
    ("visual_reviewer",5, ["screenshot", "image", "visual", "ui review", "layout"]),
    ("writer",         5, ["write", "document", "email", "report", "readme"]),
    ("researcher",     5, ["search", "find", "research", "compare", "look up"]),
    ("summarizer",     4, ["summarize", "tldr", "key points", "condense"]),
    ("executor",       4, ["deploy", "run", "install", "execute", "download"]),
    ("assistant",      3, ["what is", "what does", "how many", "define", "explain"]),
]


def _classify_by_keywords(title: str, description: str) -> TaskClassification:
    """Fast keyword-based classification. Last resort only."""
    text = f"{title} {description}".lower()

    for agent_type, difficulty, keywords in _KEYWORD_RULES:
        if any(kw in text for kw in keywords):
            needs_tools = agent_type in (
                "fixer", "coder", "test_generator", "executor", "researcher",
            )
            return TaskClassification(
                agent_type=agent_type,
                difficulty=difficulty,
                needs_tools=needs_tools,
                needs_vision=(agent_type == "visual_reviewer"),
                confidence=0.4,
                method="keyword",
            )

    return TaskClassification(
        agent_type="executor",
        difficulty=5,
        confidence=0.3,
        method="keyword_default",
    )
