# task_classifier.py
"""
Unified Task Classification — LLM-based with keyword fallback.

Returns a TaskClassification dataclass used by the orchestrator to
set agent_type, difficulty, and feature flags before execution.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from src.infra.logging_config import get_logger
from .router import ModelRequirements

logger = get_logger("core.task_classifier")


def _extract_json(text: str) -> dict:
    """Extract JSON from LLM output that may contain think tags, markdown, or preamble."""
    # Strip <think>...</think> blocks (Qwen3/DeepSeek thinking)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Strip markdown code fences
    if "```" in text:
        match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    # Find first { ... } block
    match = re.search(r"\{[^{}]*\}", text)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"No JSON object found in: {text[:200]}")


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
    shopping_sub_intent: str | None = None


# ─── LLM Classification Prompt ────────────────────────────────────────────

CLASSIFIER_PROMPT = """You are a task classifier for an AI agent system. Classify this task.
Respond ONLY with valid JSON, no markdown.

Available agent types:
- "planner": mission decomposition, project planning, step ordering
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
- "analyst": data analysis, feasibility studies, structured evaluation, risk assessment
- "error_recovery": recovering from failures, retrying operations, fallback strategies
- "shopping_advisor": product research, price checks, deal finding, purchase advice, comparisons

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


# ─── Shopping Sub-Intent Detection ────────────────────────────────────────

_SHOPPING_SUB_INTENT_RULES: list[tuple[str, list[str]]] = [
    ("price_check",           ["fiyat", "fiyatı", "price", "how much", "ne kadar"]),
    ("compare",               ["karşılaştır", "compare", "vs ", " vs ", "fark"]),
    ("purchase_advice",       ["should i buy", "almalı mıyım", "tavsiye", "recommendation", "öneri"]),
    ("deal_hunt",             ["en ucuz", "cheapest", "indirim", "kampanya", "deal", "discount"]),
    ("research",              ["araştır", "research", "review", "inceleme"]),
    ("upgrade",               ["upgrade", "yükseltme", "geçiş", "switch from"]),
    ("gift",                  ["hediye", "gift", "gift idea", "hediye fikri"]),
    ("exploration",           ["want to buy", "almak istiyorum", "bakıyorum", "looking for"]),
    ("complaint_return_help", ["iade", "return", "şikayet", "complaint", "arıza", "defect"]),
]


def _classify_shopping_sub_intent(text: str) -> str | None:
    """Determine shopping sub-intent from text. Returns None if not shopping."""
    text_lower = text.lower()
    for sub_intent, keywords in _SHOPPING_SUB_INTENT_RULES:
        if any(kw in text_lower for kw in keywords):
            return sub_intent
    return "exploration"  # default sub-intent for shopping tasks


# ─── Public API ────────────────────────────────────────────────────────────

async def classify_task(title: str, description: str) -> TaskClassification:
    """
    Classify a task. Tries LLM first, falls back to keywords.
    """
    try:
        cls = await _classify_with_llm(title, description)
    except Exception as e:
        logger.warning("llm classification failed fallback to keyword", error=str(e))
        cls = _classify_by_keywords(title, description)

    # Attach shopping sub-intent if classified as shopping_advisor
    if cls.agent_type == "shopping_advisor":
        cls.shopping_sub_intent = _classify_shopping_sub_intent(
            f"{title} {description}"
        )

    return cls


# ─── LLM-Based Classification ─────────────────────────────────────────────

async def _classify_with_llm(title: str, description: str) -> TaskClassification:
    """Classify using the standard router — just another LLM call."""
    reqs = ModelRequirements(
        task="router",
        agent_type="classifier",
        difficulty=3,
        prefer_speed=True,
        needs_json_mode=True,
        priority=3,                    # yield to real work tasks on GPU
        estimated_input_tokens=500,
        estimated_output_tokens=200,
    )

    messages = [{
        "role": "user",
        "content": CLASSIFIER_PROMPT.format(
            task_description=f"{title}: {description[:500]}"
        ),
    }]

    from src.core.llm_dispatcher import get_dispatcher, CallCategory
    response = await get_dispatcher().request(
        CallCategory.OVERHEAD, reqs, messages,
    )

    raw = response.get("content", "").strip()
    result = _extract_json(raw)

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
        "task classified",
        agent_type=cls.agent_type,
        difficulty=cls.difficulty,
        needs_tools=cls.needs_tools,
        needs_vision=cls.needs_vision,
        needs_thinking=cls.needs_thinking,
        local_only=cls.local_only,
        priority=cls.priority,
        confidence=cls.confidence,
    )

    # Record 429s for adaptive rate limiting
    return cls


# ─── Keyword Fallback ─────────────────────────────────────────────────────

_KEYWORD_RULES: list[tuple[str, int, list[str]]] = [
    # (agent_type, difficulty, keywords)
    # Difficulty is a HINT for model quality — keep moderate (3-6) so
    # available models don't get filtered out. Only genuinely hard tasks get 7+.
    ("shopping_advisor", 5, [
        "fiyat", "fiyatı", "price", "how much", "ne kadar", "en ucuz",
        "cheapest", "indirim", "kampanya", "deal", "discount",
        "almak istiyorum", "want to buy", "should i buy", "almalı mıyım",
        "karşılaştır", "compare", "vs ", " vs ", "upgrade", "yükseltme",
        "hediye", "gift", "tavsiye", "recommendation", "öneri",
    ]),
    ("fixer",          5, ["fix", "bug", "error", "debug", "traceback", "crash"]),
    ("error_recovery", 5, ["recover", "retry", "fallback", "roll back", "revert failure"]),
    ("architect",      6, ["architect", "system design", "api design", "scalability"]),
    ("coder",          5, ["implement", "create", "build", "write code", "refactor"]),
    ("implementer",    5, ["follow plan", "implement plan", "step by step", "execute plan"]),
    ("test_generator", 4, ["test", "spec", "coverage", "unit test"]),
    ("reviewer",       5, ["review", "analyze", "audit", "critique"]),
    ("planner",        5, ["plan", "design", "roadmap", "schema", "decompose"]),
    ("visual_reviewer",4, ["screenshot", "image", "visual", "ui review", "layout"]),
    ("writer",         4, ["write", "document", "email", "report", "readme"]),
    ("researcher",     4, ["search", "find", "research", "compare", "look up"]),
    ("analyst",        5, ["analyze", "evaluate", "assess", "feasibility", "risk analysis"]),
    ("summarizer",     3, ["summarize", "tldr", "key points", "condense"]),
    ("executor",       3, ["deploy", "run", "install", "execute", "download"]),
    ("assistant",      3, ["what is", "what does", "how many", "define", "explain"]),
]


def _classify_by_keywords(title: str, description: str) -> TaskClassification:
    """Fast keyword-based classification. Last resort only."""
    text = f"{title} {description}".lower()

    for agent_type, difficulty, keywords in _KEYWORD_RULES:
        if any(kw in text for kw in keywords):
            needs_tools = agent_type in (
                "fixer", "coder", "implementer", "test_generator",
                "executor", "researcher", "error_recovery",
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
