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
    search_depth: str = "none"


# ─── LLM Classification Prompt (now in rubrics/classifier.yaml) ───────────


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


# ─── Search Depth Detection ──────────────────────────────────────────────

_SEARCH_DEPTH_RULES: list[tuple[str, list[str]]] = [
    ("deep", [
        "analyze", "analyse", "research in detail", "in detail", "market analysis",
        "competitor analysis", "in-depth", "comprehensive",
        "multi-source", "synthesize", "synthesis",
    ]),
    ("standard", [
        "price", "fiyat", "fiyatı", "ne kadar", "how much",
        "compare", "karşılaştır", "vs ", " vs ",
        "review", "inceleme", "best ", "en iyi",
        "recommend", "tavsiye", "öneri",
    ]),
    ("none", [
        "write ", "fix ", "implement", "refactor", "debug",
        "create ", "build ", "deploy", "test ",
    ]),
]

# ─── Time-Sensitive Query Detection ─────────────────────────────────────

# Patterns that indicate time-sensitive queries requiring fresh web data.
# Organized by upgrade level: "standard" patterns get higher priority than "quick".

_TIME_SENSITIVE_STANDARD: list[str] = [
    # Match lineups / predicted XIs
    "predicted xi", "lineup", "kadro", "ilk 11", "ilk onbir",
    # Live scores / match results
    "score", "skor", "maç", "match result",
    # Financial prices (change frequently)
    "stock price", "share price", "exchange rate", "dolar kuru",
    "altın fiyatı", "euro kuru", "borsa",
]

_TIME_SENSITIVE_QUICK: list[str] = [
    # Temporal words (English)
    "today", "tonight", "tomorrow", "next week", "this week", "this month",
    "next monday", "next tuesday", "next wednesday", "next thursday",
    "next friday", "next saturday", "next sunday", "this weekend",
    # Temporal words (Turkish)
    "bugün", "yarın", "bu hafta", "gelecek hafta", "bu akşam",
    "bu ay", "gelecek ay", "bu gece", "haftaya",
    # Current/live events
    "weather", "hava durumu",
    # Recency markers (English)
    "latest", "current", "recent", "right now", "live", "breaking",
    # Recency markers (Turkish)
    "güncel", "son dakika", "şu an", "anlık", "şu anda",
]

# Depth ordering for comparison
_DEPTH_ORDER = {"none": 0, "quick": 1, "standard": 2, "deep": 3}


def _apply_time_sensitivity(text_lower: str, depth: str) -> str:
    """Upgrade search depth if the query contains time-sensitive patterns.

    - _TIME_SENSITIVE_STANDARD patterns upgrade to at least "standard"
    - _TIME_SENSITIVE_QUICK patterns upgrade to at least "quick"
    """
    current = _DEPTH_ORDER.get(depth, 0)

    # Check standard-level patterns first (higher upgrade)
    if current < _DEPTH_ORDER["standard"]:
        if any(kw in text_lower for kw in _TIME_SENSITIVE_STANDARD):
            return "standard"

    # Check quick-level patterns
    if current < _DEPTH_ORDER["quick"]:
        if any(kw in text_lower for kw in _TIME_SENSITIVE_QUICK):
            return "quick"

    return depth


def _classify_search_depth(text: str) -> str:
    """Classify how much web search depth a task needs.

    Returns: "deep", "standard", "quick", or "none".
    Applies time-sensitivity upgrades after initial classification.
    """
    text_lower = text.lower()

    # Initial classification from keyword rules
    depth = "none"  # default: no web search unless keywords match
    for d, keywords in _SEARCH_DEPTH_RULES:
        if any(kw in text_lower for kw in keywords):
            depth = d
            break

    # Upgrade if time-sensitive patterns detected
    depth = _apply_time_sensitivity(text_lower, depth)

    return depth


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

async def _enqueue_inline_classifier(
    *,
    title: str,
    description: str,
    llm_call_kwargs: dict,
) -> dict:
    """Enqueue a classifier LLM call as a Beckman task with await_inline=True.

    Returns the dispatcher response dict so callers can do
    `response.get("content", ...)` unchanged.
    """
    import general_beckman
    spec = {
        "title": title,
        "description": description,
        "agent_type": "classifier",
        "kind": "classifier",
        "context": {
            "llm_call": {
                "raw_dispatch": True,
                **llm_call_kwargs,
            },
        },
    }
    # SP5-DEFERRED: this is the one edge-group await_inline site SP2 keeps,
    # because classify_task's caller (add_task) consumes the returned
    # TaskClassification synchronously. CPS-migrating this requires
    # redesigning task admission — see SP2 spec §Site 2 special case.
    tr = await general_beckman.enqueue(spec, parent_id=None, await_inline=True)
    if tr.status == "failed":
        from src.core.router import ModelCallFailed
        raise ModelCallFailed(tr.error or "call failed", error_category="availability")
    res = tr.result
    if isinstance(res, str):
        try:
            res = json.loads(res)
        except Exception:
            res = {"content": res}
    return res or {}


async def _classify_with_llm(title: str, description: str) -> TaskClassification:
    """Classify using the standard router — just another LLM call."""
    from prompt_foundry import build_messages
    _msgs = build_messages("classifier", {
        "task_description": f"{title}: {description[:500]}",
    })
    # Original sends a single user message (no system) — preserve that structure.
    messages = [_msgs[1]]

    response = await _enqueue_inline_classifier(
        title="task-classifier",
        description=f"Classify task: {title[:80]!r}",
        llm_call_kwargs={
            "task": "router",
            "agent_type": "classifier",
            "difficulty": 3,
            "messages": messages,
            "prefer_speed": True,
            "needs_json_mode": True,
            "priority": 3,
            "estimated_input_tokens": 500,
            "estimated_output_tokens": 200,
            "call_category": "overhead",
        },
    )

    raw = response.get("content", "").strip()
    result = _extract_json(raw)

    search_depth = result.get("search_depth") or _classify_search_depth(title + " " + description)

    agent_type = result.get("agent_type", "executor")

    # Only visual_reviewer actually uses analyze_image. The LLM classifier
    # often over-tags needs_vision for tasks mentioning "UI" or "design".
    # A false needs_vision triggers a 60s server restart to load the 876MB
    # mmproj projector — extremely wasteful for text-only work.
    needs_vision = result.get("needs_vision", False) and agent_type == "visual_reviewer"

    cls = TaskClassification(
        agent_type=agent_type,
        difficulty=max(1, min(10, int(result.get("difficulty", 5)))),
        needs_tools=result.get("needs_tools", False),
        needs_vision=needs_vision,
        needs_thinking=result.get("needs_thinking", False),
        local_only=result.get("local_only", False),
        priority=PRIORITY_MAP.get(result.get("priority", "normal"), 5),
        confidence=0.85,
        method="llm",
        search_depth=search_depth,
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
    #
    # ORDER MATTERS: first match wins. Put more specific agents before generic ones.
    # Rule: shopping > fixer > analyst > architect > test_generator > coder >
    #       implementer > summarizer > grader > reviewer > planner >
    #       visual_reviewer > researcher > writer > assistant > executor
    ("shopping_advisor", 5, [
        "fiyat", "fiyatı", "price", "how much", "ne kadar", "en ucuz",
        "cheapest", "indirim", "kampanya", "deal", "discount",
        "almak istiyorum", "want to buy", "should i buy", "almalı mıyım",
        "karşılaştır", " vs. ", " vs ", "upgrade", "yükseltme",
        "hediye", "gift", "tavsiye", "recommendation", "öneri",
        # English price patterns
        "under 5000", "under tl", "tl budget", "coffee machine",
        # Turkish product category nouns (bare product names = shopping)
        "makinesi", "makinası", "makine", "telefon", "laptop", "tablet",
        "buzdolabı", "bulaşık", "çamaşır", "kulaklık", "hoparlör",
        "monitör", "klavye", "mouse", "ayakkabı", "kahve", "espresso",
        "klima", "süpürge", "fırın", "ocak", "televizyon",
    ]),
    ("fixer",          5, ["fix ", "bug", "error", "debug", "traceback", "crash"]),
    # analyst before reviewer so "analyze" doesn't accidentally match reviewer
    ("analyst",        5, ["analyze ", "analyse ", "feasibility", "risk analysis",
                            "fee structure", "evaluate ", "assess ", "structured report"]),
    ("architect",      6, ["architect", "system design", "api design", "scalability"]),
    # test_generator before coder so "write tests" doesn't hit coder via "write code"
    ("test_generator", 4, ["write tests", "add tests", "unit test", "write pytest",
                            "test coverage", "edge case"]),
    # coder: explicit code-construction phrases; avoid bare "write" (→ writer)
    ("coder",          5, ["implement ", "create ", "build ", "write code",
                            "write a parser", "write a script", "write a module",
                            "make a ", "refactor", "new project"]),
    ("implementer",    5, ["follow plan", "implement plan", "step by step", "execute plan",
                            "implement the ", "from architecture.md", "from spec"]),
    ("summarizer",     3, ["summarize", "tldr", "key points", "condense", "shorten"]),
    ("reviewer",       5, ["review", "audit", "critique", "check for issues"]),
    ("planner",        5, ["plan ", "roadmap", "schema", "decompose", "break down",
                            "step ordering", "subtasks"]),
    ("visual_reviewer",4, ["screenshot", "ui image", "visual", "ui review", "layout image"]),
    ("researcher",     4, ["research ", "search for", "find information", "look up",
                            "find out about"]),
    ("writer",         4, ["write ", "write blog", "draft ", "document", "email", "readme"]),
    ("assistant",      3, ["what is", "what's the", "what does", "how many",
                            "define ", "explain ", "capital of", "who is"]),
    ("executor",       3, ["deploy", "run ", "install", "execute", "download"]),
]


def _classify_by_keywords(title: str, description: str) -> TaskClassification:
    """Fast keyword-based classification. Last resort only."""
    text = f"{title} {description}".lower()
    search_depth = _classify_search_depth(text)

    for agent_type, difficulty, keywords in _KEYWORD_RULES:
        if any(kw in text for kw in keywords):
            needs_tools = agent_type in (
                "fixer", "coder", "implementer", "test_generator",
                "executor", "researcher",
            )
            return TaskClassification(
                agent_type=agent_type,
                difficulty=difficulty,
                needs_tools=needs_tools,
                needs_vision=(agent_type == "visual_reviewer"),
                confidence=0.4,
                method="keyword",
                search_depth=search_depth,
            )

    return TaskClassification(
        agent_type="executor",
        difficulty=5,
        confidence=0.3,
        method="keyword_default",
        search_depth=search_depth,
    )
