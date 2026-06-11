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


async def _enqueue(spec: dict, **kwargs):
    """Thin, monkeypatchable wrapper over general_beckman.enqueue."""
    import general_beckman
    return await general_beckman.enqueue(spec, **kwargs)


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


# ─── LLM Classification Prompt ────────────────────────────────────────────

CLASSIFIER_PROMPT = """You are a task classifier for an AI agent system. Classify this task.
Respond ONLY with valid JSON, no markdown.

AGENT PICK/REJECT RULES (read carefully — overlapping types have explicit tie-breakers):

CODING CLUSTER:
- "coder": Pick for ad-hoc multi-file builds, standalone projects, "build a X", "create a project", "make a small app", writing new code with no existing spec. Has git_commit + run_code. NOT when there's an existing ARCHITECTURE.md or a detailed spec file — that's "implementer".
- "implementer": Pick for "implement <file>", "write the X module per spec", ONE file following an existing ARCHITECTURE.md or detailed spec. No git, no run. NOT for greenfield builds or vague requests.
- "fixer": Pick when description references "fix", "bug", "feedback", "test failure", "review found", "error", "crash", debugging driven by review or test output. NOT for new features.
- "test_generator": Pick for "write tests", "add tests for X", "generate pytest", edge case identification. NOT for running tests.

REVIEW CLUSTER:
- "reviewer": Pick for general code/content quality review, "review this PR", "check for issues", critique, structured review. NOT for numeric scoring.
- "integration_reviewer": Pick for cross-file / cross-module consistency checks — "check integration", "verify signatures match", "cross-module consistency", "caller callee alignment", "migration model alignment", "interface contract", "type contract across modules", "check boundary". NOT for single-file reviews.

RESEARCH & ANALYSIS CLUSTER:
- "researcher": Pick for general web-search synthesis, "research X", "find information about", "look up", non-shopping topics. NOT for structured data analysis.
- "analyst": Pick for structured data/feasibility/contract/risk analysis on non-shopping topics — "analyze", "feasibility study", "risk assessment", "structured report", "fee structure". NOT for simple Q&A.

SHOPPING CLUSTER:
- "shopping_advisor": Pick for Turkish market product queries, "find me a X under Y TL", price checks, deal finding, purchase advice, product comparisons. Bare product nouns (laptop, buzdolabı, kahve makinesi, ayakkabı) count as shopping. NOT for general research.
- "shopping_clarifier": Pick for one-turn clarification of vague shopping intent — typically called when user invokes /shop with no clear product. NOT for actual product search.
- "product_researcher": Pick for deep multi-vendor product research, "research X product in detail", thorough spec comparison across many sources. NOT for quick price checks.
- "deal_analyst": Pick for analyzing existing deal/discount data — "analyze this deal", "is this a good price", "compare these offers". NOT for finding new products.

CONTENT CREATION CLUSTER:
- "writer": Pick for prose, docs, markdown — "write blog post", "write docs", "draft an article", "write an email". NOT for code.
- "summarizer": Pick for condensing long content — "summarize", "TLDR", "shorten", "key points", "condense". NOT for creating new content.

PLANNING & DESIGN CLUSTER:
- "planner": Pick for decomposing missions into ordered subtasks — "plan", "break down", "decompose", "roadmap", "step ordering". NOT for system design.
- "architect": Pick for system design and ARCHITECTURE.md creation — "design the X module", "design the auth system", "system layout", "architecture", "API design". NOT for implementation.

UTILITIES:
- "assistant": Pick for general Q&A, conversation, simple factual questions — "what is", "what's the capital of", "explain", "define". Pick when no other type clearly fits and no execution is needed.
- "executor": Pick for tool-execution fallback — file operations, deploy, run, install, when no specific role fits but tools are needed.
- "visual_reviewer": Pick ONLY when image/screenshot analysis is explicitly required — "analyze this screenshot", "review this UI image", "look at this diagram".

Determine:
- agent_type: best matching type from above rules
- difficulty (1-10): 1-3 trivial, 4-6 moderate, 7-8 complex, 9-10 critical
- needs_tools: does this need to execute actions (files, shell, search)?
- needs_vision: does this need to look at images/screenshots?
- needs_thinking: does this need deep multi-step reasoning?
- local_only: personal/sensitive data that shouldn't go to cloud?
- priority: "critical" | "high" | "normal" | "low" | "background"
- search_depth: "deep" | "standard" | "quick" | "none"

BIAS: Most tasks need difficulty 4-6. Default needs_tools=false, local_only=false.

Task: {task_description}

Respond as: {{"agent_type": "coder", "difficulty": 6, "needs_tools": true, "needs_vision": false, "needs_thinking": false, "local_only": false, "priority": "normal", "search_depth": "none"}}"""


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


# ─── Public API (CPS) ────────────────────────────────────────────────────────

async def classify_task(
    title: str,
    description: str,
    *,
    on_complete: str = "task_classifier.classify.resume",
    cont_state: dict | None = None,
) -> int | None:
    """CPS kickoff: enqueue the classifier LLM child and return its task id.

    The classification result is delivered asynchronously to the named
    ``on_complete`` continuation handler (default: this module's
    ``_classify_resume``), which rebuilds the TaskClassification via
    ``parse_classification``. No synchronous return value, no await_inline.

    Keyword-only callers that want a synchronous classification with no LLM
    can call ``_classify_by_keywords`` directly. ``parse_classification`` is the
    pure mapping used by the resume handler.
    """
    messages = [{
        "role": "user",
        "content": CLASSIFIER_PROMPT.format(
            task_description=f"{title}: {description[:500]}"
        ),
    }]
    state = dict(cont_state or {})
    state.setdefault("title", title)
    state.setdefault("description", description)
    spec = {
        "title": "task-classifier",
        "description": f"Classify task: {title[:80]!r}",
        "agent_type": "classifier",
        "kind": "classifier",
        "context": {"llm_call": {
            "raw_dispatch": True,
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
        }},
    }
    return await _enqueue(spec, on_complete=on_complete, cont_state=state)


def parse_classification(result: dict, *, title: str, description: str) -> "TaskClassification":
    """Map a raw classifier-LLM result dict -> TaskClassification.

    Pure + synchronous (no LLM, no Beckman). On any parse failure, degrades to
    the keyword classifier. This is the intelligence formerly inlined in
    _classify_with_llm's post-await block; extracted so it is unit-testable and
    reusable by the CPS resume handler.
    """
    content = result.get("content", "")
    if isinstance(content, list):
        content = "\n".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        )
    raw = str(content or "").strip()
    try:
        parsed = _extract_json(raw)
    except Exception:
        cls = _classify_by_keywords(title, description)
        if cls.agent_type == "shopping_advisor":
            cls.shopping_sub_intent = _classify_shopping_sub_intent(f"{title} {description}")
        return cls

    search_depth = parsed.get("search_depth") or _classify_search_depth(title + " " + description)
    agent_type = parsed.get("agent_type", "executor")
    # Only visual_reviewer actually uses analyze_image. The LLM classifier
    # often over-tags needs_vision for tasks mentioning "UI" or "design". A
    # false needs_vision triggers a 60s mmproj reload — guard it.
    needs_vision = parsed.get("needs_vision", False) and agent_type == "visual_reviewer"

    cls = TaskClassification(
        agent_type=agent_type,
        difficulty=max(1, min(10, int(parsed.get("difficulty", 5)))),
        needs_tools=parsed.get("needs_tools", False),
        needs_vision=needs_vision,
        needs_thinking=parsed.get("needs_thinking", False),
        local_only=parsed.get("local_only", False),
        priority=PRIORITY_MAP.get(parsed.get("priority", "normal"), 5),
        confidence=0.85,
        method="llm",
        search_depth=search_depth,
    )
    if cls.agent_type == "shopping_advisor":
        cls.shopping_sub_intent = _classify_shopping_sub_intent(f"{title} {description}")
    return cls


def _on_classified(cls: "TaskClassification", state: dict) -> None:
    """Default consumer for a completed classification. No live caller wires a
    real consumer yet (telegram + /task admit typed tasks directly), so this
    logs. Future wiring: pass your own on_complete to classify_task and consume
    the TaskClassification there."""
    logger.info(
        "task classified (cps)",
        agent_type=cls.agent_type, difficulty=cls.difficulty,
        method=cls.method, title=str(state.get("title", ""))[:60],
    )


async def _classify_resume(child_task_id: int, result: dict, state: dict) -> None:
    """Continuation: rebuild the classification from the classifier LLM result."""
    cls = parse_classification(
        result or {},
        title=str(state.get("title", "")),
        description=str(state.get("description", "")),
    )
    _on_classified(cls, state)


def register_continuations() -> None:
    """Register CPS handlers for the task classifier (called at import + by
    general_beckman.continuations.register_startup_handlers)."""
    from general_beckman.continuations import register_resume
    register_resume("task_classifier.classify.resume", _classify_resume)


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


# Register CPS continuation handlers at import time (mirrors the substrate's
# register_startup_handlers contract — module is listed in _HANDLER_MODULES).
register_continuations()
