"""Layered resolution: resolve tasks via API registry before LLM dispatch.

Layer 0 (try_resolve): Full resolution — call API, format, return answer. No LLM.
Layer 1 (enrich_context): Partial match — fetch data, return as context for agent.
"""

import re
import time
import logging

from src.tools.free_apis import (
    find_apis,
    call_api,
    get_api,
    TURKISH_CATEGORY_PATTERNS,
    tokenize_api_description,
)

logger = logging.getLogger(__name__)

_LAYER0_THRESHOLD = 0.6
_LAYER1_THRESHOLD = 0.3


async def try_resolve(task: dict) -> str | None:
    """Layer 0: Try to fully resolve a task via API fast-path."""
    task_text = f"{task.get('title', '')} {task.get('description', '')}".strip()
    if not task_text:
        return None

    try:
        match = await _find_best_match(task_text)
        if not match or match["score"] < _LAYER0_THRESHOLD:
            return None

        api = match["api"]
        params = _extract_params(task_text, match["category"])

        start = time.time()
        raw = await _call_best_api(api, params)
        elapsed_ms = int((time.time() - start) * 1000)

        if not raw:
            return None

        formatted = _format_response(raw, match["category"], api.name)

        try:
            from src.infra.db import log_smart_search, record_api_call
            await log_smart_search(task_text, layer=0, source=api.name, success=True, response_ms=elapsed_ms)
            await record_api_call(api.name, success=True)
        except Exception:
            pass

        logger.info("fast-path resolved via %s (category=%s, ms=%d)", api.name, match["category"], elapsed_ms)
        return formatted

    except Exception as exc:
        logger.info("fast-path failed, falling through: %s", exc)
        return None


async def enrich_context(task: dict) -> str | None:
    """Layer 1: Fetch relevant API data as context for the agent."""
    task_text = f"{task.get('title', '')} {task.get('description', '')}".strip()
    if not task_text:
        return None

    try:
        match = await _find_best_match(task_text)
        if not match or match["score"] < _LAYER1_THRESHOLD:
            return None

        api = match["api"]
        params = _extract_params(task_text, match["category"])

        start = time.time()
        raw = await _call_best_api(api, params)
        elapsed_ms = int((time.time() - start) * 1000)

        if not raw:
            return None

        formatted = _format_response(raw, match["category"], api.name)

        try:
            from src.infra.db import log_smart_search, record_api_call
            await log_smart_search(task_text, layer=1, source=api.name, success=True, response_ms=elapsed_ms)
            await record_api_call(api.name, success=True)
        except Exception:
            pass

        return f"### Available Data\n{formatted}\n(Source: {api.name}, fetched just now)"

    except Exception as exc:
        logger.debug("context enrichment failed: %s", exc)
        return None


async def _find_best_match(task_text: str) -> dict | None:
    """Find the best matching API using keywords + Turkish patterns."""
    task_lower = task_text.lower()
    best = None

    # 1. Turkish category patterns (strong signal)
    try:
        from src.infra.db import get_api_category_patterns
        db_patterns = await get_api_category_patterns()
    except Exception:
        db_patterns = {}

    all_patterns = {**TURKISH_CATEGORY_PATTERNS, **db_patterns}

    for category, pattern in all_patterns.items():
        try:
            if re.search(pattern, task_lower, re.IGNORECASE):
                apis = find_apis(category=category)
                api = await _pick_most_reliable(apis)
                if api:
                    score = 0.8
                    if not best or score > best["score"]:
                        best = {"api": api, "category": category, "score": score}
        except re.error:
            continue

    # 2. Keyword index matching
    try:
        from src.infra.db import find_apis_by_keywords
        task_keywords = tokenize_api_description(task_text)
        if task_keywords:
            matches = await find_apis_by_keywords(task_keywords, limit=3)
            for m in matches:
                score = m["match_count"] / max(len(task_keywords), 1)
                score = min(score * 1.5, 1.0)
                api = get_api(m["api_name"])
                if api and (not best or score > best["score"]):
                    best = {"api": api, "category": api.category, "score": score}
    except Exception as exc:
        logger.debug("keyword matching failed: %s", exc)

    # 3. Check reliability
    if best:
        try:
            from src.infra.db import get_api_reliability
            rel = await get_api_reliability(best["api"].name)
            if rel:
                if rel["status"] == "suspended":
                    return None
                elif rel["status"] == "demoted":
                    best["score"] *= 0.3
                elif rel["status"] == "warning":
                    best["score"] *= 0.5
        except Exception:
            pass

    return best


async def _pick_most_reliable(apis: list) -> "FreeAPI | None":
    """From a list of APIs in the same category, pick the most reliable."""
    if not apis:
        return None
    if len(apis) == 1:
        return apis[0]

    try:
        from src.infra.db import get_api_reliability
        scored = []
        for api in apis:
            rel = await get_api_reliability(api.name)
            if rel and rel["status"] == "suspended":
                continue
            reliability = 0.5
            if rel:
                total = rel["success_count"] + rel["failure_count"]
                if total > 0:
                    reliability = rel["success_count"] / total
            scored.append((api, reliability))
        if scored:
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[0][0]
    except Exception:
        pass

    return apis[0]


def _extract_params(task_text: str, category: str) -> dict:
    """Extract API call parameters from task text."""
    params = {}
    text_lower = task_text.lower()

    cities = [
        "istanbul", "ankara", "izmir", "bursa", "antalya", "adana",
        "konya", "gaziantep", "mersin", "kayseri", "eskisehir",
        "trabzon", "samsun", "denizli", "diyarbakir", "erzurum",
    ]
    for city in cities:
        if city in text_lower:
            params["city"] = city.capitalize()
            break

    if category == "currency":
        currencies = {
            "dolar": "USD", "dollar": "USD", "usd": "USD",
            "euro": "EUR", "eur": "EUR",
            "sterlin": "GBP", "pound": "GBP", "gbp": "GBP",
            "yen": "JPY", "jpy": "JPY",
        }
        for term, code in currencies.items():
            if term in text_lower:
                params["currency"] = code
                break
        if "currency" not in params:
            params["currency"] = "USD"

    return params


async def _call_best_api(api, params: dict) -> dict | str | None:
    """Call an API with extracted params."""
    endpoint = api.example_endpoint

    city = params.get("city", "Istanbul")
    currency = params.get("currency", "USD")
    endpoint = endpoint.replace("Istanbul", city)
    endpoint = endpoint.replace("USD", currency)

    result = await call_api(api, endpoint=endpoint)
    if not result:
        return None
    return result


def _format_response(raw, category: str, api_name: str) -> str:
    """Format raw API response into clean text."""
    if isinstance(raw, str):
        if len(raw) > 2000:
            raw = raw[:2000] + "..."
        return raw

    if isinstance(raw, dict):
        import json
        return json.dumps(raw, ensure_ascii=False, indent=2)[:2000]

    return str(raw)[:2000]
