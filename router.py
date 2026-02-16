# router.py
import litellm
import json
import asyncio
import logging
import time
from config import MODEL_TIERS, CLASSIFIER_MODEL, FALLBACK_ORDER

logger = logging.getLogger(__name__)

# Simple rate limiter
class RateLimiter:
    def __init__(self, calls_per_minute=25):
        self.calls_per_minute = calls_per_minute
        self.timestamps = []

    async def wait(self):
        now = time.time()
        # Remove timestamps older than 60 seconds
        self.timestamps = [t for t in self.timestamps if now - t < 60]

        if len(self.timestamps) >= self.calls_per_minute:
            wait_time = 60 - (now - self.timestamps[0]) + 0.5
            logger.info(f"Rate limiter: waiting {wait_time:.1f}s")
            await asyncio.sleep(wait_time)

        self.timestamps.append(time.time())

# Global rate limiter — Groq free = ~30 requests/min
rate_limiter = RateLimiter(calls_per_minute=20)


ROUTER_PROMPT = """You are a task router. Classify the given task into a complexity tier.
Respond ONLY with valid JSON, no markdown.

Tiers:
- "cheap": Simple/quick tasks - factual questions, formatting, lookups, 
  classification, yes/no, translations, simple math, definitions.
  MOST tasks should be cheap. When in doubt, choose cheap.
- "medium": Moderate tasks - multi-paragraph summaries, content drafting, 
  data analysis, detailed explanations, comparisons
- "expensive": Complex tasks ONLY - multi-step reasoning, full code generation,
  critical business decisions, creative strategy requiring nuance

IMPORTANT: Bias heavily toward "cheap". Only escalate if truly necessary.
A simple question like "what is X" is ALWAYS cheap.

Also determine if human approval is needed:
- needs_approval: true ONLY if task involves external actions, spending money,
  sending communications, or irreversible operations
- Simple information requests NEVER need approval

Task: {task_description}

Respond as: {{"tier": "cheap", "needs_approval": false, "reasoning": "brief"}}"""


async def classify_task(title: str, description: str) -> dict:
    """Use the cheapest model to classify the task tier."""
    await rate_limiter.wait()

    try:
        response = await litellm.acompletion(
            model=CLASSIFIER_MODEL,
            messages=[{
                "role": "user",
                "content": ROUTER_PROMPT.format(
                    task_description=f"{title}: {description}"
                )
            }],
            max_tokens=150,
            temperature=0
        )

        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            raw = raw.rsplit("```", 1)[0]

        result = json.loads(raw)
        tier = result.get("tier", "cheap")

        # Map tier to what's actually available
        tier = _resolve_tier(tier)

        logger.info(f"Classification: '{tier}' | Reason: {result.get('reasoning', 'N/A')}")

        return {
            "tier": tier,
            "needs_approval": result.get("needs_approval", False),
            "reasoning": result.get("reasoning", "")
        }
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        return {
            "tier": _get_cheapest_available_tier(),
            "needs_approval": False,
            "reasoning": f"Classification failed, defaulting to cheapest: {e}"
        }


def _resolve_tier(requested_tier: str) -> str:
    """Map a requested tier to the best available tier."""
    if requested_tier in MODEL_TIERS:
        return requested_tier

    # If expensive requested but not available, try medium, then cheap
    # If medium requested but not available, try cheap
    priority = ["expensive", "medium", "cheap"]
    try:
        start_idx = priority.index(requested_tier)
    except ValueError:
        start_idx = 0

    # Look for the requested tier or the next best one going DOWN
    for i in range(start_idx, len(priority)):
        if priority[i] in MODEL_TIERS:
            if priority[i] != requested_tier:
                logger.info(f"Tier '{requested_tier}' unavailable, using '{priority[i]}'")
            return priority[i]

    return _get_cheapest_available_tier()


def _get_cheapest_available_tier() -> str:
    for tier in reversed(FALLBACK_ORDER):
        if tier in MODEL_TIERS:
            return tier
    return "cheap"


def _get_fallback_tier(current_tier: str) -> str | None:
    try:
        current_idx = FALLBACK_ORDER.index(current_tier)
    except ValueError:
        return _get_cheapest_available_tier()
    for i in range(current_idx + 1, len(FALLBACK_ORDER)):
        candidate = FALLBACK_ORDER[i]
        if candidate in MODEL_TIERS:
            return candidate
    return None


async def call_model(tier: str, messages: list, task_context: str = "") -> dict:
    """Call the appropriate model with rate limiting and retry."""

    # Resolve tier to what's available
    tier = _resolve_tier(tier)
    tier_config = MODEL_TIERS[tier]

    # Retry with backoff for rate limits
    max_retries = 3
    for attempt in range(max_retries):
        try:
            await rate_limiter.wait()

            logger.info(f"Calling {tier_config['model']} (tier: {tier})")

            response = await litellm.acompletion(
                model=tier_config["model"],
                messages=messages,
                max_tokens=tier_config["max_tokens"],
                temperature=0.3
            )

            try:
                cost = litellm.completion_cost(completion_response=response)
            except Exception:
                cost = 0.0

            return {
                "content": response.choices[0].message.content,
                "model": tier_config["model"],
                "tier": tier,
                "cost": cost or 0.0,
                "usage": dict(response.usage) if response.usage else {}
            }

        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit = any(kw in error_str for kw in [
                "rate limit", "rate_limit", "429", "too many requests",
                "tokens per minute"
            ])

            if is_rate_limit and attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5  # 5s, 10s, 15s
                logger.warning(
                    f"Rate limited on {tier_config['model']}, "
                    f"waiting {wait_time}s (attempt {attempt + 1}/{max_retries})"
                )
                await asyncio.sleep(wait_time)
                continue

            # Not a rate limit, or exhausted retries — try fallback
            is_auth_error = any(kw in error_str for kw in [
                "api key", "authentication", "unauthorized"
            ])
            is_billing_error = any(kw in error_str for kw in [
                "credit balance", "billing", "quota", "insufficient"
            ])

            logger.warning(f"Model {tier_config['model']} failed: {e}")

            fallback_tier = _get_fallback_tier(tier)
            if fallback_tier and fallback_tier != tier:
                logger.info(f"Falling back: {tier} → {fallback_tier}")
                return await call_model(fallback_tier, messages, task_context)

            raise RuntimeError(
                f"All models failed. Last error from "
                f"{tier_config['model']}: {e}"
            )

    raise RuntimeError("Exhausted all retries")