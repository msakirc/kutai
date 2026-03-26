# Message Classification — Root Cause Analysis

## The Bug (exact trace of misclassification)

User sends: **"How is the coffee machine search going"**

### Path A: LLM classification succeeds (the expected path)

1. Line 1619: `classification = await self._classify_user_message(text)` is called.
2. Line 1805-1813: ModelRequirements uses `difficulty=2, prefer_speed=True, needs_json_mode=True`. This routes to the **smallest/fastest available model** (likely a local Qwen or similar small model).
3. The LLM prompt (line 1757-1789) DOES list `status_query` and even includes the exact example "how is the coffee machine search going". The prompt also has the IMPORTANT note at line 1780.
4. **However**: The small model sees "coffee machine" and "search" in the message. The `shopping` category description says *"buying, comparing prices, finding deals, product recommendations"* — and a coffee machine is clearly a product. A difficulty-2 model with `prefer_speed=True` is likely a tiny model that pattern-matches "coffee machine" -> product -> "shopping" rather than understanding the nuanced distinction that the user is asking about an **existing** search.
5. The LLM returns `{"type": "shopping", "confidence": 0.7}`.
6. Line 1826: `msg_type = "shopping"`, confidence 0.7 > 0.4 threshold.
7. Line 1647-1660: The `shopping` handler creates a **brand new** shopping task with `add_task(agent_type="shopping_advisor")`.

### Path B: LLM classification fails (fallback path)

1. Line 1837-1840: Exception caught, falls to `self._classify_message_by_keywords(text)`.
2. Line 1843: Keyword classifier runs.
3. Line 1854-1869: Status phrases checked FIRST. `"how is the"` is in `_status_phrases`. **This would correctly return `{"type": "status_query"}`.**
4. So the keyword fallback actually WORKS correctly.

### The real failure: the LLM path succeeds with the WRONG answer

The bug is NOT that `status_query` is missing from the router or the keyword fallback. **The keyword fallback is correct.** The problem is:

1. The LLM classification **succeeds** (no exception), so the keyword fallback is never reached.
2. The small/fast model returns `"shopping"` instead of `"status_query"` because it sees "coffee machine" as a product keyword and doesn't properly weigh the "how is...going" framing.
3. There is **no second-opinion or cross-check** — the LLM result is trusted directly.

## Why Previous Fix Failed

The previous Opus agent added:
- `status_query` to the LLM prompt (line 1760) -- **correct but insufficient**
- `status_query` to the keyword fallback (lines 1854-1869) -- **correct but never reached**
- `status_query` to the router (line 1631) -- **correct**
- A `_handle_status_query` handler (line 2067) -- **correct and well-implemented**

The fix was **architecturally complete** — every piece is in place. But it fails because:

1. **The keyword fallback is only reached on LLM exception** (line 1837-1840). The LLM doesn't throw an exception; it returns the wrong classification confidently.
2. The LLM prompt tells the model about `status_query` but a difficulty-2 speed-optimized model doesn't reliably distinguish "asking about a coffee machine search" from "wanting to buy a coffee machine."
3. There is **no hybrid check** — the keyword classifier's knowledge is wasted when the LLM path succeeds.

## All Misclassification Scenarios

| User message | Expected type | Likely LLM result | Why |
|---|---|---|---|
| "How is the coffee machine search going" | status_query | shopping | "coffee machine" triggers product association |
| "Any update on the motherboard" | status_query | shopping | "motherboard" is a product; line 1877 has it as a shopping keyword |
| "Did you find anything for the GPU" | status_query | shopping | "GPU" is literally in shopping keywords at line 1877 |
| "What happened with the laptop comparison" | status_query | shopping | "laptop" is in shopping keywords |
| "How far along is the CPU task" | status_query | shopping | "CPU" in shopping keywords |
| "Tell me about X" | question | mission | Ambiguous; small model may default to mission |
| "Find me a good deal on Y" | shopping | status_query (false positive possible) | If "find me" is about checking existing search |
| "What's the status of the phone search" | status_query | shopping | "phone"/"telefon" in shopping keywords |

**Critical pattern**: Every product-related status query is at risk because the LLM sees the product noun and classifies as shopping. The shopping keyword list (line 1877) includes `motherboard, cpu, gpu, laptop, phone, telefon` — these are the exact items users are most likely to ask status about.

## Root Cause

**The architecture is "LLM-first, keywords-only-on-exception."** This means:

1. High-quality keyword rules (which correctly handle status queries) are **dead code** in the happy path.
2. The LLM model used (difficulty=2, prefer_speed=True) is too weak to reliably distinguish status-about-products from new-product-requests.
3. There is no confidence threshold where the system falls back to keywords — only a 0.4 floor that converts to generic "task" (line 1831-1832), not to keyword fallback.
4. The low-confidence fallback returns `{"type": "task"}` (line 1832) instead of consulting the keyword classifier.

## Proposed Fix (with exact code changes needed)

### Fix 1: Keywords-first for high-confidence patterns, LLM for ambiguous cases (RECOMMENDED)

Change `_handle_free_text` (around line 1616-1619) to run keyword classification FIRST, and only use LLM when keywords return a low-confidence/generic result.

**At line 1616-1620, replace:**
```python
# PRIORITY 2: LLM-based message classification
classification = await self._classify_user_message(text)
msg_type = classification["type"]
msg_workflow = classification.get("workflow")
```

**With:**
```python
# PRIORITY 2: Keyword pre-check for high-confidence patterns
keyword_result = self._classify_message_by_keywords(text)
keyword_type = keyword_result["type"]

# High-confidence keyword matches skip the LLM entirely
_KEYWORD_AUTHORITATIVE_TYPES = {
    "status_query", "todo", "load_control", "bug_report",
    "feature_request", "casual",
}
if keyword_type in _KEYWORD_AUTHORITATIVE_TYPES:
    classification = keyword_result
else:
    # Ambiguous — use LLM classification
    classification = await self._classify_user_message(text)

msg_type = classification["type"]
msg_workflow = classification.get("workflow")
```

**Why this works**: Status queries have very reliable keyword patterns ("how is the X going", "any update on", "what's the status of"). These should be authoritative. The LLM is only needed for ambiguous cases like distinguishing "mission" from "task" from "shopping" for genuinely new requests.

### Fix 2: Hybrid classification — use keywords as a tiebreaker (ALTERNATIVE)

Change `_classify_user_message` (around line 1826-1836) to cross-check the LLM result against keywords when the LLM returns "shopping" or "mission" but keywords say "status_query".

**After line 1826 (`msg_type = result.get("type", "task")`), add:**
```python
# Cross-check: if keywords strongly say status_query but LLM says
# shopping/mission, trust keywords (product names confuse small models)
keyword_result = self._classify_message_by_keywords(text)
if keyword_result["type"] == "status_query" and msg_type in ("shopping", "mission", "task"):
    logger.info("keyword override: LLM said %s but keywords say status_query", msg_type)
    msg_type = "status_query"
```

### Fix 3: Low-confidence fallback should use keywords, not default to "task"

**At line 1831-1832, replace:**
```python
if confidence < 0.4:
    return {"type": "task"}
```

**With:**
```python
if confidence < 0.4:
    return self._classify_message_by_keywords(text)
```

**This ensures the keyword classifier is consulted when the LLM is uncertain**, rather than blindly defaulting to "task" which falls through to PRIORITY 3 and creates a task.

### Fix 4: Remove product nouns from shopping keyword fallback

**At line 1877, remove the product-specific nouns** that cause false positives in the keyword classifier itself (even though it currently works due to ordering, it's fragile):

```python
# REMOVE these from shopping keywords:
"motherboard", "cpu", "gpu", "laptop", "phone", "telefon",
```

These bare product nouns without buying-intent verbs should NOT force a shopping classification. "GPU" alone doesn't mean shopping — "I want to buy a GPU" does. The LLM can handle the intent distinction for messages that don't match status patterns.

### Recommended implementation order

1. **Fix 3** (5 min, zero risk) — low-confidence LLM uses keywords instead of defaulting to "task"
2. **Fix 1** (10 min, low risk) — keywords-first for authoritative patterns
3. **Fix 4** (5 min, low risk) — remove bare product nouns from shopping keywords
4. Fix 2 is an alternative to Fix 1; implement one or the other, not both.

### Line references (telegram_bot.py)

| What | Line(s) |
|---|---|
| LLM classification called | 1619 |
| msg_type extracted | 1620 |
| Router: status_query handled | 1631-1633 |
| Router: shopping creates new task | 1647-1660 |
| Router: mission creates new mission | 1711-1738 |
| Fallthrough: anything else creates task | 1739-1753 |
| LLM prompt with categories | 1757-1789 |
| LLM call with difficulty=2 | 1805-1813 |
| Low confidence -> "task" default | 1831-1832 |
| Exception -> keyword fallback | 1837-1840 |
| Keyword classifier | 1843-1931 |
| Keyword: status phrases (correct) | 1854-1869 |
| Keyword: shopping (has product nouns) | 1871-1879 |
| _handle_status_query handler | 2067-2177 |
