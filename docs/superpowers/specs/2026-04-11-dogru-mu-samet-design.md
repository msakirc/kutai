# Content Quality Module — Design Spec

## Problem

Local LLMs (9B models) produce degenerate output: repeated markdown sections, oversized files (56K+), low-entropy garbage, raw envelope leaks. The current system has scattered, narrow checks — header-only repetition detection in hooks.py, a 10-char auto-fail in grading.py — that miss most degenerate patterns. No single module owns content quality, so gaps compound: bad output gets stored, memorized, summarized, and injected into downstream tasks.

## Solution

A standalone, zero-dependency content quality package (`packages/dogru_mu_samet/`) that assesses any text — agent answers, file content, streaming buffers, summaries, memories — and returns a structured quality result. KutAI integrates it at 13 surgical points across the execution chain.

## Package: `packages/dogru_mu_samet/`

### Structure

```
packages/dogru_mu_samet/
  pyproject.toml
  src/dogru_mu_samet/
    __init__.py          # re-exports: assess, salvage, make_stream_callback, ContentQualityResult
    assessor.py          # ContentQualityResult dataclass + assess() orchestrator
    salvager.py          # salvage() — section deduplication
    detectors.py         # individual detection functions
    streaming.py         # make_stream_callback() factory
```

### Dependencies

Zero external dependencies. Pure stdlib: `re`, `collections`, `math`, `dataclasses`.

### Public API

```python
def assess(text: str, max_size: int = 20_000) -> ContentQualityResult:
    """Run all heuristic checks on any text. Returns structured result.
    
    No side effects. No LLM calls. Pure string analysis.
    """

def salvage(text: str) -> str:
    """Deduplicate repeated sections, return cleaned text.
    
    Keeps first occurrence of each normalized section header.
    Returns empty string if nothing structurally complete survives
    (at least one ## Header with content underneath required).
    """

def make_stream_callback(
    max_size: int = 20_000,
    check_interval: int = 4096,
) -> Callable[[str], bool]:
    """Returns callback(accumulated_text) -> should_abort.
    
    Runs assess() every check_interval chars. Returns True (abort) 
    when is_degenerate becomes True.
    """
```

### ContentQualityResult

```python
@dataclass
class ContentQualityResult:
    size: int                       # len(text)
    max_size: int                   # ceiling used for this assessment
    repetition_ratio: float         # 0.0-1.0, header-level (## sections)
    paragraph_repetition: float     # 0.0-1.0, paragraph-block level
    token_entropy: float            # Shannon entropy in bits on whitespace tokens
    is_degenerate: bool             # True if ANY threshold breached
    reasons: list[str]              # e.g. ["size_exceeded", "header_repetition"]

    @property
    def summary(self) -> str:
        """One-line human-readable summary for logs and error messages.
        
        Example: "degenerate: size_exceeded (34201 > 20000), header_repetition (0.62)"
        """
```

### Detectors (`detectors.py`)

Four pure functions, each returns a tuple of `(score, breached: bool, reason_tag: str | None)`:

| Function | Catches | Method | Threshold |
|----------|---------|--------|-----------|
| `check_size(text, max_size)` | Oversized output (56K files) | `len(text) > max_size` | 20K default, 50K hard cap |
| `check_header_repetition(text)` | Duplicate `##` sections | Current `_detect_repetition_ratio` logic — split by `\n## `, normalize headers (strip "summary/examples/notes/details" suffixes), count duplicates via Counter | >0.4 (40% duplicate headers) |
| `check_paragraph_repetition(text)` | Copy-paste blocks without headers | Split text into paragraph blocks (double-newline separated), hash each block (normalized whitespace), count blocks sharing hash with 2+ others | >0.3 (30%+ paragraph blocks duplicated) |
| `check_token_entropy(text)` | Low-diversity garbage ("the the the...") | Shannon entropy on whitespace-split tokens: `-sum(p * log2(p))` over token frequency distribution | <3.0 bits (natural English ~9-10 bits, degenerate <3) |

`check_header_repetition` requires 5+ sections to trigger (same as current implementation — avoids false positives on short outputs with a few sections).

`assess()` runs all four, collects results, sets `is_degenerate = any(breached)`.

### Salvager (`salvager.py`)

```python
def salvage(text: str) -> str:
    """Deduplicate repeated markdown sections.
    
    1. Split text by \n## headers
    2. Normalize each header (strip summary/examples/notes suffixes, lowercase)
    3. Keep first occurrence of each normalized header
    4. Reassemble
    5. Return empty string if no complete section survives
       (a complete section = ## Header + at least one non-empty line below it)
    """
```

Only operates on markdown-structured text (the known degenerate pattern from 9B models). Non-markdown text passes through unchanged — `salvage()` on prose returns the original text since there are no `##` headers to deduplicate.

### Streaming Callback (`streaming.py`)

```python
def make_stream_callback(
    max_size: int = 20_000,
    check_interval: int = 4096,
) -> Callable[[str], bool]:
    """Factory for streaming abort callbacks.
    
    Returns a stateful callback that:
    - Tracks last-checked length
    - Every check_interval chars, runs assess() on accumulated text
    - Returns True (abort) when is_degenerate becomes True
    - Always returns True if len(text) > max_size (immediate abort on size)
    """
```

Size check runs on every call (cheap `len()` comparison). Full `assess()` runs only at intervals to avoid overhead on every chunk.

## KutAI Integration: 13 Surgical Points

Each integration replaces an existing inline check OR adds a check where none exists. Pattern is always: `assess()` → act on `is_degenerate`. No new control flow, no restructuring.

### Group 1: hooks.py (5 points)

**1. Final quality gate** (lines 905-917 in `post_execute_workflow_step`)

Replace:
```python
rep = _detect_repetition_ratio(output_value)
if rep > 0.4:
    result["status"] = "failed"
    result["error"] = f"Output is {rep:.0%} repetitive ..."
```

With:
```python
from dogru_mu_samet import assess
step_max = ctx.get("artifact_schema", {}).get("max_output_chars", 20_000)
cq = assess(output_value, max_size=step_max)
if cq.is_degenerate:
    result["status"] = "failed"
    result["error"] = f"Degenerate content rejected: {cq.summary}"
```

**2. Workspace file recovery** (lines 862-876)

Replace: delete degenerate files.

With:
```python
cq = assess(file_content)
if cq.is_degenerate:
    cleaned = salvage(file_content)
    if cleaned:  # has at least one complete section
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(cleaned)
        file_parts.append(cleaned)
    else:
        os.remove(fpath)
    break
```

**3. `_prev_output` sanitization** (lines 563-577)

Replace: discard if header rep > 0.4.

With:
```python
cq = assess(_prev)
if cq.is_degenerate:
    cleaned = salvage(_prev)
    _prev = cleaned if cleaned else None
```

**4. Context injection filter** (lines 611-622)

Replace: skip if header rep > 0.4.

With:
```python
cq = assess(file_content)
if cq.is_degenerate:
    continue  # skip — don't inject garbage into prompt
```

**5a. `_llm_summarize()` output validation** (lines 387-440)

Add after LLM summarization returns, before storing:
```python
cq = assess(summary)
if cq.is_degenerate:
    logger.warning(f"[Workflow Hook] LLM summary degenerate, using structural fallback")
    summary = _structural_summary(content)
```

**5b. `_generate_phase_summary()` output validation** (lines 1137-1201)

Add after phase summary LLM call returns, before storing as `phase_N_summary`:
```python
cq = assess(phase_summary)
if cq.is_degenerate:
    logger.warning(f"[Workflow Hook] Phase summary degenerate, using structural fallback")
    phase_summary = _structural_summary(combined_artifacts)
```

### Group 2: grading.py (2 points)

**6. Pre-grading auto-fail** (lines 188-190)

Add below existing `len < 10` check:
```python
from dogru_mu_samet import assess
cq = assess(str(result_text))
if cq.is_degenerate:
    return GradeResult(passed=False, raw=f"auto-fail: {cq.summary}")
```

Existing `len < 10` check stays. This adds broader detection without changing the existing guard.

**7. Grading prompt — add quality fields**

Add to `GRADING_PROMPT` after existing 8 fields:
```
WELL_FORMED: PASS or FAIL (no repeated sections, no garbage, structurally sound)
COHERENT: PASS or FAIL (output makes logical sense end-to-end)
```

Uses PASS/FAIL keywords (not YES/NO) to match VERDICT format and reduce small-model format confusion. Parsed via existing `_parse_yes_no()` which already handles both YES/NO and PASS/FAIL.

Parsing: add to optional field zone (lines 101-117). If `WELL_FORMED` parses as FAIL, override VERDICT to FAIL regardless. If fields are missing (small model can't handle them), cascade is unaffected — VERDICT still works as the required field.

Add `well_formed` and `coherent` fields to `GradeResult` dataclass (both `Optional[bool]`, default `None`).

Skill extraction (lines 265-299) reads SITUATION/STRATEGY/TOOLS — completely independent. Zero conflict.

### Group 3: router.py (1 point)

**8. Streaming abort callback**

In `_stream_with_partial_buf()`, add optional `on_chunk` parameter:
```python
async def _stream_with_partial_buf(completion_kwargs, partial_buf, on_chunk=None):
    accumulated = ""
    async for chunk in response:
        chunk_text = chunk.choices[0].delta.content or ""
        accumulated += chunk_text
        partial_buf._partial_content = accumulated
        if on_chunk and on_chunk(accumulated):
            break  # callback said abort
```

Router does NOT import dogru_mu_samet. The callback is created upstream by the caller:
```python
from dogru_mu_samet import make_stream_callback
callback = make_stream_callback(max_size=20_000)
```

The callback is threaded through the existing call chain: `LLMDispatcher.request()` accepts an optional `on_chunk` kwarg, passes it to `call_model()`, which passes it to `_stream_with_partial_buf()`. Each layer just forwards it — no logic changes. Non-streaming calls unaffected (callback is optional, defaults to None). The callback is only created for workflow step execution (where degenerate output is most common), not for all LLM calls.

### Group 4: base.py (3 points)

**9. Checkpoint recovery validation**

After recovering partial output from timeout checkpoint, before injecting as `_prev_output`:
```python
from dogru_mu_samet import assess, salvage
cq = assess(recovered_output)
if cq.is_degenerate:
    recovered_output = salvage(recovered_output) or None
```

**10. Dependency result injection**

In `_build_context()` where dependency results are injected (lines 856-862), assess each result:
```python
cq = assess(dep_result)
if cq.is_degenerate:
    dep_result = salvage(dep_result) or "(dependency output was degenerate — skipped)"
```

**11. Self-reflection corrected_result**

In `_self_reflect()` (lines 2828-2881), after parsing reflection output, before accepting corrected_result:
```python
if corrected_result:
    cq = assess(corrected_result)
    if cq.is_degenerate:
        logger.warning("Self-reflection produced degenerate corrected_result, keeping original")
        corrected_result = None  # fall back to original
```

### Group 5: episodic.py (1 point)

**12. Episodic memory storage gate**

In `store_task_result()` (line ~82), before storing the result snippet:
```python
from dogru_mu_samet import assess
cq = assess(result_snippet)
if cq.is_degenerate:
    logger.info("Skipping episodic storage for degenerate result")
    return  # don't memorize garbage
```

### Artifact Schema Extension

Add optional `max_output_chars` field to artifact schema definitions in workflow JSON:
```json
{
  "output_artifacts": {
    "implementation_plan": {
      "type": "markdown",
      "required_sections": ["Overview", "Components"],
      "max_output_chars": 30000
    }
  }
}
```

Default: 20,000. Hard cap: 50,000 (enforced in `assess()` — `max_size = min(max_size, 50_000)`).

### Deprecation

`_detect_repetition_ratio()` in hooks.py gets a deprecation comment:
```python
# DEPRECATED: Use dogru_mu_samet.assess() instead. Kept for reference.
# Replaced by dogru_mu_samet.detectors.check_header_repetition()
```

All 4 callsites in hooks.py switch to `assess()`. Function body preserved but unused.

## What Does NOT Change

- Schema validation flow (`validate_artifact_schema`) — unchanged
- Retry decision logic (`retry.py`, `RetryContext`) — unchanged
- DLQ/quarantine flow — unchanged
- Phase completion checks — unchanged
- Disguised failure detection (`_is_disguised_failure`) — unchanged
- Skill extraction (reads SITUATION/STRATEGY/TOOLS, independent path) — unchanged
- Preference/insight extraction from grading — unchanged
- All existing control flow, status routing, and error handling — unchanged
- `_unwrap_envelope()` — unchanged (dogru_mu_samet operates on already-unwrapped text)

## Testing Strategy

### Package unit tests (`packages/dogru_mu_samet/tests/`)

- `test_detectors.py`: Each detector with known-good and known-bad inputs
  - Header repetition: text with 3x `## Component Usage` sections
  - Paragraph repetition: 5 identical paragraph blocks
  - Token entropy: "the the the the" vs natural English paragraph
  - Size: at/above/below threshold
- `test_assessor.py`: `assess()` with combinations of violations, verify `is_degenerate` and `reasons`
- `test_salvager.py`: Degenerate markdown in, deduplicated markdown out. Verify first occurrence kept. Verify empty return when nothing salvageable.
- `test_streaming.py`: Callback with accumulating text, verify abort triggers at right point

### Integration tests (`tests/`)

- `test_dogru_mu_samet_integration.py`: Verify each of the 13 integration points with mocked `assess()` returning degenerate result, confirm the callsite acts correctly (rejects/salvages/skips)

## Rollout

The package can be developed and unit-tested independently. Integration points can be wired in one at a time — each is independent. Suggested order:

1. Package implementation + unit tests
2. hooks.py integrations (highest impact — prevents garbage propagation)
3. grading.py integrations (catches degenerate output at quality gate)
4. router.py streaming callback (prevents wasted GPU time)
5. base.py integrations (checkpoint, dependency, self-reflection)
6. episodic.py integration (prevents long-term memory poisoning)
