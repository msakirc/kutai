# Inference Stability Fixes — Issue Log

## Active Issues (from 2026-03-29 01:33-02:00 session)

### 1. GPU Queue Cascade Failure (CRITICAL)
Self-reflection tasks pile up in GPU scheduler (queue_depth=10), each timing out at 120s. Backpressure retries create an infinite loop. Triggered when model swap to 35B happens during active inference.
**Status**: Opus agent implementing fix (skip local during swap in _route_overhead)

### 2. --fit VRAM Instability
Benchmark: 25.4 tok/s. Runtime: 12.9 tok/s for same model (Qwen3.5-9B). `--fit` allocates fewer GPU layers when VRAM is tighter at runtime.

### 3. Perplexica Returns "I Don't Know" as Success
`perplexica search ok | answer_len=1956 source_count=0` — Perplexica returned a 1956-char answer that says "no data available, purely speculative fantasy". web_search treated this as success (answer_len > 0), never fell through to SearXNG direct or DuckDuckGo.
**Fix needed**: Quality gate — if Perplexica answer has 0 sources OR contains "no data"/"cannot provide"/"unavailable" phrases, treat as failure and fall through to next backend.

### 4. SearXNG Direct Never Called
The search chain stops at the first "successful" backend. Perplexica returned text (even though useless), so SearXNG and DuckDuckGo were skipped.
**Fix needed**: When Perplexica returns 0 sources, also try SearXNG direct.

### 5. Task Timeout Too Short for Web Search
180s total timeout for researcher agent. Perplexica alone takes 40s + 2 LLM iterations at 12.9 tok/s with large context = timeout.
**Fix needed**: Increase researcher timeout to 300s. Or: if iteration 1 has a result, use it as final answer on timeout instead of failing.

### 6. Timeout Discards Good Checkpoint
The agent completed iteration 1 with web search results. Started iteration 2 for refinement. Timed out at 180s. The checkpoint from iteration 1 was discarded instead of returned as the best-available result.
**Fix needed**: On timeout, check for checkpoint and return the last successful iteration's result.

### 7. chromadb Not Installed
`No module named 'chromadb'` — vector skill search and RAG broken.

### 8. All Models `excluded` for Classifier on Cold Start
OVERHEAD classifier runs before any model loaded. Keyword fallback works but degrades routing quality.
**Fix needed**: Run proactive loader BEFORE processing first task.

### 9. Thinking Mode on 9B Wastes Tokens
`think=YES` in benchmark — model generates reasoning_content even with enable_thinking=false.

### 10. gemma GGUF deleted but scanner still found renamed file
Fixed by deleting file entirely.
