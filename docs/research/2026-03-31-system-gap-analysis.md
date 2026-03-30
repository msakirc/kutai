# System Gap Analysis (March 2026)

> High-impact gaps identified by comparing KutAI against competitor frameworks.
> Focus on gaps that are fixable with reasonable effort.

## Priority Matrix

| # | Recommendation | Impact | Effort | Status |
|---|---|---|---|---|
| 1 | Enable self-reflection on coder/researcher/writer | High | Trivial | Implementing |
| 2 | Add "task started" Telegram notification | High | Trivial | Implementing |
| 3 | Add elapsed time to progress messages | High | Trivial | Implementing |
| 4 | Check cancellation status each iteration | High | Small | Implementing |
| 5 | Add `/trace <task_id>` command | Medium | Small | TODO |
| 6 | Internet connectivity pre-check before web tasks | Medium | Small | TODO |
| 7 | Tool circuit breaker (disable after repeated failures) | Medium | Small | TODO |
| 8 | Context budget warning when context is filling up | Medium | Small | TODO |
| 9 | Quality grade notification for low scores | Medium | Trivial | TODO |
| 10 | Change permission check to fail-closed | Low | Trivial | TODO |

## Detailed Findings

### Error Recovery (Current: 7/10)
- Three-layer architecture: in-loop retry, backpressure queue, dead-letter queue
- Checkpoint/resume for interrupted tasks
- **Gap**: No retry at agent level when dispatcher throws
- **Gap**: No circuit breaker for consistently failing tools

### Task Result Quality (Current: 6/10)
- Custom validation, hallucination guard, search guard all exist
- Self-reflection and confidence gating exist but are DISABLED
- **Gap**: Self-reflection only enabled on error_recovery agent
- **Gap**: Confidence gating `min_confidence=0` everywhere (disabled)
- **Gap**: Quality grades don't trigger re-execution or alerts

### Context Management (Current: 7/10)
- Context window detection, 80% trim threshold, RAG injection
- **Gap**: No LLM-based summarization (purely mechanical truncation)
- **Gap**: No priority weighting for which messages to keep
- **Gap**: No context budget warning to the agent

### Observability (Current: 7/10)
- Structured JSON logging, 4 sinks, per-task tracing, audit log, metrics
- **Gap**: No dashboard (data exists, needs presentation)
- **Gap**: No end-to-end task timeline view
- **Gap**: No statistical anomaly alerting

### Security (Current: 8/10)
- Agent permission matrix, shell blocklist, workspace containment, Docker sandbox
- **Gap**: Shell blocklist is substring-based (bypassable)
- **Gap**: Permission check is fail-open (should be fail-closed)
- **Gap**: No rate limiting on tool calls

### Offline/Degraded Mode (Current: 5/10)
- Backpressure queue, dead-letter queue, Docker fallback
- **Gap**: No internet connectivity check before web tasks
- **Gap**: No GPU health monitoring
- **Gap**: No capability-based task deferral

### User Experience (Current: 6/10)
- 30-second progress updates, cancel command, DLQ UI
- **Gap**: No ETA or elapsed time in progress messages
- **Gap**: Cancel doesn't abort running agent
- **Gap**: No "task started" notification
- **Gap**: First 30 seconds have no feedback
