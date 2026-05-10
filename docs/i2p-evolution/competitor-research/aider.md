# Aider

**Date:** 2026-05-09
**Confidence:** H (widely used OSS, current with 2026 features)
**Category:** Terminal-based AI pair programmer with autonomous loops

---

## 1. What it is

Aider is an open-source CLI pair programmer that lives in the terminal and edits the user's codebase with full git integration. By 2026 it has crossed from "pair programming helper" into "agentic test-driven autonomous loop" territory: it can run test suites, debug its own errors, and iterate without human intervention until tests pass. Best-in-class on Claude 3.7 / 4.x Sonnet, DeepSeek R1/V3, and OpenAI o-series.

## 2. Input flow

- CLI prompt in terminal; images and web pages can be attached as context.
- Voice input (request features / tests / fixes by speaking).
- Files explicitly added to chat context.

## 3. Output type

Edits to user's codebase, committed automatically with sensible commit messages.

## 4. Iteration loop

- Edit → auto-lint → auto-test → auto-fix loop.
- 2026 addition: **autonomous bug-fix loops** — runs the test suite, identifies failures, attempts fix, retries until pass or budget exhausted.
- Tree-sitter ranked symbol graph provides intelligent context without loading whole files.

## 5. Spec / charter

None. Aider is downstream of spec.

## 6. Multi-screen / multi-page

N/A — code-level tool.

## 7. Style / theming

Inherits codebase style; lints with whatever the project uses.

## 8. Deploy / export

Direct git commits; deploy is downstream.

## 9. Pricing

OSS / free. User pays model provider directly (no Aider markup).

## 10. Underlying model

Any. Recommended: Claude 4.x Sonnet, DeepSeek R1/V3, GPT-4.x / o-series.

## 11. Recent updates

- Autonomous test-driven debug loops (2026).
- Multi-modal: voice input, image/web-page context.
- Expanded model support.
- Tree-sitter symbol graph maturity.

## 12. Notable strengths / limitations

- **Strength:** Cleanest "BYOM, pay provider directly, no platform tax" model in the OSS coding-agent space (Cline shares this stance). Git-native loop is genuinely good.
- **Limitation:** No spec phase; no UI/design generation; no multi-mission memory. Pure code-edit loop.
- **Lesson for KutAI:** Aider's auto-test-auto-fix loop is the same pattern KutAI's coulson runtime is converging on. Two specifics worth borrowing: (a) tree-sitter ranked symbol graph for context selection (Z2 review-density doc should consider this for context window management) and (b) voice input — KutAI already has Telegram voice; the agent should *act on* voice, not just transcribe it.

## 13. Sources

- [Aider chat homepage](https://aider.chat/)
- [Aider docs](https://aider.chat/docs/)
- [Aider-AI/aider on GitHub](https://github.com/Aider-AI/aider)
- [Aider Review 2026 — AIAgentsList](https://aiagentslist.com/agents/aider)
- [State of AI Coding Agents 2026 — Dave Patten / Medium](https://medium.com/@dave-patten/the-state-of-ai-coding-agents-2026-from-pair-programming-to-autonomous-ai-teams-b11f2b39232a)
