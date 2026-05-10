# Windsurf — Cascade

Date: 2026-05-09
Confidence: High on Cascade flow + pricing; Medium on async/background story (Windsurf has not pushed a Cursor-style cloud-agent product as visibly).

## 1. Input flow
- **Cascade panel** in the Windsurf IDE — chat box that doubles as the agent surface.
- **Workflows** with auto-generated **slash commands** let you invoke saved multi-step recipes; teams build their own.
- **@-mentions** for files, **Codemaps**, terminal output, browser elements; selections from the integrated browser flow back into the prompt.

## 2. Output type
- Multi-file plan → multi-file diffs applied with inline accept/reject.
- Can run terminal commands, edit files, and trigger deploys all in one turn.
- "Flow Awareness": tracks edits/clipboard/terminal/conversation to infer current task without re-priming.

## 3. Iteration loop
- Cascade proposes a plan, then applies edits with **inline diffs you can accept or reject** per hunk.
- Memory generation happens autonomously between sessions — Cascade writes its own memory files for next time.

## 4. Charter / spec / PRD generation
- No dedicated spec generator. Equivalent surface is **`.windsurfrules`** + project Rules (must be kept short — every line costs tokens per turn).
- Workflows act as semi-formal recipes; closest thing to a "playbook" artifact.

## 5. Multi-screen UI generation
- Possible via Cascade plus the integrated browser preview. No dedicated multi-screen-from-prompt generator like a v0 / Stitch.
- Mobile: same React Native / Expo path as Cursor; nothing mobile-native.

## 6. Style awareness
- **Codemaps** (powered by SWE-1.5) build hierarchical maps of how components actually compose, not just symbol lists; @-mentioned in Cascade for context.
- **Fast Context** retrieval is the marketed differentiator over Cursor's index.
- Project Rules carry style intent; no automatic design-token extraction.

## 7. Async / background
- **Weakest of the four** on background async. Cascade is fundamentally an interactive in-editor agent; no public Cursor-style "cloud VM + PR" mode as of May 2026.
- **Cascade Hooks** (Nov 2025) let you trigger Cascade from external events, but execution is still tied to an active editor session.
- Web/PR review surface exists for teams (GitHub-integrated), but it is a review agent, not an autonomous remote builder.

## 8. Deploy
- **Has a deploy path**: Cascade can deploy via integrated tooling (Netlify / similar) directly from the chat, which Cursor does not do natively.

## 9. Pricing (2026)
- **Free** — 25 credits/mo.
- **Pro** $15/mo — 500 credits, all premium models (SWE-1.5, Claude Sonnet 4.6, GPT-5, Gemini 3.1 Pro), unlimited Tab + Command.
- **Pro Plus** $35/mo — larger quota.
- **Teams** $25–$30/user/mo — sharing, analytics, GitHub PR review.
- **Enterprise** $60/user/mo — SSO, larger quotas.
- (Sources disagree slightly — Windsurf's own docs vs third-party trackers — Pro is consistently $15.)

## 10. Underlying model
- **SWE-1.5** — Windsurf's own agentic coding model, ~Claude 4.5-tier quality at ~13× speed (~950 tok/s), most coding turns under 5s. Powers Codemaps.
- Plus Claude Sonnet 4.6, Claude Opus, GPT-5, Gemini 3.1 Pro selectable per turn.

## 11. Recent updates (late 2025 – May 2026)
- Acquired by Cognition (post-Devin) — review coverage notes the change in trajectory.
- Nov 2025: SWE-1.5 + Cascade Hooks.
- Late 2025: pricing refresh ($15 Pro / $35 Pro+ / $25 Teams) and credit-based billing.
- 2026: Codemaps maturation; team analytics ("% of code written by Cascade").

## 12. Limitations
- No first-class background/remote agent — interactive only.
- No spec/PRD artifact; Rules-only style guidance.
- Credit system can surprise on heavy use.
- Cognition acquisition introduces strategic uncertainty (some reviews flag this).
- "90% of code per user is written by Cascade" marketing claim is unverifiable, treat with skepticism.

## Sources
- [Cascade product page](https://windsurf.com/cascade)
- [Windsurf changelog](https://windsurf.com/changelog)
- [AI Models docs](https://docs.windsurf.com/windsurf/models)
- [Plans & usage docs](https://docs.windsurf.com/windsurf/accounts/usage)
- [SWE-1.5 + Cascade Hooks guide (DigitalApplied)](https://www.digitalapplied.com/blog/windsurf-swe-1-5-cascade-hooks-november-2025)
- [Windsurf Review 2026 (vibecoding.app)](https://vibecoding.app/blog/windsurf-review)
- [Windsurf Review 2026 (Taskade)](https://www.taskade.com/blog/windsurf-review)
- [Windsurf vs Cursor 2026 (Vibecoding Academy)](https://www.vibecodingacademy.ai/blog/windsurf-vs-cursor)
- [Windsurf Rules guide (understandingAI)](https://understandingai.net/windsurf-master-prompt/)
- [Windsurf pricing 2026 (Verdent)](https://www.verdent.ai/guides/windsurf-pricing-2026)
