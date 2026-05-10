# GPT Engineer / GPT-Pilot / Pythagora (the OSS-to-commercial lineage)

**Date:** 2026-05-09
**Confidence:** M-H (well-documented public lineage)
**Category:** Multi-agent autonomous app builders (OSS predecessors → commercial product)

---

## 1. What they are

**GPT Engineer** (2023, Anton Osika) and **GPT-Pilot** (Pythagora-io, also 2023) were the two OSS pioneers of "describe an app, agent writes the whole thing." Both reached tens-of-thousands of stars; both showed the limits of one-shot codegen and motivated the multi-agent / step-by-step iteration paradigm now standard.

- **GPT Engineer** evolved into the commercial **gptengineer.app** (now branded "Lovable" in many UI surfaces — same company; "Lovable" is the dominant brand by 2026).
- **GPT-Pilot** evolved into **Pythagora** — a YC-backed VS Code / Cursor extension with 14 specialized agents (planner, dev, reviewer, debugger, tester, etc.) that walks a project end-to-end with founder approval at each stage. The OSS GPT-Pilot repo is now archived.

These two represent the **two opposing philosophies** in autonomous app building:

| | Lovable (ex-GPT Engineer) | Pythagora (ex-GPT-Pilot) |
|---|---|---|
| Speed | 60 seconds to first app | 15-20 min — deliberate |
| Process | One prompt → full codebase, iterate | Spec first → step-by-step → human approval per step |
| Surface | Web | VS Code / Cursor extension |
| Predictability | Low — fix what AI gets wrong | High — approve plan before code |
| Pricing | $19/mo starter | $49/mo starter, $89/mo team |

This split is the most important architectural axis in the category.

## 2. Input flow

- Lovable: web prompt + image attachments.
- Pythagora: prompt + iterative spec dialog inside IDE.

## 3. Output type

- Lovable: hosted preview + editable codebase + GitHub sync.
- Pythagora: real codebase in user's IDE, one-click deploy, automatic breakpoints, AI code review, CRUD scaffolding, API integration, doc ingestion.

## 4. Iteration loop

- Lovable: chat-to-fix loop after generation.
- Pythagora: 14 agents own different lifecycle phases; the loop is "approve the plan → agents do the step → review → next step."

## 5. Spec / charter

- Lovable: implicit; the prompt + image is the spec.
- **Pythagora: explicit detailed technical spec generated first**, before any code. This is the closest commercial analogue to KutAI's i2p phase 0-6 ambition — and it validates the bet that founders prefer "see the plan first" if it's quick enough.

## 6. Multi-screen / multi-page

Both yes.

## 7. Style / theming

- Lovable: shadcn/ui + Tailwind defaults.
- Pythagora: framework-of-choice; less opinionated on style.

## 8. Deploy / export

- Lovable: Vercel/Netlify integrations + GitHub.
- Pythagora: one-click deploy.

## 9. Pricing

- Lovable: $19/mo starter.
- Pythagora: free tier, $49/mo standard, $89/mo team.

## 10. Underlying model

Both model-agnostic, default to Claude / GPT-4 family.

## 11. Recent updates

- Lovable continues iteration on hosted preview + Supabase integration.
- Pythagora's 14-agent architecture is mature in 2026 and the explicit-spec-first flow is the headline differentiator.
- Original GPT-Pilot repo archived (signal: commercial product ate the OSS).

## 12. Notable strengths / limitations

- **Strength of Pythagora's approach:** generating a detailed technical spec before any code matches what KutAI is reaching for in i2p evolution. Pythagora has commercially proven that founders will tolerate the slower path *if* the spec is fast enough to feel responsive.
- **Limitation of both:** still single-mission, single-session. No cross-mission inheritance, no persistent founder context across products, no async/AFK loop.
- **Lesson for KutAI:** Pythagora is the closest competitor in *philosophy* (spec-first, step-by-step, approve-each-stage). The differentiation KutAI must hold:
  - **Async/AFK** (Pythagora needs founder at IDE; KutAI needs founder reachable on Telegram, can wait hours).
  - **Cross-mission inheritance** (P9/A7) — Pythagora restarts cold each project.
  - **Local-first inference** + Turkish.
  - **Real workforce** (mechanical executors, scheduled tasks, watchdogs) vs. just-codegen.

## 13. Sources

- [Pythagora-io/gpt-pilot on GitHub](https://github.com/Pythagora-io/gpt-pilot)
- [Pythagora platform](https://www.pythagora.ai/)
- [Pythagora Review 2026 — AIChief](https://aichief.com/ai-development-tools/pythagora/)
- [GPT Pilot YC profile](https://www.ycombinator.com/companies/pythagora-gpt-pilot)
- [GPT Engineer / Lovable](https://gptengineer.app/)
- [Lovable AI Review 2026 + Alternatives — Capacity](https://capacity.so/blog/lovable-ai-review-and-alternatives-2026-2)
