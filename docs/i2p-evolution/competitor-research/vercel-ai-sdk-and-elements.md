# Vercel AI SDK + AI Elements

**Date:** 2026-05-09
**Confidence:** H (widely used building blocks; strong public docs)
**Category:** Building-block library for AI apps (NOT an app builder itself; sibling to v0)

---

## 1. What it is

The **Vercel AI SDK** (now at v5) is an open-source TypeScript toolkit for building AI-powered apps and agents — abstracts model providers (OpenAI, Anthropic, Google, etc.), provides typed chat hooks for React/Svelte/Vue/Angular, agentic loop control, tool calling, streaming, speech, image gen.

**AI Elements** (introduced 2025) is a 20+ component React UI kit on top of `shadcn/ui`, designed specifically for AI interfaces — message parts, streaming states, tool calls, reasoning displays, markdown rendering, all integrated with `useChat`.

This pair sits one layer below v0/Bolt/Lovable: it's what *they* could build on. Important to cover because it shapes what an AI app *looks* and *behaves* like in 2026 — and any KutAI codegen targeting AI-app missions should know these primitives exist.

## 2. Input flow

Developer code; not a builder.

## 3. Output type

Library code in their app.

## 4. Iteration loop

Standard SDK dev loop.

## 5. Spec / charter

N/A — but the SDK's typed chat schema effectively defines the contract for any AI-app i2p mission would build. KutAI should treat this as a baseline schema when generating AI-feature code.

## 6. Multi-screen / multi-page

N/A.

## 7. Style / theming

shadcn/ui-based; inherits Tailwind. Editable by developer.

## 8. Deploy / export

Library; deployment is the dev's responsibility (Vercel's pitch is to also host it).

## 9. Pricing

OSS, free. Vercel monetizes hosting.

## 10. Underlying model

Provider-agnostic — that's the point.

## 11. Recent updates

- AI SDK v5 (typed chat, agentic loop control, speech, tool enhancements).
- AI Elements component kit (20+ shadcn-based AI UI parts).
- Cross-framework support (React, Svelte, Vue, Angular).

## 12. Notable strengths / limitations

- **Strength:** De-facto standard for AI app shape in TS/React world. If KutAI ever generates a chat-driven product, building on AI SDK + Elements skips months of plumbing.
- **Limitation:** Library, not a builder. Doesn't address the spec/charter side.
- **Lesson for KutAI:** When phase 5 codegen produces an AI feature, *strongly* prefer AI SDK + Elements as the default stack rather than custom-rolling streaming chat plumbing. Add a recipe entry: "AI feature in mission ⇒ AI SDK + AI Elements." This is the kind of opinionated default the C7 ADR (component library decision) should already encode.

## 13. Sources

- [AI SDK by Vercel](https://ai-sdk.dev/)
- [AI SDK 5 announcement](https://vercel.com/blog/ai-sdk-5)
- [Introducing AI Elements](https://vercel.com/changelog/introducing-ai-elements)
- [AI Elements docs](https://docs.vercel.com/academy/ai-sdk/ai-elements)
- [vercel/ai on GitHub](https://github.com/vercel/ai)
