# Designer-Grade Cluster Summary — Subframe / Magic Patterns / Tempo

**Research date:** 2026-05-09
**Cluster definition:** Tools sitting between "raw prompt → vibe-codegen" (v0/Lovable) and "Figma + manual handoff." All three emphasize design-system fidelity, token-aware generation, and component-library reuse.

---

## At-a-glance comparison

| Dimension | Subframe | Magic Patterns | Tempo |
|---|---|---|---|
| **Primary input** | Visual canvas + prompt | Prompt + screenshot + user-story | Prompt + image + Figma + repo |
| **Code output** | React/TS/Tailwind/Radix | React/Vue/Tailwind | React + React Native (Expo) |
| **Iteration** | Visual + AI variants | Chat + Select Mode | Visual + IDE round-trip |
| **Token system** | First-class, syncs to `tailwind.config` | Imports your tokens, semantic-naming hint | Visual editor over your codebase |
| **BYO component lib** | Yes (Radix-based starters extensible) | Storybook + Figma + Chrome scrape | Storybook + existing repo import |
| **Two-way sync** | One-way (design → code) | One-way + GitHub commits | **True round-trip** (Tempo ↔ GitHub ↔ VSCode) |
| **Figma export** | No | Yes (layered files out) | Plugin (Figma → Tempo) |
| **PRD/spec layer** | None | User-stories accepted | None |
| **Mobile** | No | No | Yes (Expo) |
| **Pricing entry** | $29/editor/mo | $19–$20/mo | $30/mo |
| **Premium tier** | Custom enterprise | $75–$100/mo | **$4,500/mo Agent+ (human-assisted)** |
| **Underlying model** | Undisclosed | Undisclosed | GPT-4 + Claude 3.5 + Gemini (disclosed via credits) |
| **Killer differentiator** | Deterministic export from real component lib (no LLM at export) | Chrome-scrape any live app into a library | GitHub round-trip + Expo mobile + SaaS templates |

---

## Cross-tool patterns (what they all agree on)

1. **Tokens-first generation.** All three constrain the AI to your token palette rather than letting it invent hex values. Magic Patterns is most explicit ("`color-action`, not `blue-500`"); Subframe is most structural (tokens compile to `tailwind.config`).
2. **Component library is the unit of reuse, not the screen.** Generation references library components, not regenerates from primitives.
3. **React + Tailwind is the default code stack.** Vue (MP) and React Native (Tempo) are extensions, not the center.
4. **Multiplayer + comments** are table stakes — all three ship Figma-grade collaboration.
5. **Chat-only iteration is dead.** All three offer visual + chat hybrids; pure chat (v0-style) is deliberately not enough for designers.

## Where they diverge

- **Spec layer is universally missing.** None of them ingest a PRD, derive acceptance criteria, or generate user-flow charters. This is the cluster's blind spot — and the natural seat for an i2p layer above them.
- **Round-trip story:** Tempo wins (full GitHub IDE loop). Subframe is one-way and proud of it (deterministic export). Magic Patterns is in-between.
- **Model transparency:** Tempo discloses (GPT-4 + Claude + Gemini); the other two don't. Implies different strategic bets — Tempo treats the model as commodity, others treat it as moat.
- **"Done" definition:** Subframe = pixel-perfect React component. Magic Patterns = clean UI for a screen. Tempo = full SaaS app with Stripe + auth + DB wired. Increasing scope.

---

## What KutAI i2p should learn from this tier (vs code-gen tier)

### Adopt from designer-grade tier

1. **Token-aware generation as a hard constraint, not a polish step.** Inject tokens into the system prompt; reject outputs that introduce new hex values. Cheap to implement, huge fidelity gain.
2. **Component-library-as-source-of-truth.** Generate by composition over named library components, not raw JSX. Mirrors Subframe's "deterministic export" philosophy and kills hallucination.
3. **`@LibraryName/Component` prompt grammar** (Magic Patterns) — give the LLM a referenceable namespace, don't expect it to remember.
4. **Select Mode** (Magic Patterns) — single-element edits as the iteration default, full-screen regeneration as the exception. Saves tokens and matches designer mental model.
5. **GitHub round-trip** (Tempo) — the canvas is a view, the repo is the source. Solves the drift-after-customization problem that makes one-way tools brittle.
6. **Multi-model stack** (Tempo) — different prompts hit different models. KutAI already does this (Fatih Hoca); validates the approach.
7. **BYO via multiple paths** (Magic Patterns) — Storybook + Figma + Chrome-scrape. Never assume one import format.

### KutAI's natural advantages over this tier

1. **PRD/spec layer is the cluster blind spot.** All three start at "I want a dashboard." KutAI's i2p flow can author user stories, charters, and acceptance criteria upstream and feed those into a designer-grade tool (or generate that layer itself).
2. **Backend + workflows + persistence.** All three are frontend-first; even Tempo's SaaS templates only wire the integration shell. KutAI's mission/workflow/agent stack covers the back half they don't.
3. **Mobile track parity.** Only Tempo competes on mobile (via Expo). KutAI's `05-build-mobile-track.md` should treat Tempo as the direct comparable.
4. **Local-first model economics.** Subframe and Magic Patterns hide their model costs in monthly subscriptions (likely thin margins on heavy users); KutAI's local llama-server + cloud-fallback story is structurally cheaper at high usage.

### Things NOT to copy

- **One-way sync as a deliberate choice** (Subframe) — works for them because their canvas IS the IDE; for KutAI's broader scope it would be a footgun. Choose Tempo's round-trip model.
- **Per-generation credit metering** (Magic Patterns, Tempo Pro) — friction that punishes the iteration the tools are supposed to enable. KutAI's local-first inference makes unlimited iteration economically viable; lean into it.
- **Closed-source model selection.** Subframe and Magic Patterns hide their LLM. KutAI should be transparent (it already is via Fatih Hoca telemetry); turn it into a trust differentiator.

---

## Sources

See per-tool files for full citation lists:
- `subframe.md`
- `magic-patterns.md`
- `tempo-labs.md`
