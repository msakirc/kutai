# Next Session Prompt

Paste this to start the next session:

---

Read `docs/superpowers/specs/2026-04-15-classification-refactor.md` — it documents findings from a deep investigation of KutAI's classification system. The classification is unreliable: "kahve makinesi" (coffee machine) got routed to a coder agent that built a FastAPI app instead of searching for products.

The findings doc covers:
- The full classification flow (telegram message classifier → task classifier → dispatch fallback)
- 10 specific problems identified with file paths and line numbers
- Shopping pipeline fixes already committed on main
- Design direction for the refactor

**Your task**: Design and implement a unified classification system. The current system has two independent classifiers with scattered rules across 4 files. We need one reliable system that:
1. Handles bare product nouns without enumerating them
2. Has proper fallback when the LLM is unreliable (local models misclassify often)
3. Propagates context (button press, chat history) through the classification chain
4. Has a single source of truth for rules instead of 4 files with different keyword subsets

Start by reading the spec, then brainstorm approaches. Focus on shopping classification reliability first — that's the acute pain point — but design for all agent types.
