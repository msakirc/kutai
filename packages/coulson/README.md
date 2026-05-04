# coulson — agent runtime

Phil Coulson handles agents. This package owns multi-call orchestration: the
ReAct loop, single-shot path, tool VM, context assembly, sub-iteration
guards, escalation, reflection, checkpoint.

Sits between Orchestrator/Beckman (which decides *what* to run) and
Dispatcher (which makes one LLM call). Runtime is one-task-many-calls —
fills the architectural gap between admission and per-call dispatch.

## Public API

```python
from coulson import execute

result = await execute(profile, task, progress_callback=None)
```

`profile` is duck-typed. Needs:

- `name: str`
- `description: str`
- `allowed_tools: list[str] | None`
- `max_iterations: int`
- `execution_pattern: str`  ("react_loop" | "single_shot")
- `enable_self_reflection: bool`
- `min_confidence: int`
- `can_create_subtasks: bool`
- `_suppress_clarification: bool`
- `default_tier: str`, `min_tier: str`
- `get_system_prompt(task: dict) -> str`

`BaseAgent` in `src/agents/base.py` is the canonical Profile shape.

## Modules

| Module | LOC | Purpose |
|---|---|---|
| `runner.py` | – | public `execute()` entry, lane routing |
| `react.py` | ~1370 | ReAct loop |
| `single_shot.py` | ~110 | one-call path |
| `tools.py` | ~140 | tool VM helpers + registry |
| `context.py` | ~990 | system + user prompt assembly, RAG, skills |
| `parsing.py` | ~330 | ReAct DSL parser, alias map, function-call response |
| `window.py` | ~230 | context-window count/trim/prune |
| `guards.py` | ~260 | sub-iteration guards (hallucination, search, clarify) |
| `escalation.py` | ~85 | message trim on mid-task escalate |
| `reflection.py` | ~105 | self-reflect post-final review |
| `checkpoint.py` | ~110 | state save/restore + log |
| `validation.py` | ~85 | refusal/length/empty + samet wrappers |

## History

Extracted from `src/agents/base.py` in Phase A (2026-05-04, see
`docs/superpowers/specs/2026-05-04-runtime-extraction-design.md`).
Phase A landed as `src/runtime/`; Phase B (this package) is the
mechanical move to a sibling package alongside `general_beckman`,
`fatih_hoca`, `dallama`, `hallederiz_kadir`.

`src/runtime/__init__.py` remains as a thin re-export shim.
