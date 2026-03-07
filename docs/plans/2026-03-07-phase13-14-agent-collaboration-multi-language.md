# Phase 13 + 14: Agent Collaboration & Multi-Language Coding

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add agent collaboration (shared blackboard, plan verification, agent-to-agent queries, parallel tasks, interactive approval) and multi-language coding support (language toolkits, language-aware prompts, multi-runtime sandbox, dependency detection).

**Architecture:** Phase 13 introduces a shared blackboard (per-goal structured state) backed by DB, a new `ask_agent` action type, plan verification pass in orchestrator, dynamic concurrency, and Telegram inline approval. Phase 14 adds an abstract `LanguageToolkit` interface with concrete implementations for Python/JS/TS/Go/Rust/Java/Kotlin, dynamic language-specific prompt injection, multi-runtime sandbox support, and polyglot dependency detection.

**Tech Stack:** Python 3.12, aiosqlite, asyncio, Pydantic, python-telegram-bot, Docker (sandbox)

---

### Task 1: Shared Blackboard — Module + DB Table (13.1)

**Files:**
- Create: `collaboration/__init__.py`
- Create: `collaboration/blackboard.py`
- Modify: `db.py` (add `blackboards` table to `init_db`)
- Test: `tests/test_phase13.py`

**Step 1: Write the failing tests**

```python
# tests/test_phase13.py — initial blackboard tests

import unittest
import asyncio
import json

class TestBlackboard(unittest.TestCase):
    """13.1 — Shared Blackboard."""

    def test_blackboard_module_exists(self):
        import collaboration.blackboard as bb
        self.assertTrue(hasattr(bb, "read_blackboard"))
        self.assertTrue(hasattr(bb, "write_blackboard"))
        self.assertTrue(hasattr(bb, "update_blackboard_entry"))
        self.assertTrue(hasattr(bb, "get_or_create_blackboard"))

    def test_blackboard_schema(self):
        from collaboration.blackboard import DEFAULT_BLACKBOARD
        self.assertIn("architecture", DEFAULT_BLACKBOARD)
        self.assertIn("files", DEFAULT_BLACKBOARD)
        self.assertIn("decisions", DEFAULT_BLACKBOARD)
        self.assertIn("open_issues", DEFAULT_BLACKBOARD)
        self.assertIn("constraints", DEFAULT_BLACKBOARD)

    def test_write_and_read_blackboard(self):
        from collaboration.blackboard import (
            write_blackboard, read_blackboard, get_or_create_blackboard
        )
        loop = asyncio.new_event_loop()
        # Test in-memory (bypass DB for unit test)
        # use internal dict cache
        from collaboration.blackboard import _BLACKBOARD_CACHE
        _BLACKBOARD_CACHE[99] = {"architecture": {}, "files": {}, "decisions": [], "open_issues": [], "constraints": []}
        loop.run_until_complete(
            write_blackboard(99, "decisions", [{"what": "Use FastAPI", "why": "performance", "by": "architect"}])
        )
        result = loop.run_until_complete(read_blackboard(99, "decisions"))
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["what"], "Use FastAPI")
        loop.close()

    def test_update_blackboard_entry(self):
        from collaboration.blackboard import update_blackboard_entry, _BLACKBOARD_CACHE
        loop = asyncio.new_event_loop()
        _BLACKBOARD_CACHE[100] = {"architecture": {}, "files": {}, "decisions": [], "open_issues": [], "constraints": []}
        loop.run_until_complete(
            update_blackboard_entry(100, "files", "app.py", {"status": "implemented", "interface_hash": "abc"})
        )
        result = _BLACKBOARD_CACHE[100]["files"]
        self.assertEqual(result["app.py"]["status"], "implemented")
        loop.close()
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_phase13.py -v -x --tb=short 2>&1 | head -30`
Expected: FAIL with `ModuleNotFoundError: No module named 'collaboration'`

**Step 3: Create `collaboration/__init__.py`**

```python
# collaboration/__init__.py
"""Phase 13 — Agent Collaboration."""
```

**Step 4: Create `collaboration/blackboard.py`**

```python
# collaboration/blackboard.py
"""
Phase 13.1 — Shared Blackboard.

Per-goal structured state store with typed entries. Backed by a
`blackboards` DB table (goal_id → data JSON). Agents read/write structured
data instead of parsing prior_steps text blobs.
"""
import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_BLACKBOARD: dict = {
    "architecture": {},
    "files": {},
    "decisions": [],
    "open_issues": [],
    "constraints": [],
}

# In-memory cache to reduce DB round-trips within a single run cycle.
_BLACKBOARD_CACHE: dict[int, dict] = {}


async def get_or_create_blackboard(goal_id: int) -> dict:
    """Load the blackboard for a goal, creating a fresh one if needed."""
    if goal_id in _BLACKBOARD_CACHE:
        return _BLACKBOARD_CACHE[goal_id]

    try:
        from db import get_db
        db = await get_db()

        # Ensure table exists
        await db.execute("""
            CREATE TABLE IF NOT EXISTS blackboards (
                goal_id INTEGER PRIMARY KEY,
                data JSON NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor = await db.execute(
            "SELECT data FROM blackboards WHERE goal_id = ?", (goal_id,)
        )
        row = await cursor.fetchone()
        if row:
            board = json.loads(row[0])
        else:
            board = json.loads(json.dumps(DEFAULT_BLACKBOARD))  # deep copy
            await db.execute(
                "INSERT INTO blackboards (goal_id, data) VALUES (?, ?)",
                (goal_id, json.dumps(board)),
            )
            await db.commit()
    except Exception as exc:
        logger.debug(f"Blackboard DB access failed, using defaults: {exc}")
        board = json.loads(json.dumps(DEFAULT_BLACKBOARD))

    _BLACKBOARD_CACHE[goal_id] = board
    return board


async def read_blackboard(goal_id: int, key: Optional[str] = None) -> Any:
    """Read the entire blackboard or a specific key."""
    board = await get_or_create_blackboard(goal_id)
    if key is None:
        return board
    return board.get(key)


async def write_blackboard(goal_id: int, key: str, value: Any) -> None:
    """Overwrite a top-level key in the blackboard."""
    board = await get_or_create_blackboard(goal_id)
    board[key] = value
    _BLACKBOARD_CACHE[goal_id] = board
    await _persist(goal_id, board)


async def update_blackboard_entry(
    goal_id: int, key: str, sub_key: str, value: Any
) -> None:
    """Update a nested entry (e.g., files["app.py"] = {...})."""
    board = await get_or_create_blackboard(goal_id)
    section = board.get(key)
    if isinstance(section, dict):
        section[sub_key] = value
    elif isinstance(section, list):
        section.append({sub_key: value})
    else:
        board[key] = {sub_key: value}
    _BLACKBOARD_CACHE[goal_id] = board
    await _persist(goal_id, board)


async def append_blackboard(goal_id: int, key: str, item: Any) -> None:
    """Append an item to a list-typed key (decisions, open_issues, etc.)."""
    board = await get_or_create_blackboard(goal_id)
    section = board.get(key, [])
    if not isinstance(section, list):
        section = [section]
    section.append(item)
    board[key] = section
    _BLACKBOARD_CACHE[goal_id] = board
    await _persist(goal_id, board)


def format_blackboard_for_prompt(board: dict, max_chars: int = 3000) -> str:
    """Format a blackboard for injection into an agent system prompt."""
    if not board:
        return ""
    parts = ["## Shared Blackboard (Project State)"]

    # Architecture
    arch = board.get("architecture", {})
    if arch:
        parts.append(f"### Architecture\n```json\n{json.dumps(arch, indent=2)[:500]}\n```")

    # Files
    files = board.get("files", {})
    if files:
        file_lines = []
        for path, info in list(files.items())[:20]:
            status = info.get("status", "?") if isinstance(info, dict) else str(info)
            file_lines.append(f"  - `{path}`: {status}")
        parts.append("### File Status\n" + "\n".join(file_lines))

    # Decisions
    decisions = board.get("decisions", [])
    if decisions:
        dec_lines = [
            f"  - **{d.get('what', '?')}** — {d.get('why', '')} (by {d.get('by', '?')})"
            for d in decisions[-5:]  # last 5
        ]
        parts.append("### Key Decisions\n" + "\n".join(dec_lines))

    # Open Issues
    issues = board.get("open_issues", [])
    if issues:
        parts.append("### Open Issues\n" + "\n".join(f"  - {i}" for i in issues[-5:]))

    # Constraints
    constraints = board.get("constraints", [])
    if constraints:
        parts.append("### Constraints\n" + "\n".join(f"  - {c}" for c in constraints))

    text = "\n\n".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... [blackboard truncated]"
    return text


async def _persist(goal_id: int, board: dict) -> None:
    """Write board state back to DB."""
    try:
        from db import get_db
        db = await get_db()
        await db.execute("""
            CREATE TABLE IF NOT EXISTS blackboards (
                goal_id INTEGER PRIMARY KEY,
                data JSON NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.execute(
            """INSERT OR REPLACE INTO blackboards (goal_id, data, updated_at)
               VALUES (?, ?, CURRENT_TIMESTAMP)""",
            (goal_id, json.dumps(board)),
        )
        await db.commit()
    except Exception as exc:
        logger.debug(f"Blackboard persist failed (non-critical): {exc}")
```

**Step 5: Add `blackboards` table to `db.py` `init_db()`**

Add after the `scheduled_tasks` table creation:

```python
    # Blackboards (Phase 13.1)
    await db.execute("""
        CREATE TABLE IF NOT EXISTS blackboards (
            goal_id INTEGER PRIMARY KEY,
            data JSON NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
```

**Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_phase13.py::TestBlackboard -v`
Expected: PASS (4 tests)

**Step 7: Commit**

```bash
git add collaboration/__init__.py collaboration/blackboard.py tests/test_phase13.py
git commit -m "phase 13.1: shared blackboard module"
```

---

### Task 2: Blackboard as Agent Tools + Prompt Injection (13.1 continued)

**Files:**
- Modify: `tools/__init__.py` (register `read_blackboard`, `write_blackboard`)
- Modify: `agents/base.py` (inject blackboard into context)
- Test: `tests/test_phase13.py` (add tool registration + injection tests)

**Step 1: Write failing tests**

```python
class TestBlackboardToolRegistration(unittest.TestCase):
    """13.1 — Blackboard tools registered."""

    def test_blackboard_tools_in_registry(self):
        source = open("tools/__init__.py").read()
        self.assertIn("read_blackboard", source)
        self.assertIn("write_blackboard", source)

    def test_blackboard_injection_in_base_agent(self):
        source = open("agents/base.py").read()
        self.assertIn("blackboard", source.lower())
        self.assertIn("format_blackboard_for_prompt", source)
```

**Step 2: Run to verify failure**

Run: `python -m pytest tests/test_phase13.py::TestBlackboardToolRegistration -v -x`
Expected: FAIL

**Step 3: Register blackboard tools in `tools/__init__.py`**

Add import and TOOL_REGISTRY entries for `read_blackboard` and `write_blackboard`. The tool wrappers accept `goal_id` and optionally `key`/`value`.

```python
# In imports section:
from collaboration.blackboard import (
    read_blackboard as _read_blackboard,
    write_blackboard as _write_blackboard,
)

# Tool wrappers (since the raw functions need goal_id from task context):
async def tool_read_blackboard(goal_id: int, key: str = "") -> str:
    """Read from the shared project blackboard."""
    import json
    result = await _read_blackboard(goal_id, key=key or None)
    return json.dumps(result, indent=2) if result else "{}"

async def tool_write_blackboard(goal_id: int, key: str, value: str) -> str:
    """Write to the shared project blackboard."""
    import json
    try:
        parsed_value = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        parsed_value = value
    await _write_blackboard(goal_id, key, parsed_value)
    return f"✅ Blackboard key '{key}' updated."

# In TOOL_REGISTRY:
"read_blackboard": {
    "function": tool_read_blackboard,
    "description": "Read from the shared project blackboard (structured state shared between agents). Args: goal_id (int), key (str, optional — e.g. 'architecture', 'files', 'decisions').",
    "example": '{"tool": "read_blackboard", "args": {"goal_id": 1, "key": "decisions"}}',
},
"write_blackboard": {
    "function": tool_write_blackboard,
    "description": "Write to the shared project blackboard. Args: goal_id (int), key (str — 'architecture'|'files'|'decisions'|'open_issues'|'constraints'), value (str — JSON string).",
    "example": '{"tool": "write_blackboard", "args": {"goal_id": 1, "key": "decisions", "value": "[{\\"what\\": \\"Use FastAPI\\", \\"why\\": \\"speed\\", \\"by\\": \\"architect\\"}]"}}',
},
```

**Step 4: Inject blackboard into agent context in `agents/base.py`**

In `_build_context()`, after the project profile injection block and before RAG context injection, add:

```python
        # ── Phase 13.1: Blackboard injection ──
        if goal_id:
            try:
                from collaboration.blackboard import (
                    get_or_create_blackboard,
                    format_blackboard_for_prompt,
                )
                board = await get_or_create_blackboard(goal_id)
                bb_block = format_blackboard_for_prompt(board)
                if bb_block:
                    parts.append(bb_block)
            except Exception as exc:
                logger.debug(f"Blackboard injection failed (non-critical): {exc}")
```

**Step 5: Run tests**

Run: `python -m pytest tests/test_phase13.py::TestBlackboardToolRegistration -v`
Expected: PASS

**Step 6: Commit**

```bash
git add tools/__init__.py agents/base.py tests/test_phase13.py
git commit -m "phase 13.1: blackboard tools + prompt injection"
```

---

### Task 3: Plan Verification (13.2)

**Files:**
- Create: `collaboration/plan_verification.py`
- Modify: `orchestrator.py` (call verifier in `_handle_subtasks`)
- Test: `tests/test_phase13.py` (add verification tests)

**Step 1: Write failing tests**

```python
class TestPlanVerification(unittest.TestCase):
    """13.2 — Plan Verification."""

    def test_verification_module_exists(self):
        import collaboration.plan_verification as pv
        self.assertTrue(hasattr(pv, "verify_plan"))

    def test_verify_acyclic_deps(self):
        from collaboration.plan_verification import verify_plan
        subtasks = [
            {"title": "A", "agent_type": "coder", "depends_on_step": None},
            {"title": "B", "agent_type": "coder", "depends_on_step": 0},
            {"title": "C", "agent_type": "coder", "depends_on_step": 1},
        ]
        issues = verify_plan(subtasks, goal_budget=10.0)
        self.assertEqual(len(issues), 0, f"Expected no issues: {issues}")

    def test_verify_cyclic_deps_detected(self):
        from collaboration.plan_verification import verify_plan
        subtasks = [
            {"title": "A", "agent_type": "coder", "depends_on_step": 2},
            {"title": "B", "agent_type": "coder", "depends_on_step": 0},
            {"title": "C", "agent_type": "coder", "depends_on_step": 1},
        ]
        issues = verify_plan(subtasks, goal_budget=10.0)
        self.assertTrue(any("cycl" in i.lower() for i in issues))

    def test_verify_agent_type_mismatch(self):
        from collaboration.plan_verification import verify_plan
        subtasks = [
            {"title": "Write tests for auth module", "description": "Write unit tests for auth.py",
             "agent_type": "writer"},  # should be coder/pipeline
        ]
        issues = verify_plan(subtasks, goal_budget=10.0)
        self.assertTrue(any("agent" in i.lower() for i in issues))

    def test_verify_duplicate_detection(self):
        from collaboration.plan_verification import verify_plan
        subtasks = [
            {"title": "Build the API", "agent_type": "coder"},
            {"title": "Build the API", "agent_type": "coder"},  # duplicate
        ]
        issues = verify_plan(subtasks, goal_budget=10.0)
        self.assertTrue(any("duplicate" in i.lower() for i in issues))

    def test_orchestrator_calls_verify(self):
        source = open("orchestrator.py").read()
        self.assertIn("verify_plan", source)
```

**Step 2: Run to verify failure**

Run: `python -m pytest tests/test_phase13.py::TestPlanVerification -v -x`
Expected: FAIL

**Step 3: Create `collaboration/plan_verification.py`**

```python
# collaboration/plan_verification.py
"""
Phase 13.2 — Plan Verification.

After planner creates subtasks, verify:
1. Agent type assignments are sensible
2. Dependency graph is acyclic
3. No duplicate subtasks
4. Estimated cost fits within budget
"""
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Heuristic keywords → expected agent_type mapping
_TASK_AGENT_HINTS: dict[str, set[str]] = {
    "coder": {"write code", "implement", "build", "create.*api", "add.*endpoint",
              "fix.*bug", "write.*test", "unit test", "integration test",
              "refactor", "write.*function", "write.*class", "code"},
    "pipeline": {"complex", "multi-file", "full.*implementation",
                 "architect.*and.*implement"},
    "researcher": {"research", "find", "search", "compare", "investigate",
                   "look up", "analyze.*option", "evaluate"},
    "writer": {"document", "readme", "write.*doc", "create.*doc",
               "documentation", "write.*guide"},
    "reviewer": {"review", "audit", "check.*quality"},
}

# Estimated cost per task by tier (rough USD)
_TIER_COST_ESTIMATE: dict[str, float] = {
    "cheap": 0.01,
    "medium": 0.05,
    "expensive": 0.20,
    "auto": 0.03,
}


def verify_plan(
    subtasks: list[dict],
    goal_budget: float = 10.0,
) -> list[str]:
    """Verify a plan's subtasks, returning a list of issue strings (empty = OK)."""
    issues: list[str] = []

    if not subtasks:
        return issues

    # 1. Check for cycles in dependency graph
    cycle_issues = _check_cycles(subtasks)
    issues.extend(cycle_issues)

    # 2. Agent type assignment sanity check
    for i, st in enumerate(subtasks):
        title = (st.get("title", "") + " " + st.get("description", "")).lower()
        assigned = st.get("agent_type", "executor")
        suggested = _suggest_agent_type(title)
        if suggested and assigned not in suggested and assigned != "executor":
            issues.append(
                f"Subtask {i} '{st.get('title', '?')[:40]}': assigned "
                f"'{assigned}' but content suggests {suggested}"
            )

    # 3. Duplicate detection
    titles = [st.get("title", "").strip().lower() for st in subtasks]
    seen = set()
    for i, t in enumerate(titles):
        if t in seen and t:
            issues.append(f"Subtask {i} duplicate title: '{t[:40]}'")
        seen.add(t)

    # 4. Budget check
    total_cost = sum(
        _TIER_COST_ESTIMATE.get(st.get("tier", "auto"), 0.03)
        for st in subtasks
    )
    if total_cost > goal_budget:
        issues.append(
            f"Estimated plan cost ${total_cost:.2f} exceeds budget ${goal_budget:.2f}"
        )

    return issues


def _check_cycles(subtasks: list[dict]) -> list[str]:
    """Detect cycles in the dependency graph."""
    n = len(subtasks)
    adj: dict[int, list[int]] = {i: [] for i in range(n)}

    for i, st in enumerate(subtasks):
        dep = st.get("depends_on_step")
        if dep is None:
            continue
        if isinstance(dep, int):
            if 0 <= dep < n:
                adj[dep].append(i)
        elif isinstance(dep, list):
            for d in dep:
                if isinstance(d, int) and 0 <= d < n:
                    adj[d].append(i)

    # Topological sort via Kahn's algorithm
    in_degree = [0] * n
    for i, st in enumerate(subtasks):
        dep = st.get("depends_on_step")
        if isinstance(dep, int) and 0 <= dep < n:
            in_degree[i] += 1
        elif isinstance(dep, list):
            for d in dep:
                if isinstance(d, int) and 0 <= d < n:
                    in_degree[i] += 1

    queue = [i for i in range(n) if in_degree[i] == 0]
    visited = 0
    while queue:
        node = queue.pop(0)
        visited += 1
        for neighbor in adj.get(node, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if visited < n:
        return [f"Dependency cycle detected in plan ({n - visited} tasks in cycle)"]
    return []


def _suggest_agent_type(text: str) -> set[str] | None:
    """Suggest acceptable agent types from task text."""
    suggestions = set()
    for agent, keywords in _TASK_AGENT_HINTS.items():
        for kw in keywords:
            if re.search(kw, text):
                suggestions.add(agent)
                break
    return suggestions or None
```

**Step 4: Add verification call to `orchestrator.py` `_handle_subtasks()`**

After the `MAX_SUBTASKS` cap and before `processed` list construction, add:

```python
        # ── Phase 13.2: Plan verification ──
        try:
            from collaboration.plan_verification import verify_plan
            issues = verify_plan(subtasks, goal_budget=10.0)
            if issues:
                logger.warning(
                    f"[Task #{task_id}] Plan verification found {len(issues)} issue(s): "
                    + "; ".join(issues)
                )
                # Send issues back to planner as feedback (non-blocking)
                await self.telegram.send_notification(
                    f"⚠️ *Plan Issues (Task #{task_id})*\n"
                    + "\n".join(f"  • {i}" for i in issues)
                )
        except Exception as e:
            logger.debug(f"Plan verification failed (non-critical): {e}")
```

**Step 5: Run tests**

Run: `python -m pytest tests/test_phase13.py::TestPlanVerification -v`
Expected: PASS

**Step 6: Commit**

```bash
git add collaboration/plan_verification.py orchestrator.py tests/test_phase13.py
git commit -m "phase 13.2: plan verification"
```

---

### Task 4: Agent-to-Agent Queries (13.3)

**Files:**
- Modify: `models.py` (add `AskAgentAction`)
- Modify: `agents/base.py` (handle `ask_agent` action in the ReAct loop)
- Modify: `orchestrator.py` (process inline ask_agent subtask)
- Test: `tests/test_phase13.py`

**Step 1: Write failing tests**

```python
class TestAskAgentAction(unittest.TestCase):
    """13.3 — Agent-to-Agent Queries."""

    def test_ask_agent_action_model_exists(self):
        from models import AskAgentAction, ACTION_MODELS
        self.assertTrue(hasattr(AskAgentAction, "action"))
        self.assertIn("ask_agent", ACTION_MODELS)

    def test_ask_agent_action_fields(self):
        from models import AskAgentAction
        a = AskAgentAction(target="researcher", question="What is the best ORM?")
        self.assertEqual(a.action, "ask_agent")
        self.assertEqual(a.target, "researcher")
        self.assertEqual(a.question, "What is the best ORM?")

    def test_ask_agent_handled_in_base_agent(self):
        source = open("agents/base.py").read()
        self.assertIn("ask_agent", source)

    def test_normalize_aliases(self):
        from agents.base import BaseAgent
        agent = BaseAgent()
        result = agent._normalize_action({"action": "ask_agent", "target": "researcher", "question": "test?"})
        self.assertIsNotNone(result)
        self.assertEqual(result["action"], "ask_agent")
```

**Step 2: Run to verify failure**

Run: `python -m pytest tests/test_phase13.py::TestAskAgentAction -v -x`
Expected: FAIL

**Step 3: Add `AskAgentAction` to `models.py`**

```python
class AskAgentAction(BaseModel):
    """Agent wants to ask another agent a question (Phase 13.3)."""
    action: Literal["ask_agent"] = "ask_agent"
    target: str = Field(..., description="Target agent type (e.g., 'researcher')")
    question: str = Field(..., description="Question to ask the target agent")
    reasoning: Optional[str] = Field(None, description="Agent's reasoning")

# Add to ACTION_MODELS:
"ask_agent": AskAgentAction,

# Add to get_action_json_schema properties:
"target": {"type": "string", "description": "Target agent type (for ask_agent)"},
```

**Step 4: Handle `ask_agent` in `agents/base.py` execute loop**

In the main ReAct loop, after handling `tool_call` and before `final_answer`, add a handler for `ask_agent`:

```python
            elif action_type == "ask_agent":
                # Phase 13.3: Agent-to-Agent query
                target = parsed.get("target", "researcher")
                question = parsed.get("question", "")
                logger.info(f"[{self.name}] Asking agent '{target}': {question[:80]}")

                try:
                    from agents import get_agent
                    target_agent = get_agent(target)
                    # Create an inline mini-task
                    inline_task = {
                        "id": task.get("id", 0),
                        "title": f"Query from {self.name}: {question[:60]}",
                        "description": question,
                        "agent_type": target,
                        "goal_id": task.get("goal_id"),
                        "context": task.get("context", "{}"),
                        "tier": "cheap",
                        "priority": task.get("priority", 5),
                    }
                    # Execute with timeout
                    import asyncio
                    answer_result = await asyncio.wait_for(
                        target_agent.execute(inline_task), timeout=120
                    )
                    answer_text = answer_result.get("result", "(no answer)")
                    if len(answer_text) > 3000:
                        answer_text = answer_text[:3000] + "\n... [truncated]"
                except asyncio.TimeoutError:
                    answer_text = f"(Agent '{target}' timed out after 120s)"
                except Exception as exc:
                    answer_text = f"(Failed to query agent '{target}': {exc})"

                messages.append({
                    "role": "assistant",
                    "content": content,
                })
                messages.append({
                    "role": "user",
                    "content": f"## Answer from {target} agent\n{answer_text}",
                })
                continue
```

**Step 5: Run tests**

Run: `python -m pytest tests/test_phase13.py::TestAskAgentAction -v`
Expected: PASS

**Step 6: Commit**

```bash
git add models.py agents/base.py tests/test_phase13.py
git commit -m "phase 13.3: agent-to-agent queries (ask_agent action)"
```

---

### Task 5: Parallel Independent Tasks (13.4)

**Files:**
- Modify: `orchestrator.py` (dynamic concurrency + file overlap detection)
- Test: `tests/test_phase13.py`

**Step 1: Write failing tests**

```python
class TestParallelTasks(unittest.TestCase):
    """13.4 — Parallel Independent Tasks."""

    def test_dynamic_concurrency_in_orchestrator(self):
        source = open("orchestrator.py").read()
        self.assertIn("_compute_max_concurrent", source)

    def test_file_overlap_detection(self):
        from orchestrator import _detect_file_overlap
        task_a = {"description": "Edit app.py and routes.py"}
        task_b = {"description": "Edit models.py and db.py"}
        task_c = {"description": "Edit app.py and config.py"}  # overlaps with a
        self.assertFalse(_detect_file_overlap(task_a, task_b))
        self.assertTrue(_detect_file_overlap(task_a, task_c))

    def test_max_concurrent_default(self):
        source = open("orchestrator.py").read()
        self.assertIn("MAX_CONCURRENT_TASKS", source)
```

**Step 2: Run to verify failure**

Run: `python -m pytest tests/test_phase13.py::TestParallelTasks -v -x`
Expected: FAIL

**Step 3: Add file overlap detection and dynamic concurrency to `orchestrator.py`**

```python
import re as _re

def _extract_file_refs(task: dict) -> set[str]:
    """Extract file path references from task description/title."""
    text = f"{task.get('title', '')} {task.get('description', '')}"
    # Match common file patterns
    return set(_re.findall(r'[\w/\\.-]+\.\w{1,10}', text))

def _detect_file_overlap(task_a: dict, task_b: dict) -> bool:
    """Check if two tasks reference overlapping files."""
    files_a = _extract_file_refs(task_a)
    files_b = _extract_file_refs(task_b)
    return bool(files_a & files_b)

def _compute_max_concurrent(tasks: list[dict]) -> int:
    """Dynamically compute how many tasks to run concurrently."""
    base = MAX_CONCURRENT_TASKS  # 2

    # Group by goal_id — different goals can safely run in parallel
    goal_ids = set(t.get("goal_id") for t in tasks if t.get("goal_id"))
    if len(goal_ids) > 1:
        base = min(len(goal_ids), 4)

    return base
```

Update `run_loop` to use `_compute_max_concurrent` and filter out overlapping tasks.

**Step 4: Run tests**

Run: `python -m pytest tests/test_phase13.py::TestParallelTasks -v`
Expected: PASS

**Step 5: Commit**

```bash
git add orchestrator.py tests/test_phase13.py
git commit -m "phase 13.4: parallel independent tasks + file overlap detection"
```

---

### Task 6: Interactive Plan Approval (13.5)

**Files:**
- Modify: `orchestrator.py` (add approval flow in `_handle_subtasks`)
- Modify: `telegram_bot.py` (add inline keyboard for plan approval)
- Test: `tests/test_phase13.py`

**Step 1: Write failing tests**

```python
class TestInteractivePlanApproval(unittest.TestCase):
    """13.5 — Interactive Plan Approval."""

    def test_approval_flow_in_orchestrator(self):
        source = open("orchestrator.py").read()
        self.assertIn("plan_approval", source.lower().replace("_", ""))

    def test_telegram_approval_buttons(self):
        source = open("telegram_bot.py").read()
        self.assertIn("approve_plan", source)
        self.assertIn("modify_plan", source)
        self.assertIn("reject_plan", source)

    def test_auto_approve_timeout(self):
        source = open("orchestrator.py").read()
        # Should have configurable auto-approve timeout
        self.assertIn("auto_approve", source.lower().replace("_", "").replace("-", ""))
```

**Step 2: Run to verify failure**

Run: `python -m pytest tests/test_phase13.py::TestInteractivePlanApproval -v -x`
Expected: FAIL

**Step 3: Add plan approval to `orchestrator.py`**

Add an `_await_plan_approval` method that:
1. Sends plan to Telegram with ✅ Approve / ✏️ Modify / ❌ Reject buttons
2. Waits up to `AUTO_APPROVE_TIMEOUT` seconds (default 600 = 10 min)
3. On timeout → auto-approve
4. On reject → cancel subtasks
5. On modify → create correction task

```python
    AUTO_APPROVE_TIMEOUT: int = 600  # 10 minutes

    async def _await_plan_approval(self, task_id: int, subtasks: list[dict], plan_summary: str) -> str:
        """Send plan for approval and wait for response. Returns: 'approved'|'rejected'|'modified:<feedback>'"""
        approval_key = f"plan_{task_id}"
        event = asyncio.Event()
        self.telegram._approval_events[approval_key] = {
            "event": event,
            "response": "approved",  # default on timeout
        }

        # Send approval message
        await self.telegram.send_plan_approval(task_id, subtasks, plan_summary)

        try:
            await asyncio.wait_for(event.wait(), timeout=self.AUTO_APPROVE_TIMEOUT)
        except asyncio.TimeoutError:
            logger.info(f"[Task #{task_id}] Plan auto-approved after timeout")

        result = self.telegram._approval_events.pop(approval_key, {})
        return result.get("response", "approved")
```

**Step 4: Add Telegram approval UI in `telegram_bot.py`**

```python
    async def send_plan_approval(self, task_id: int, subtasks: list[dict], plan_summary: str):
        """Send plan with inline approval buttons."""
        subtask_list = "\n".join(
            f"  {i+1}. [{st.get('agent_type', '?')}] {st.get('title', '?')[:50]}"
            for i, st in enumerate(subtasks)
        )
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("✅ Approve", callback_data=f"approve_plan:{task_id}"),
                InlineKeyboardButton("✏️ Modify", callback_data=f"modify_plan:{task_id}"),
                InlineKeyboardButton("❌ Reject", callback_data=f"reject_plan:{task_id}"),
            ]
        ])
        await self.app.bot.send_message(
            chat_id=TELEGRAM_ADMIN_CHAT_ID,
            text=(
                f"📋 *Plan Approval — Task #{task_id}*\n\n"
                f"{plan_summary}\n\n"
                f"{subtask_list}\n\n"
                f"_Auto-approves in 10 minutes_"
            ),
            reply_markup=keyboard,
            parse_mode="Markdown",
        )
```

Handle callback in `handle_callback`:

```python
        if data.startswith("approve_plan:"):
            task_id = int(data.split(":")[1])
            key = f"plan_{task_id}"
            if key in self._approval_events:
                self._approval_events[key]["response"] = "approved"
                self._approval_events[key]["event"].set()
            await query.answer("✅ Plan approved")

        elif data.startswith("reject_plan:"):
            task_id = int(data.split(":")[1])
            key = f"plan_{task_id}"
            if key in self._approval_events:
                self._approval_events[key]["response"] = "rejected"
                self._approval_events[key]["event"].set()
            await query.answer("❌ Plan rejected")

        elif data.startswith("modify_plan:"):
            task_id = int(data.split(":")[1])
            await query.answer("✏️ Reply with modifications")
            await query.message.reply_text(
                f"Reply to this message with your modifications for plan #{task_id}:"
            )
```

**Step 5: Run tests**

Run: `python -m pytest tests/test_phase13.py::TestInteractivePlanApproval -v`
Expected: PASS

**Step 6: Commit**

```bash
git add orchestrator.py telegram_bot.py tests/test_phase13.py
git commit -m "phase 13.5: interactive plan approval with Telegram buttons"
```

---

### Task 7: Phase 13 Full Test Run + Commit

**Files:**
- Test: `tests/test_phase13.py` (complete file)

**Step 1: Run all Phase 13 tests**

Run: `python -m pytest tests/test_phase13.py -v`
Expected: All PASS

**Step 2: Run full test suite to check for regressions**

Run: `python -m pytest tests/ -v --tb=short 2>&1 | tail -30`
Expected: No regressions (all 713+ tests pass)

**Step 3: Fix any failures**

If failures occur, fix them before proceeding.

**Step 4: Commit Phase 13 complete**

```bash
git add -A
git commit -m "phase 13: agent collaboration (blackboard, verification, ask_agent, parallel tasks, approval)"
```

---

### Task 8: Language Toolkit Interface (14.1)

**Files:**
- Create: `languages/__init__.py`
- Create: `languages/base.py`
- Create: `languages/python_lang.py`
- Create: `languages/javascript.py`
- Create: `languages/typescript.py`
- Create: `languages/go_lang.py`
- Create: `languages/rust_lang.py`
- Create: `languages/java_lang.py`
- Create: `languages/kotlin_lang.py`
- Test: `tests/test_phase14.py`

**Step 1: Write failing tests**

```python
# tests/test_phase14.py

import unittest

class TestLanguageToolkitInterface(unittest.TestCase):
    """14.1 — Language Toolkit Interface."""

    def test_base_toolkit_exists(self):
        from languages.base import LanguageToolkit
        import abc
        # Should be abstract
        self.assertTrue(hasattr(LanguageToolkit, "lint"))
        self.assertTrue(hasattr(LanguageToolkit, "format_code"))
        self.assertTrue(hasattr(LanguageToolkit, "test"))
        self.assertTrue(hasattr(LanguageToolkit, "typecheck"))
        self.assertTrue(hasattr(LanguageToolkit, "detect_imports"))
        self.assertTrue(hasattr(LanguageToolkit, "install_deps"))
        self.assertTrue(hasattr(LanguageToolkit, "compile"))
        self.assertTrue(hasattr(LanguageToolkit, "run"))

    def test_python_toolkit(self):
        from languages.python_lang import PythonToolkit
        tk = PythonToolkit()
        self.assertEqual(tk.name, "python")
        self.assertIn(".py", tk.extensions)

    def test_javascript_toolkit(self):
        from languages.javascript import JavaScriptToolkit
        tk = JavaScriptToolkit()
        self.assertEqual(tk.name, "javascript")
        self.assertIn(".js", tk.extensions)

    def test_typescript_toolkit(self):
        from languages.typescript import TypeScriptToolkit
        tk = TypeScriptToolkit()
        self.assertEqual(tk.name, "typescript")
        self.assertIn(".ts", tk.extensions)

    def test_go_toolkit(self):
        from languages.go_lang import GoToolkit
        tk = GoToolkit()
        self.assertEqual(tk.name, "go")
        self.assertIn(".go", tk.extensions)

    def test_rust_toolkit(self):
        from languages.rust_lang import RustToolkit
        tk = RustToolkit()
        self.assertEqual(tk.name, "rust")
        self.assertIn(".rs", tk.extensions)

    def test_java_toolkit(self):
        from languages.java_lang import JavaToolkit
        tk = JavaToolkit()
        self.assertEqual(tk.name, "java")
        self.assertIn(".java", tk.extensions)

    def test_kotlin_toolkit(self):
        from languages.kotlin_lang import KotlinToolkit
        tk = KotlinToolkit()
        self.assertEqual(tk.name, "kotlin")
        self.assertIn(".kt", tk.extensions)

    def test_get_toolkit_by_language(self):
        from languages import get_toolkit
        tk = get_toolkit("python")
        self.assertIsNotNone(tk)
        self.assertEqual(tk.name, "python")

    def test_get_toolkit_unknown_returns_none(self):
        from languages import get_toolkit
        tk = get_toolkit("brainfuck")
        self.assertIsNone(tk)

    def test_toolkit_has_idiomatic_patterns(self):
        from languages.python_lang import PythonToolkit
        tk = PythonToolkit()
        patterns = tk.get_idiomatic_patterns()
        self.assertIsInstance(patterns, str)
        self.assertIn("PEP", patterns)

    def test_toolkit_has_common_pitfalls(self):
        from languages.python_lang import PythonToolkit
        tk = PythonToolkit()
        pitfalls = tk.get_common_pitfalls()
        self.assertIsInstance(pitfalls, str)
        self.assertTrue(len(pitfalls) > 0)
```

**Step 2: Run to verify failure**

Run: `python -m pytest tests/test_phase14.py -v -x --tb=short 2>&1 | head -20`
Expected: FAIL

**Step 3: Create `languages/__init__.py`**

```python
# languages/__init__.py
"""Phase 14 — Multi-Language Coding Support."""
from typing import Optional

_TOOLKIT_REGISTRY: dict[str, type] = {}


def register_toolkit(name: str):
    """Decorator to register a language toolkit."""
    def decorator(cls):
        _TOOLKIT_REGISTRY[name] = cls
        return cls
    return decorator


def get_toolkit(language: str) -> Optional["languages.base.LanguageToolkit"]:
    """Get a toolkit instance by language name."""
    # Lazy import all toolkit modules to populate registry
    _ensure_loaded()
    cls = _TOOLKIT_REGISTRY.get(language.lower())
    if cls:
        return cls()
    return None


def get_all_toolkits() -> dict[str, "languages.base.LanguageToolkit"]:
    """Get all registered toolkit instances."""
    _ensure_loaded()
    return {name: cls() for name, cls in _TOOLKIT_REGISTRY.items()}


_loaded = False

def _ensure_loaded():
    global _loaded
    if _loaded:
        return
    _loaded = True
    try:
        import languages.python_lang
        import languages.javascript
        import languages.typescript
        import languages.go_lang
        import languages.rust_lang
        import languages.java_lang
        import languages.kotlin_lang
    except ImportError:
        pass
```

**Step 4: Create `languages/base.py`**

```python
# languages/base.py
"""
Abstract base class for language-specific toolkits.

Each language implements: lint, format, test, typecheck, detect_imports,
install_deps, compile, run.
"""
from abc import ABC, abstractmethod
from typing import Optional


class LanguageToolkit(ABC):
    """Abstract interface for language-specific development tools."""

    name: str = "unknown"
    extensions: tuple[str, ...] = ()

    # ── Tool commands (override in subclasses) ──

    @abstractmethod
    async def lint(self, filepath: str, **kwargs) -> str:
        """Run linter on a file. Returns output or error."""
        ...

    @abstractmethod
    async def format_code(self, filepath: str, **kwargs) -> str:
        """Auto-format a file. Returns output or error."""
        ...

    @abstractmethod
    async def test(self, path: str, **kwargs) -> str:
        """Run tests. Path can be file or directory."""
        ...

    @abstractmethod
    async def typecheck(self, path: str, **kwargs) -> str:
        """Run type checker. Returns output or error."""
        ...

    @abstractmethod
    async def detect_imports(self, filepath: str) -> list[str]:
        """Detect imported packages/modules from a file."""
        ...

    @abstractmethod
    async def install_deps(self, path: str, **kwargs) -> str:
        """Install dependencies for the project at path."""
        ...

    async def compile(self, path: str, **kwargs) -> str:
        """Compile project/file. Default: no-op for interpreted languages."""
        return "(no compilation needed)"

    async def run(self, filepath: str, **kwargs) -> str:
        """Run a file. Returns stdout/stderr."""
        return "(not implemented)"

    # ── Prompt helpers ──

    def get_idiomatic_patterns(self) -> str:
        """Return language-specific idiomatic patterns for prompt injection."""
        return ""

    def get_common_pitfalls(self) -> str:
        """Return common pitfalls to avoid for this language."""
        return ""

    def get_test_runner_info(self) -> str:
        """Return test runner commands and conventions."""
        return ""

    def get_import_conventions(self) -> str:
        """Return import style/ordering conventions."""
        return ""

    def get_prompt_rules(self) -> str:
        """Build full language rules block for agent prompt injection."""
        parts = [f"## Language: {self.name.title()}"]
        patterns = self.get_idiomatic_patterns()
        if patterns:
            parts.append(f"### Idiomatic Patterns\n{patterns}")
        pitfalls = self.get_common_pitfalls()
        if pitfalls:
            parts.append(f"### Common Pitfalls\n{pitfalls}")
        test_info = self.get_test_runner_info()
        if test_info:
            parts.append(f"### Testing\n{test_info}")
        import_info = self.get_import_conventions()
        if import_info:
            parts.append(f"### Import Conventions\n{import_info}")
        return "\n\n".join(parts)
```

**Step 5: Create all language toolkit implementations**

Create `languages/python_lang.py`, `languages/javascript.py`, `languages/typescript.py`, `languages/go_lang.py`, `languages/rust_lang.py`, `languages/java_lang.py`, `languages/kotlin_lang.py`.

Each follows the same pattern — see `python_lang.py` as the example:

```python
# languages/python_lang.py
"""Python language toolkit."""
import asyncio
from languages.base import LanguageToolkit
from languages import register_toolkit


@register_toolkit("python")
class PythonToolkit(LanguageToolkit):
    name = "python"
    extensions = (".py", ".pyi", ".pyw")

    async def lint(self, filepath, **kwargs):
        proc = await asyncio.create_subprocess_exec(
            "ruff", "check", filepath,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        return (stdout or stderr or b"").decode()[:5000]

    async def format_code(self, filepath, **kwargs):
        proc = await asyncio.create_subprocess_exec(
            "ruff", "format", filepath,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        return (stdout or stderr or b"✅ Formatted").decode()[:5000]

    async def test(self, path, **kwargs):
        proc = await asyncio.create_subprocess_exec(
            "python", "-m", "pytest", path, "-v", "--tb=short",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        return (stdout.decode() + stderr.decode())[:10000]

    async def typecheck(self, path, **kwargs):
        proc = await asyncio.create_subprocess_exec(
            "mypy", path, "--no-error-summary",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        return (stdout or stderr or b"").decode()[:5000]

    async def detect_imports(self, filepath):
        import ast
        try:
            with open(filepath) as f:
                tree = ast.parse(f.read())
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
            return imports
        except Exception:
            return []

    async def install_deps(self, path, **kwargs):
        proc = await asyncio.create_subprocess_exec(
            "pip", "install", "-r", f"{path}/requirements.txt",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        return (stdout or stderr or b"").decode()[:5000]

    async def run(self, filepath, **kwargs):
        proc = await asyncio.create_subprocess_exec(
            "python", filepath,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        return (stdout.decode() + stderr.decode())[:10000]

    def get_idiomatic_patterns(self):
        return (
            "- Follow PEP 8 naming conventions (snake_case for functions/variables, PascalCase for classes)\n"
            "- Use type hints for all function signatures\n"
            "- Prefer list/dict comprehensions over loops for simple transformations\n"
            "- Use context managers (with) for resource management\n"
            "- Use f-strings for string formatting\n"
            "- Use pathlib.Path over os.path for file operations"
        )

    def get_common_pitfalls(self):
        return (
            "- Mutable default arguments (def f(x=[]))\n"
            "- Late binding closures in loops\n"
            "- Not handling None returns properly\n"
            "- Forgetting to await async functions\n"
            "- Import cycles between modules"
        )

    def get_test_runner_info(self):
        return (
            "- Test runner: `pytest`\n"
            "- Test file naming: `test_*.py` or `*_test.py`\n"
            "- Run: `python -m pytest tests/ -v`\n"
            "- Coverage: `python -m pytest --cov=src tests/`"
        )

    def get_import_conventions(self):
        return (
            "- Standard library imports first\n"
            "- Third-party imports second\n"
            "- Local imports third\n"
            "- Blank line between each group\n"
            "- Alphabetical within each group"
        )
```

Other language implementations follow the same pattern with language-specific commands:
- **javascript.py**: eslint, prettier, jest/vitest, npm
- **typescript.py**: extends JS + tsc
- **go_lang.py**: go vet, gofmt, go test, go build
- **rust_lang.py**: cargo clippy, rustfmt, cargo test, cargo build
- **java_lang.py**: checkstyle/spotless, google-java-format, gradle test/mvn test, javac/gradle build
- **kotlin_lang.py**: ktlint, ktlint --format, gradle test, kotlinc/gradle build

**Step 6: Run tests**

Run: `python -m pytest tests/test_phase14.py::TestLanguageToolkitInterface -v`
Expected: PASS

**Step 7: Commit**

```bash
git add languages/__init__.py languages/base.py languages/python_lang.py languages/javascript.py languages/typescript.py languages/go_lang.py languages/rust_lang.py languages/java_lang.py languages/kotlin_lang.py tests/test_phase14.py
git commit -m "phase 14.1: language toolkit interface + 7 implementations"
```

---

### Task 9: Language-Aware Agent Prompts (14.2)

**Files:**
- Modify: `agents/base.py` (detect language, inject toolkit prompt rules)
- Test: `tests/test_phase14.py`

**Step 1: Write failing tests**

```python
class TestLanguageAwarePrompts(unittest.TestCase):
    """14.2 — Language-Aware Agent Prompts."""

    def test_language_detection_in_base_agent(self):
        source = open("agents/base.py").read()
        self.assertIn("get_toolkit", source)
        self.assertIn("get_prompt_rules", source)

    def test_python_toolkit_prompt_rules(self):
        from languages.python_lang import PythonToolkit
        tk = PythonToolkit()
        rules = tk.get_prompt_rules()
        self.assertIn("Python", rules)
        self.assertIn("PEP", rules)
        self.assertIn("pytest", rules)
```

**Step 2: Run to verify failure**

Run: `python -m pytest tests/test_phase14.py::TestLanguageAwarePrompts -v -x`
Expected: FAIL

**Step 3: Inject language rules in `agents/base.py` `_build_context()`**

After the project profile injection block:

```python
        # ── Phase 14.2: Language-aware prompt injection ──
        try:
            from languages import get_toolkit
            # Detect language from project profile or task context
            lang = None
            if task_context.get("language"):
                lang = task_context["language"]
            elif project_profile and project_profile.get("language"):
                lang = project_profile.get("language")

            if lang and lang != "unknown":
                toolkit = get_toolkit(lang)
                if toolkit:
                    lang_rules = toolkit.get_prompt_rules()
                    if lang_rules:
                        parts.append(lang_rules)
        except Exception as exc:
            logger.debug(f"Language prompt injection failed (non-critical): {exc}")
```

Note: Need to capture `project_profile` from the Phase 12.6 block (move it to a local variable).

**Step 4: Run tests**

Run: `python -m pytest tests/test_phase14.py::TestLanguageAwarePrompts -v`
Expected: PASS

**Step 5: Commit**

```bash
git add agents/base.py tests/test_phase14.py
git commit -m "phase 14.2: language-aware agent prompts"
```

---

### Task 10: Multi-Runtime Sandbox (14.3)

**Files:**
- Modify: `tools/sandbox.py` (or wherever `run_code` is defined — add `language` param)
- Create: `Dockerfile.multi` (multi-runtime sandbox Dockerfile)
- Test: `tests/test_phase14.py`

**Step 1: Write failing tests**

```python
class TestMultiRuntimeSandbox(unittest.TestCase):
    """14.3 — Multi-Runtime Sandbox."""

    def test_run_code_accepts_language(self):
        source = open("tools/sandbox.py").read()
        self.assertIn("language", source)

    def test_language_runtime_map(self):
        from tools.sandbox import LANGUAGE_RUNTIMES
        self.assertIn("python", LANGUAGE_RUNTIMES)
        self.assertIn("javascript", LANGUAGE_RUNTIMES)
        self.assertIn("go", LANGUAGE_RUNTIMES)
        self.assertIn("rust", LANGUAGE_RUNTIMES)
        self.assertIn("java", LANGUAGE_RUNTIMES)

    def test_dockerfile_multi_exists(self):
        import os
        self.assertTrue(os.path.exists("Dockerfile.multi") or os.path.exists("sandbox/Dockerfile.multi"))
```

**Step 2: Run to verify failure**

Run: `python -m pytest tests/test_phase14.py::TestMultiRuntimeSandbox -v -x`
Expected: FAIL

**Step 3: Add `LANGUAGE_RUNTIMES` and `language` parameter to `tools/sandbox.py`**

```python
# Add near top of tools/sandbox.py:
LANGUAGE_RUNTIMES: dict[str, dict] = {
    "python": {"cmd": ["python"], "ext": ".py", "image": "python:3.12-slim"},
    "javascript": {"cmd": ["node"], "ext": ".js", "image": "node:20-slim"},
    "typescript": {"cmd": ["npx", "tsx"], "ext": ".ts", "image": "node:20-slim"},
    "go": {"cmd": ["go", "run"], "ext": ".go", "image": "golang:1.22-alpine"},
    "rust": {"cmd": ["rustc", "--edition", "2021", "-o", "/tmp/out"], "ext": ".rs", "image": "rust:1.77-slim", "compile_then_run": True},
    "java": {"cmd": ["java"], "ext": ".java", "image": "eclipse-temurin:21-jdk-alpine"},
    "kotlin": {"cmd": ["kotlinc", "-script"], "ext": ".kts", "image": "eclipse-temurin:21-jdk-alpine"},
}
```

Update `run_code()` to accept a `language` parameter and select the appropriate runtime.

**Step 4: Create `Dockerfile.multi`**

```dockerfile
FROM python:3.12-slim

# Node.js 20 LTS
RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    npm install -g typescript tsx

# Go 1.22
RUN curl -fsSL https://go.dev/dl/go1.22.0.linux-amd64.tar.gz | tar -C /usr/local -xzf - && \
    ln -s /usr/local/go/bin/go /usr/bin/go

# Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    ln -s /root/.cargo/bin/rustc /usr/bin/rustc && \
    ln -s /root/.cargo/bin/cargo /usr/bin/cargo

# Java 21
RUN apt-get install -y default-jdk

# Python packages
RUN pip install ruff pytest mypy

WORKDIR /workspace
```

**Step 5: Run tests**

Run: `python -m pytest tests/test_phase14.py::TestMultiRuntimeSandbox -v`
Expected: PASS

**Step 6: Commit**

```bash
git add tools/sandbox.py Dockerfile.multi tests/test_phase14.py
git commit -m "phase 14.3: multi-runtime sandbox"
```

---

### Task 11: Language-Aware Dependency Detection (14.4)

**Files:**
- Modify: `tools/deps.py` (add multi-language dependency detection)
- Test: `tests/test_phase14.py`

**Step 1: Write failing tests**

```python
class TestLanguageDependencyDetection(unittest.TestCase):
    """14.4 — Language-Aware Dependency Detection."""

    def test_deps_supports_multiple_languages(self):
        source = open("tools/deps.py").read()
        self.assertIn("package.json", source)
        self.assertIn("go.mod", source)
        self.assertIn("Cargo.toml", source)
        self.assertIn("build.gradle", source)
        self.assertIn("pom.xml", source)

    def test_detect_project_type(self):
        from tools.deps import detect_project_type
        import tempfile, os
        with tempfile.TemporaryDirectory() as td:
            open(os.path.join(td, "package.json"), "w").write("{}")
            result = detect_project_type(td)
            self.assertEqual(result, "javascript")

    def test_detect_project_type_python(self):
        from tools.deps import detect_project_type
        import tempfile, os
        with tempfile.TemporaryDirectory() as td:
            open(os.path.join(td, "requirements.txt"), "w").write("flask")
            result = detect_project_type(td)
            self.assertEqual(result, "python")

    def test_detect_project_type_go(self):
        from tools.deps import detect_project_type
        import tempfile, os
        with tempfile.TemporaryDirectory() as td:
            open(os.path.join(td, "go.mod"), "w").write("module example.com/foo")
            result = detect_project_type(td)
            self.assertEqual(result, "go")

    def test_detect_project_type_rust(self):
        from tools.deps import detect_project_type
        import tempfile, os
        with tempfile.TemporaryDirectory() as td:
            open(os.path.join(td, "Cargo.toml"), "w").write("[package]")
            result = detect_project_type(td)
            self.assertEqual(result, "rust")

    def test_detect_project_type_java_gradle(self):
        from tools.deps import detect_project_type
        import tempfile, os
        with tempfile.TemporaryDirectory() as td:
            open(os.path.join(td, "build.gradle"), "w").write("apply plugin: 'java'")
            result = detect_project_type(td)
            self.assertIn(result, ("java", "kotlin"))

    def test_detect_project_type_java_maven(self):
        from tools.deps import detect_project_type
        import tempfile, os
        with tempfile.TemporaryDirectory() as td:
            open(os.path.join(td, "pom.xml"), "w").write("<project>")
            result = detect_project_type(td)
            self.assertEqual(result, "java")
```

**Step 2: Run to verify failure**

Run: `python -m pytest tests/test_phase14.py::TestLanguageDependencyDetection -v -x`
Expected: FAIL

**Step 3: Add multi-language detection to `tools/deps.py`**

```python
# Project type detection by indicator files
_PROJECT_INDICATORS: list[tuple[str, str]] = [
    ("requirements.txt", "python"),
    ("setup.py", "python"),
    ("pyproject.toml", "python"),
    ("Pipfile", "python"),
    ("package.json", "javascript"),
    ("go.mod", "go"),
    ("Cargo.toml", "rust"),
    ("build.gradle", "java"),
    ("build.gradle.kts", "kotlin"),
    ("pom.xml", "java"),
    ("settings.gradle", "java"),
    ("settings.gradle.kts", "kotlin"),
]


def detect_project_type(path: str) -> str:
    """Detect project type from indicator files."""
    import os
    for indicator, lang in _PROJECT_INDICATORS:
        if os.path.exists(os.path.join(path, indicator)):
            return lang
    return "unknown"


# Install commands per language
_INSTALL_COMMANDS: dict[str, list[str]] = {
    "python": ["pip", "install", "-r", "requirements.txt"],
    "javascript": ["npm", "install"],
    "go": ["go", "mod", "download"],
    "rust": ["cargo", "build"],
    "java": ["gradle", "dependencies"],  # or mvn dependency:resolve
    "kotlin": ["gradle", "dependencies"],
}
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_phase14.py::TestLanguageDependencyDetection -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tools/deps.py tests/test_phase14.py
git commit -m "phase 14.4: language-aware dependency detection"
```

---

### Task 12: Phase 14 Full Test Run + Final Commit

**Files:**
- Test: All phase 14 tests

**Step 1: Run all Phase 14 tests**

Run: `python -m pytest tests/test_phase14.py -v`
Expected: All PASS

**Step 2: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short 2>&1 | tail -30`
Expected: All previous tests + new tests pass (730+ tests)

**Step 3: Fix any failures**

**Step 4: Final commit for Phase 13 + 14**

```bash
git add -A
git commit -m "phase 14: multi-language coding (toolkits, prompts, sandbox, deps)"
```

---

## Implementation Notes

### Key Integration Points

1. **Blackboard ↔ Agent Context**: Blackboard state is injected into agent prompts in `_build_context()` so agents have awareness of project state without parsing text blobs.

2. **Plan Verification ↔ Orchestrator**: Verification runs synchronously before subtask creation. Issues are logged + sent via Telegram but don't block execution (warning-only for now).

3. **ask_agent ↔ ReAct Loop**: When an agent emits `ask_agent`, the base agent creates an inline sub-execution. This is synchronous within the requesting agent's loop but limited to 120s.

4. **Language Toolkit ↔ Agent Prompts**: Language detection cascades: task context → project profile → file extensions. Toolkit prompt rules are appended to the user message, not the system prompt.

5. **Multi-Runtime ↔ Sandbox**: The `run_code` tool accepts a `language` parameter and selects the appropriate runtime from `LANGUAGE_RUNTIMES`.

### Dependencies Between Tasks

```
Task 1 (Blackboard module) ─→ Task 2 (Tools + injection)
                                    ↓
Task 3 (Verification) ────────→ Task 7 (Phase 13 commit)
                                    ↑
Task 4 (ask_agent) ────────────→ ┤
                                    │
Task 5 (Parallel tasks) ──────→ ┤
                                    │
Task 6 (Interactive approval) ─→ ┘

Task 8 (Toolkit interface) ──→ Task 9 (Language prompts)
                                    ↓
Task 10 (Multi-runtime) ─────→ Task 12 (Phase 14 commit)
                                    ↑
Task 11 (Dependency detect) ──→ ┘
```
