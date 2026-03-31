# Extraction-Ready Refactoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make 7 KutAI modules extraction-ready — clean imports, injectable deps, internal packaging boundaries — without changing any external behavior.

**Architecture:** Each module gets two changes: (1) replace hard `src.infra.logging_config` import with stdlib `logging.getLogger(__name__)`, and (2) make cross-module dependencies injectable with backward-compatible defaults. No classes where functions work fine. No behavior changes.

**Tech Stack:** Python stdlib logging, existing pytest suite

**Why stdlib logging works:** `get_logger("x")` returns a `_ContextLogger` wrapping `logging.getLogger("x")`. The JSON formatter and file/ntfy sinks are attached to the **root logger** by `init_logging()`, so they capture output from all loggers regardless of how they were created. The only loss is `logger.info("msg", key=val)` keyword-context syntax — but all 7 target modules use f-strings only (except `llm_dispatcher.py` which needs ~5 lines converted).

**Backward compatibility guarantee:** Every module keeps its existing public API. Callers don't change. Existing tests must pass without modification.

---

### Task 1: Decouple `embeddings.py` from `src.infra.logging_config`

**Files:**
- Modify: `src/memory/embeddings.py:19-21`
- Test: `tests/test_embeddings_standalone.py` (new — verifies module imports without src.infra)

- [ ] **Step 1: Write the import-isolation test**

```python
# tests/test_embeddings_standalone.py
"""Verify embeddings module has no hard coupling to src.infra."""
import importlib
import sys


def test_embeddings_uses_stdlib_logging():
    """embeddings.py must use stdlib logging, not src.infra.logging_config."""
    import src.memory.embeddings as mod
    source = importlib.util.find_spec("src.memory.embeddings").origin
    with open(source) as f:
        text = f.read()
    assert "from src.infra.logging_config" not in text
    assert "import logging" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_embeddings_standalone.py -v`
Expected: FAIL — embeddings.py still imports from src.infra.logging_config

- [ ] **Step 3: Replace logging import in embeddings.py**

In `src/memory/embeddings.py`, replace lines 19-21:

```python
# OLD:
from src.infra.logging_config import get_logger

logger = get_logger("memory.embeddings")
```

With:

```python
# NEW:
import logging

logger = logging.getLogger(__name__)
```

- [ ] **Step 4: Run the isolation test**

Run: `python -m pytest tests/test_embeddings_standalone.py -v`
Expected: PASS

- [ ] **Step 5: Run existing tests to verify no regression**

Run: `python -m pytest tests/ -x -q --timeout=30 2>&1 | head -40`
Expected: All previously-passing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add src/memory/embeddings.py tests/test_embeddings_standalone.py
git commit -m "refactor(embeddings): decouple from src.infra.logging_config

Use stdlib logging.getLogger(__name__) instead of custom get_logger().
Root logger sinks (JSON, ntfy) still capture all output."
```

---

### Task 2: Decouple `vector_store.py` — logging + injectable embeddings

**Files:**
- Modify: `src/memory/vector_store.py:28-31, 59, 87, 162, 247`
- Test: `tests/test_vector_store_standalone.py` (new)

- [ ] **Step 1: Write the isolation test**

```python
# tests/test_vector_store_standalone.py
"""Verify vector_store has no hard coupling to src.infra and accepts injectable deps."""
import importlib


def test_vector_store_uses_stdlib_logging():
    source = importlib.util.find_spec("src.memory.vector_store").origin
    with open(source) as f:
        text = f.read()
    assert "from src.infra.logging_config" not in text
    assert "import logging" in text


def test_init_store_accepts_embed_fn():
    """init_store should accept an optional embed_fn parameter."""
    import inspect
    from src.memory.vector_store import init_store
    sig = inspect.signature(init_store)
    assert "embed_fn" in sig.parameters, "init_store must accept embed_fn parameter"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_vector_store_standalone.py -v`
Expected: FAIL on both assertions

- [ ] **Step 3: Replace logging and make embedding dependency injectable**

In `src/memory/vector_store.py`, replace lines 24-31:

```python
# OLD:
import os
import time
from typing import Optional

from src.infra.logging_config import get_logger
from src.memory.embeddings import get_embedding, get_expected_dimension

logger = get_logger("memory.vector_store")
```

With:

```python
# NEW:
import logging
import os
import time
from typing import Optional, Callable, Awaitable

logger = logging.getLogger(__name__)

# --- Injectable dependencies (default to src.memory.embeddings) -----------
_embed_fn: Callable[..., Awaitable[Optional[list[float]]]] | None = None
_dimension_fn: Callable[[], int] | None = None


def _get_embed_fn():
    global _embed_fn
    if _embed_fn is None:
        from src.memory.embeddings import get_embedding
        _embed_fn = get_embedding
    return _embed_fn


def _get_dimension_fn():
    global _dimension_fn
    if _dimension_fn is None:
        from src.memory.embeddings import get_expected_dimension
        _dimension_fn = get_expected_dimension
    return _dimension_fn
```

Then update `init_store` signature (line ~59):

```python
async def init_store(persist_dir: str | None = None, embed_fn=None, dimension_fn=None) -> bool:
    global _embed_fn, _dimension_fn
    if embed_fn is not None:
        _embed_fn = embed_fn
    if dimension_fn is not None:
        _dimension_fn = dimension_fn
```

Replace the `get_expected_dimension()` call in init_store (line ~87):

```python
# OLD:
expected_dim = get_expected_dimension()
# NEW:
expected_dim = _get_dimension_fn()()
```

Replace `get_embedding` calls in `embed_and_store` (line ~162) and `query` (line ~247):

```python
# OLD:
embedding = await get_embedding(text, is_query=False)
# NEW:
embedding = await _get_embed_fn()(text, is_query=False)
```

```python
# OLD:
embedding = await get_embedding(text, is_query=True)
# NEW:
embedding = await _get_embed_fn()(text, is_query=True)
```

- [ ] **Step 4: Run isolation test**

Run: `python -m pytest tests/test_vector_store_standalone.py -v`
Expected: PASS

- [ ] **Step 5: Run existing tests**

Run: `python -m pytest tests/ -x -q --timeout=30 2>&1 | head -40`
Expected: All previously-passing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add src/memory/vector_store.py tests/test_vector_store_standalone.py
git commit -m "refactor(vector_store): decouple logging + injectable embeddings

- stdlib logging instead of src.infra.logging_config
- embed_fn/dimension_fn injectable via init_store() params
- lazy import from src.memory.embeddings as default fallback"
```

---

### Task 3: Decouple `shell.py` from `src.infra.logging_config`

**Files:**
- Modify: `src/tools/shell.py:10-13`
- Test: `tests/test_shell_standalone.py` (new)

- [ ] **Step 1: Write the isolation test**

```python
# tests/test_shell_standalone.py
"""Verify shell.py has no hard coupling to src.infra."""
import importlib


def test_shell_uses_stdlib_logging():
    source = importlib.util.find_spec("src.tools.shell").origin
    with open(source) as f:
        text = f.read()
    assert "from src.infra.logging_config" not in text
    assert "import logging" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_shell_standalone.py -v`
Expected: FAIL

- [ ] **Step 3: Replace logging import in shell.py**

In `src/tools/shell.py`, replace lines 10-13:

```python
# OLD:
from src.infra.logging_config import get_logger

logger = get_logger("tools.shell")
```

With:

```python
# NEW:
import logging

logger = logging.getLogger(__name__)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_shell_standalone.py tests/ -x -q --timeout=30 2>&1 | head -40`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tools/shell.py tests/test_shell_standalone.py
git commit -m "refactor(shell): decouple from src.infra.logging_config"
```

---

### Task 4: Decouple `free_apis.py` — logging + lazy DB

**Files:**
- Modify: `src/tools/free_apis.py:17-19`
- Test: `tests/test_free_apis_standalone.py` (new)

- [ ] **Step 1: Write the isolation test**

```python
# tests/test_free_apis_standalone.py
"""Verify free_apis.py has no hard coupling to src.infra at import time."""
import importlib


def test_free_apis_uses_stdlib_logging():
    source = importlib.util.find_spec("src.tools.free_apis").origin
    with open(source) as f:
        text = f.read()
    assert "from src.infra.logging_config" not in text
    assert "import logging" in text


def test_free_apis_no_toplevel_db_import():
    """DB imports must be lazy (inside functions), not at module level."""
    source = importlib.util.find_spec("src.tools.free_apis").origin
    with open(source) as f:
        lines = f.readlines()
    # Check only non-indented lines (module level)
    for i, line in enumerate(lines, 1):
        stripped = line.lstrip()
        if stripped != line:
            continue  # indented = inside function, OK
        assert "from src.infra.db" not in line, f"Top-level DB import at line {i}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_free_apis_standalone.py -v`
Expected: FAIL on logging assertion

- [ ] **Step 3: Replace logging import**

In `src/tools/free_apis.py`, replace lines 17-19:

```python
# OLD:
from src.infra.logging_config import get_logger

logger = get_logger("tools.free_apis")
```

With:

```python
# NEW:
import logging

logger = logging.getLogger(__name__)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_free_apis_standalone.py tests/test_free_apis.py tests/test_api_discovery.py -x -v --timeout=30 2>&1 | head -40`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tools/free_apis.py tests/test_free_apis_standalone.py
git commit -m "refactor(free_apis): decouple from src.infra.logging_config

DB imports were already lazy (inside functions). Only logging needed fixing."
```

---

### Task 5: Decouple `llm_dispatcher.py` — logging + convert kv-arg log calls

**Files:**
- Modify: `src/core/llm_dispatcher.py:32-34` and ~18 kv-arg log call sites
- Test: `tests/test_dispatcher_standalone.py` (new)

This module uses `logger.info("msg", key=val)` syntax (ContextLogger feature) in ~18 call sites. Those must be converted to f-strings before switching to stdlib.

- [ ] **Step 1: Write the isolation test**

```python
# tests/test_dispatcher_standalone.py
"""Verify llm_dispatcher has no hard coupling to src.infra at top level."""
import importlib


def test_dispatcher_uses_stdlib_logging():
    source = importlib.util.find_spec("src.core.llm_dispatcher").origin
    with open(source) as f:
        text = f.read()
    assert "from src.infra.logging_config" not in text
    assert "import logging" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dispatcher_standalone.py -v`
Expected: FAIL

- [ ] **Step 3: Find and convert all kv-arg log calls**

Read `src/core/llm_dispatcher.py` fully first. Find every `logger.xxx("msg", key=val)` call and convert to f-string format.

Example conversions:

```python
# OLD:
logger.info("swap recorded", recent_swaps=len(self._timestamps), budget_remaining=max(0, self.max_swaps - len(self._timestamps)))
# NEW:
logger.info(f"swap recorded | recent_swaps={len(self._timestamps)} budget_remaining={max(0, self.max_swaps - len(self._timestamps))}")
```

Then replace the import:

```python
# OLD:
from src.infra.logging_config import get_logger

logger = get_logger("core.llm_dispatcher")
```

With:

```python
# NEW:
import logging

logger = logging.getLogger(__name__)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_dispatcher_standalone.py tests/test_llm_dispatcher.py -x -v --timeout=30 2>&1 | head -40`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/core/llm_dispatcher.py tests/test_dispatcher_standalone.py
git commit -m "refactor(llm_dispatcher): decouple from src.infra.logging_config

Convert kv-arg log calls to f-strings for stdlib compatibility.
Lazy imports for router/models already in place."
```

---

### Task 6: Decouple `web_search.py` — logging + injectable shell executor

**Files:**
- Modify: `src/tools/web_search.py:22-25`
- Test: `tests/test_web_search_standalone.py` (new)

- [ ] **Step 1: Write the isolation test**

```python
# tests/test_web_search_standalone.py
"""Verify web_search.py has no hard coupling to src.infra or src.tools at top level."""
import importlib


def test_web_search_uses_stdlib_logging():
    source = importlib.util.find_spec("src.tools.web_search").origin
    with open(source) as f:
        text = f.read()
    assert "from src.infra.logging_config" not in text
    assert "import logging" in text


def test_web_search_no_toplevel_tools_import():
    """run_shell import must be lazy, not at module level."""
    source = importlib.util.find_spec("src.tools.web_search").origin
    with open(source) as f:
        lines = f.readlines()
    for i, line in enumerate(lines, 1):
        stripped = line.lstrip()
        if stripped != line:
            continue  # indented = inside function
        assert "from src.tools import" not in line, f"Top-level src.tools import at line {i}"
        assert "from src.tools." not in line, f"Top-level src.tools import at line {i}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_web_search_standalone.py -v`
Expected: FAIL on both assertions

- [ ] **Step 3: Replace logging and make shell dependency lazy**

In `src/tools/web_search.py`, replace lines 22-25:

```python
# OLD:
from src.infra.logging_config import get_logger
from src.tools import run_shell

logger = get_logger("tools.web_search")
```

With:

```python
# NEW:
import logging

logger = logging.getLogger(__name__)

# Injectable shell executor — lazy import from src.tools by default
_shell_fn = None


def _get_shell_fn():
    global _shell_fn
    if _shell_fn is None:
        from src.tools import run_shell
        _shell_fn = run_shell
    return _shell_fn
```

Then find all calls to `run_shell(...)` in the file and replace with `_get_shell_fn()(...)`. These should be inside async functions, not at module level.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_web_search_standalone.py tests/test_search_guard.py tests/test_search_depth.py -x -v --timeout=30 2>&1 | head -40`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tools/web_search.py tests/test_web_search_standalone.py
git commit -m "refactor(web_search): decouple logging + lazy shell import

- stdlib logging instead of src.infra.logging_config
- run_shell imported lazily via _get_shell_fn() for extraction readiness"
```

---

### Task 7: Decouple `rag.py` — logging + injectable vector_store/embeddings

**Files:**
- Modify: `src/memory/rag.py:15-22`
- Test: `tests/test_rag_standalone.py` (new)

- [ ] **Step 1: Write the isolation test**

```python
# tests/test_rag_standalone.py
"""Verify rag.py has no hard coupling to src.infra or src.memory at top level."""
import importlib


def test_rag_uses_stdlib_logging():
    source = importlib.util.find_spec("src.memory.rag").origin
    with open(source) as f:
        text = f.read()
    assert "from src.infra.logging_config" not in text
    assert "import logging" in text


def test_rag_no_toplevel_memory_imports():
    """vector_store and embeddings imports must be lazy."""
    source = importlib.util.find_spec("src.memory.rag").origin
    with open(source) as f:
        lines = f.readlines()
    for i, line in enumerate(lines, 1):
        stripped = line.lstrip()
        if stripped != line:
            continue  # indented
        assert "from src.memory.vector_store" not in line, f"Top-level vector_store import at line {i}"
        assert "from src.memory.embeddings" not in line, f"Top-level embeddings import at line {i}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_rag_standalone.py -v`
Expected: FAIL

- [ ] **Step 3: Replace logging and make deps lazy**

In `src/memory/rag.py`, replace lines 15-22:

```python
# OLD:
import time
from typing import Optional

from src.infra.logging_config import get_logger
from src.memory.vector_store import is_ready, query, embed_and_store
from src.memory.embeddings import get_embedding

logger = get_logger("memory.rag")
```

With:

```python
# NEW:
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

# --- Lazy imports for extraction readiness --------------------------------
_vs_is_ready = None
_vs_query = None
_vs_embed_and_store = None
_emb_get_embedding = None


def _load_deps():
    global _vs_is_ready, _vs_query, _vs_embed_and_store, _emb_get_embedding
    if _vs_is_ready is None:
        from src.memory.vector_store import is_ready, query, embed_and_store
        from src.memory.embeddings import get_embedding
        _vs_is_ready = is_ready
        _vs_query = query
        _vs_embed_and_store = embed_and_store
        _emb_get_embedding = get_embedding
```

Then at the top of each public function that uses these deps, add `_load_deps()` and replace bare function calls:

```python
# OLD:
if not is_ready():
# NEW:
_load_deps()
if not _vs_is_ready():
```

```python
# OLD:
results = await query(text, collection, top_k)
# NEW:
results = await _vs_query(text, collection, top_k)
```

(Apply same pattern to all `is_ready`, `query`, `embed_and_store`, `get_embedding` calls in the file.)

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_rag_standalone.py tests/ -x -q --timeout=30 2>&1 | head -40`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/memory/rag.py tests/test_rag_standalone.py
git commit -m "refactor(rag): decouple logging + lazy vector_store/embeddings

All src.memory imports now lazy-loaded via _load_deps().
Extraction-ready: can swap deps without changing call sites."
```

---

### Task 8: Full regression test

**Files:**
- No file changes — verification only

- [ ] **Step 1: Run complete test suite**

Run: `python -m pytest tests/ -x -q --timeout=60 2>&1 | tail -20`
Expected: Same pass/fail count as before refactoring. No new failures.

- [ ] **Step 2: Verify import-time isolation for all 7 modules**

Run all standalone tests together:

```bash
python -m pytest tests/test_embeddings_standalone.py tests/test_vector_store_standalone.py tests/test_shell_standalone.py tests/test_free_apis_standalone.py tests/test_dispatcher_standalone.py tests/test_web_search_standalone.py tests/test_rag_standalone.py -v
```

Expected: All 7+ tests PASS.

- [ ] **Step 3: Smoke-test module imports in isolation**

```bash
python -c "
import src.memory.embeddings
import src.memory.vector_store
import src.tools.shell
import src.tools.free_apis
import src.core.llm_dispatcher
import src.tools.web_search
import src.memory.rag
print('All 7 modules import successfully')
"
```

Expected: No import errors.

- [ ] **Step 4: Commit test results confirmation**

No commit needed — this is a verification gate only.
