# tools/__init__.py
"""
Tool registry — every tool an agent can invoke.

Each entry bundles the async callable, a human-readable description, and
a JSON usage example.  The full catalogue can be injected into agent
prompts via ``get_tool_descriptions()``.
"""

import inspect
from typing import Any, Optional

from ..parsing.code_embeddings import reindex_file
from ..infra.logging_config import get_logger

logger = get_logger("tools")

# ---------------------------------------------------------------------------
# Imports — must match the public API of each module we built
# ---------------------------------------------------------------------------
from .shell import run_shell, run_shell_with_stdin, run_shell_sequential
from .workspace import get_file_tree, read_file, write_file, detect_project
from .edit_file import edit_file
from .patch_file import patch_file
from .apply_diff import apply_diff
from .linting import auto_lint
from .deps import verify_dependencies
from .git_ops import (
    git_init,
    git_commit,
    git_branch,
    git_log,
    git_diff,
    git_rollback,
    git_status,
)
from ..memory.ingest import ingest_document as _ingest_fn

# AST-aware code tools (Phase 8)
from .ast_tools import (
    get_function,
    replace_function,
    list_classes,
    list_functions,
    get_imports,
)

# Codebase indexing tools (Phase 8)
from .codebase_index import (
    index_workspace,
    query_codebase,
    codebase_map,
)

# Optional / pre-existing tools — degrade gracefully if absent
_optional_tools: dict[str, dict[str, Any]] = {}

try:
    from .web_search import web_search

    _optional_tools["web_search"] = {
        "function": web_search,
        "description": "Search the web. Args: query (str)",
        "example": '{"action": "tool_call", "tool": "web_search", "args": {"query": "FastAPI websocket tutorial"}}',
    }
except Exception as e:
    logger.warning(f"web_search tool not available — {type(e).__name__}: {e}")

try:
    from .code_runner import run_code

    _optional_tools["run_code"] = {
        "function": run_code,
        "description": (
            "Run a code snippet directly. "
            "Args: code (str), language (str, optional, default 'python')"
        ),
        "example": '{"action": "tool_call", "tool": "run_code", "args": {"code": "print(2+2)"}}',
    }
except ImportError as e:
    logger.warning(f"run_code tool not available — {type(e).__name__}: {e}")

try:
    from .http_client import http_request

    _optional_tools["http_request"] = {
        "function": http_request,
        "description": (
            "Make an HTTP request to a URL. "
            "Args: method (str: GET/POST/PUT/DELETE), url (str), "
            "headers (dict, optional), body (str, optional), "
            "timeout (int, optional, default 30)"
        ),
        "example": (
            '{"action": "tool_call", "tool": "http_request", '
            '"args": {"method": "GET", "url": "https://api.example.com/data"}}'
        ),
    }
except Exception as e:
    logger.warning(f"http_request tool not available — {type(e).__name__}: {e}")

try:
    from .download import download_file

    _optional_tools["download_file"] = {
        "function": download_file,
        "description": (
            "Download a file from a URL into the workspace. "
            "Args: url (str), save_as (str), timeout (int, optional)"
        ),
        "example": (
            '{"action": "tool_call", "tool": "download_file", '
            '"args": {"url": "https://example.com/data.csv", "save_as": "data.csv"}}'
        ),
    }
except Exception as e:
    logger.warning(f"download_file tool not available — {type(e).__name__}: {e}")

# Document / web tools (Phase 5)
try:
    from .web_extract import extract_url

    _optional_tools["extract_url"] = {
        "function": extract_url,
        "description": "Extract clean text from a URL. Args: url (str)",
        "example": '{"action": "tool_call", "tool": "extract_url", "args": {"url": "https://example.com/article"}}',
    }
except Exception as e:
    logger.warning(f"extract_url tool not available — {e}")

try:
    from .vision import analyze_image

    _optional_tools["analyze_image"] = {
        "function": analyze_image,
        "description": "Analyze an image file with a vision model. Args: filepath (str), question (str, optional)",
        "example": '{"action": "tool_call", "tool": "analyze_image", "args": {"filepath": "screenshot.png", "question": "What does this UI show?"}}',
    }
except Exception as e:
    logger.warning(f"analyze_image tool not available — {e}")

try:
    from .documents import extract_text, read_pdf, read_docx, read_spreadsheet

    for _name, _fn, _desc in [
        ("read_pdf", read_pdf, "Extract text from a PDF. Args: filepath (str), max_pages (int, optional)"),
        ("read_docx", read_docx, "Extract text from a Word .docx file. Args: filepath (str)"),
        ("read_spreadsheet", read_spreadsheet, "Extract content from Excel/CSV. Args: filepath (str), sheet (str, optional)"),
        ("extract_text", extract_text, "Auto-detect file type and extract text. Args: filepath (str)"),
    ]:
        _optional_tools[_name] = {
            "function": _fn,
            "description": _desc,
            "example": f'{{"action": "tool_call", "tool": "{_name}", "args": {{"filepath": "file.pdf"}}}}',
        }
except Exception as e:
    logger.warning(f"document tools not available — {e}")

# Phase 13.1: Blackboard tools
try:
    from ..collaboration.blackboard import (
        read_blackboard as _read_blackboard_fn,
        write_blackboard as _write_blackboard_fn,
    )

    async def _tool_read_blackboard(mission_id: int, key: str = "") -> str:
        """Read from the shared project blackboard."""
        import json as _json
        result = await _read_blackboard_fn(int(mission_id), key=key or None)
        return _json.dumps(result, indent=2) if result else "{}"

    async def _tool_write_blackboard(mission_id: int, key: str, value: str) -> str:
        """Write to the shared project blackboard."""
        import json as _json
        try:
            parsed_value = _json.loads(value)
        except (ValueError, TypeError):
            parsed_value = value
        await _write_blackboard_fn(int(mission_id), key, parsed_value)
        return f"✅ Blackboard key '{key}' updated."

    _optional_tools["read_blackboard"] = {
        "function": _tool_read_blackboard,
        "description": (
            "Read from the shared project blackboard (structured state shared "
            "between agents). "
            "Args: mission_id (int), key (str, optional — e.g. 'architecture', "
            "'files', 'decisions', 'open_issues', 'constraints')"
        ),
        "example": (
            '{"action": "tool_call", "tool": "read_blackboard", '
            '"args": {"mission_id": 1, "key": "decisions"}}'
        ),
    }
    _optional_tools["write_blackboard"] = {
        "function": _tool_write_blackboard,
        "description": (
            "Write to the shared project blackboard. "
            "Args: mission_id (int), key (str — 'architecture'|'files'|'decisions'"
            "|'open_issues'|'constraints'), value (str — JSON string)"
        ),
        "example": (
            '{"action": "tool_call", "tool": "write_blackboard", '
            '"args": {"mission_id": 1, "key": "decisions", '
            '"value": "[{\\"what\\": \\"Use FastAPI\\", \\"why\\": \\"speed\\", \\"by\\": \\"architect\\"}]"}}'
        ),
    }
except Exception as e:
    logger.debug(f"blackboard tools not available — {type(e).__name__}: {e}")

# External service integration tool
try:
    async def service_call(service: str, action: str, params: str = "{}") -> str:
        """Call an external service API via the integration registry."""
        import json as _json
        from ..integrations.registry import get_integration_registry

        registry = get_integration_registry()
        integration = registry.get(service)
        if integration is None:
            available = registry.list_services()
            return (
                f"Unknown service '{service}'. "
                f"Available: {', '.join(available) if available else '(none)'}"
            )

        # Parse params — accept both dict and JSON string
        if isinstance(params, str):
            try:
                parsed_params = _json.loads(params)
            except (ValueError, TypeError):
                parsed_params = {}
        else:
            parsed_params = params

        result = await integration.execute(action, parsed_params)
        return _json.dumps(result, indent=2, default=str)

    _optional_tools["service_call"] = {
        "function": service_call,
        "description": (
            "Call an external service API (GitHub, Vercel, Railway, etc.). "
            "Args: service (str: service name), action (str: action to perform), "
            "params (str: JSON string of action parameters)"
        ),
        "example": (
            '{"action": "tool_call", "tool": "service_call", '
            '"args": {"service": "github", "action": "list_repos", "params": "{}"}}'
        ),
    }
except Exception as e:
    logger.debug(f"service_call tool not available — {type(e).__name__}: {e}")

# Deployment tool (Gap 6)
try:
    from .deploy import deploy as _deploy_fn

    _optional_tools["deploy"] = {
        "function": _deploy_fn,
        "description": (
            "Deploy application to cloud platform (Vercel, Railway). "
            "Args: target (str: 'vercel' or 'railway'), "
            "project_path (str: path to project), "
            "env_vars (dict, optional: environment variables), "
            "mission_id (int, optional: workflow mission ID for validation)"
        ),
        "example": (
            '{"action": "tool_call", "tool": "deploy", '
            '"args": {"target": "vercel", "project_path": "./my-app"}}'
        ),
    }
except Exception as e:
    logger.debug(f"deploy tool not available — {type(e).__name__}: {e}")

# Phase 11.5: Document ingestion tool
try:
    async def _ingest_tool_wrapper(source: str, source_type: str = "auto") -> str:
        """Wrapper to return string result for tool system."""
        result = await _ingest_fn(source, source_type)
        if result["status"] == "ok":
            return f"Ingested {result['chunks']} chunks from {result['source']}"
        return f"Ingestion failed: {result.get('error', 'unknown error')}"

    _optional_tools["ingest_document"] = {
        "function": _ingest_tool_wrapper,
        "description": (
            "Ingest a document (URL or file) into the knowledge base. "
            "Args: source (str: URL or filepath), "
            "source_type (str: 'url', 'file', or 'auto')"
        ),
        "example": (
            '{"action": "tool_call", "tool": "ingest_document", '
            '"args": {"source": "https://docs.example.com/api"}}'
        ),
    }
except Exception as e:
    logger.debug(f"ingest_document tool not available — {type(e).__name__}: {e}")

# Coverage tool
try:
    from .coverage import get_coverage_summary

    _optional_tools["coverage"] = {
        "function": get_coverage_summary,
        "description": "Run code coverage analysis (pytest --cov or jest --coverage)",
        "example": '{"action": "tool_call", "tool": "coverage", "args": {"project_path": "myapp", "language": "python"}}',
    }
except Exception as e:
    logger.debug(f"coverage tool not available — {type(e).__name__}: {e}")

# MCP client singleton (for external tool servers)
try:
    from .mcp_client import mcp_client  # noqa: F401 — re-exported
except Exception as e:
    mcp_client = None  # type: ignore[assignment]
    logger.debug(f"MCP client not available — {type(e).__name__}: {e}")

# Shopping intelligence tools
try:
    import json as _json_shopping

    from ..shopping.intelligence.query_analyzer import analyze_query as _analyze_query
    from ..shopping.intelligence.search_planner import generate_search_plan as _generate_search_plan
    from ..shopping.intelligence.product_matcher import match_products as _match_products
    from ..shopping.intelligence.value_scorer import score_products as _score_products
    from ..shopping.intelligence.delivery_compare import compare_delivery as _compare_delivery
    from ..shopping.intelligence.review_synthesizer import synthesize_reviews as _synthesize_reviews
    from ..shopping.intelligence.constraints import check_constraints as _check_constraints
    from ..shopping.intelligence.timing import advise_timing as _advise_timing
    from ..shopping.intelligence.alternatives import generate_alternatives as _generate_alternatives
    from ..shopping.memory.user_profile import get_user_profile as _get_user_profile, update_user_profile as _update_user_profile
    from ..shopping.memory.price_watch import add_price_watch as _add_price_watch, get_active_watches as _get_active_watches, remove_watch as _remove_watch

    async def _tool_shopping_search(query: str) -> str:
        """Analyze a shopping query, then execute searches via scrapers and return actual product results."""
        import asyncio as _asyncio
        try:
            from ..shopping.resilience.fallback_chain import get_product_with_fallback

            # Run query analysis with a timeout — fall back to empty analysis if LLM is slow
            analyzed = {}
            plan = []
            try:
                analyzed = await _asyncio.wait_for(_analyze_query(query), timeout=15)
                plan = await _asyncio.wait_for(_generate_search_plan(analyzed), timeout=15)
            except (_asyncio.TimeoutError, Exception):
                pass  # proceed with direct search using query as-is

            # Extract sources from the plan (flat list of task dicts with "phase" key)
            sources: list[str] = []
            if plan:
                phase_1_tasks = [t for t in plan if isinstance(t, dict) and t.get("phase") == 1] if isinstance(plan, list) else plan.get("phase_1", plan.get("tasks", []))
                for task in phase_1_tasks:
                    for src in task.get("sources", []):
                        if src and src not in sources:
                            sources.append(src)

            # Execute actual product search via fallback chain (30s cap)
            try:
                products = await _asyncio.wait_for(
                    get_product_with_fallback(
                        query,
                        sources=sources if sources else None,
                    ),
                    timeout=30,
                )
            except _asyncio.TimeoutError:
                products = []

            return _json_shopping.dumps({
                "analysis": analyzed,
                "search_plan": plan,
                "products": products,
                "product_count": len(products),
            }, indent=2, default=str)
        except Exception as exc:
            return _json_shopping.dumps({"error": f"{type(exc).__name__}: {exc}"})

    async def _tool_shopping_compare(products: str) -> str:
        """Compare products with value scoring and delivery comparison."""
        try:
            product_list = _json_shopping.loads(products)
            scores = await _score_products(product_list)
            delivery = await _compare_delivery(product_list)
            return _json_shopping.dumps({"scores": scores, "delivery": delivery}, indent=2, default=str)
        except Exception as exc:
            return _json_shopping.dumps({"error": f"{type(exc).__name__}: {exc}"})

    async def _tool_shopping_reviews(product_name: str, reviews: str) -> str:
        """Synthesize product reviews into a summary."""
        try:
            review_list = _json_shopping.loads(reviews)
            result = await _synthesize_reviews(review_list, product_name)
            return _json_shopping.dumps(result, indent=2, default=str)
        except Exception as exc:
            return _json_shopping.dumps({"error": f"{type(exc).__name__}: {exc}"})

    async def _tool_shopping_constraints(products: str, constraints: str) -> str:
        """Filter products against user constraints."""
        try:
            product_list = _json_shopping.loads(products)
            constraint_list = _json_shopping.loads(constraints)
            result = await _check_constraints(product_list, constraint_list)
            return _json_shopping.dumps(result, indent=2, default=str)
        except Exception as exc:
            return _json_shopping.dumps({"error": f"{type(exc).__name__}: {exc}"})

    async def _tool_shopping_timing(category: str) -> str:
        """Get market timing advice for a product category."""
        try:
            result = await _advise_timing(category)
            return _json_shopping.dumps(result, indent=2, default=str)
        except Exception as exc:
            return _json_shopping.dumps({"error": f"{type(exc).__name__}: {exc}"})

    async def _tool_shopping_alternatives(query: str, category: str = "") -> str:
        """Generate alternative product suggestions."""
        try:
            result = await _generate_alternatives(query, category)
            return _json_shopping.dumps(result, indent=2, default=str)
        except Exception as exc:
            return _json_shopping.dumps({"error": f"{type(exc).__name__}: {exc}"})

    async def _tool_shopping_user_profile(user_id: str, action: str, data: str = "{}") -> str:
        """Get or update user shopping profile."""
        try:
            uid = int(user_id)
            if action == "get":
                result = await _get_user_profile(uid)
                return _json_shopping.dumps(result, indent=2, default=str)
            elif action == "update":
                fields = _json_shopping.loads(data)
                await _update_user_profile(uid, **fields)
                return _json_shopping.dumps({"status": "updated"})
            else:
                return _json_shopping.dumps({"error": f"Unknown action: {action}. Use 'get' or 'update'."})
        except Exception as exc:
            return _json_shopping.dumps({"error": f"{type(exc).__name__}: {exc}"})

    async def _tool_shopping_price_watch(user_id: str, action: str, product_name: str = "", target_price: str = "") -> str:
        """Manage price watches for a user."""
        try:
            uid = int(user_id)
            if action == "add":
                tp = float(target_price) if target_price else None
                watch_id = await _add_price_watch(uid, product_name, current_price=0.0, target_price=tp)
                return _json_shopping.dumps({"status": "added", "watch_id": watch_id})
            elif action == "list":
                watches = await _get_active_watches(uid)
                return _json_shopping.dumps(watches, indent=2, default=str)
            elif action == "remove":
                # Find watch by product name
                watches = await _get_active_watches(uid)
                removed = False
                for w in watches:
                    if product_name.lower() in w.get("product_name", "").lower():
                        await _remove_watch(w["id"])
                        removed = True
                        break
                return _json_shopping.dumps({"status": "removed" if removed else "not_found"})
            else:
                return _json_shopping.dumps({"error": f"Unknown action: {action}. Use 'add', 'list', or 'remove'."})
        except Exception as exc:
            return _json_shopping.dumps({"error": f"{type(exc).__name__}: {exc}"})

    _optional_tools["shopping_search"] = {
        "function": _tool_shopping_search,
        "description": (
            "Analyze a shopping query and generate a search plan. "
            "Args: query (str: the user's shopping query)"
        ),
        "example": (
            '{"action": "tool_call", "tool": "shopping_search", '
            '"args": {"query": "best wireless headphones under 2000 TL"}}'
        ),
    }
    _optional_tools["shopping_compare"] = {
        "function": _tool_shopping_compare,
        "description": (
            "Compare products with value scoring and delivery comparison. "
            "Args: products (str: JSON list of product dicts)"
        ),
        "example": (
            '{"action": "tool_call", "tool": "shopping_compare", '
            '"args": {"products": "[{\\"name\\": \\"Product A\\", \\"price\\": 1500}]"}}'
        ),
    }
    _optional_tools["shopping_reviews"] = {
        "function": _tool_shopping_reviews,
        "description": (
            "Synthesize product reviews into a structured summary. "
            "Args: product_name (str), reviews (str: JSON list of review dicts)"
        ),
        "example": (
            '{"action": "tool_call", "tool": "shopping_reviews", '
            '"args": {"product_name": "Sony WH-1000XM5", "reviews": "[{\\"text\\": \\"Great sound\\", \\"rating\\": 5}]"}}'
        ),
    }
    _optional_tools["shopping_constraints"] = {
        "function": _tool_shopping_constraints,
        "description": (
            "Filter products against user constraints (budget, dimensions, compatibility). "
            "Args: products (str: JSON list of product dicts), constraints (str: JSON list of constraint dicts)"
        ),
        "example": (
            '{"action": "tool_call", "tool": "shopping_constraints", '
            '"args": {"products": "[...]", "constraints": "[{\\"type\\": \\"budget\\", \\"value\\": 5000}]"}}'
        ),
    }
    _optional_tools["shopping_timing"] = {
        "function": _tool_shopping_timing,
        "description": (
            "Get market timing advice — buy now vs wait for sales. "
            "Args: category (str: product category like 'electronics', 'appliances')"
        ),
        "example": (
            '{"action": "tool_call", "tool": "shopping_timing", '
            '"args": {"category": "electronics"}}'
        ),
    }
    _optional_tools["shopping_alternatives"] = {
        "function": _tool_shopping_alternatives,
        "description": (
            "Generate alternative product suggestions for a query. "
            "Args: query (str: product query), category (str, optional)"
        ),
        "example": (
            '{"action": "tool_call", "tool": "shopping_alternatives", '
            '"args": {"query": "macbook air", "category": "electronics"}}'
        ),
    }
    _optional_tools["shopping_user_profile"] = {
        "function": _tool_shopping_user_profile,
        "description": (
            "Get or update user shopping profile (preferences, owned items). "
            "Args: user_id (str), action (str: 'get' or 'update'), "
            "data (str: JSON dict of fields to update, optional)"
        ),
        "example": (
            '{"action": "tool_call", "tool": "shopping_user_profile", '
            '"args": {"user_id": "123", "action": "get"}}'
        ),
    }
    _optional_tools["shopping_price_watch"] = {
        "function": _tool_shopping_price_watch,
        "description": (
            "Manage price watches — add, list, or remove. "
            "Args: user_id (str), action (str: 'add'|'list'|'remove'), "
            "product_name (str, optional), target_price (str, optional)"
        ),
        "example": (
            '{"action": "tool_call", "tool": "shopping_price_watch", '
            '"args": {"user_id": "123", "action": "add", "product_name": "iPhone 15", "target_price": "40000"}}'
        ),
    }
except Exception as e:
    logger.debug(f"Shopping tools not available — {type(e).__name__}: {e}")

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
TOOL_REGISTRY: dict[str, dict[str, Any]] = {
    # ── Shell ──────────────────────────────────────────────────────────────
    "shell": {
        "function": run_shell,
        "description": (
            "Execute a shell command in the Docker sandbox. "
            "Args: command (str), timeout (int, optional, default 60), "
            "workdir (str, optional)"
        ),
        "example": '{"action": "tool_call", "tool": "shell", "args": {"command": "python3 main.py"}}',
    },
    "shell_stdin": {
        "function": run_shell_with_stdin,
        "description": (
            "Execute a shell command and pipe data to its stdin. "
            "Args: command (str), stdin_data (str), "
            "timeout (int, optional), workdir (str, optional)"
        ),
        "example": (
            '{"action": "tool_call", "tool": "shell_stdin", '
            '"args": {"command": "cat > hello.txt", "stdin_data": "Hello world"}}'
        ),
    },
    "shell_sequential": {
        "function": run_shell_sequential,
        "description": (
            "Run multiple commands in order, stopping on first failure. "
            "Args: commands (list[str]), timeout (int, optional, per-command), "
            "workdir (str, optional), stop_on_error (bool, default true)"
        ),
        "example": (
            '{"action": "tool_call", "tool": "shell_sequential", '
            '"args": {"commands": ["pip install -r requirements.txt", "python main.py"]}}'
        ),
    },

    # ── File Operations ────────────────────────────────────────────────────
    "file_tree": {
        "function": get_file_tree,
        "description": (
            "List files and directories as a visual tree. "
            "Args: path (str, optional), max_depth (int, optional, default 5), "
            "max_items (int, optional, default 200)"
        ),
        "example": '{"action": "tool_call", "tool": "file_tree", "args": {"path": "."}}',
    },
    "read_file": {
        "function": read_file,
        "description": (
            "Read a file with line numbers. "
            "Args: filepath (str), max_lines (int, optional, default 200)"
        ),
        "example": '{"action": "tool_call", "tool": "read_file", "args": {"filepath": "src/main.py"}}',
    },
    "write_file": {
        "function": write_file,
        "description": (
            "Create or update a file. "
            'Args: filepath (str), content (str), '
            'mode ("write" or "append", default "write")'
        ),
        "example": '{"action": "tool_call", "tool": "write_file", "args": {"filepath": "src/main.py", "content": "print(\'hello\')"}}',
    },
    "project_info": {
        "function": detect_project,
        "description": (
            "Analyze the workspace to detect languages, frameworks, "
            "dependencies, and file statistics. "
            "Args: path (str, optional)"
        ),
        "example": '{"action": "tool_call", "tool": "project_info", "args": {}}',
    },
    "edit_file": {
        "function": edit_file,
        "description": (
            "Replace a range of lines in a file. start_line and end_line are 1-indexed and inclusive. "
            "Args: filepath (str), start_line (int), end_line (int), new_content (str)"
        ),
        "example": '{"action": "tool_call", "tool": "edit_file", "args": {"filepath": "src/main.py", "start_line": 10, "end_line": 15, "new_content": "def foo():\\n    pass\\n"}}',
    },
    "patch_file": {
        "function": patch_file,
        "description": (
            "Search-and-replace in a file. Find exact text and replace it. "
            "More reliable than line-number editing. "
            "Args: filepath (str), search_block (str), replace_block (str)"
        ),
        "example": (
            '{"action": "tool_call", "tool": "patch_file", '
            '"args": {"filepath": "src/main.py", '
            '"search_block": "def old_func():", '
            '"replace_block": "def new_func():"}}'
        ),
    },
    "apply_diff": {
        "function": apply_diff,
        "description": (
            "Apply a unified diff patch to a file. Validates syntax after applying. "
            "Args: filepath (str), unified_diff (str: unified diff format with @@ hunks)"
        ),
        "example": (
            '{"action": "tool_call", "tool": "apply_diff", '
            '"args": {"filepath": "src/main.py", '
            '"unified_diff": "@@ -10,3 +10,4 @@\\n context\\n-old line\\n+new line\\n+added line"}}'
        ),
    },
    "lint": {
        "function": auto_lint,
        "description": (
            "Auto-lint and format a Python file using ruff. "
            "Args: filepath (str)"
        ),
        "example": '{"action": "tool_call", "tool": "lint", "args": {"filepath": "src/main.py"}}',
    },
    "verify_deps": {
        "function": verify_dependencies,
        "description": (
            "Scan Python files, extract imports, and auto-install any missing "
            "third-party packages via pip. "
            "Args: path (str, optional)"
        ),
        "example": '{"action": "tool_call", "tool": "verify_deps", "args": {"path": "."}}',
    },

    # ── Git ─────────────────────────────────────────────────────────────────
    "git_init": {
        "function": git_init,
        "description": (
            "Initialize a git repo (with .gitignore and initial commit). "
            "Idempotent — safe to call on an existing repo. "
            "Args: path (str, optional)"
        ),
        "example": '{"action": "tool_call", "tool": "git_init", "args": {}}',
    },
    "git_commit": {
        "function": git_commit,
        "description": (
            "Stage all changes and commit. "
            "Args: message (str), path (str, optional), "
            "add_all (bool, optional, default true)"
        ),
        "example": '{"action": "tool_call", "tool": "git_commit", "args": {"message": "feat: add login endpoint"}}',
    },
    "git_branch": {
        "function": git_branch,
        "description": (
            "Create and switch to a branch (or switch if it already exists). "
            "Args: branch_name (str), path (str, optional)"
        ),
        "example": '{"action": "tool_call", "tool": "git_branch", "args": {"branch_name": "feat/auth"}}',
    },
    "git_log": {
        "function": git_log,
        "description": (
            "Show recent commits (one-line format). "
            "Args: path (str, optional), count (int, optional, default 10)"
        ),
        "example": '{"action": "tool_call", "tool": "git_log", "args": {"count": 5}}',
    },
    "git_diff": {
        "function": git_diff,
        "description": (
            "Show uncommitted changes. "
            "Args: path (str, optional), stat_only (bool, optional, default false)"
        ),
        "example": '{"action": "tool_call", "tool": "git_diff", "args": {}}',
    },
    "git_rollback": {
        "function": git_rollback,
        "description": (
            "Soft-reset the last N commits (keeps files staged). "
            "Args: steps (int, default 1), path (str, optional)"
        ),
        "example": '{"action": "tool_call", "tool": "git_rollback", "args": {"steps": 1}}',
    },
    "git_status": {
        "function": git_status,
        "description": (
            "Show current branch and working-tree status. "
            "Args: path (str, optional)"
        ),
        "example": '{"action": "tool_call", "tool": "git_status", "args": {}}',
    },

    # ── AST-Aware Code Tools (Phase 8) ────────────────────────────────────
    "get_function": {
        "function": get_function,
        "description": (
            "Extract a function/method source by name from a Python file. "
            "For methods use 'ClassName.method_name' format. "
            "Args: filepath (str), function_name (str)"
        ),
        "example": (
            '{"action": "tool_call", "tool": "get_function", '
            '"args": {"filepath": "src/main.py", "function_name": "process_data"}}'
        ),
    },
    "replace_function": {
        "function": replace_function,
        "description": (
            "Replace an entire function/method with new code. "
            "For methods use 'ClassName.method_name' format. "
            "Args: filepath (str), function_name (str), new_code (str)"
        ),
        "example": (
            '{"action": "tool_call", "tool": "replace_function", '
            '"args": {"filepath": "src/main.py", "function_name": "old_func", '
            '"new_code": "def old_func():\\n    return 42\\n"}}'
        ),
    },
    "list_classes": {
        "function": list_classes,
        "description": (
            "List all classes in a Python file with methods and base classes. "
            "Args: filepath (str)"
        ),
        "example": '{"action": "tool_call", "tool": "list_classes", "args": {"filepath": "src/models.py"}}',
    },
    "list_functions": {
        "function": list_functions,
        "description": (
            "List all top-level functions in a Python file. "
            "Args: filepath (str)"
        ),
        "example": '{"action": "tool_call", "tool": "list_functions", "args": {"filepath": "src/utils.py"}}',
    },
    "get_imports": {
        "function": get_imports,
        "description": (
            "Extract all import statements from a Python file. "
            "Args: filepath (str)"
        ),
        "example": '{"action": "tool_call", "tool": "get_imports", "args": {"filepath": "src/main.py"}}',
    },

    # ── Codebase Indexing (Phase 8) ────────────────────────────────────────
    "index_workspace": {
        "function": index_workspace,
        "description": (
            "Scan workspace and build a structural index of all Python files. "
            "Index includes functions, classes, imports per file. "
            "Args: path (str, optional, default '.')"
        ),
        "example": '{"action": "tool_call", "tool": "index_workspace", "args": {}}',
    },
    "query_codebase": {
        "function": query_codebase,
        "description": (
            "Search the codebase index for functions, classes, or modules. "
            "Much faster than reading files manually. "
            "Args: query (str), search_type (str: all/function/class/import/file)"
        ),
        "example": (
            '{"action": "tool_call", "tool": "query_codebase", '
            '"args": {"query": "authenticate", "search_type": "function"}}'
        ),
    },
    "codebase_map": {
        "function": codebase_map,
        "description": (
            "Generate a high-level map of the codebase showing modules, "
            "classes, and functions. Good for understanding project structure. "
            "Args: path (str, optional)"
        ),
        "example": '{"action": "tool_call", "tool": "codebase_map", "args": {}}',
    },

    # ── Optional tools injected below ──────────────────────────────────────
    **_optional_tools,
}

# ---------------------------------------------------------------------------
# Pre-compute accepted parameter names per tool (once at import time)
# ---------------------------------------------------------------------------
_TOOL_PARAMS: dict[str, Optional[set[str]]] = {}

for _name, _info in TOOL_REGISTRY.items():
    try:
        _sig = inspect.signature(_info["function"])
        _TOOL_PARAMS[_name] = set(_sig.parameters.keys())
    except (ValueError, TypeError):
        # If introspection fails, don't filter — let Python raise naturally
        _TOOL_PARAMS[_name] = None

# Clean up module namespace
del _name, _info, _sig
logger.info(f"📦 Loaded tools: {sorted(TOOL_REGISTRY.keys())}")

# ---------------------------------------------------------------------------
# LiteLLM Tool Schemas (auto-generated from TOOL_REGISTRY)
# ---------------------------------------------------------------------------
_PYTHON_TYPE_MAP = {
    int: "integer", float: "number", bool: "boolean",
    str: "string", list: "array", dict: "object",
}

TOOL_SCHEMAS: list[dict] = []

for _name, _info in TOOL_REGISTRY.items():
    try:
        _sig = inspect.signature(_info["function"])
        _properties: dict = {}
        _required: list[str] = []
        for _pname, _param in _sig.parameters.items():
            _ptype = "string"  # safe default
            _annotation = _param.annotation
            if _annotation is not inspect.Parameter.empty:
                _ptype = _PYTHON_TYPE_MAP.get(_annotation, "string")
            _properties[_pname] = {
                "type": _ptype,
                "description": f"Parameter: {_pname}",
            }
            if _param.default is inspect.Parameter.empty:
                _required.append(_pname)
        TOOL_SCHEMAS.append({
            "type": "function",
            "function": {
                "name": _name,
                "description": _info["description"],
                "parameters": {
                    "type": "object",
                    "properties": _properties,
                    "required": _required,
                },
            },
        })
    except (ValueError, TypeError):
        pass

# Pseudo-tools for structured agent actions
TOOL_SCHEMAS.append({
    "type": "function",
    "function": {
        "name": "final_answer",
        "description": (
            "Provide your final answer to the task. Use this when you are "
            "done and ready to submit your complete result."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "result": {
                    "type": "string",
                    "description": "Your complete answer / result for the task.",
                },
                "memories": {
                    "type": "object",
                    "description": "Optional key-value pairs to remember for future tasks.",
                },
            },
            "required": ["result"],
        },
    },
})

TOOL_SCHEMAS.append({
    "type": "function",
    "function": {
        "name": "clarify",
        "description": (
            "Ask the user a clarifying question when you need more information."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The clarification question to ask.",
                },
            },
            "required": ["question"],
        },
    },
})

# Clean up
del _name, _info, _sig, _properties, _required, _pname, _param, _ptype, _annotation

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------
def get_tool_descriptions() -> str:
    """
    Format every tool's description and example into a prompt-friendly
    string suitable for injection into an agent's system message.
    """
    lines: list[str] = []
    for name, info in TOOL_REGISTRY.items():
        lines.append(f"• **{name}**: {info['description']}")
        lines.append(f"  Example: {info['example']}")
    return "\n".join(lines)


def list_tool_names() -> list[str]:
    """Return a sorted list of all registered tool names."""
    return sorted(TOOL_REGISTRY.keys())


def _trigger_reindex(filepath: str | None) -> None:
    """Re-index a single file after a write/edit/patch/diff tool call.

    Runs synchronously-safe — fires and forgets so the tool isn't slowed.
    """
    if not filepath:
        return
    try:
        import asyncio

        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(reindex_file(filepath))
        else:
            loop.run_until_complete(reindex_file(filepath))
    except Exception:
        # Non-critical — embedding re-index is best-effort
        pass


async def execute_tool(tool_name: str, agent_type: str | None = None, task_hints: dict | None = None, **kwargs: Any) -> str:
    """
    Execute a registered tool by name.

    Unknown keyword arguments are silently dropped so that extra fields
    the LLM may include (e.g. ``"tool"``, ``"thought"``) don't crash
    the underlying function.

    Args:
        tool_name:  Key in TOOL_REGISTRY.
        agent_type: The calling agent's type (for per-agent shell allowlists).
        **kwargs:   Arguments forwarded to the tool function.

    Returns:
        The tool's string output, or a descriptive error message.
    """
    if tool_name not in TOOL_REGISTRY:
        available = ", ".join(sorted(TOOL_REGISTRY.keys()))
        return f"❌ Unknown tool: '{tool_name}'. Available: {available}"

    # Phase 8.2: Per-agent shell command allowlist enforcement
    if agent_type and tool_name in ("shell", "shell_stdin", "shell_sequential"):
        _cmd = kwargs.get("command", "") or kwargs.get("commands", "")
        if isinstance(_cmd, str) and _cmd.strip():
            try:
                from .shell import _is_command_allowed_for_agent
                if not _is_command_allowed_for_agent(_cmd, agent_type):
                    return (
                        f"🚫 BLOCKED: Command not in allowlist for agent type "
                        f"'{agent_type}'. Use only approved commands."
                    )
            except ImportError:
                pass

    func = TOOL_REGISTRY[tool_name]["function"]

    # Inject task_hints for web_search (before kwargs filtering so it passes through)
    if tool_name == "web_search" and task_hints:
        kwargs["_task_hints"] = task_hints

    # ── Filter kwargs to accepted parameters ──
    valid_params = _TOOL_PARAMS.get(tool_name)
    if valid_params is not None:
        filtered = {k: v for k, v in kwargs.items() if k in valid_params}
    else:
        # Introspection failed at import — pass everything through
        filtered = kwargs

    logger.debug(f"Executing tool '{tool_name}' with args: {list(filtered.keys())}")

    try:
        result = await func(**filtered)

        # ── Phase 12.2: Trigger code re-indexing after file-change tools ──
        if tool_name in ("write_file", "edit_file", "patch_file", "apply_diff"):
            _trigger_reindex(filtered.get("filepath"))

        return str(result)

    except TypeError as exc:
        # Almost always a missing required argument
        expected = (
            ", ".join(sorted(valid_params)) if valid_params else "(unknown)"
        )
        logger.error(f"Tool '{tool_name}' argument error: {exc}", exc_info=True)
        return (
            f"❌ Argument error ({tool_name}): {exc}\n"
            f"Expected parameters: {expected}"
        )

    except Exception as exc:
        logger.error(f"Tool '{tool_name}' error: {exc}", exc_info=True)
        return f"❌ Tool error ({tool_name}): {type(exc).__name__}: {exc}"
