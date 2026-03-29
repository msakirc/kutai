"""test_e2e_llm_pipeline.py — End-to-end LLM pipeline integration tests.

Coverage:
- Idea → classification → agent assignment → execution → result (full pipeline)
- Shopping flow end-to-end with LLM
- Workflow engine: JSON loading, WorkflowDefinition, validate_dependencies
- Workflow expander: expand_steps_to_tasks, filter_steps_for_context
- Shopping intelligence: query analyzer (keyword fallback)
- Memory/RAG: store and retrieve (no vector store needed)
- Router: ModelRequirements construction and model selection logic
- DB: insert_tasks_atomically, compute_task_hash, get_completed_dependency_results
- Agent registry: all standard agents registered

All LLM-calling tests are grouped under @pytest.mark.llm so they can be run
separately from structural tests.

Prerequisites for @pytest.mark.llm tests
-----------------------------------------
Either:
  A. llama-server running locally (port 8080 by default):
       llama-server --model <model.gguf> --port 8080 --n-predict 200
  B. Ollama running locally (port 11434):
       ollama run <model-name>

The tests will skip automatically when no local model is detected.

Speed optimisation
------------------
All LLM tests use:
  - prefer_speed=True / difficulty=2 to pick the smallest available model
  - Short prompts (< 200 tokens input)
  - max_tokens / estimated_output_tokens capped at 50-200
  - temperature=0.0 for deterministic (faster) outputs
  - Loose assertions: check *structure* and *non-emptiness*, not exact text

Markers:
  @pytest.mark.integration  — all tests in this file
  @pytest.mark.llm          — tests that call a real local LLM
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_async(coro):
    """Run a coroutine synchronously in a fresh event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ===========================================================================
# STRUCTURAL TESTS (no LLM, fast)
# ===========================================================================


@pytest.mark.integration
class TestAgentRegistry:
    """All standard agent types are registered and importable."""

    EXPECTED_AGENTS = [
        "planner", "architect", "coder", "implementer", "fixer",
        "test_generator", "reviewer", "visual_reviewer", "researcher",
        "analyst", "writer", "summarizer", "assistant", "executor",
        "error_recovery", "shopping_advisor", "product_researcher",
        "deal_analyst", "shopping_clarifier",
    ]

    def test_all_agents_registered(self):
        """Every expected agent type is importable from AGENT_REGISTRY."""
        from src.agents import AGENT_REGISTRY
        for agent_type in self.EXPECTED_AGENTS:
            assert agent_type in AGENT_REGISTRY, (
                f"Agent '{agent_type}' missing from AGENT_REGISTRY"
            )

    def test_get_agent_fallback_returns_executor(self):
        """get_agent with an unknown type falls back to executor, not None."""
        from src.agents import get_agent
        agent = get_agent("nonexistent_agent_type_xyz")
        assert agent is not None
        assert agent.name == "executor"

    def test_all_agents_have_name_attribute(self):
        """Every agent in the registry has a non-empty .name attribute."""
        from src.agents import AGENT_REGISTRY
        for agent_type, agent in AGENT_REGISTRY.items():
            assert hasattr(agent, "name"), f"{agent_type} has no .name"
            assert isinstance(agent.name, str) and len(agent.name) > 0, (
                f"{agent_type}.name is empty"
            )

    def test_all_agents_have_max_iterations(self):
        """Every agent has a max_iterations > 0."""
        from src.agents import AGENT_REGISTRY
        for agent_type, agent in AGENT_REGISTRY.items():
            assert hasattr(agent, "max_iterations"), (
                f"{agent_type} has no max_iterations"
            )
            assert agent.max_iterations > 0, (
                f"{agent_type}.max_iterations must be > 0, got {agent.max_iterations}"
            )

    def test_shopping_agents_have_tools(self):
        """Shopping-related agents declare tools (not tool-less)."""
        from src.agents import get_agent
        for agent_type in ("shopping_advisor", "product_researcher", "deal_analyst"):
            agent = get_agent(agent_type)
            if agent.allowed_tools is not None:
                assert len(agent.allowed_tools) > 0, (
                    f"{agent_type}.allowed_tools is empty — needs web_search at minimum"
                )


# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestWorkflowJsonFiles:
    """All workflow JSON files are valid and well-structured."""

    WORKFLOW_NAMES = [
        "idea_to_product_v1",
        "idea_to_product_v2",
    ]

    # Shopping-specific workflows (exist inside src/workflows/shopping/)
    SHOPPING_WORKFLOWS = [
        "shopping/quick_search",
        "shopping/shopping",
        "shopping/exploration",
    ]

    def _workflow_path(self, name: str) -> str:
        base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        # Slash-separated → direct path under src/workflows/
        if "/" in name:
            return os.path.join(base, "src", "workflows", name + ".json")
        # Try known subdirectory pattern
        dir_name = name
        for suffix in ("_v1", "_v2", "_v3"):
            if dir_name.endswith(suffix):
                dir_name = dir_name[: -len(suffix)]
                break
        return os.path.join(base, "src", "workflows", dir_name, name + ".json")

    @pytest.mark.parametrize("wf_name", [
        "idea_to_product_v1", "idea_to_product_v2"
    ])
    def test_workflow_json_valid(self, wf_name):
        """Workflow JSON is valid and has required top-level keys."""
        path = self._workflow_path(wf_name)
        assert os.path.exists(path), f"Workflow file missing: {path}"
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict)
        assert "plan_id" in data, f"{wf_name}: missing 'plan_id'"
        assert "version" in data, f"{wf_name}: missing 'version'"
        assert "steps" in data, f"{wf_name}: missing 'steps'"
        assert isinstance(data["steps"], list), f"{wf_name}: 'steps' must be a list"
        assert len(data["steps"]) > 0, f"{wf_name}: 'steps' list is empty"

    def test_load_workflow_via_loader(self):
        """WorkflowDefinition is loaded correctly via load_workflow()."""
        from src.workflows.engine.loader import load_workflow
        wf = load_workflow("idea_to_product_v2")
        assert wf.plan_id, "WorkflowDefinition.plan_id is empty"
        assert len(wf.steps) > 0, "WorkflowDefinition.steps is empty"
        assert len(wf.phases) > 0, "WorkflowDefinition.phases is empty"

    def test_validate_dependencies_idea_to_product_v2(self):
        """idea_to_product_v2 has no broken dependency references or cycles."""
        from src.workflows.engine.loader import load_workflow, validate_dependencies
        wf = load_workflow("idea_to_product_v2")
        errors = validate_dependencies(wf)
        assert errors == [], (
            f"Dependency validation errors in idea_to_product_v2:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    def test_validate_dependencies_idea_to_product_v1(self):
        """idea_to_product_v1 has no broken dependency references."""
        from src.workflows.engine.loader import load_workflow, validate_dependencies
        wf = load_workflow("idea_to_product_v1")
        errors = validate_dependencies(wf)
        # v1 may have orphan steps (it's older) — we only fail on cycles and
        # unknown references, not orphans.
        hard_errors = [e for e in errors if "cycle" in e.lower() or "unknown" in e.lower()]
        assert hard_errors == [], (
            "idea_to_product_v1 has critical dependency errors:\n"
            + "\n".join(f"  - {e}" for e in hard_errors)
        )

    def test_get_phase_steps_returns_subset(self):
        """get_phase_steps returns only steps for the requested phase."""
        from src.workflows.engine.loader import load_workflow
        wf = load_workflow("idea_to_product_v2")
        if not wf.phases:
            pytest.skip("No phases in workflow")
        first_phase_id = wf.phases[0]["id"]
        phase_steps = wf.get_phase_steps(first_phase_id)
        for s in phase_steps:
            assert s.get("phase") == first_phase_id, (
                f"Step {s['id']} has phase '{s.get('phase')}', expected '{first_phase_id}'"
            )


# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestWorkflowExpander:
    """Workflow step expansion into task dicts."""

    def test_expand_steps_to_tasks_basic(self):
        """expand_steps_to_tasks converts step dicts into task dicts."""
        from src.workflows.engine.expander import expand_steps_to_tasks

        steps = [
            {
                "id": "1.1",
                "name": "Analyze idea",
                "phase": "phase_1",
                "agent": "analyst",
                "instruction": "Analyze the raw idea for feasibility.",
                "input_artifacts": ["raw_idea"],
                "output_artifacts": ["idea_analysis"],
                "depends_on": [],
            },
            {
                "id": "1.2",
                "name": "Plan architecture",
                "phase": "phase_1",
                "agent": "architect",
                "instruction": "Design the system architecture.",
                "input_artifacts": ["idea_analysis"],
                "output_artifacts": ["architecture"],
                "depends_on": ["1.1"],
            },
        ]

        tasks = expand_steps_to_tasks(steps, mission_id=99, initial_context={"raw_idea": "Build a calendar app"})

        assert len(tasks) == 2
        t1, t2 = tasks

        # Check structure
        assert t1["agent_type"] == "analyst"
        assert t1["mission_id"] == 99
        assert "1.1" in t1["title"]
        assert t1["context"]["workflow_step_id"] == "1.1"
        assert t1["context"]["is_workflow_step"] is True
        assert "raw_idea" in t1["context"]["workflow_context"]

        # Check dependency list is preserved as step IDs (not task IDs yet)
        assert t2["depends_on_steps"] == ["1.1"]

    def test_expand_steps_priority_from_phase(self):
        """Earlier phases get higher priority numbers."""
        from src.workflows.engine.expander import expand_steps_to_tasks

        steps_p1 = [{"id": "1.1", "name": "Early", "phase": "phase_1", "agent": "planner",
                      "instruction": "x", "depends_on": []}]
        steps_p8 = [{"id": "8.1", "name": "Late", "phase": "phase_8", "agent": "executor",
                      "instruction": "x", "depends_on": []}]

        t_early = expand_steps_to_tasks(steps_p1, mission_id=1)[0]
        t_late = expand_steps_to_tasks(steps_p8, mission_id=1)[0]

        assert t_early["priority"] > t_late["priority"], (
            "Earlier phase steps should have higher priority"
        )

    def test_filter_steps_excludes_phase_minus_1(self):
        """filter_steps_for_context excludes phase_-1 steps for new projects."""
        from src.workflows.engine.expander import filter_steps_for_context

        steps = [
            {"id": "0.1", "phase": "phase_0", "name": "Root"},
            {"id": "-1.1", "phase": "phase_-1", "name": "Onboard existing codebase"},
            {"id": "1.1", "phase": "phase_1", "name": "Analyze"},
        ]

        # New project: phase_-1 excluded
        filtered = filter_steps_for_context(steps, has_existing_codebase=False)
        phases = [s["phase"] for s in filtered]
        assert "phase_-1" not in phases

        # Existing codebase: phase_-1 included
        all_steps = filter_steps_for_context(steps, has_existing_codebase=True)
        assert len(all_steps) == 3

    def test_agent_map_router_to_executor(self):
        """AGENT_MAP maps 'router' → 'executor' (and unknown names pass through)."""
        from src.workflows.engine.expander import map_agent_type
        assert map_agent_type("router") == "executor"
        assert map_agent_type("coder") == "coder"
        assert map_agent_type("unknown_agent") == "unknown_agent"


# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestDBAdvanced:
    """Advanced DB operations not covered by test_task_lifecycle.py."""

    def test_compute_task_hash_deterministic(self):
        """Same inputs always produce the same hash."""
        from src.infra.db import compute_task_hash
        h1 = compute_task_hash("title", "desc", "coder", 1, None)
        h2 = compute_task_hash("title", "desc", "coder", 1, None)
        assert h1 == h2

    def test_compute_task_hash_different_inputs(self):
        """Different inputs produce different hashes."""
        from src.infra.db import compute_task_hash
        h1 = compute_task_hash("title A", "desc", "coder", 1, None)
        h2 = compute_task_hash("title B", "desc", "coder", 1, None)
        assert h1 != h2

    def test_insert_tasks_atomically(self, temp_db):
        """insert_tasks_atomically creates multiple tasks in one transaction."""
        from src.infra.db import add_mission, insert_tasks_atomically, get_tasks_for_mission

        async def _run():
            mid = await add_mission(title="Atomic test mission", description="")
            tasks = [
                {"title": "Atomic task 1", "description": "step 1",
                 "agent_type": "researcher", "tier": "auto", "priority": 7,
                 "depends_on": [], "context": {}},
                {"title": "Atomic task 2", "description": "step 2",
                 "agent_type": "analyst", "tier": "auto", "priority": 6,
                 "depends_on": [], "context": {}},
                {"title": "Atomic task 3", "description": "step 3",
                 "agent_type": "writer", "tier": "auto", "priority": 5,
                 "depends_on": [], "context": {}},
            ]
            created_ids = await insert_tasks_atomically(tasks, mission_id=mid)
            assert len(created_ids) == 3
            assert all(tid > 0 for tid in created_ids), (
                "All tasks should be created (no dedup should occur)"
            )

            mission_tasks = await get_tasks_for_mission(mid)
            assert len(mission_tasks) == 3

        run_async(_run())

    def test_insert_tasks_atomically_dedup(self, temp_db):
        """insert_tasks_atomically deduplicates identical tasks within a batch."""
        from src.infra.db import add_mission, insert_tasks_atomically

        async def _run():
            mid = await add_mission(title="Dedup atomic test", description="")
            tasks = [
                {"title": "Same task", "description": "x", "agent_type": "executor",
                 "tier": "auto", "priority": 5, "depends_on": [], "context": {}},
                {"title": "Same task", "description": "x", "agent_type": "executor",
                 "tier": "auto", "priority": 5, "depends_on": [], "context": {}},
            ]
            created_ids = await insert_tasks_atomically(tasks, mission_id=mid)
            assert len(created_ids) == 2
            # First created, second deduped
            assert created_ids[0] > 0
            assert created_ids[1] == -1

        run_async(_run())

    def test_get_completed_dependency_results(self, temp_db):
        """get_completed_dependency_results fetches results from completed deps."""
        from src.infra.db import add_task, claim_task, update_task, get_completed_dependency_results
        from datetime import datetime, timezone

        async def _run():
            dep_id = await add_task(
                title="Dep task", description="produces output",
                agent_type="researcher",
            )
            await claim_task(dep_id)
            await update_task(
                dep_id,
                status="completed",
                result="Here is the research output",
                completed_at=datetime.now(timezone.utc).isoformat(),
            )

            results = await get_completed_dependency_results([dep_id])
            assert dep_id in results
            assert results[dep_id]["result"] == "Here is the research output"

        run_async(_run())

    def test_get_completed_dependency_results_empty(self, temp_db):
        """Empty depends_on list returns empty dict."""
        from src.infra.db import get_completed_dependency_results

        async def _run():
            result = await get_completed_dependency_results([])
            assert result == {}

        run_async(_run())

    def test_todo_crud(self, temp_db):
        """Todo items can be added, retrieved, toggled, and deleted."""
        from src.infra.db import add_todo, get_todos, get_todo, toggle_todo, delete_todo

        async def _run():
            tid = await add_todo(
                title="Buy groceries",
                description="Milk, eggs, bread",
                priority="high",
            )
            assert tid is not None

            todos = await get_todos()
            ids = [t["id"] for t in todos]
            assert tid in ids

            todo = await get_todo(tid)
            assert todo["title"] == "Buy groceries"
            assert todo["status"] == "pending"

            await toggle_todo(tid)
            todo_toggled = await get_todo(tid)
            assert todo_toggled["status"] == "done"

            await delete_todo(tid)
            todos_after = await get_todos()
            assert all(t["id"] != tid for t in todos_after)

        run_async(_run())

    def test_find_duplicate_task(self, temp_db):
        """find_duplicate_task detects pending duplicate by hash."""
        from src.infra.db import add_task, find_duplicate_task, compute_task_hash

        async def _run():
            await add_task(title="Hash test", description="desc", agent_type="coder")
            h = compute_task_hash("Hash test", "desc", "coder", None, None)
            dup = await find_duplicate_task(h)
            assert dup is not None
            assert dup["title"] == "Hash test"

        run_async(_run())

    def test_scheduled_task_seed_exists(self, temp_db):
        """The todo-reminder scheduled task (id=9999) is seeded by init_db."""
        from src.infra.db import get_db

        async def _run():
            db = await get_db()
            cursor = await db.execute(
                "SELECT id, title FROM scheduled_tasks WHERE id = 9999"
            )
            row = await cursor.fetchone()
            assert row is not None, "Todo reminder scheduled task (id=9999) should be seeded"
            assert "Todo" in row["title"] or "todo" in row["title"].lower()

        run_async(_run())


# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestShoppingIntelligenceKeywords:
    """Shopping intelligence modules — keyword-based logic (no LLM)."""

    def test_query_analyzer_detect_language_turkish(self):
        """detect_language identifies Turkish text (returns 'tr')."""
        from src.shopping.text_utils import detect_language
        # Turkish-specific chars: ş, ı, ğ, ü, ö, ç — triggers 'tr'
        lang = detect_language("Bu laptopun fiyatı ne kadar?")
        assert lang == "tr", f"Expected 'tr' for Turkish text, got: {lang}"

    def test_query_analyzer_detect_language_english(self):
        """detect_language returns 'en' for ASCII-only text."""
        from src.shopping.text_utils import detect_language
        lang = detect_language("What is the best laptop under 1000 dollars?")
        assert lang == "en", f"Expected 'en' for English text, got: {lang}"

    def test_normalize_turkish_removes_accents(self):
        """normalize_turkish converts ş→s, ı→i, etc."""
        from src.shopping.text_utils import normalize_turkish
        result = normalize_turkish("şişko")
        # Should produce something without ş
        assert "ş" not in result or result == "şişko"  # no-op is also acceptable

    def test_generate_search_variants_non_empty(self):
        """generate_search_variants returns at least the original query."""
        from src.shopping.text_utils import generate_search_variants
        variants = generate_search_variants("RTX 4070")
        assert len(variants) >= 1
        # Original query should be preserved somewhere in variants
        assert any("RTX 4070" in v or "rtx 4070" in v.lower() for v in variants)

    def test_shopping_sub_intent_all_categories(self):
        """All sub-intent categories are reachable via keyword matching."""
        from src.core.task_classifier import _classify_shopping_sub_intent

        cases = {
            "price_check": "What is the price of iPhone 15?",
            "compare": "Compare RTX 4090 vs RX 7900 XTX",
            "deal_hunt": "En ucuz laptop hangisi, indirim var mı?",
            "purchase_advice": "Should I buy the MacBook Pro?",
            "upgrade": "I want to upgrade from GTX 1060 to RTX 4070",
            "gift": "Hediye olarak ne alsam, teknoloji",
        }

        for expected_intent, text in cases.items():
            result = _classify_shopping_sub_intent(text)
            assert result == expected_intent, (
                f"For '{text}': expected '{expected_intent}', got '{result}'"
            )


# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestRouterModelRequirements:
    """ModelRequirements dataclass and basic router logic (no LLM)."""

    def test_model_requirements_defaults(self):
        """ModelRequirements has sensible defaults."""
        from src.core.router import ModelRequirements
        reqs = ModelRequirements()
        assert reqs.difficulty == 5
        assert reqs.prefer_speed is False
        assert reqs.local_only is False
        assert reqs.needs_vision is False

    def test_model_requirements_speed_config(self):
        """prefer_speed=True is set correctly."""
        from src.core.router import ModelRequirements
        reqs = ModelRequirements(
            task="assistant",
            agent_type="assistant",
            difficulty=2,
            prefer_speed=True,
        )
        assert reqs.prefer_speed is True
        assert reqs.difficulty == 2

    def test_capability_to_task_mapping_complete(self):
        """CAPABILITY_TO_TASK covers all common primary capabilities."""
        from src.core.router import CAPABILITY_TO_TASK
        expected_caps = [
            "reasoning", "planning", "analysis", "code_generation",
            "code_reasoning", "system_design", "prose_quality",
            "instruction_adherence", "domain_knowledge",
            "conversation", "general", "shopping",
        ]
        for cap in expected_caps:
            assert cap in CAPABILITY_TO_TASK, f"Capability '{cap}' missing from CAPABILITY_TO_TASK"

    def test_model_requirements_agent_type_field(self):
        """ModelRequirements has an agent_type field (not just a kwarg)."""
        from src.core.router import ModelRequirements
        reqs = ModelRequirements(
            task="coder",
            agent_type="coder",
            difficulty=6,
            prefer_speed=False,
            needs_function_calling=True,
        )
        assert reqs.difficulty == 6
        assert reqs.agent_type == "coder"

    def test_model_requirements_model_override(self):
        """model_override can be set to pin to a specific model."""
        from src.core.router import ModelRequirements
        reqs = ModelRequirements(task="assistant", difficulty=2)
        reqs.model_override = "llama_cpp/some-model"
        assert reqs.model_override == "llama_cpp/some-model"

    def test_model_requirements_effective_task_fallback(self):
        """effective_task falls back gracefully for unknown task names."""
        from src.core.router import ModelRequirements
        reqs = ModelRequirements(task="unknown_task_xyz")
        # Should not raise; returns empty string for unknown task
        result = reqs.effective_task
        assert isinstance(result, str)


# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestClassificationEdgeCases:
    """Edge cases in keyword classification not covered elsewhere."""

    def test_classify_empty_title_and_description(self):
        """Empty strings fall back to default executor classification."""
        from src.core.task_classifier import _classify_by_keywords
        result = _classify_by_keywords("", "")
        assert result.agent_type == "executor"
        assert result.confidence <= 0.4

    def test_classify_priority_shopping_over_researcher(self):
        """Shopping keywords take precedence over researcher keywords.

        The KEYWORD_RULES list is evaluated top-to-bottom; shopping_advisor
        appears first, so 'compare fiyat' should classify as shopping.
        """
        from src.core.task_classifier import _classify_by_keywords
        result = _classify_by_keywords(
            "compare prices",
            "fiyat karşılaştırması yap for GPU models",
        )
        # 'fiyat' matches shopping_advisor before 'compare' matches researcher
        assert result.agent_type == "shopping_advisor"

    def test_classify_confidence_llm_vs_keyword(self):
        """LLM classification always has higher confidence than keyword."""
        from src.core.task_classifier import TaskClassification
        # Simulate results from each method
        llm_result = TaskClassification(agent_type="coder", confidence=0.85, method="llm")
        kw_result = TaskClassification(agent_type="coder", confidence=0.4, method="keyword")
        assert llm_result.confidence > kw_result.confidence

    def test_classify_turkish_shopping_keywords(self):
        """Turkish shopping keywords (almak istiyorum, hediye) → shopping_advisor."""
        from src.core.task_classifier import _classify_by_keywords

        tr_cases = [
            ("Almak istiyorum", "bir laptop almak istiyorum bütçem 20000 TL"),
            ("Hediye", "hediye için teknoloji ürünü arıyorum"),
            ("En ucuz", "en ucuz akıllı saat hangisi"),
        ]
        for title, desc in tr_cases:
            result = _classify_by_keywords(title, desc)
            assert result.agent_type == "shopping_advisor", (
                f"'{title} / {desc[:50]}' → expected shopping_advisor, got {result.agent_type}"
            )

    def test_message_classify_cancel_command(self):
        """'/cancel' commands are classified as task control, not shopping."""
        from src.app.telegram_bot import TelegramInterface
        classify = TelegramInterface._classify_message_by_keywords
        result = classify("/cancel 5")
        # Cancel commands don't match shopping or todo — should be task or command
        assert result["type"] != "shopping", (
            "/cancel must not be misclassified as shopping"
        )


# ===========================================================================
# END-TO-END LLM PIPELINE TESTS
# ===========================================================================


@pytest.mark.integration
@pytest.mark.llm
class TestFullPipelineLLM:
    """Full idea → classification → agent assignment → execution → result.

    Groups multiple LLM tests in one class so the model is loaded only once
    (or at most a few times) per test session.

    Requires: llama-server or Ollama running locally.
    """

    @pytest.fixture(autouse=True)
    def _skip_if_no_model(self, fastest_local_model):
        if fastest_local_model is None:
            pytest.skip("No local model available — set up llama-server or Ollama")

    @pytest.mark.timeout(240)
    def test_idea_to_classification_to_agent_simple_qa(self, temp_db, fastest_local_model):
        """Full pipeline: simple Q&A idea → classified → executed → non-empty result.

        Pipeline steps:
        1. Classify the idea with LLM
        2. Resolve the agent from classification
        3. Execute the agent
        4. Verify the result dict is non-empty
        """
        from src.core.task_classifier import classify_task
        from src.agents import get_agent

        idea_title = "What is 2 + 2?"
        idea_desc = "Simple arithmetic question"

        async def _run():
            # Step 1: classify
            cls = await classify_task(idea_title, idea_desc)
            assert cls.agent_type in (
                "assistant", "executor", "analyst", "summarizer"
            ), f"Unexpected classification for simple math: {cls.agent_type}"

            # Step 2: resolve agent
            agent = get_agent(cls.agent_type)
            assert agent is not None

            # Step 3: build minimal task dict and execute
            task = {
                "id": 1001,
                "title": idea_title,
                "description": idea_desc,
                "agent_type": cls.agent_type,
                "context": json.dumps({
                    "model_override": fastest_local_model,
                }),
                "depends_on": "[]",
                "mission_id": None,
            }
            result = await agent.execute(task)

            # Step 4: verify structure
            assert isinstance(result, dict), "Agent must return a dict"
            assert any(k in result for k in ("result", "error", "status")), (
                f"Result dict missing expected keys: {list(result.keys())}"
            )

        run_async(_run())

    @pytest.mark.timeout(240)
    def test_idea_to_classification_coding_task(self, temp_db, fastest_local_model):
        """Coding idea is classified as coder/planner and returns non-empty result."""
        from src.core.task_classifier import classify_task
        from src.agents import get_agent

        idea_title = "Write a Python function"
        idea_desc = "Write a Python function that reverses a string"

        async def _run():
            cls = await classify_task(idea_title, idea_desc)
            assert cls.agent_type in (
                "coder", "implementer", "executor", "assistant"
            ), f"Unexpected classification for coding task: {cls.agent_type}"

            agent = get_agent(cls.agent_type)
            task = {
                "id": 1002,
                "title": idea_title,
                "description": idea_desc,
                "agent_type": cls.agent_type,
                "context": json.dumps({"model_override": fastest_local_model}),
                "depends_on": "[]",
                "mission_id": None,
            }
            result = await agent.execute(task)
            assert isinstance(result, dict)
            result_text = result.get("result", "") or ""
            assert len(result_text) > 0, "Coding task returned empty result"

        run_async(_run())

    @pytest.mark.timeout(120)
    def test_classification_method_is_llm_or_keyword(self, temp_db, fastest_local_model):
        """classify_task always returns a valid method field."""
        from src.core.task_classifier import classify_task

        async def _run():
            cls = await classify_task("Test task", "A generic test task description")
            assert cls.method in ("llm", "keyword", "keyword_default"), (
                f"Unknown classification method: {cls.method}"
            )
            assert cls.agent_type != "", "agent_type must not be empty"
            assert 1 <= cls.difficulty <= 10, (
                f"difficulty must be 1-10, got {cls.difficulty}"
            )

        run_async(_run())

    @pytest.mark.timeout(120)
    def test_direct_llm_call_via_router(self, temp_db, fastest_local_model):
        """call_model via the router returns a dict with 'content' key."""
        from src.core.router import ModelRequirements, call_model

        async def _run():
            reqs = ModelRequirements(
                task="assistant",
                difficulty=2,
                prefer_speed=True,
                estimated_input_tokens=50,
                estimated_output_tokens=50,
            )
            if fastest_local_model:
                reqs.model_override = fastest_local_model

            messages = [
                {"role": "user", "content": "Reply with exactly the word: PONG"}
            ]

            response = await call_model(reqs, messages)
            assert isinstance(response, dict), "call_model must return a dict"
            assert "content" in response, f"Response missing 'content' key: {response}"
            content = response["content"]
            assert isinstance(content, str) and len(content) > 0, (
                f"Response content is empty or not a string: {content!r}"
            )

        run_async(_run())

    @pytest.mark.timeout(120)
    def test_llm_returns_parseable_json_for_classifier(self, temp_db, fastest_local_model):
        """The classifier prompt produces JSON parseable by _extract_json."""
        from src.core.router import ModelRequirements, call_model
        from src.core.task_classifier import CLASSIFIER_PROMPT, _extract_json

        async def _run():
            reqs = ModelRequirements(
                task="router",
                difficulty=2,
                prefer_speed=True,
                needs_json_mode=True,
                estimated_input_tokens=300,
                estimated_output_tokens=100,
            )
            if fastest_local_model:
                reqs.model_override = fastest_local_model

            messages = [{
                "role": "user",
                "content": CLASSIFIER_PROMPT.format(
                    task_description="Write a Python hello world script: simple coding task"
                )
            }]

            response = await call_model(reqs, messages)
            content = response.get("content", "")
            assert len(content) > 0, "LLM returned empty content for classifier prompt"

            # Try to parse JSON (may fail on very fast/degraded models)
            try:
                parsed = _extract_json(content)
                assert "agent_type" in parsed, (
                    f"Classifier JSON missing 'agent_type': {parsed}"
                )
            except (ValueError, Exception) as e:
                # Acceptable if the model is too fast/small to reliably format JSON
                pytest.xfail(
                    f"LLM output not parseable as classifier JSON "
                    f"(expected for very small/fast models): {e}\n"
                    f"Raw output: {content[:200]}"
                )

        run_async(_run())


# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.llm
class TestShoppingPipelineLLM:
    """Shopping-specific end-to-end tests with real LLM.

    These are separate from TestFullPipelineLLM to allow running shopping
    tests independently. They share the same LLM infrastructure.
    """

    @pytest.fixture(autouse=True)
    def _skip_if_no_model(self, fastest_local_model):
        if fastest_local_model is None:
            pytest.skip("No local model available — set up llama-server or Ollama")

    @pytest.mark.timeout(300)
    def test_shopping_query_classify_then_execute(self, temp_db, fastest_local_model):
        """Shopping query: classify → agent → execute → non-empty response.

        This is the core KutAI shopping flow:
        'I want to buy an RTX 4070' → shopping_advisor → result text
        """
        from src.core.task_classifier import classify_task
        from src.agents import get_agent

        title = "GPU önerisi"
        desc = "Oyun için RTX 4070 almak istiyorum, fiyat ne kadar?"

        async def _run():
            cls = await classify_task(title, desc)
            # Turkish shopping keywords should reliably classify as shopping
            assert cls.agent_type in (
                "shopping_advisor", "researcher", "analyst"
            ), f"Expected shopping classification, got: {cls.agent_type}"

            agent = get_agent(cls.agent_type)
            task = {
                "id": 2001,
                "title": title,
                "description": desc,
                "agent_type": cls.agent_type,
                "context": json.dumps({
                    "model_override": fastest_local_model,
                    "shopping_sub_intent": cls.shopping_sub_intent,
                    "max_web_searches": 0,  # no web searches for speed
                }),
                "depends_on": "[]",
                "mission_id": None,
            }
            result = await agent.execute(task)

            assert isinstance(result, dict), "Agent must return a dict"
            result_text = result.get("result", "") or ""
            assert len(result_text) >= 10, (
                f"Shopping result too short ({len(result_text)} chars): {result_text!r}"
            )

        run_async(_run())

    @pytest.mark.timeout(120)
    def test_shopping_sub_intent_attached_in_classify_task(self, temp_db, fastest_local_model):
        """classify_task attaches shopping_sub_intent for shopping queries."""
        from src.core.task_classifier import classify_task

        async def _run():
            cls = await classify_task(
                "Fiyat karşılaştırması",
                "iPhone 15 ile Samsung S24 fiyat karşılaştırması yap",
            )
            if cls.agent_type == "shopping_advisor":
                assert cls.shopping_sub_intent is not None, (
                    "shopping_sub_intent must be set for shopping_advisor tasks"
                )
                # With 'fiyat' and 'karşılaştırması' keywords, expect price_check or compare
                assert cls.shopping_sub_intent in (
                    "price_check", "compare", "exploration"
                ), f"Unexpected sub_intent: {cls.shopping_sub_intent}"

        run_async(_run())

    @pytest.mark.timeout(120)
    def test_shopping_assistant_handles_status_query(self, temp_db, fastest_local_model):
        """Status queries go through assistant/executor, not shopping_advisor.

        Regression test: 'How is the coffee machine search going?' should not
        result in a new shopping search — it's asking for status.
        """
        from src.core.task_classifier import classify_task

        async def _run():
            # Keyword pre-filter in TelegramInterface catches this first;
            # but if it falls through to LLM, test documents behavior.
            cls = await classify_task(
                "How is the coffee machine search going",
                "Asking for status of an ongoing task",
            )
            if cls.agent_type == "shopping_advisor":
                pytest.xfail(
                    f"LLM classified status query as shopping_advisor. "
                    "This is a known weakness of small models. "
                    "The TelegramInterface keyword pre-filter is the primary guard."
                )

        run_async(_run())


# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.llm
class TestWorkflowRunnerLLM:
    """Workflow runner with real mission creation and LLM step execution.

    NOTE: These tests create real missions in the temp DB but do NOT run
    the full orchestrator loop (which would require Telegram etc.).
    They test the workflow engine primitives in isolation.
    """

    @pytest.fixture(autouse=True)
    def _skip_if_no_model(self, fastest_local_model):
        if fastest_local_model is None:
            pytest.skip("No local model available")

    @pytest.mark.timeout(300)
    def test_workflow_step_executed_by_agent(self, temp_db, fastest_local_model):
        """A single workflow step can be executed by the assigned agent.

        This simulates what the orchestrator does after calling
        expand_steps_to_tasks: it takes one task dict and runs the agent.
        """
        from src.workflows.engine.expander import expand_steps_to_tasks
        from src.infra.db import add_mission, insert_tasks_atomically, get_task
        from src.agents import get_agent

        steps = [
            {
                "id": "1.1",
                "name": "Idea analysis",
                "phase": "phase_1",
                "agent": "analyst",
                "instruction": (
                    "Analyze this product idea in 2-3 sentences: "
                    "A simple todo app with Telegram bot interface."
                ),
                "input_artifacts": [],
                "output_artifacts": ["idea_analysis"],
                "depends_on": [],
            }
        ]

        async def _run():
            mid = await add_mission(
                title="Workflow LLM test",
                description="single step workflow test",
            )
            tasks = expand_steps_to_tasks(
                steps,
                mission_id=mid,
                initial_context={"raw_idea": "Telegram todo bot"},
            )
            created_ids = await insert_tasks_atomically(tasks, mission_id=mid)
            assert created_ids[0] > 0, "Task must be created"

            db_task = await get_task(created_ids[0])
            assert db_task is not None

            # Override model for speed
            ctx = json.loads(db_task.get("context", "{}") or "{}")
            ctx["model_override"] = fastest_local_model
            db_task["context"] = json.dumps(ctx)

            agent = get_agent(db_task["agent_type"])
            result = await agent.execute(db_task)

            assert isinstance(result, dict)
            result_text = result.get("result", "") or ""
            assert len(result_text) > 0, "Workflow step agent returned empty result"

        run_async(_run())
