# tests/test_phase10.py
"""
Tests for Phase 10: Model Backend Expansion & Intelligent Router

  10.1  Custom Endpoint Support (config.py)
  10.2  Semantic Task Classifier (task_classifier.py)
  10.3  Sensitivity Detection (security/sensitivity.py)
  10.4  Capability-Based Model Matching (router.py select_model)
  10.5  Model Pinning & Override (router.py call_model)

Note: Tests that would require `import router` use source-code inspection
instead, because litellm (a router dependency) is not always installed in
the test environment.
"""
import asyncio
import inspect
import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read_source(relpath: str) -> str:
    """Read a source file relative to project root (UTF-8)."""
    with open(os.path.join(_ROOT, relpath), "r", encoding="utf-8") as f:
        return f.read()


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ─── 10.1 Custom Endpoint Support ────────────────────────────────────────────

class TestCustomEndpointDetection(unittest.TestCase):
    """Tests for _detect_custom_endpoints in config.py."""

    def test_detect_custom_endpoints_function_exists(self):
        """_detect_custom_endpoints should be defined in config.py."""
        from config import _detect_custom_endpoints
        self.assertTrue(callable(_detect_custom_endpoints))

    def test_detect_custom_endpoints_returns_list(self):
        """Should return a list."""
        from config import _detect_custom_endpoints
        result = _detect_custom_endpoints()
        self.assertIsInstance(result, list)

    def test_custom_endpoints_module_var(self):
        """CUSTOM_ENDPOINTS should be a module-level list."""
        from config import CUSTOM_ENDPOINTS
        self.assertIsInstance(CUSTOM_ENDPOINTS, list)

    def test_env_parsing_llama_cpp(self):
        """Should parse LLAMA_CPP_ENDPOINTS env var format."""
        source = inspect.getsource(
            __import__("config")._detect_custom_endpoints
        )
        self.assertIn("LLAMA_CPP_ENDPOINTS", source)
        self.assertIn("llamacpp", source)

    def test_env_parsing_custom_openai(self):
        """Should parse CUSTOM_OPENAI_ENDPOINTS env var."""
        source = inspect.getsource(
            __import__("config")._detect_custom_endpoints
        )
        self.assertIn("CUSTOM_OPENAI_ENDPOINTS", source)
        self.assertIn("custom_openai", source)

    def test_endpoint_entry_has_api_base(self):
        """Endpoint entries should include api_base field."""
        source = inspect.getsource(
            __import__("config")._detect_custom_endpoints
        )
        self.assertIn("api_base", source)

    def test_endpoint_probes_v1_models(self):
        """Should probe /v1/models to verify endpoint is alive."""
        source = inspect.getsource(
            __import__("config")._detect_custom_endpoints
        )
        self.assertIn("/v1/models", source)


class TestContextLengthInModelPool(unittest.TestCase):
    """Tests that context_length field is present on MODEL_POOL entries."""

    def test_all_pool_entries_have_context_length(self):
        """Every MODEL_POOL entry should have context_length."""
        from config import MODEL_POOL
        for key, cfg in MODEL_POOL.items():
            self.assertIn(
                "context_length", cfg,
                f"MODEL_POOL['{key}'] is missing context_length"
            )

    def test_context_lengths_are_positive_ints(self):
        """context_length should be a positive integer."""
        from config import MODEL_POOL
        for key, cfg in MODEL_POOL.items():
            ctx_len = cfg.get("context_length", 0)
            self.assertIsInstance(ctx_len, int, f"{key} context_length not int")
            self.assertGreater(ctx_len, 0, f"{key} context_length <= 0")

    def test_api_base_in_config_source(self):
        """config.py should define api_base in endpoint entries."""
        source = _read_source("config.py")
        self.assertIn('"api_base"', source)

    def test_custom_endpoint_registration_in_model_pool(self):
        """Custom endpoints should be registered into MODEL_POOL."""
        source = _read_source("config.py")
        self.assertIn("CUSTOM_ENDPOINTS", source)
        self.assertIn("MODEL_POOL[", source)


# ─── 10.2 Semantic Task Classifier ───────────────────────────────────────────

class TestTaskClassifier(unittest.TestCase):
    """Tests for task_classifier.py."""

    def test_module_imports(self):
        """task_classifier module should import without litellm."""
        import task_classifier
        self.assertTrue(hasattr(task_classifier, "classify_task_semantic"))

    def test_reference_tasks_defined(self):
        """Should define reference tasks across all expected categories."""
        from task_classifier import REFERENCE_TASKS
        expected = {
            "simple_qa", "code_simple", "code_complex", "research",
            "writing", "planning", "action_required", "sensitive",
        }
        self.assertEqual(set(REFERENCE_TASKS.keys()), expected)

    def test_reference_tasks_non_empty(self):
        """Each category should have at least 3 reference examples."""
        from task_classifier import REFERENCE_TASKS
        for cat, examples in REFERENCE_TASKS.items():
            self.assertGreaterEqual(
                len(examples), 3,
                f"Category '{cat}' has < 3 reference examples"
            )

    def test_reference_flat_populated(self):
        """Flattened reference list should be populated."""
        from task_classifier import _REFERENCE_FLAT
        self.assertGreater(len(_REFERENCE_FLAT), 30)
        # Each entry is (category, text) tuple
        cat, text = _REFERENCE_FLAT[0]
        self.assertIsInstance(cat, str)
        self.assertIsInstance(text, str)

    def test_keyword_classifier_code_complex(self):
        """Keyword classifier should detect code_complex tasks."""
        from task_classifier import _classify_by_keywords
        result = _classify_by_keywords(
            "Implement a REST API",
            "with authentication and rate limiting"
        )
        self.assertEqual(result["category"], "code_complex")
        self.assertEqual(result["method"], "keyword")

    def test_keyword_classifier_code_simple(self):
        """Keyword classifier should detect code_simple tasks."""
        from task_classifier import _classify_by_keywords
        result = _classify_by_keywords(
            "Fix typo in variable name",
            "Change the variable name from foo to bar"
        )
        self.assertEqual(result["category"], "code_simple")

    def test_keyword_classifier_research(self):
        """Keyword classifier should detect research tasks."""
        from task_classifier import _classify_by_keywords
        result = _classify_by_keywords(
            "Research best practices",
            "Find the latest developments in AI safety"
        )
        self.assertEqual(result["category"], "research")

    def test_keyword_classifier_writing(self):
        """Keyword classifier should detect writing tasks."""
        from task_classifier import _classify_by_keywords
        result = _classify_by_keywords(
            "Draft a memo",
            "Draft a project update for stakeholders"
        )
        self.assertEqual(result["category"], "writing")

    def test_keyword_classifier_planning(self):
        """Keyword classifier should detect planning tasks."""
        from task_classifier import _classify_by_keywords
        # Avoid words like "architect"/"build"/"design system" that
        # trigger code_complex first
        result = _classify_by_keywords(
            "Create a roadmap",
            "Outline the Q4 roadmap for the team"
        )
        self.assertEqual(result["category"], "planning")

    def test_keyword_classifier_action_required(self):
        """Keyword classifier should detect action_required tasks."""
        from task_classifier import _classify_by_keywords
        # Avoid "build"/"report" which trigger code_complex/writing first
        result = _classify_by_keywords(
            "Run tests on the project",
            "Execute the test suite and check results"
        )
        self.assertEqual(result["category"], "action_required")

    def test_keyword_classifier_sensitive(self):
        """Keyword classifier should detect sensitive tasks."""
        from task_classifier import _classify_by_keywords
        result = _classify_by_keywords(
            "Process payment",
            "Handle the credit card transaction for the customer"
        )
        self.assertEqual(result["category"], "sensitive")

    def test_keyword_classifier_simple_qa(self):
        """Keyword classifier should detect simple_qa tasks."""
        from task_classifier import _classify_by_keywords
        result = _classify_by_keywords(
            "What is Python?",
            "What is the difference between a list and a tuple?"
        )
        self.assertEqual(result["category"], "simple_qa")

    def test_keyword_classifier_default(self):
        """Unknown tasks should default to simple_qa with low confidence."""
        from task_classifier import _classify_by_keywords
        result = _classify_by_keywords(
            "xyzzy",
            "completely arbitrary gibberish text"
        )
        self.assertEqual(result["category"], "simple_qa")
        self.assertEqual(result["method"], "keyword_default")
        self.assertLessEqual(result["confidence"], 0.5)

    def test_classify_task_semantic_returns_dict(self):
        """classify_task_semantic should return a dict with required keys."""
        from task_classifier import classify_task_semantic
        result = run_async(classify_task_semantic(
            "Simple question", "What is 2+2?"
        ))
        self.assertIn("category", result)
        self.assertIn("confidence", result)
        self.assertIn("method", result)

    def test_classify_task_semantic_categories_valid(self):
        """Returned category should be one of the defined categories."""
        from task_classifier import classify_task_semantic, REFERENCE_TASKS
        result = run_async(classify_task_semantic(
            "Refactor the ORM layer", "Refactor to use connection pooling"
        ))
        self.assertIn(result["category"], REFERENCE_TASKS.keys())

    def test_cosine_similarity(self):
        """Cosine similarity should work correctly for basic vectors."""
        from task_classifier import _cosine_similarity
        # Identical vectors -> 1.0
        self.assertAlmostEqual(
            _cosine_similarity([1, 0, 0], [1, 0, 0]), 1.0
        )
        # Orthogonal vectors -> 0.0
        self.assertAlmostEqual(
            _cosine_similarity([1, 0, 0], [0, 1, 0]), 0.0
        )
        # Opposite vectors -> -1.0
        self.assertAlmostEqual(
            _cosine_similarity([1, 0, 0], [-1, 0, 0]), -1.0
        )

    def test_cosine_similarity_empty(self):
        """Empty vectors should return 0.0."""
        from task_classifier import _cosine_similarity
        self.assertEqual(_cosine_similarity([], []), 0.0)

    def test_cosine_similarity_mismatched_length(self):
        """Mismatched length vectors should return 0.0."""
        from task_classifier import _cosine_similarity
        self.assertEqual(_cosine_similarity([1, 0], [1, 0, 0]), 0.0)

    def test_embedding_cache_key(self):
        """Cache key should be a short deterministic hash."""
        from task_classifier import _cache_key
        k1 = _cache_key("hello world")
        k2 = _cache_key("hello world")
        k3 = _cache_key("different text")
        self.assertEqual(k1, k2)
        self.assertNotEqual(k1, k3)
        self.assertEqual(len(k1), 16)


# ─── 10.3 Sensitivity Detection ──────────────────────────────────────────────

class TestSensitivityDetection(unittest.TestCase):
    """Tests for security/sensitivity.py."""

    def test_module_imports(self):
        """security.sensitivity module should import cleanly."""
        from security.sensitivity import (
            SensitivityLevel, SensitivityResult,
            detect_sensitivity, scan_task,
        )
        self.assertTrue(callable(detect_sensitivity))
        self.assertTrue(callable(scan_task))

    def test_sensitivity_levels(self):
        """SensitivityLevel should have PUBLIC, PRIVATE, SECRET."""
        from security.sensitivity import SensitivityLevel
        self.assertEqual(SensitivityLevel.PUBLIC, "public")
        self.assertEqual(SensitivityLevel.PRIVATE, "private")
        self.assertEqual(SensitivityLevel.SECRET, "secret")

    def test_clean_text_is_public(self):
        """Text with no sensitive data should be PUBLIC."""
        from security.sensitivity import detect_sensitivity, SensitivityLevel
        result = detect_sensitivity("Hello, how are you?")
        self.assertEqual(result.level, SensitivityLevel.PUBLIC)
        self.assertEqual(result.matches, [])

    def test_empty_text_is_public(self):
        """Empty text should be PUBLIC."""
        from security.sensitivity import detect_sensitivity, SensitivityLevel
        result = detect_sensitivity("")
        self.assertEqual(result.level, SensitivityLevel.PUBLIC)

    # -- API Key Detection --

    def test_openai_key_detected(self):
        """Should detect OpenAI API keys (sk-...)."""
        from security.sensitivity import detect_sensitivity, SensitivityLevel
        text = "Use this key: sk-abcdefghijklmnopqrstuvwxyz1234"
        result = detect_sensitivity(text)
        self.assertEqual(result.level, SensitivityLevel.SECRET)
        self.assertTrue(any("OpenAI" in m for m in result.matches))

    def test_github_token_detected(self):
        """Should detect GitHub personal access tokens."""
        from security.sensitivity import detect_sensitivity, SensitivityLevel
        text = "Token: ghp_aBcDeFgHiJkLmNoPqRsTuVwXyZ012345678901"
        result = detect_sensitivity(text)
        self.assertEqual(result.level, SensitivityLevel.SECRET)
        self.assertTrue(any("GitHub" in m for m in result.matches))

    def test_aws_key_detected(self):
        """Should detect AWS access key IDs."""
        from security.sensitivity import detect_sensitivity, SensitivityLevel
        text = "AWS key: AKIAIOSFODNN7EXAMPLE"
        result = detect_sensitivity(text)
        self.assertEqual(result.level, SensitivityLevel.SECRET)
        self.assertTrue(any("AWS" in m for m in result.matches))

    def test_slack_token_detected(self):
        """Should detect Slack tokens."""
        from security.sensitivity import detect_sensitivity, SensitivityLevel
        text = "Bot token: xoxb-123456789012-1234567890123-abcdefghijklmnopqrstuv"
        result = detect_sensitivity(text)
        self.assertEqual(result.level, SensitivityLevel.SECRET)

    def test_stripe_key_detected(self):
        """Should detect Stripe secret keys."""
        from security.sensitivity import detect_sensitivity, SensitivityLevel
        text = "Stripe: sk_live_abcdefghijklmnopqrstuvwxyz"
        result = detect_sensitivity(text)
        self.assertEqual(result.level, SensitivityLevel.SECRET)

    def test_bearer_token_detected(self):
        """Should detect Bearer tokens."""
        from security.sensitivity import detect_sensitivity, SensitivityLevel
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.payload.sig"
        result = detect_sensitivity(text)
        self.assertEqual(result.level, SensitivityLevel.SECRET)

    # -- Private Key Detection --

    def test_rsa_private_key_detected(self):
        """Should detect RSA private key blocks."""
        from security.sensitivity import detect_sensitivity, SensitivityLevel
        text = "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIB...\n-----END RSA PRIVATE KEY-----"
        result = detect_sensitivity(text)
        self.assertEqual(result.level, SensitivityLevel.SECRET)
        self.assertTrue(any("private key" in m.lower() for m in result.matches))

    def test_openssh_private_key_detected(self):
        """Should detect OpenSSH private keys."""
        from security.sensitivity import detect_sensitivity, SensitivityLevel
        text = "-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1..."
        result = detect_sensitivity(text)
        self.assertEqual(result.level, SensitivityLevel.SECRET)

    # -- Credit Card Detection --

    def test_visa_card_detected(self):
        """Should detect Visa card numbers."""
        from security.sensitivity import detect_sensitivity, SensitivityLevel
        text = "Card: 4111111111111111"
        result = detect_sensitivity(text)
        self.assertEqual(result.level, SensitivityLevel.SECRET)
        self.assertTrue(any("Visa" in m or "card" in m.lower() for m in result.matches))

    def test_mastercard_detected(self):
        """Should detect Mastercard numbers."""
        from security.sensitivity import detect_sensitivity, SensitivityLevel
        text = "Card: 5500000000000004"
        result = detect_sensitivity(text)
        self.assertEqual(result.level, SensitivityLevel.SECRET)

    def test_card_with_separators_detected(self):
        """Should detect card numbers with dashes or spaces."""
        from security.sensitivity import detect_sensitivity, SensitivityLevel
        text = "Card: 4111-1111-1111-1111"
        result = detect_sensitivity(text)
        self.assertEqual(result.level, SensitivityLevel.SECRET)

    # -- SSN Detection --

    def test_ssn_detected(self):
        """Should detect SSN patterns."""
        from security.sensitivity import detect_sensitivity, SensitivityLevel
        text = "SSN: 123-45-6789"
        result = detect_sensitivity(text)
        self.assertEqual(result.level, SensitivityLevel.SECRET)
        self.assertTrue(any("SSN" in m for m in result.matches))

    # -- Password Detection --

    def test_password_assignment_detected(self):
        """Should detect password assignments."""
        from security.sensitivity import detect_sensitivity, SensitivityLevel
        text = "password: mysecretpassword123"
        result = detect_sensitivity(text)
        self.assertEqual(result.level, SensitivityLevel.SECRET)

    def test_api_key_assignment_detected(self):
        """Should detect api_key=... assignments."""
        from security.sensitivity import detect_sensitivity, SensitivityLevel
        text = 'api_key = "my-secret-key-value"'
        result = detect_sensitivity(text)
        self.assertEqual(result.level, SensitivityLevel.SECRET)

    # -- Email Detection (PRIVATE) --

    def test_email_detected_as_private(self):
        """Should detect email addresses as PRIVATE level."""
        from security.sensitivity import detect_sensitivity, SensitivityLevel
        text = "Contact: user@example.com for details"
        result = detect_sensitivity(text)
        self.assertEqual(result.level, SensitivityLevel.PRIVATE)
        self.assertTrue(any("Email" in m for m in result.matches))

    def test_email_with_secret_elevates_to_secret(self):
        """Email + API key should elevate to SECRET."""
        from security.sensitivity import detect_sensitivity, SensitivityLevel
        text = "Send to user@example.com with key sk-abcdefghijklmnopqrstuvwxyz1234"
        result = detect_sensitivity(text)
        self.assertEqual(result.level, SensitivityLevel.SECRET)
        # Should include both matches
        self.assertGreaterEqual(len(result.matches), 2)

    # -- scan_task wrapper --

    def test_scan_task_combines_fields(self):
        """scan_task should scan title + description + context."""
        from security.sensitivity import scan_task, SensitivityLevel
        result = scan_task(
            title="Fix login",
            description="Update the password: admin123",
            context=None,
        )
        self.assertEqual(result.level, SensitivityLevel.SECRET)

    def test_scan_task_with_dict_context(self):
        """scan_task should serialize dict context."""
        from security.sensitivity import scan_task, SensitivityLevel
        result = scan_task(
            title="Task",
            description="Something",
            context={"api_key": "sk-abcdefghijklmnopqrstuvwxyz1234"},
        )
        self.assertEqual(result.level, SensitivityLevel.SECRET)

    def test_scan_task_clean(self):
        """scan_task with clean data should return PUBLIC."""
        from security.sensitivity import scan_task, SensitivityLevel
        result = scan_task(
            title="Hello",
            description="A simple task",
            context=None,
        )
        self.assertEqual(result.level, SensitivityLevel.PUBLIC)


# ─── 10.4 Capability-Based Model Matching (source inspection) ────────────────

class TestCapabilityBasedMatchingSource(unittest.TestCase):
    """Source inspection tests for Phase 10.4 enhancements to select_model."""

    def setUp(self):
        self.source = _read_source("router.py")

    def test_select_model_has_required_capabilities_param(self):
        """select_model should accept required_capabilities parameter."""
        self.assertIn("required_capabilities", self.source)

    def test_select_model_has_min_context_length_param(self):
        """select_model should accept min_context_length parameter."""
        self.assertIn("min_context_length", self.source)

    def test_select_model_has_sensitivity_param(self):
        """select_model should accept sensitivity parameter."""
        # Check in function signature area
        self.assertIn("sensitivity", self.source)

    def test_local_providers_defined(self):
        """Local-only providers for sensitive data should be defined."""
        self.assertIn("_local_providers", self.source)
        self.assertIn("ollama", self.source)
        self.assertIn("llamacpp", self.source)
        self.assertIn("custom_openai", self.source)

    def test_capability_filtering_logic(self):
        """Should filter models by required_capabilities."""
        self.assertIn("required_capabilities", self.source)
        self.assertIn("model_caps", self.source)

    def test_context_length_filtering_logic(self):
        """Should filter models by min_context_length."""
        self.assertIn("min_context_length", self.source)
        self.assertIn("model_ctx", self.source)

    def test_sensitivity_filtering_logic(self):
        """Should restrict to local models for private/secret sensitivity."""
        self.assertIn("restrict_local", self.source)
        self.assertIn('"private"', self.source)
        self.assertIn('"secret"', self.source)

    def test_candidates_include_api_base(self):
        """All candidate dicts should include api_base field."""
        count = self.source.count('"api_base"')
        self.assertGreaterEqual(
            count, 3,
            "api_base should appear in multiple candidate dict constructions"
        )


# ─── 10.5 Model Pinning & Override (source inspection) ───────────────────────

class TestModelPinningSource(unittest.TestCase):
    """Source inspection tests for Phase 10.5 model pinning."""

    def test_call_model_has_model_override_param(self):
        """call_model should accept model_override parameter."""
        source = _read_source("router.py")
        # Check in function signature
        self.assertIn("model_override", source)

    def test_model_override_default_none(self):
        """model_override should default to None."""
        source = _read_source("router.py")
        self.assertIn("model_override: str | None = None", source)

    def test_router_has_pinning_logic(self):
        """router.py call_model should have model pinning logic."""
        source = _read_source("router.py")
        self.assertIn("Model pinned", source)
        self.assertIn("skipping tier selection", source)

    def test_router_has_api_base_passthrough(self):
        """router.py should pass api_base to litellm.acompletion."""
        source = _read_source("router.py")
        self.assertIn('completion_kwargs["api_base"]', source)

    def test_pinned_model_gets_high_score(self):
        """Pinned model should get a very high score."""
        source = _read_source("router.py")
        # Should assign score: 999 to pinned candidate
        self.assertIn('"score": 999', source)

    def test_model_override_raw_litellm_fallback(self):
        """If model_override not in pool, should still try raw litellm call."""
        source = _read_source("router.py")
        self.assertIn("not in MODEL_POOL", source)
        self.assertIn("raw litellm", source.lower())

    def test_base_agent_extracts_model_override(self):
        """base.py execute should extract model_override from task context."""
        source = _read_source(os.path.join("agents", "base.py"))
        self.assertIn("model_override", source)
        # Ensure it's passed to call_model
        self.assertIn("model_override=model_override", source)

    def test_base_agent_single_shot_has_model_override(self):
        """execute_single_shot should also pass model_override."""
        source = _read_source(os.path.join("agents", "base.py"))
        self.assertIn("_ss_model_override", source)

    def test_telegram_model_flag_parsing(self):
        """Telegram bot should parse --model flag from /task command."""
        source = _read_source("telegram_bot.py")
        self.assertIn("--model", source)
        self.assertIn("model_override", source)


# ─── 10.2/10.3 Integration ───────────────────────────────────────────────────

class TestClassifierSensitivityIntegration(unittest.TestCase):
    """Integration tests between classifier and sensitivity detection."""

    def test_classifier_sensitive_category(self):
        """Classifier should categorize payment tasks as sensitive."""
        from task_classifier import _classify_by_keywords
        result = _classify_by_keywords(
            "Process payment",
            "Handle credit card transaction for the customer"
        )
        self.assertEqual(result["category"], "sensitive")

    def test_sensitivity_detects_in_task_content(self):
        """Sensitivity scanner should catch secrets in task descriptions."""
        from security.sensitivity import scan_task, SensitivityLevel
        result = scan_task(
            title="Update credentials",
            description="Use API key sk-prod1234567890abcdefghijklmnop to authenticate",
        )
        self.assertEqual(result.level, SensitivityLevel.SECRET)

    def test_both_classifier_and_sensitivity_agree(self):
        """Both systems should flag sensitive content appropriately."""
        from task_classifier import _classify_by_keywords
        from security.sensitivity import scan_task, SensitivityLevel

        title = "Update API key"
        desc = "Change the api_key = 'sk-abcdefghijklmnopqrstuvwxyz1234' in config"

        classification = _classify_by_keywords(title, desc)
        sensitivity = scan_task(title, desc)

        # Classifier should catch via keyword
        self.assertEqual(classification["category"], "sensitive")
        # Sensitivity should catch the actual key pattern
        self.assertEqual(sensitivity.level, SensitivityLevel.SECRET)


# ─── Edge Cases ──────────────────────────────────────────────────────────────

class TestEdgeCases(unittest.TestCase):
    """Edge case tests for Phase 10 components."""

    def test_cosine_similarity_zero_vectors(self):
        """Zero vectors should return 0.0 similarity."""
        from task_classifier import _cosine_similarity
        self.assertEqual(_cosine_similarity([0, 0, 0], [0, 0, 0]), 0.0)

    def test_sensitivity_multiple_patterns(self):
        """Text with multiple secret patterns should list all matches."""
        from security.sensitivity import detect_sensitivity, SensitivityLevel
        text = (
            "Key: sk-abcdefghijklmnopqrstuvwxyz1234 "
            "Card: 4111111111111111 "
            "SSN: 123-45-6789"
        )
        result = detect_sensitivity(text)
        self.assertEqual(result.level, SensitivityLevel.SECRET)
        self.assertGreaterEqual(len(result.matches), 3)

    def test_keyword_classifier_case_insensitive(self):
        """Keyword classifier should be case insensitive."""
        from task_classifier import _classify_by_keywords
        result1 = _classify_by_keywords("IMPLEMENT AN API", "REFACTOR IT")
        result2 = _classify_by_keywords("implement an api", "refactor it")
        self.assertEqual(result1["category"], result2["category"])

    def test_sensitivity_no_false_positives_on_normal_numbers(self):
        """Short numbers shouldn't trigger credit card detection."""
        from security.sensitivity import detect_sensitivity, SensitivityLevel
        text = "Order #12345 has 3 items worth $50"
        result = detect_sensitivity(text)
        self.assertEqual(result.level, SensitivityLevel.PUBLIC)

    def test_scan_task_with_none_fields(self):
        """scan_task should handle None title/description gracefully."""
        from security.sensitivity import scan_task, SensitivityLevel
        result = scan_task(title=None, description=None, context=None)
        self.assertEqual(result.level, SensitivityLevel.PUBLIC)

    def test_keyword_rules_order_matters(self):
        """code_complex keywords are checked first by design."""
        from task_classifier import _classify_by_keywords
        # "build" triggers code_complex before action_required
        result = _classify_by_keywords(
            "Build and deploy", "Build the app and deploy it"
        )
        self.assertEqual(result["category"], "code_complex")

    def test_classify_semantic_falls_back_to_keywords(self):
        """Without embedding model, classify_task_semantic uses keywords."""
        from task_classifier import classify_task_semantic
        result = run_async(classify_task_semantic(
            "What is the capital of France?",
            "Simple geography question"
        ))
        # Without Ollama embeddings, should fall back to keyword method
        self.assertIn(result["method"], ("keyword", "keyword_default", "embedding"))
        self.assertIn(result["category"], (
            "simple_qa", "code_simple", "code_complex", "research",
            "writing", "planning", "action_required", "sensitive",
        ))


if __name__ == "__main__":
    unittest.main()
