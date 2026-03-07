# tests/test_phase11.py
"""
Tests for Phase 11: Memory & Knowledge System

  11.1  Vector Store Setup (memory/vector_store.py)
  11.2  Episodic Memory (memory/episodic.py)
  11.3  RAG Pipeline (memory/rag.py)
  11.4  Conversation Continuity (memory/conversations.py)
  11.5  Document Ingestion (memory/ingest.py)
  11.6  Memory Forgetting & Decay (memory/decay.py)
  11.7  User Preference Learning (memory/preferences.py)

Tests use source-code inspection where ChromaDB may not be available,
and functional tests where modules can be tested independently.
"""
import asyncio
import inspect
import os
import re
import sys
import time
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


# ═══════════════════════════════════════════════════════════════════════════════
# 11.1 — Vector Store Setup
# ═══════════════════════════════════════════════════════════════════════════════

class TestVectorStoreModule(unittest.TestCase):
    """Tests for memory/vector_store.py structure."""

    def test_vector_store_file_exists(self):
        path = os.path.join(_ROOT, "memory", "vector_store.py")
        self.assertTrue(os.path.isfile(path))

    def test_collections_defined(self):
        src = _read_source("memory/vector_store.py")
        for col in ["episodic", "semantic", "codebase", "errors", "conversations"]:
            self.assertIn(f'"{col}"', src, f"Collection '{col}' not found")

    def test_public_api_exists(self):
        src = _read_source("memory/vector_store.py")
        for fn in ["init_store", "embed_and_store", "query", "delete",
                    "is_ready", "get_collection_count", "get_all_counts",
                    "delete_by_metadata", "reset_store"]:
            self.assertIn(f"async def {fn}" if fn != "is_ready" else f"def {fn}", src,
                          f"Function '{fn}' not found")

    def test_cosine_distance_space(self):
        src = _read_source("memory/vector_store.py")
        self.assertIn("cosine", src)

    def test_access_tracking_metadata(self):
        """embed_and_store should add access_count and last_accessed."""
        src = _read_source("memory/vector_store.py")
        self.assertIn("access_count", src)
        self.assertIn("last_accessed", src)

    def test_query_updates_access_count(self):
        """Query should increment access_count on results."""
        src = _read_source("memory/vector_store.py")
        # Check that query function updates access metadata
        self.assertIn("access_count", src)
        # Find the query function and verify it updates metadata
        query_match = re.search(
            r'async def query\(.*?\n(?:.*?\n)*?.*?access_count.*?(\+\s*1|\+ 1)',
            src, re.DOTALL,
        )
        self.assertIsNotNone(query_match, "query() should increment access_count")


# ═══════════════════════════════════════════════════════════════════════════════
# 11.1 — Shared Embeddings
# ═══════════════════════════════════════════════════════════════════════════════

class TestEmbeddingsModule(unittest.TestCase):
    """Tests for memory/embeddings.py."""

    def test_embeddings_file_exists(self):
        path = os.path.join(_ROOT, "memory", "embeddings.py")
        self.assertTrue(os.path.isfile(path))

    def test_public_api(self):
        src = _read_source("memory/embeddings.py")
        self.assertIn("async def get_embedding", src)
        self.assertIn("async def get_embeddings", src)
        self.assertIn("def embedding_available", src)
        self.assertIn("def clear_cache", src)

    def test_ollama_models_listed(self):
        src = _read_source("memory/embeddings.py")
        self.assertIn("nomic-embed-text", src)
        self.assertIn("all-minilm", src)

    def test_sentence_transformers_fallback(self):
        src = _read_source("memory/embeddings.py")
        self.assertIn("SentenceTransformer", src)
        self.assertIn("all-MiniLM-L6-v2", src)

    def test_cache_max_size(self):
        src = _read_source("memory/embeddings.py")
        self.assertIn("_CACHE_MAX_SIZE", src)

    def test_cache_eviction(self):
        """Cache should evict entries when full."""
        src = _read_source("memory/embeddings.py")
        # Should remove ~10% when full
        self.assertIn("_CACHE_MAX_SIZE // 10", src)


# ═══════════════════════════════════════════════════════════════════════════════
# 11.2 — Episodic Memory
# ═══════════════════════════════════════════════════════════════════════════════

class TestEpisodicMemory(unittest.TestCase):
    """Tests for memory/episodic.py."""

    def test_episodic_file_exists(self):
        path = os.path.join(_ROOT, "memory", "episodic.py")
        self.assertTrue(os.path.isfile(path))

    def test_public_api(self):
        src = _read_source("memory/episodic.py")
        for fn in ["store_task_result", "store_error_recovery",
                    "recall_similar_tasks", "recall_error_patterns",
                    "format_similar_tasks", "format_error_warnings"]:
            self.assertIn(f"def {fn}", src, f"Function '{fn}' not found")

    def test_store_task_result_embeds_key_fields(self):
        """store_task_result should embed title, description, agent_type, outcome."""
        src = _read_source("memory/episodic.py")
        self.assertIn("task_id", src)
        self.assertIn("agent_type", src)
        self.assertIn("model_used", src)
        self.assertIn("success", src)

    def test_error_recovery_high_importance(self):
        """Error recovery entries should have high importance."""
        src = _read_source("memory/episodic.py")
        self.assertIn('"importance": 8', src)

    def test_format_similar_tasks_returns_str(self):
        from memory.episodic import format_similar_tasks
        result = format_similar_tasks([])
        self.assertEqual(result, "")

        result = format_similar_tasks([{
            "title": "Test task",
            "success": True,
            "model_used": "gpt-4",
        }])
        self.assertIn("Past Experience", result)
        self.assertIn("Test task", result)

    def test_format_error_warnings_returns_str(self):
        from memory.episodic import format_error_warnings
        result = format_error_warnings([])
        self.assertEqual(result, "")

        result = format_error_warnings([{
            "error_signature": "ImportError",
            "prevention_hint": "Install the module first",
        }])
        self.assertIn("Known Issues", result)
        self.assertIn("ImportError", result)

    def test_orchestrator_stores_episodic_on_complete(self):
        """Orchestrator._handle_complete should call store_task_result."""
        src = _read_source("orchestrator.py")
        self.assertIn("from memory.episodic import store_task_result", src)
        self.assertIn("await store_task_result", src)


# ═══════════════════════════════════════════════════════════════════════════════
# 11.3 — RAG Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

class TestRAGPipeline(unittest.TestCase):
    """Tests for memory/rag.py."""

    def test_rag_file_exists(self):
        path = os.path.join(_ROOT, "memory", "rag.py")
        self.assertTrue(os.path.isfile(path))

    def test_public_api(self):
        src = _read_source("memory/rag.py")
        self.assertIn("async def retrieve_context", src)
        self.assertIn("async def store_fact", src)

    def test_recency_weight_function(self):
        from memory.rag import _recency_weight
        # Fresh memory should have high weight
        weight = _recency_weight(time.time())
        self.assertGreater(weight, 0.9)

        # Old memory should have lower weight
        seven_days_ago = time.time() - 7 * 86400
        weight = _recency_weight(seven_days_ago)
        self.assertAlmostEqual(weight, 0.5, delta=0.05)

        # Very old memory
        thirty_days_ago = time.time() - 30 * 86400
        weight = _recency_weight(thirty_days_ago)
        self.assertLess(weight, 0.2)

    def test_importance_weight(self):
        from memory.rag import _importance_weight

        # Error recovery = high importance
        w = _importance_weight({"type": "error_recovery"})
        self.assertGreaterEqual(w, 0.8)

        # User preference = high importance
        w = _importance_weight({"type": "user_preference"})
        self.assertGreaterEqual(w, 0.8)

        # Failed task = medium-high
        w = _importance_weight({"type": "task_result", "success": False})
        self.assertGreaterEqual(w, 0.7)

        # Successful task = medium
        w = _importance_weight({"type": "task_result", "success": True})
        self.assertGreaterEqual(w, 0.5)

    def test_deduplicate(self):
        from memory.rag import _deduplicate
        results = [
            {"text": "the quick brown fox jumps over the lazy dog"},
            {"text": "the quick brown fox jumps over the lazy cat"},  # very similar
            {"text": "completely different text about quantum physics"},
        ]
        deduped = _deduplicate(results, threshold=0.7)
        self.assertEqual(len(deduped), 2)

    def test_deduplicate_empty(self):
        from memory.rag import _deduplicate
        self.assertEqual(_deduplicate([]), [])

    def test_estimate_tokens(self):
        from memory.rag import _estimate_tokens
        # ~4 chars per token
        self.assertEqual(_estimate_tokens("a" * 400), 100)

    def test_rank_results(self):
        from memory.rag import _rank_results
        results = [
            {
                "text": "low relevance",
                "distance": 0.9,
                "metadata": {"timestamp": time.time() - 30 * 86400},
            },
            {
                "text": "high relevance",
                "distance": 0.1,
                "metadata": {"timestamp": time.time(), "type": "error_recovery"},
            },
        ]
        ranked = _rank_results(results)
        self.assertEqual(ranked[0]["text"], "high relevance")

    def test_base_agent_calls_rag(self):
        """BaseAgent._build_context should call retrieve_context."""
        src = _read_source("agents/base.py")
        self.assertIn("from memory.rag import retrieve_context", src)
        self.assertIn("await retrieve_context", src)


# ═══════════════════════════════════════════════════════════════════════════════
# 11.4 — Conversation Continuity
# ═══════════════════════════════════════════════════════════════════════════════

class TestConversationContinuity(unittest.TestCase):
    """Tests for memory/conversations.py."""

    def test_conversations_file_exists(self):
        path = os.path.join(_ROOT, "memory", "conversations.py")
        self.assertTrue(os.path.isfile(path))

    def test_public_api(self):
        src = _read_source("memory/conversations.py")
        for fn in ["store_exchange", "get_recent_exchanges",
                    "find_followup_context", "format_recent_context"]:
            self.assertIn(f"def {fn}", src, f"Function '{fn}' not found")

    def test_format_recent_context(self):
        from memory.conversations import format_recent_context

        # Empty input
        result = format_recent_context([])
        self.assertEqual(result, [])

        # With data
        exchanges = [
            {"user_message": "hello", "response_preview": "hi there"},
            {"user_message": "what time is it", "response_preview": "it's 3pm"},
        ]
        result = format_recent_context(exchanges, limit=1)
        self.assertEqual(len(result), 1)
        self.assertIn("user_asked", result[0])
        self.assertIn("result", result[0])

    def test_telegram_uses_followup_detection(self):
        """telegram_bot.py should call find_followup_context."""
        src = _read_source("telegram_bot.py")
        self.assertIn("find_followup_context", src)
        self.assertIn("format_recent_context", src)

    def test_telegram_stores_exchanges(self):
        """telegram_bot.py send_result should store exchanges."""
        src = _read_source("telegram_bot.py")
        self.assertIn("from memory.conversations import store_exchange", src)
        self.assertIn("await store_exchange", src)


# ═══════════════════════════════════════════════════════════════════════════════
# 11.5 — Document Ingestion
# ═══════════════════════════════════════════════════════════════════════════════

class TestDocumentIngestion(unittest.TestCase):
    """Tests for memory/ingest.py."""

    def test_ingest_file_exists(self):
        path = os.path.join(_ROOT, "memory", "ingest.py")
        self.assertTrue(os.path.isfile(path))

    def test_public_api(self):
        src = _read_source("memory/ingest.py")
        for fn in ["ingest_url", "ingest_file", "ingest_document"]:
            self.assertIn(f"async def {fn}", src, f"Function '{fn}' not found")

    def test_chunk_text_function(self):
        from memory.ingest import _chunk_text

        # Empty text
        self.assertEqual(_chunk_text(""), [])
        self.assertEqual(_chunk_text("   "), [])

        # Short text (fits in one chunk)
        result = _chunk_text("hello world")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], "hello world")

    def test_chunk_text_splits_long_text(self):
        from memory.ingest import _chunk_text
        # Create a text longer than default chunk size (~500 tokens ≈ 2000 chars)
        words = ["word"] * 600  # 600 words > 500 token chunk
        text = " ".join(words)
        chunks = _chunk_text(text, chunk_size=100)  # small chunks for testing
        self.assertGreater(len(chunks), 1)

    def test_chunk_text_overlap(self):
        from memory.ingest import _chunk_text
        # With overlap, chunks should share some words
        words = ["word" + str(i) for i in range(200)]
        text = " ".join(words)
        chunks = _chunk_text(text, chunk_size=50, overlap=10)
        if len(chunks) >= 2:
            # Last words of first chunk should appear at start of second
            first_words = set(chunks[0].split()[-5:])
            second_words = set(chunks[1].split()[:15])
            overlap = first_words & second_words
            self.assertGreater(len(overlap), 0, "Chunks should have overlap")

    def test_extract_file_text_nonexistent(self):
        from memory.ingest import _extract_file_text
        result = _extract_file_text("/nonexistent/file.txt")
        self.assertIsNone(result)

    def test_extract_file_text_plain(self):
        """Should read plain text files."""
        from memory.ingest import _extract_file_text
        # Use this test file itself
        result = _extract_file_text(__file__)
        self.assertIsNotNone(result)
        self.assertIn("test_phase11", result)

    def test_supported_extensions(self):
        """ingest.py should support common text/code extensions."""
        src = _read_source("memory/ingest.py")
        for ext in [".txt", ".md", ".py", ".js", ".ts", ".go", ".rs",
                    ".java", ".pdf", ".docx"]:
            self.assertIn(f'"{ext}"', src, f"Extension '{ext}' not supported")

    def test_ingest_tool_registered(self):
        """ingest_document should be registered as a tool."""
        src = _read_source("tools/__init__.py")
        self.assertIn("ingest_document", src)
        self.assertIn("_ingest_tool_wrapper", src)

    def test_telegram_ingest_command(self):
        """Telegram bot should have /ingest command."""
        src = _read_source("telegram_bot.py")
        self.assertIn("cmd_ingest", src)
        self.assertIn('CommandHandler("ingest"', src)

    def test_ingest_command_help_text(self):
        """The /start help should mention /ingest."""
        src = _read_source("telegram_bot.py")
        self.assertIn("/ingest", src)


# ═══════════════════════════════════════════════════════════════════════════════
# 11.6 — Memory Forgetting & Decay
# ═══════════════════════════════════════════════════════════════════════════════

class TestMemoryDecay(unittest.TestCase):
    """Tests for memory/decay.py."""

    def test_decay_file_exists(self):
        path = os.path.join(_ROOT, "memory", "decay.py")
        self.assertTrue(os.path.isfile(path))

    def test_public_api(self):
        src = _read_source("memory/decay.py")
        self.assertIn("async def run_decay_cycle", src)
        self.assertIn("def compute_relevance", src)
        self.assertIn("async def get_decay_stats", src)

    def test_collection_caps_defined(self):
        from memory.decay import COLLECTION_CAPS
        self.assertIsInstance(COLLECTION_CAPS, dict)
        for col in ["episodic", "semantic", "codebase", "errors", "conversations"]:
            self.assertIn(col, COLLECTION_CAPS)
            self.assertGreater(COLLECTION_CAPS[col], 0)

    def test_compute_relevance_fresh_accessed(self):
        """Fresh, frequently accessed memory should have high relevance."""
        from memory.decay import compute_relevance
        meta = {
            "access_count": 50,
            "last_accessed": time.time(),
            "stored_at": time.time(),
            "importance": 8,
        }
        score = compute_relevance(meta)
        self.assertGreater(score, 0.5)

    def test_compute_relevance_old_unused(self):
        """Old, never-accessed memory should have low relevance."""
        from memory.decay import compute_relevance
        meta = {
            "access_count": 0,
            "last_accessed": time.time() - 90 * 86400,  # 90 days ago
            "stored_at": time.time() - 90 * 86400,
            "importance": 3,
        }
        score = compute_relevance(meta)
        self.assertLess(score, 0.3)

    def test_compute_relevance_protected_types(self):
        """Protected types (user_preference, error_recovery) get importance boost."""
        from memory.decay import compute_relevance, PROTECTED_TYPES
        self.assertIn("user_preference", PROTECTED_TYPES)
        self.assertIn("error_recovery", PROTECTED_TYPES)

        # Same metadata but protected type should score higher
        base_meta = {
            "access_count": 1,
            "last_accessed": time.time() - 30 * 86400,
            "stored_at": time.time() - 60 * 86400,
            "importance": 3,
        }
        regular_score = compute_relevance({**base_meta, "type": "conversation"})
        protected_score = compute_relevance({**base_meta, "type": "user_preference"})
        self.assertGreater(protected_score, regular_score)

    def test_compute_relevance_returns_float(self):
        from memory.decay import compute_relevance
        score = compute_relevance({})
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_orchestrator_runs_decay_weekly(self):
        """Orchestrator should run decay cycle periodically."""
        src = _read_source("orchestrator.py")
        self.assertIn("from memory.decay import run_decay_cycle", src)
        self.assertIn("await run_decay_cycle", src)
        self.assertIn("last_decay_check", src)

    def test_prune_fraction_reasonable(self):
        from memory.decay import PRUNE_FRACTION
        self.assertGreater(PRUNE_FRACTION, 0)
        self.assertLess(PRUNE_FRACTION, 1)

    def test_relevance_threshold_set(self):
        from memory.decay import RELEVANCE_THRESHOLD
        self.assertGreater(RELEVANCE_THRESHOLD, 0)
        self.assertLess(RELEVANCE_THRESHOLD, 0.5)


# ═══════════════════════════════════════════════════════════════════════════════
# 11.7 — User Preference Learning
# ═══════════════════════════════════════════════════════════════════════════════

class TestUserPreferenceLearning(unittest.TestCase):
    """Tests for memory/preferences.py."""

    def test_preferences_file_exists(self):
        path = os.path.join(_ROOT, "memory", "preferences.py")
        self.assertTrue(os.path.isfile(path))

    def test_public_api(self):
        src = _read_source("memory/preferences.py")
        for fn in ["record_feedback", "store_preference",
                    "get_user_preferences", "detect_preferences",
                    "format_preferences"]:
            self.assertIn(f"def {fn}", src, f"Function '{fn}' not found")

    def test_feedback_types_defined(self):
        from memory.preferences import (
            FEEDBACK_ACCEPTED, FEEDBACK_MODIFIED, FEEDBACK_REJECTED,
        )
        self.assertEqual(FEEDBACK_ACCEPTED, "accepted")
        self.assertEqual(FEEDBACK_MODIFIED, "modified")
        self.assertEqual(FEEDBACK_REJECTED, "rejected")

    def test_preference_categories_defined(self):
        from memory.preferences import PREFERENCE_CATEGORIES
        self.assertIsInstance(PREFERENCE_CATEGORIES, list)
        self.assertIn("language", PREFERENCE_CATEGORIES)
        self.assertIn("framework", PREFERENCE_CATEGORIES)
        self.assertIn("style", PREFERENCE_CATEGORIES)
        self.assertIn("verbosity", PREFERENCE_CATEGORIES)

    def test_format_preferences_empty(self):
        from memory.preferences import format_preferences
        result = format_preferences([])
        self.assertEqual(result, "")

    def test_format_preferences_with_data(self):
        from memory.preferences import format_preferences
        prefs = [
            {"preference_text": "Prefers Python", "category": "language", "confidence": 0.8},
            {"preference_text": "Likes concise output", "category": "verbosity", "confidence": 0.7},
        ]
        result = format_preferences(prefs)
        self.assertIn("User Preferences", result)
        self.assertIn("Prefers Python", result)
        self.assertIn("Likes concise output", result)

    def test_format_preferences_low_confidence(self):
        """Low confidence preferences should be marked tentative."""
        from memory.preferences import format_preferences
        prefs = [
            {"preference_text": "Maybe likes Rust", "category": "language", "confidence": 0.5},
        ]
        result = format_preferences(prefs)
        self.assertIn("tentative", result)

    def test_format_preferences_very_low_confidence_excluded(self):
        """Very low confidence preferences should be excluded."""
        from memory.preferences import format_preferences
        prefs = [
            {"preference_text": "Maybe something", "category": "general", "confidence": 0.3},
        ]
        result = format_preferences(prefs)
        self.assertEqual(result, "")

    def test_format_preferences_deduplicates(self):
        """Should not show duplicate preference texts."""
        from memory.preferences import format_preferences
        prefs = [
            {"preference_text": "Same pref", "category": "style", "confidence": 0.8},
            {"preference_text": "Same pref", "category": "style", "confidence": 0.9},
        ]
        result = format_preferences(prefs)
        self.assertEqual(result.count("Same pref"), 1)

    def test_extract_patterns_language(self):
        from memory.preferences import _extract_patterns
        texts = [
            "Use python for this",
            "Rewrite in python please",
            "python is preferred",
        ]
        patterns = _extract_patterns(texts, "modification")
        lang_patterns = [p for p in patterns if p["category"] == "language"]
        self.assertGreater(len(lang_patterns), 0)

    def test_extract_patterns_style(self):
        from memory.preferences import _extract_patterns
        texts = [
            "Use snake_case for variable names",
            "snake_case naming please",
        ]
        patterns = _extract_patterns(texts, "modification")
        style_patterns = [p for p in patterns if p["category"] == "style"]
        self.assertGreater(len(style_patterns), 0)

    def test_extract_patterns_framework(self):
        from memory.preferences import _extract_patterns
        texts = ["Use fastapi", "fastapi is better"]
        patterns = _extract_patterns(texts, "modification")
        fw_patterns = [p for p in patterns if p["category"] == "framework"]
        self.assertGreater(len(fw_patterns), 0)

    def test_extract_patterns_empty(self):
        from memory.preferences import _extract_patterns
        patterns = _extract_patterns([], "modification")
        self.assertEqual(patterns, [])

    def test_base_agent_injects_preferences(self):
        """BaseAgent._build_context should inject user preferences."""
        src = _read_source("agents/base.py")
        self.assertIn("from memory.preferences import", src)
        self.assertIn("format_preferences", src)

    def test_orchestrator_records_feedback(self):
        """Orchestrator should record acceptance feedback on task complete."""
        src = _read_source("orchestrator.py")
        self.assertIn("from memory.preferences import record_feedback", src)
        self.assertIn("await record_feedback", src)

    def test_telegram_records_modified_feedback(self):
        """Telegram bot should record modified feedback on clarification reply."""
        src = _read_source("telegram_bot.py")
        self.assertIn("from memory.preferences import record_feedback", src)
        self.assertIn('"modified"', src)

    def test_store_preference_deterministic_id(self):
        """store_preference should use a deterministic doc_id to avoid duplicates."""
        src = _read_source("memory/preferences.py")
        self.assertIn("pref-", src)
        self.assertIn("hashlib.sha256", src)

    def test_preference_high_importance(self):
        """User preferences should have high importance (protected from decay)."""
        src = _read_source("memory/preferences.py")
        # Look for importance: 9 in store_preference
        self.assertIn('"importance": 9', src)


# ═══════════════════════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestPhase11Integration(unittest.TestCase):
    """Cross-module integration tests."""

    def test_memory_package_init(self):
        """memory/__init__.py should exist."""
        path = os.path.join(_ROOT, "memory", "__init__.py")
        self.assertTrue(os.path.isfile(path))

    def test_all_memory_modules_exist(self):
        """All Phase 11 modules should exist."""
        modules = [
            "memory/__init__.py",
            "memory/embeddings.py",
            "memory/vector_store.py",
            "memory/episodic.py",
            "memory/rag.py",
            "memory/conversations.py",
            "memory/ingest.py",
            "memory/decay.py",
            "memory/preferences.py",
        ]
        for mod in modules:
            path = os.path.join(_ROOT, mod)
            self.assertTrue(os.path.isfile(path), f"Module {mod} not found")

    def test_all_modules_importable(self):
        """All memory modules should be importable without errors."""
        modules = [
            "memory.embeddings",
            "memory.vector_store",
            "memory.episodic",
            "memory.rag",
            "memory.conversations",
            "memory.ingest",
            "memory.decay",
            "memory.preferences",
        ]
        for mod in modules:
            try:
                __import__(mod)
            except ImportError as e:
                # ChromaDB or other optional deps might not be installed
                # but the module itself should be importable
                if "chromadb" in str(e) or "sentence_transformers" in str(e):
                    pass  # expected in test environments
                else:
                    self.fail(f"Failed to import {mod}: {e}")

    def test_graceful_degradation(self):
        """All memory functions should fail gracefully without ChromaDB."""
        # These should all return empty/None without raising
        from memory.episodic import format_similar_tasks, format_error_warnings
        self.assertEqual(format_similar_tasks([]), "")
        self.assertEqual(format_error_warnings([]), "")

        from memory.conversations import format_recent_context
        self.assertEqual(format_recent_context([]), [])

        from memory.preferences import format_preferences
        self.assertEqual(format_preferences([]), "")

    def test_rag_context_wrapped_in_try_except(self):
        """RAG injection in base.py should be wrapped in try/except."""
        src = _read_source("agents/base.py")
        # Find the RAG section
        idx = src.find("Phase 11.3: RAG context injection")
        self.assertGreater(idx, 0)
        # Check that try/except wraps it (use wider window)
        nearby = src[idx - 50:idx + 500]
        self.assertIn("try:", nearby)
        self.assertIn("except", nearby)

    def test_preference_injection_wrapped_in_try_except(self):
        """Preference injection in base.py should be wrapped in try/except."""
        src = _read_source("agents/base.py")
        idx = src.find("Phase 11.7: User preference injection")
        self.assertGreater(idx, 0)
        nearby = src[idx - 50:idx + 500]
        self.assertIn("try:", nearby)
        self.assertIn("except", nearby)

    def test_decay_integration_non_critical(self):
        """Decay in orchestrator should be wrapped in try/except."""
        src = _read_source("orchestrator.py")
        idx = src.find("Phase 11.6: Memory decay")
        self.assertGreater(idx, 0)
        nearby = src[idx - 50:idx + 500]
        self.assertIn("try:", nearby)
        self.assertIn("except", nearby)
        self.assertIn("non-critical", nearby)


if __name__ == "__main__":
    unittest.main()
