import pytest
from src.memory.rag import RAG_CONFIG, get_rag_collections


class TestRAGConfig:
    def test_relevance_threshold_raised(self):
        assert RAG_CONFIG["min_relevance"] >= 0.72

    def test_top_k_reduced(self):
        assert RAG_CONFIG["top_k_per_collection"] <= 2


class TestRAGCollectionGating:
    def test_coder_gets_errors_and_codebase(self):
        collections = get_rag_collections("coder")
        assert "errors" in collections
        assert "codebase" in collections
        assert "shopping" not in collections

    def test_shopping_advisor_gets_shopping(self):
        collections = get_rag_collections("shopping_advisor")
        assert "shopping" in collections
        assert "errors" not in collections

    def test_unknown_gets_default(self):
        collections = get_rag_collections("nonexistent")
        assert "episodic" in collections
        assert "semantic" in collections

    def test_assistant_gets_semantic_and_conversations(self):
        collections = get_rag_collections("assistant")
        assert "semantic" in collections
        assert "conversations" in collections


class TestHyDERemoved:
    def test_hyde_disabled_or_removed(self):
        """Fake HyDE must not exist — raw queries are better."""
        import src.memory.rag as rag_mod
        assert not getattr(rag_mod, "HYDE_ENABLED", False), "HYDE_ENABLED should be False or removed"
        assert not hasattr(rag_mod, "_hyde_expand"), "_hyde_expand should be removed"


class TestRerankerConfig:
    def test_reranker_enabled(self):
        from src.memory.rag import RERANKER_ENABLED
        assert RERANKER_ENABLED is True

    def test_reranker_skips_small_result_sets(self):
        """Reranking <3 results has no value."""
        import asyncio
        from src.memory.rag import _rerank_results
        results = [
            {"text": "result 1", "id": "1"},
            {"text": "result 2", "id": "2"},
        ]
        out = asyncio.get_event_loop().run_until_complete(
            _rerank_results("test query", results)
        )
        assert out == results


class TestSkillThreshold:
    def test_match_threshold_raised(self):
        from src.memory.skills import MATCH_SIMILARITY_THRESHOLD
        assert MATCH_SIMILARITY_THRESHOLD >= 0.75
