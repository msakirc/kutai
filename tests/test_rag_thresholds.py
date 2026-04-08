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


class TestSkillThreshold:
    def test_match_threshold_raised(self):
        from src.memory.skills import MATCH_SIMILARITY_THRESHOLD
        assert MATCH_SIMILARITY_THRESHOLD >= 0.75
