"""Tests for workflow dispatch logic."""

import pytest

from src.workflows.engine.dispatch import (
    should_start_workflow,
    detect_onboarding_path,
    extract_idea_text,
)


class TestWorkflowKeywordsClassify:
    """Positive matches for product-building requests."""

    @pytest.mark.parametrize(
        "message",
        [
            "build me a product",
            "create a saas platform",
            "make me an app",
            "develop a tool for tracking tasks",
            "build a task management app",
            "idea for a project management app",
            "idea to build a saas product",
            "idea about a fitness platform",
            "idea.to.product",
            "full product development",
            "from scratch build an app",
            "mvp for build quickly",
            "launch a product next month",
            "launch my startup idea",
            "create me a website for my business",
            "develop me a service",
            "/product Build me a task management app",
        ],
    )
    def test_workflow_keywords_classify(self, message: str) -> None:
        assert should_start_workflow(message) is True


class TestNonWorkflowKeywords:
    """Negative matches -- these should NOT trigger a workflow."""

    @pytest.mark.parametrize(
        "message",
        [
            "fix the login bug",
            "update the README",
            "refactor the database module",
            "what is the weather today",
            "hello",
            "deploy to production",
            "run the tests",
            "add a button to the settings page",
        ],
    )
    def test_non_workflow_keywords(self, message: str) -> None:
        assert should_start_workflow(message) is False


class TestDetectOnboarding:
    """Extract onboarding path from messages."""

    def test_detect_onboarding(self) -> None:
        result = detect_onboarding_path("onboard /home/user/myrepo")
        assert result == "/home/user/myrepo"

    def test_detect_onboarding_with_product_prefix(self) -> None:
        result = detect_onboarding_path("/product onboard /tmp/project")
        assert result == "/tmp/project"

    def test_detect_onboarding_none(self) -> None:
        result = detect_onboarding_path("build me a task management app")
        assert result is None

    def test_detect_onboarding_none_empty(self) -> None:
        result = detect_onboarding_path("hello world")
        assert result is None


class TestExtractIdeaText:
    """Extract idea text, stripping /product prefix if present."""

    def test_extract_idea_text(self) -> None:
        result = extract_idea_text("/product Build me a task management app")
        assert result == "Build me a task management app"

    def test_extract_idea_text_no_prefix(self) -> None:
        result = extract_idea_text("Build me a task management app")
        assert result == "Build me a task management app"

    def test_extract_idea_text_strips_whitespace(self) -> None:
        result = extract_idea_text("/product   lots of spaces  ")
        assert result == "lots of spaces"


class TestCaseInsensitive:
    """Workflow detection should be case insensitive."""

    def test_case_insensitive(self) -> None:
        assert should_start_workflow("BUILD me an APP") is True

    def test_case_insensitive_mixed(self) -> None:
        assert should_start_workflow("Create A SaaS Platform") is True

    def test_case_insensitive_idea(self) -> None:
        assert should_start_workflow("IDEA FOR a fitness App") is True
