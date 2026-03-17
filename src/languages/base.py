# languages/base.py
"""Abstract base class for language toolkits."""
from __future__ import annotations
from abc import ABC, abstractmethod


class LanguageToolkit(ABC):
    """Abstract language toolkit interface."""

    name: str = "unknown"
    extensions: tuple[str, ...] = ()

    @abstractmethod
    def lint_command(self, filepath: str) -> str:
        """Return shell command to lint a file."""

    @abstractmethod
    def format_command(self, filepath: str) -> str:
        """Return shell command to format a file."""

    @abstractmethod
    def test_command(self, project_root: str) -> str:
        """Return shell command to run tests."""

    @abstractmethod
    def typecheck_command(self, project_root: str) -> str:
        """Return shell command to run type checking."""

    @abstractmethod
    def install_deps_command(self, project_root: str) -> str:
        """Return shell command to install dependencies."""

    def get_prompt_hints(self) -> str:
        """Return language-specific hints for agent system prompts."""
        return ""
