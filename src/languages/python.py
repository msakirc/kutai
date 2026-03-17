# languages/python.py
from .base import LanguageToolkit

class PythonToolkit(LanguageToolkit):
    name = "python"
    extensions = (".py",)

    def lint_command(self, filepath: str) -> str:
        return f"ruff check --fix {filepath} && ruff format {filepath}"

    def format_command(self, filepath: str) -> str:
        return f"ruff format {filepath}"

    def test_command(self, project_root: str) -> str:
        return "pytest -x --tb=short 2>&1 | head -50"

    def typecheck_command(self, project_root: str) -> str:
        return "mypy . --ignore-missing-imports --no-error-summary 2>&1 | head -30"

    def install_deps_command(self, project_root: str) -> str:
        return "pip install -r requirements.txt 2>&1 | tail -5"

    def get_prompt_hints(self) -> str:
        return (
            "## Python Guidelines\n"
            "- Run tests with: `pytest -x --tb=short`\n"
            "- Lint with: `ruff check --fix . && ruff format .`\n"
            "- Type check with: `mypy . --ignore-missing-imports`\n"
            "- Follow PEP 8; use type hints; prefer pathlib over os.path\n"
            "- Use `async/await` for I/O; avoid blocking calls in async code\n"
        )
