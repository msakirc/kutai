# languages/javascript.py
from .base import LanguageToolkit

class JavaScriptToolkit(LanguageToolkit):
    name = "javascript"
    extensions = (".js", ".mjs", ".cjs")

    def lint_command(self, filepath: str) -> str:
        return f"eslint --fix {filepath} 2>&1 | head -20"

    def format_command(self, filepath: str) -> str:
        return f"prettier --write {filepath}"

    def test_command(self, project_root: str) -> str:
        return "npm test 2>&1 | tail -30"

    def typecheck_command(self, project_root: str) -> str:
        return "echo 'No type checking for plain JavaScript'"

    def install_deps_command(self, project_root: str) -> str:
        return "npm install 2>&1 | tail -5"

    def get_prompt_hints(self) -> str:
        return (
            "## JavaScript Guidelines\n"
            "- Run tests with: `npm test`\n"
            "- Lint with: `eslint --fix .`\n"
            "- Format with: `prettier --write .`\n"
            "- Use ES6+ syntax; prefer const/let over var\n"
            "- Use async/await over callbacks\n"
        )
