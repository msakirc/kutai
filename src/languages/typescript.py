# languages/typescript.py
from .javascript import JavaScriptToolkit

class TypeScriptToolkit(JavaScriptToolkit):
    name = "typescript"
    extensions = (".ts", ".tsx")

    def lint_command(self, filepath: str) -> str:
        return f"eslint --fix {filepath} 2>&1 | head -20"

    def typecheck_command(self, project_root: str) -> str:
        return "npx tsc --noEmit 2>&1 | head -30"

    def get_prompt_hints(self) -> str:
        return (
            "## TypeScript Guidelines\n"
            "- Run tests with: `npm test`\n"
            "- Type check with: `npx tsc --noEmit`\n"
            "- Lint with: `eslint --fix .`\n"
            "- Use strict TypeScript; avoid `any`; define interfaces\n"
            "- Prefer explicit return types on exported functions\n"
        )
