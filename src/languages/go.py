# languages/go.py
from .base import LanguageToolkit

class GoToolkit(LanguageToolkit):
    name = "go"
    extensions = (".go",)

    def lint_command(self, filepath: str) -> str:
        return f"gofmt -w {filepath} && go vet ./..."

    def format_command(self, filepath: str) -> str:
        return f"gofmt -w {filepath}"

    def test_command(self, project_root: str) -> str:
        return "go test ./... 2>&1 | tail -20"

    def typecheck_command(self, project_root: str) -> str:
        return "go vet ./... 2>&1 | head -20"

    def install_deps_command(self, project_root: str) -> str:
        return "go mod download 2>&1 | tail -5"

    def get_prompt_hints(self) -> str:
        return (
            "## Go Guidelines\n"
            "- Run tests with: `go test ./...`\n"
            "- Format with: `gofmt -w .`\n"
            "- Vet with: `go vet ./...`\n"
            "- Handle errors explicitly; avoid panic in production code\n"
            "- Use context.Context for cancellation\n"
        )
