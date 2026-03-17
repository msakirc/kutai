# languages/rust.py
from .base import LanguageToolkit

class RustToolkit(LanguageToolkit):
    name = "rust"
    extensions = (".rs",)

    def lint_command(self, filepath: str) -> str:
        return "cargo clippy 2>&1 | head -30"

    def format_command(self, filepath: str) -> str:
        return "cargo fmt"

    def test_command(self, project_root: str) -> str:
        return "cargo test 2>&1 | tail -20"

    def typecheck_command(self, project_root: str) -> str:
        return "cargo check 2>&1 | head -20"

    def install_deps_command(self, project_root: str) -> str:
        return "cargo build 2>&1 | tail -10"

    def get_prompt_hints(self) -> str:
        return (
            "## Rust Guidelines\n"
            "- Run tests with: `cargo test`\n"
            "- Lint with: `cargo clippy`\n"
            "- Format with: `cargo fmt`\n"
            "- Use Result/Option properly; avoid unwrap() in production\n"
            "- Prefer ownership over cloning when possible\n"
        )
