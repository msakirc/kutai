# tools/code_runner.py
"""
Run code snippets inside the Docker sandbox — never on the host.
"""
import logging

logger = logging.getLogger(__name__)


async def run_code(code: str, language: str = "python", timeout: int = 30) -> str:
    """Write code to a temp file inside the sandbox and execute it."""
    if language != "python":
        return f"Only Python execution is supported, got: {language}"

    from shell import run_shell_with_stdin, run_shell

    # Write code into the container via stdin (avoids all escaping issues)
    write_result = await run_shell_with_stdin(
        "cat > /tmp/_run_code.py", code, timeout=10,
    )
    if write_result.startswith("❌"):
        return f"Failed to write code to sandbox: {write_result}"

    # Execute inside sandbox
    return await run_shell("python3 /tmp/_run_code.py", timeout=timeout)
