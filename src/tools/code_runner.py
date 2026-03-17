# tools/code_runner.py
"""
Run code snippets inside the Docker sandbox — never on the host.
"""
from src.infra.logging_config import get_logger
from src.tools import run_shell_with_stdin, run_shell

logger = get_logger("tools.code_runner")


async def run_code(code: str, language: str = "python", timeout: int = 30) -> str:
    """Write code to a temp file inside the sandbox and execute it."""
    if language != "python":
        return f"Only Python execution is supported, got: {language}"

    # Write code into the container via stdin (avoids all escaping issues)
    write_result = await run_shell_with_stdin(
        "cat > /tmp/_run_code.py", code, timeout=10,
    )
    if write_result.startswith("❌"):
        return f"Failed to write code to sandbox: {write_result}"

    # Execute inside sandbox
    logger.debug("executing python code", timeout=timeout)
    result = await run_shell("python3 /tmp/_run_code.py", timeout=timeout)
    # Extract exit code from result
    if result.startswith("✅"):
        logger.info("python code executed successfully", timeout=timeout)
    elif result.startswith("❌"):
        logger.warning("python code execution failed", timeout=timeout)
    return result
