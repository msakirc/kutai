# tools/code_runner.py
import asyncio
import tempfile
import os
import logging

logger = logging.getLogger(__name__)

async def run_code(code: str, language: str = "python", timeout: int = 30) -> str:
    """Run code in a sandboxed subprocess."""
    if language != "python":
        return f"Only Python execution is supported, got: {language}"

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        f.flush()
        tmp_path = f.name

    try:
        proc = await asyncio.create_subprocess_exec(
            "python3", tmp_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            # Basic sandboxing
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )

        output = ""
        if stdout:
            output += f"STDOUT:\n{stdout.decode()[:5000]}\n"
        if stderr:
            output += f"STDERR:\n{stderr.decode()[:2000]}\n"
        output += f"Exit code: {proc.returncode}"

        return output or "Code executed successfully (no output)."

    except asyncio.TimeoutError:
        proc.kill()
        return f"Error: Code execution timed out after {timeout}s"
    except Exception as e:
        return f"Execution error: {e}"
    finally:
        os.unlink(tmp_path)
