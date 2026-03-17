# tools/download.py
"""
File download tool — download files from URLs into the workspace.
"""

import asyncio
import os

from src.infra.logging_config import get_logger
from .workspace import _safe_resolve, WORKSPACE_DIR

logger = get_logger("tools.download")

MAX_FILE_SIZE = 50_000_000  # 50 MB
ALLOWED_EXTENSIONS = {
    ".py", ".js", ".ts", ".json", ".yaml", ".yml", ".toml",
    ".txt", ".md", ".csv", ".xml", ".html", ".css",
    ".sh", ".bash", ".zsh", ".fish",
    ".go", ".rs", ".java", ".c", ".cpp", ".h",
    ".sql", ".graphql",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
    ".pdf", ".zip", ".tar", ".gz",
    ".env.example", ".gitignore", ".dockerignore",
    ".dockerfile", ".makefile",
}


async def download_file(
    url: str,
    save_as: str,
    timeout: int = 30,
) -> str:
    """
    Download a file from *url* and save it to the workspace.

    Args:
        url:     URL to download from.
        save_as: Filename/path relative to workspace root.
        timeout: Download timeout in seconds (max 120).

    Returns:
        Confirmation or error message.
    """
    if not url or not url.startswith(("http://", "https://")):
        return "❌ URL must start with http:// or https://"

    if not save_as:
        return "❌ save_as is required."

    full_path = _safe_resolve(save_as)
    if full_path is None:
        return "❌ Access denied: save_as path is outside workspace."

    # Ensure parent dir exists
    parent = os.path.dirname(full_path)
    os.makedirs(parent, exist_ok=True)

    timeout = min(timeout, 120)

    try:
        import httpx
        return await _download_httpx(url, full_path, save_as, timeout)
    except ImportError:
        return await _download_curl(url, full_path, save_as, timeout)


async def _download_httpx(
    url: str, full_path: str, save_as: str, timeout: int,
) -> str:
    """Download using httpx."""
    import httpx

    try:
        async with httpx.AsyncClient(
            timeout=timeout, follow_redirects=True,
        ) as client:
            response = await client.get(url)
            response.raise_for_status()

            content = response.content
            if len(content) > MAX_FILE_SIZE:
                return (
                    f"❌ File too large: {len(content)} bytes "
                    f"(max {MAX_FILE_SIZE // 1_000_000} MB)"
                )

            with open(full_path, "wb") as f:
                f.write(content)

            size = len(content)
            return (
                f"✅ Downloaded {save_as} ({_human_size(size)}) "
                f"from {url}"
            )

    except httpx.TimeoutException:
        return f"❌ Download timed out after {timeout}s"
    except httpx.HTTPStatusError as exc:
        return f"❌ HTTP {exc.response.status_code}: {exc}"
    except Exception as exc:
        return f"❌ Download failed: {exc}"


async def _download_curl(
    url: str, full_path: str, save_as: str, timeout: int,
) -> str:
    """Fallback download using curl."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "curl", "-sL", "-o", full_path,
            "-m", str(timeout),
            "--max-filesize", str(MAX_FILE_SIZE),
            url,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout + 5,
        )

        if proc.returncode != 0:
            err = stderr.decode("utf-8", errors="replace").strip()
            return f"❌ curl failed (code {proc.returncode}): {err}"

        if os.path.isfile(full_path):
            size = os.path.getsize(full_path)
            return (
                f"✅ Downloaded {save_as} ({_human_size(size)}) "
                f"from {url}"
            )
        return "❌ Download produced no file."

    except asyncio.TimeoutError:
        return f"❌ Download timed out after {timeout}s"
    except Exception as exc:
        return f"❌ Download failed: {exc}"


def _human_size(size_bytes: int) -> str:
    """Format byte count as human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes}B" if unit == "B" else f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}TB"
