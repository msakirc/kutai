# tools/file_ops.py
import os
import aiofiles

WORKSPACE = os.path.expanduser("~/ai-orchestrator/workspace")
os.makedirs(WORKSPACE, exist_ok=True)

def _safe_path(filename: str) -> str:
    """Prevent path traversal."""
    safe = os.path.normpath(filename).lstrip(os.sep)
    full = os.path.join(WORKSPACE, safe)
    if not full.startswith(WORKSPACE):
        raise ValueError("Path traversal detected")
    return full

async def read_file(filename: str) -> str:
    path = _safe_path(filename)
    if not os.path.exists(path):
        return f"File not found: {filename}"
    async with aiofiles.open(path, 'r') as f:
        content = await f.read()
    return content[:10000]

async def write_file(filename: str, content: str) -> str:
    path = _safe_path(filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    async with aiofiles.open(path, 'w') as f:
        await f.write(content)
    return f"Written {len(content)} chars to {filename}"

async def list_files(directory: str = "") -> str:
    path = _safe_path(directory) if directory else WORKSPACE
    if not os.path.isdir(path):
        return f"Directory not found: {directory}"
    files = []
    for root, dirs, filenames in os.walk(path):
        for fn in filenames:
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, WORKSPACE)
            size = os.path.getsize(full)
            files.append(f"  {rel} ({size} bytes)")
    return "\n".join(files) if files else "Workspace is empty."
