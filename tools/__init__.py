# tools/__init__.py
from tools.web_search import search_web
from tools.code_runner import run_code
from tools.file_ops import read_file, write_file, list_files

TOOL_REGISTRY = {
    "web_search": search_web,
    "code_runner": run_code,
    "read_file": read_file,
    "write_file": write_file,
    "list_files": list_files,
}

async def execute_tool(tool_name: str, **kwargs) -> str:
    if tool_name not in TOOL_REGISTRY:
        return f"Error: Unknown tool '{tool_name}'"
    try:
        result = await TOOL_REGISTRY[tool_name](**kwargs)
        return result
    except Exception as e:
        return f"Tool error ({tool_name}): {e}"
