"""Minimal stdio JSON-RPC 2.0 server emulating an MCP server for tests.

Supports two methods:
  * ``tools/list`` -> returns two fake tools
  * ``tools/call``  -> echoes the call arguments

Reads one JSON object per line from stdin, writes one JSON object per line to
stdout. Exits on EOF. If argv contains ``--unhealthy`` it never answers
``tools/list`` (to exercise the health-probe failure path).
"""
import json
import sys

_TOOLS = [
    {"name": "echo", "description": "Echo back the given text",
     "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}}},
    {"name": "add", "description": "Add two integers a and b",
     "inputSchema": {"type": "object",
                     "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}}},
]


def main() -> None:
    unhealthy = "--unhealthy" in sys.argv
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        method = req.get("method")
        req_id = req.get("id")
        if method == "tools/list":
            if unhealthy:
                continue  # never reply -> probe times out
            resp = {"jsonrpc": "2.0", "id": req_id, "result": {"tools": _TOOLS}}
        elif method == "tools/call":
            params = req.get("params") or {}
            resp = {"jsonrpc": "2.0", "id": req_id,
                    "result": {"content": [{"type": "text",
                                            "text": json.dumps(params)}]}}
        else:
            resp = {"jsonrpc": "2.0", "id": req_id,
                    "error": {"code": -32601, "message": "method not found"}}
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
