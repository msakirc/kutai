# tools/mcp_client.py
"""
MCP (Model Context Protocol) client — connect to external MCP servers
and dynamically register their tools into the local TOOL_REGISTRY.

Supports two transports:
  - **stdio**: spawn a subprocess, exchange JSON-RPC over stdin/stdout
  - **SSE**: connect to an HTTP SSE endpoint

No third-party MCP SDK required — we implement the JSON-RPC 2.0
protocol directly using asyncio subprocesses and aiohttp.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from functools import partial
from pathlib import Path
from typing import Any, Optional

from ..infra.logging_config import get_logger

logger = get_logger("tools.mcp_client")

# JSON-RPC 2.0 helpers -------------------------------------------------------

_next_id = 0


def _make_request(method: str, params: dict | None = None) -> bytes:
    """Build a JSON-RPC 2.0 request and return it as newline-terminated bytes."""
    global _next_id
    _next_id += 1
    msg: dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": _next_id,
        "method": method,
    }
    if params is not None:
        msg["params"] = params
    payload = json.dumps(msg)
    return payload.encode("utf-8") + b"\n"


def _parse_response(data: str) -> dict:
    """Parse a JSON-RPC 2.0 response, raising on protocol errors."""
    resp = json.loads(data)
    if "error" in resp:
        err = resp["error"]
        code = err.get("code", -1)
        message = err.get("message", "unknown")
        raise MCPError(f"JSON-RPC error {code}: {message}")
    return resp.get("result", {})


# Exceptions ------------------------------------------------------------------

class MCPError(Exception):
    """Any error originating from MCP protocol communication."""


class MCPConnectionError(MCPError):
    """Failed to establish or maintain a connection."""


# Connection wrappers ---------------------------------------------------------

class _StdioConnection:
    """Wraps an asyncio subprocess for MCP stdio transport."""

    def __init__(self, process: asyncio.subprocess.Process):
        self.process = process
        self._lock = asyncio.Lock()

    async def send(self, method: str, params: dict | None = None) -> dict:
        """Send a JSON-RPC request and wait for the response."""
        assert self.process.stdin is not None
        assert self.process.stdout is not None

        async with self._lock:
            request = _make_request(method, params)
            self.process.stdin.write(request)
            await self.process.stdin.drain()

            # Read response — MCP stdio uses newline-delimited JSON
            line = await asyncio.wait_for(
                self.process.stdout.readline(), timeout=30
            )
            if not line:
                raise MCPConnectionError("Server closed stdout unexpectedly")
            return _parse_response(line.decode("utf-8").strip())

    async def close(self):
        """Terminate the subprocess."""
        try:
            if self.process.stdin:
                self.process.stdin.close()
            self.process.terminate()
            await asyncio.wait_for(self.process.wait(), timeout=5)
        except (ProcessLookupError, asyncio.TimeoutError):
            self.process.kill()

    @property
    def is_alive(self) -> bool:
        return self.process.returncode is None


class _SSEConnection:
    """Wraps an aiohttp session for MCP SSE transport."""

    def __init__(self, url: str, headers: dict | None = None):
        self.url = url.rstrip("/")
        self.headers = headers or {}
        self._session: Any = None  # aiohttp.ClientSession
        self._lock = asyncio.Lock()

    async def connect(self):
        """Open the HTTP session."""
        try:
            import aiohttp
        except ImportError:
            raise MCPConnectionError(
                "aiohttp is required for SSE transport: pip install aiohttp"
            )
        self._session = aiohttp.ClientSession(headers=self.headers)

    async def send(self, method: str, params: dict | None = None) -> dict:
        """POST a JSON-RPC request to the server's /message endpoint."""
        if self._session is None:
            await self.connect()

        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": _next_id + 1,
            "method": method,
        }
        if params is not None:
            payload["params"] = params

        async with self._lock:
            async with self._session.post(
                f"{self.url}/message",
                json=payload,
                timeout=30,
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return _parse_response(json.dumps(data))

    async def close(self):
        """Close the HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None

    @property
    def is_alive(self) -> bool:
        return self._session is not None and not self._session.closed


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------

class MCPClient:
    """Connect to external MCP servers and register their tools dynamically."""

    def __init__(self):
        self.connections: dict[str, Any] = {}      # server_name -> connection
        self._tools: dict[str, dict] = {}          # tool_name -> {server, schema}

    # -- Connection management ------------------------------------------------

    async def connect(
        self,
        server_name: str,
        command: list[str],
        env: dict | None = None,
    ):
        """Connect to an MCP server via stdio transport.

        Args:
            server_name: Unique name for this connection.
            command:     Command to run, e.g.
                         ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/path"]
            env:         Optional environment variables for the subprocess.
        """
        if server_name in self.connections:
            logger.warning(f"Already connected to '{server_name}', disconnecting first")
            await self.disconnect(server_name)

        # Merge caller env with current env
        merged_env = {**os.environ, **(env or {})}

        logger.info(f"Connecting to MCP server '{server_name}' via stdio: {command}")

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=merged_env,
            )
        except FileNotFoundError:
            raise MCPConnectionError(
                f"Command not found: {command[0]}. "
                "Ensure it is installed and on PATH."
            )

        conn = _StdioConnection(process)
        self.connections[server_name] = conn

        # MCP handshake — send initialize
        try:
            result = await conn.send("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "kutay", "version": "1.0.0"},
            })
            logger.info(
                f"MCP server '{server_name}' initialized: "
                f"{result.get('serverInfo', {}).get('name', 'unknown')}"
            )

            # Notify initialized
            await conn.send("notifications/initialized")
        except Exception as exc:
            await conn.close()
            del self.connections[server_name]
            raise MCPConnectionError(
                f"Failed to initialize MCP server '{server_name}': {exc}"
            ) from exc

        # Discover tools
        await self._discover_tools(server_name)

    async def connect_sse(
        self,
        server_name: str,
        url: str,
        headers: dict | None = None,
    ):
        """Connect to an MCP server via SSE transport.

        Args:
            server_name: Unique name for this connection.
            url:         Base URL of the MCP SSE server.
            headers:     Optional HTTP headers (e.g. Authorization).
        """
        if server_name in self.connections:
            logger.warning(f"Already connected to '{server_name}', disconnecting first")
            await self.disconnect(server_name)

        logger.info(f"Connecting to MCP server '{server_name}' via SSE: {url}")

        conn = _SSEConnection(url, headers)
        await conn.connect()
        self.connections[server_name] = conn

        try:
            result = await conn.send("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "kutay", "version": "1.0.0"},
            })
            logger.info(
                f"MCP SSE server '{server_name}' initialized: "
                f"{result.get('serverInfo', {}).get('name', 'unknown')}"
            )
            await conn.send("notifications/initialized")
        except Exception as exc:
            await conn.close()
            raise MCPConnectionError(
                f"Failed to initialize MCP SSE server '{server_name}': {exc}"
            ) from exc

        self.connections[server_name] = conn
        await self._discover_tools(server_name)

    async def disconnect(self, server_name: str):
        """Disconnect from an MCP server and remove its tools."""
        conn = self.connections.pop(server_name, None)
        if conn is None:
            logger.warning(f"No connection named '{server_name}' to disconnect")
            return

        # Remove tools belonging to this server
        to_remove = [
            name for name, info in self._tools.items()
            if info["server"] == server_name
        ]
        for name in to_remove:
            del self._tools[name]
            logger.debug(f"Removed MCP tool: {name}")

        await conn.close()
        logger.info(f"Disconnected from MCP server '{server_name}'")

    async def disconnect_all(self):
        """Disconnect from all MCP servers."""
        names = list(self.connections.keys())
        for name in names:
            await self.disconnect(name)

    # -- Tool discovery -------------------------------------------------------

    async def _discover_tools(self, server_name: str):
        """Query tools/list on a connected server and cache the results."""
        conn = self.connections.get(server_name)
        if conn is None:
            return

        try:
            result = await conn.send("tools/list")
        except Exception as exc:
            logger.error(f"Failed to list tools from '{server_name}': {exc}")
            return

        tools = result.get("tools", [])
        for tool in tools:
            tool_name = tool.get("name", "")
            if not tool_name:
                continue
            qualified = f"mcp_{server_name}_{tool_name}"
            self._tools[qualified] = {
                "server": server_name,
                "original_name": tool_name,
                "description": tool.get("description", ""),
                "inputSchema": tool.get("inputSchema", {}),
            }
            logger.debug(f"Discovered MCP tool: {qualified}")

        logger.info(
            f"Discovered {len(tools)} tools from MCP server '{server_name}'"
        )

    async def list_tools(self, server_name: str | None = None) -> list[dict]:
        """List tools from one or all connected servers.

        Args:
            server_name: If provided, list only tools from this server.

        Returns:
            List of dicts with keys: name, server, description, inputSchema.
        """
        results = []
        for name, info in self._tools.items():
            if server_name and info["server"] != server_name:
                continue
            results.append({
                "name": name,
                "server": info["server"],
                "original_name": info["original_name"],
                "description": info["description"],
                "inputSchema": info["inputSchema"],
            })
        return results

    # -- Tool execution -------------------------------------------------------

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        args: dict,
    ) -> str:
        """Call a tool on a connected MCP server.

        Args:
            server_name: Which server to call.
            tool_name:   The tool's original name (not the prefixed version).
            args:        Tool arguments as a dict.

        Returns:
            Tool result as a string.
        """
        conn = self.connections.get(server_name)
        if conn is None:
            raise MCPConnectionError(f"Not connected to server '{server_name}'")

        logger.debug(f"Calling MCP tool '{tool_name}' on '{server_name}'")

        result = await conn.send("tools/call", {
            "name": tool_name,
            "arguments": args,
        })

        # MCP tools return content array — extract text
        content = result.get("content", [])
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                else:
                    text_parts.append(json.dumps(item))
            else:
                text_parts.append(str(item))

        return "\n".join(text_parts) if text_parts else json.dumps(result)

    # -- TOOL_REGISTRY integration --------------------------------------------

    def register_all_tools(self):
        """Register all discovered MCP tools into the project TOOL_REGISTRY.

        Each MCP tool is wrapped in an async callable that delegates to
        ``call_tool()``, then inserted into ``TOOL_REGISTRY`` with the
        standard ``{function, description, example}`` shape.
        """
        from . import TOOL_REGISTRY

        for qualified_name, info in self._tools.items():
            server = info["server"]
            original = info["original_name"]
            schema = info.get("inputSchema", {})

            # Build the wrapper — capture server/original via partial
            wrapper = partial(self._mcp_tool_wrapper, server, original)

            # Extract arg names from JSON Schema
            properties = schema.get("properties", {})
            arg_names = list(properties.keys())
            args_str = ", ".join(f"{a} (str)" for a in arg_names) if arg_names else "(none)"

            TOOL_REGISTRY[qualified_name] = {
                "function": wrapper,
                "description": (
                    f"[MCP/{server}] {info['description']}  "
                    f"Args: {args_str}"
                ),
                "example": json.dumps({
                    "action": "tool_call",
                    "tool": qualified_name,
                    "args": {a: f"<{a}>" for a in arg_names},
                }),
            }
            logger.debug(f"Registered MCP tool in TOOL_REGISTRY: {qualified_name}")

        logger.info(
            f"Registered {len(self._tools)} MCP tools into TOOL_REGISTRY"
        )

    def unregister_all_tools(self):
        """Remove all MCP tools from TOOL_REGISTRY."""
        from . import TOOL_REGISTRY

        for qualified_name in self._tools:
            TOOL_REGISTRY.pop(qualified_name, None)

        logger.info("Removed all MCP tools from TOOL_REGISTRY")

    async def _mcp_tool_wrapper(self, server_name: str, tool_name: str, **kwargs) -> str:
        """Async wrapper used as the callable inside TOOL_REGISTRY entries."""
        return await self.call_tool(server_name, tool_name, kwargs)


# ---------------------------------------------------------------------------
# Configuration loader
# ---------------------------------------------------------------------------

def _expand_env_vars(value: str) -> str:
    """Expand ${VAR_NAME} references in a string."""
    def _replace(match):
        var = match.group(1)
        return os.environ.get(var, "")
    return re.sub(r"\$\{(\w+)\}", _replace, value)


def load_mcp_config(config_path: str | None = None) -> dict:
    """Load MCP server definitions from a YAML config file.

    Looks in order:
      1. Explicit ``config_path``
      2. ``mcp.yaml`` in the project root
      3. ``mcp_servers`` key inside ``models.yaml``

    Returns:
        Dict of server_name -> config (command, env, auto_connect, etc.)
    """
    import yaml  # project already depends on pyyaml

    candidates: list[Path] = []
    if config_path:
        candidates.append(Path(config_path))
    else:
        project_root = Path(__file__).resolve().parent.parent.parent
        candidates.append(project_root / "mcp.yaml")
        candidates.append(project_root / "src" / "models" / "models.yaml")

    for path in candidates:
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        servers = data.get("mcp_servers", {})
        if servers:
            # Expand env vars in env blocks
            for name, cfg in servers.items():
                if "env" in cfg:
                    cfg["env"] = {
                        k: _expand_env_vars(str(v))
                        for k, v in cfg["env"].items()
                    }
            logger.info(f"Loaded MCP config from {path}: {list(servers.keys())}")
            return servers

    logger.debug("No MCP server configuration found")
    return {}


async def auto_connect_servers(client: MCPClient, config: dict | None = None):
    """Connect to all MCP servers that have ``auto_connect: true``.

    Args:
        client: An MCPClient instance.
        config: Server config dict; loaded from YAML if not provided.
    """
    if config is None:
        config = load_mcp_config()

    for name, cfg in config.items():
        if not cfg.get("auto_connect", False):
            continue

        try:
            if "url" in cfg:
                await client.connect_sse(
                    server_name=name,
                    url=cfg["url"],
                    headers=cfg.get("headers"),
                )
            elif "command" in cfg:
                await client.connect(
                    server_name=name,
                    command=cfg["command"],
                    env=cfg.get("env"),
                )
            else:
                logger.warning(
                    f"MCP server '{name}' has no 'command' or 'url' — skipping"
                )
        except MCPError as exc:
            logger.error(f"Failed to auto-connect MCP server '{name}': {exc}")


# Module-level singleton
mcp_client = MCPClient()
