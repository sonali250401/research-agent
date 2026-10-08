"""
Task 4: MCP Client
==================
Launched by the agent on startup.  It:
1. Spawns `mcp_server.py` as a subprocess over stdio.
2. Sends `initialize` + `tools/list` to discover tool schemas.
3. Exposes `call_tool()` so any graph node can invoke an MCP tool.
"""

import json
import subprocess
import sys
import threading
import logging
from pathlib import Path

log = logging.getLogger("mcp_client")


class MCPClient:
    """Minimal stdio MCP client."""

    def __init__(self, server_script: str = "mcp_server.py"):
        self._req_id = 0
        self._lock = threading.Lock()
        script_path = Path(__file__).parent / server_script

        self._proc = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        log.info(f"MCP Server PID: {self._proc.pid}")

        # Initialize the MCP session
        self._send({"method": "initialize", "params": {"protocolVersion": "2024-11-05"}})
        self.tool_schemas: list[dict] = []

    def _next_id(self) -> int:
        with self._lock:
            self._req_id += 1
            return self._req_id

    def _send(self, payload: dict) -> dict:
        req_id = self._next_id()
        payload["jsonrpc"] = "2.0"
        payload["id"] = req_id

        line = json.dumps(payload) + "\n"
        self._proc.stdin.write(line)
        self._proc.stdin.flush()

        while True:
            raw = self._proc.stdout.readline()
            if not raw:
                raise RuntimeError("MCP server closed stdout unexpectedly.")
            raw = raw.strip()
            if not raw:
                continue
            try:
                data = json.loads(raw)
                if isinstance(data, dict) and "jsonrpc" in data:
                    return data
            except json.JSONDecodeError:
                log.debug(f"[mcp_client] Skipping non-JSON line: {raw}")
                continue

    def list_tools(self) -> list[dict]:
        """Discover tools from the MCP server and cache their schemas."""
        resp = self._send({"method": "tools/list", "params": {}})
        mcp_tools = resp.get("result", {}).get("tools", [])

        # Convert MCP schema → OpenAI-style function schema for Groq
        self.tool_schemas = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["inputSchema"],
                },
            }
            for t in mcp_tools
        ]
        log.info(f"Discovered {len(self.tool_schemas)} tools via MCP: "
                 f"{[t['function']['name'] for t in self.tool_schemas]}")
        return self.tool_schemas

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        """
        Route a tools/call request to the MCP server.
        Returns the result as a JSON string (ready to paste into messages).
        """
        resp = self._send(
            {
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
            }
        )

        if "error" in resp:
            err = resp["error"]
            # Return a 400-style error string so the LLM can self-correct
            return json.dumps(
                {"error_code": err.get("code"), "message": err.get("message")}
            )

        contents = resp.get("result", {}).get("content", [])
        if contents:
            return contents[0].get("text", "{}")
        return "{}"

    def close(self):
        """Terminate the MCP server subprocess."""
        try:
            self._proc.stdin.close()
            self._proc.terminate()
        except Exception:
            pass
