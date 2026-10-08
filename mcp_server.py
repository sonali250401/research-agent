"""
Task 4: MCP Server (Model Context Protocol)
============================================
A stdio-based MCP server that exposes the agent's tools dynamically.
The agent's MCP client discovers tools via `tools/list` on startup
and routes `tools/call` payloads here at runtime.

Run standalone:  python mcp_server.py
"""

import sys
import json
import logging
from tools import TOOL_REGISTRY, TOOL_SCHEMAS, pre_tool_hook, post_tool_hook, HookError

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
log = logging.getLogger("mcp_server")


def build_tool_list_response(request_id: int) -> dict:
    """Build the MCP tools/list response from TOOL_SCHEMAS."""
    mcp_tools = []
    for schema in TOOL_SCHEMAS:
        fn = schema["function"]
        mcp_tools.append(
            {
                "name": fn["name"],
                "description": fn["description"],
                "inputSchema": fn["parameters"],
            }
        )
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {"tools": mcp_tools},
    }


def handle_tools_call(request_id: int, params: dict) -> dict:
    """Dispatch a tools/call request to the correct Python function."""
    tool_name = params.get("name", "")
    args = params.get("arguments", {})

    if tool_name not in TOOL_REGISTRY:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": f"Tool not found: {tool_name}"},
        }

    # Apply pre-tool hook
    try:
        pre_tool_hook(tool_name, args)
    except HookError as e:
        log.warning(f"Pre-hook blocked {tool_name}: {e}")
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": 400,
                "message": f"Pre-tool validation failed: {e}. Please self-correct and retry.",
            },
        }

    # Execute the tool
    try:
        result = TOOL_REGISTRY[tool_name](**args)
        post_tool_hook(tool_name, result)
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]},
        }
    except Exception as e:
        log.error(f"Tool {tool_name} raised: {e}")
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": 500, "message": str(e)},
        }


def mcp_server_loop():
    """
    Stdio JSON-RPC loop.  Reads one JSON object per line from stdin,
    writes one JSON object per line to stdout.
    """
    log.info("MCP Server started — awaiting JSON-RPC requests on stdin …")

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        try:
            request = json.loads(raw_line)
        except json.JSONDecodeError as e:
            error_resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {e}"},
            }
            print(json.dumps(error_resp), flush=True)
            continue

        req_id = request.get("id")
        method = request.get("method", "")
        params = request.get("params", {})

        log.info(f"← method={method}  id={req_id}")

        if method == "tools/list":
            response = build_tool_list_response(req_id)
        elif method == "tools/call":
            response = handle_tools_call(req_id, params)
        elif method == "initialize":
            response = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "research-agent-mcp", "version": "1.0.0"},
                },
            }
        else:
            response = {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }

        log.info(f"→ id={req_id}")
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    mcp_server_loop()
