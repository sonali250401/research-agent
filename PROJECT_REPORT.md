# ResearchBot: AI Research Agent

## 1. Problem Statement

Finding a reliable answer to a research question often requires web search,
calculation, document lookup, summarization, and context management. A basic
chatbot may answer from its existing knowledge, but it does not reliably
decide when to use external tools, preserve workflow state, request approval
for sensitive actions, or recover from tool errors.

This project solves that problem by building **ResearchBot**, a stateful AI
research agent. It accepts a user's question, reasons about the next step,
discovers and calls tools through MCP, stores useful information, and returns a
structured answer through a browser interface, CLI, or FastAPI endpoint.

## 2. Project Objective

The objective was to implement the nine production-agent architecture tasks:

- Define a shared state object.
- Build an LLM reasoning node.
- Execute tools through JSON-RPC.
- Discover tools dynamically through MCP.
- Add deterministic pre- and post-tool hooks.
- Pause sensitive actions for human approval.
- Compact long context and persist checkpoints.
- Add telemetry and Langfuse support.
- Evaluate and deploy the agent through FastAPI and Docker.

## 3. Solution Overview

The user submits a question through the web UI or `POST /webhook`. LangGraph
passes the request through the following workflow:

```text
User question
     |
     v
Reasoning node (Groq)
     |
     +--> Final answer
     |
     +--> MCP client --> MCP server --> Tool --> Tool result
                                      |
                                      v
                              Reasoning node
```

The graph can also route to human approval before `store_document`, compact
older messages when the context becomes large, and save state in SQLite.

## 4. Technology Stack

| Technology | Use |
|---|---|
| Python | Application implementation |
| LangGraph | Stateful graph, routing, interrupts, and checkpoints |
| LangChain | Messages and LLM integration |
| Groq | LLM reasoning and answer generation |
| MCP-style JSON-RPC | Dynamic tool discovery and execution |
| Tavily | Optional web search |
| Qdrant | Optional vector document storage |
| SQLite | Persistent LangGraph checkpoints |
| Langfuse | Optional production telemetry |
| FastAPI | Webhook and browser API |
| Docker | Container deployment |
| GitHub | Source-code hosting |

## 5. Task-wise Implementation

### Task 1 — State Management

**What I built:** `state.py` defines the `AgentState` `TypedDict`, the single
memory object passed between graph nodes. It contains the conversation
messages, original query, search results, summary, tool-call count, pending
approval action, approval status, final answer, error information, and trace
ID. The `messages` field uses LangGraph's `add_messages` reducer.

**Why:** A defined state contract makes every node predictable and allows the
workflow to resume after an interruption or process restart.

### Task 2 — Reasoning Node

**What I built:** `reason_node()` in `agent.py` adds the system prompt and
conversation state, initializes Groq, binds the schemas discovered from MCP,
and invokes the LLM. The node only decides the next action: it returns a tool
call or final answer content. It does not execute tools.

**Why:** Separating decision-making from execution makes the agent modular and
allows the same reasoning node to work with multiple tools and providers.

### Task 3 — Tool Provisioning and JSON-RPC Execution

**What I built:** `tools.py` implements:

- `web_search`
- `store_document`
- `retrieve_documents`
- `calculator`
- `summarise_text`

Each tool has a JSON-compatible schema. `execute_tool_node()` extracts the
tool name, arguments, and call ID, sends the request through the MCP client,
and appends the result as a `ToolMessage`.

**Why:** Tools provide deterministic capabilities that are more reliable than
asking an LLM to perform web retrieval, arithmetic, or storage directly.

### Task 4 — MCP Dynamic Tool Integration

**What I built:** `mcp_server.py` runs a local stdio JSON-RPC server supporting
`initialize`, `tools/list`, and `tools/call`. `mcp_client.py` starts the server,
requests the tool list at startup, converts the returned schemas to the LLM
tool format, and routes tool calls back to the server.

**Why:** The agent is decoupled from tool implementation details. New tools
can be exposed by the MCP server without rewriting the reasoning node.

### Task 5 — Pre- and Post-tool Hooks

**What I built:** `pre_tool_hook()` validates inputs such as search-query
length, document title/content, and calculator expressions. A failed
validation is returned as a simulated JSON-RPC 400 error so the LLM can
self-correct. `post_tool_hook()` records result metadata and appends an entry
to `tool_audit.log`.

**Why:** Deterministic validation provides a safety boundary around
non-deterministic model output.

### Task 6 — Human-in-the-Loop Routing

**What I built:** `store_document` is treated as a high-impact action.
LangGraph routes it to the `hitl_approval` node and interrupts execution
before the tool runs. The pending action is persisted with the graph state,
and the user can approve or reject it. The API also supports
`auto_approve_hitl` for non-interactive local testing.

**Why:** A human approval step prevents an autonomous agent from persisting
important information without review.

### Task 7 — Context Compaction and Checkpointing

**What I built:** When the message history exceeds the configured threshold,
`compact_node()` summarizes older messages and retains recent context.
LangGraph uses SQLite checkpointing through `checkpoints.sqlite`, with a
thread ID identifying each conversation.

**Why:** Compaction controls token growth, while checkpointing allows the
workflow to resume after HITL pauses or application restarts.

### Task 8 — Continuous Telemetry

**What I built:** `telemetry.py` wraps the agent execution in a trace and
records node/span timing, tool names, and payload metadata. When valid
Langfuse credentials are configured, the Langfuse client is enabled. When
they are not configured, the project uses a local timer tracer and logs
`TRACE START`, `SPAN`, and `TRACE END` events.

**Why:** Agent workflows contain multiple model and tool steps. Telemetry
helps identify slow nodes, failed calls, and the exact execution path.

### Task 9 — Evaluation and Deployment

**What I built:** `eval.py` provides an evaluation workflow for judging agent
responses. `api.py` exposes:

- `POST /webhook` for synchronous questions
- `POST /webhook/async` for background jobs
- `GET /status/{thread_id}` for job status
- `GET /health` for liveness
- `GET /docs` for Swagger documentation

The project also includes a Dockerfile for deployment to a container platform
such as Render or Hugging Face Spaces. Local Windows launchers are provided in
`start_server.bat`, `start_server.ps1`, and `start_server_window.bat`.

**Why:** The same graph can be used from a terminal during development and
through an HTTP endpoint in production.

## 6. Testing and Results

The following behaviors were verified during development:

- The FastAPI application starts successfully.
- The health endpoint returns HTTP 200.
- The browser UI accepts a question and displays the generated answer.
- The Groq reasoning node can generate tool calls and final responses.
- MCP dynamically discovers the available tools.
- Tool results return to the message state.
- Markdown answers are rendered as readable headings, lists, links, and
  tables instead of raw formatting characters.
- `.env`, API keys, SQLite files, and logs are excluded from GitHub.

## 7. Screenshots to Include

Add the following screenshots to the final submission document:

1. **Project structure:** VS Code Explorer showing `agent.py`, `state.py`,
   `tools.py`, `mcp_client.py`, `mcp_server.py`, `api.py`, and `Dockerfile`.
2. **Running API:** The terminal showing Uvicorn running successfully.
3. **ResearchBot UI:** The browser question box and Ask ResearchBot button.
4. **Generated answer:** The browser displaying a completed answer.
5. **MCP discovery/logs:** The terminal showing tools discovered through MCP.
6. **HITL approval:** The approval prompt for `store_document`, if demonstrated.

Suggested captions:

- *Figure 1: Modular project structure for the ResearchBot agent.*
- *Figure 2: FastAPI server running locally through Uvicorn.*
- *Figure 3: User submitting a research question through the browser UI.*
- *Figure 4: ResearchBot returning a formatted research answer.*
- *Figure 5: Dynamic MCP tool discovery and tool execution.*
- *Figure 6: Human approval before a high-impact storage action.*

## 8. Challenges and Solutions

| Challenge | Solution |
|---|---|
| LLM needs current information | Added web-search tool support |
| Tool implementation should stay decoupled | Added MCP client/server JSON-RPC layer |
| Model may send invalid arguments | Added pre-tool validation and 400 errors |
| Sensitive actions need oversight | Added LangGraph HITL interrupt |
| Long conversations increase token usage | Added context compaction |
| Workflow must survive interruptions | Added SQLite checkpointing |
| Dynamic workflows are difficult to debug | Added local tracing and Langfuse support |
| Local Windows Python alias was unavailable | Added direct Python launcher scripts |

## 9. Future Improvements

- Add authentication and rate limiting to the API.
- Add automated unit and integration tests for every tool.
- Add stronger citation verification and source deduplication.
- Add retry/backoff policies for external APIs.
- Replace the in-memory async job store with Redis or a database.

## 10. Project Links

**GitHub repository:**  
https://github.com/sonali250401/research-agent

**Live deployed application:**  
https://research-agent-igk9.onrender.com/

**Live health check:**  
https://research-agent-igk9.onrender.com/health

**Live API documentation:**  
https://research-agent-igk9.onrender.com/docs

**Local project URL:**  
http://127.0.0.1:8080/

**Local API documentation:**  
http://127.0.0.1:8080/docs

The deployed service was verified successfully. Its health endpoint returned
HTTP 200 with the response `{"status":"healthy","service":"researchbot",
"version":"1.0.0"}`. The local URLs work only while the server is running on
the development computer.

## 11. Conclusion

ResearchBot demonstrates a complete stateful agent architecture. It combines
LLM reasoning, JSON-RPC tools, dynamic MCP discovery, deterministic
guardrails, human approval, context management, checkpointing, telemetry, and
FastAPI deployment. The design separates decisions from actions, making the
system easier to extend, observe, and operate than a simple chatbot.
