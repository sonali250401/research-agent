# ResearchBot

## MCP-Powered AI Research Agent

### Project Implementation Report

**Submitted by:** Sonali Gautam

**GitHub Repository:**
https://github.com/sonali250401/research-agent

**Live Application:**
https://research-agent-igk9.onrender.com/

**API Documentation:**
https://research-agent-igk9.onrender.com/docs

**Deployment Platform:** Render

---

## 1. Introduction

### Problem Statement

Answering a research question often requires collecting current information,
evaluating sources, performing calculations, retrieving documents, summarizing
content, and maintaining context across several steps. A basic chatbot may
generate an answer from its existing knowledge, but it does not reliably know
when to use external tools, preserve workflow state, request approval for
sensitive operations, or recover from tool errors.

This project solves that problem by building **ResearchBot**, a stateful AI
research agent. ResearchBot accepts a user's question, reasons about the next
step, dynamically discovers tools through MCP, executes those tools, maintains
state, and returns a structured answer through a browser interface, CLI, or
FastAPI endpoint.

### Objectives

- Build an autonomous research assistant.
- Allow the LLM to choose tools instead of hard-coding every decision.
- Retrieve current information through web search.
- Store and retrieve research documents.
- Support calculations and text summarization.
- Maintain state throughout a LangGraph workflow.
- Pause high-impact actions for human approval.
- Compact long message histories.
- Persist graph state using SQLite checkpointing.
- Add telemetry and an LLM-as-a-judge evaluation workflow.
- Deploy the agent as a public FastAPI service.

### Solution Overview

The user submits a question through the browser UI or `POST /webhook`.
LangGraph passes the request through a reasoning node. The Groq model either
returns a final response or requests a tool. Tool schemas are discovered
dynamically from the local MCP server. The selected tool executes through
JSON-RPC, and the result is returned to the reasoning node. This loop
continues until the agent has enough information to produce a final answer.

Cross-cutting components provide human approval, context compaction, SQLite
checkpointing, and telemetry.

### High-Level Architecture

```text
                         User Query
                             |
                             v
              Browser UI or POST /webhook
                             |
                             v
                   LangGraph State Graph
                             |
                             v
                    Reasoning Node
                       Groq LLM
                             |
              +--------------+--------------+
              |                             |
         Final answer                    Tool call
              |                             |
              v                             v
        FastAPI response              MCP Client
                                            |
                                            | tools/list / tools/call
                                            v
                                       MCP Server
                                            |
                +---------------------------+--------------------------+
                |              |             |            |            |
                v              v             v            v            v
           Web search      Storage       Retrieval    Calculator   Summarizer
                |              |             |            |            |
                +--------------+-------------+------------+------------+
                                       |
                                       v
                                 Tool result
                                       |
                                       v
                              Reasoning Node again

      HITL approval, context compaction, SQLite, and telemetry
      operate across the workflow.
```

**Figure 1: High-level architecture of ResearchBot.**

### Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application development |
| LangGraph | Stateful graph, routing, loops, interrupts, and checkpoints |
| LangChain | Messages and LLM integration |
| Groq | LLM reasoning and answer generation |
| MCP-style JSON-RPC | Dynamic tool discovery and execution |
| Tavily | Optional current web search |
| Qdrant | Optional vector document storage |
| SQLite | Persistent LangGraph checkpointing |
| Langfuse | Optional production telemetry |
| FastAPI | API and browser application |
| Docker | Container deployment |
| GitHub | Version control and source hosting |

---

## 2. Task-wise Implementation

### Task 1 — Define the State

#### What I built

I created `state.py`, which defines the `AgentState` `TypedDict`. It is the
single memory object passed between every LangGraph node.

The state contains:

- `messages`: conversation and tool-message history
- `query`: original user question
- `search_results`: web-search results
- `summary`: compacted historical context
- `tool_call_count`: tool/reasoning budget
- `pending_action`: action waiting for approval
- `hitl_approved`: approval status
- `final_answer`: completed response
- `error`: workflow error information
- `trace_id`: telemetry identifier

#### How it works

The `messages` field uses LangGraph's `add_messages` reducer. Each node
returns only its state updates, and LangGraph merges those updates into the
shared state.

#### Why it matters

A defined state contract makes the graph predictable. It allows reasoning,
execution, human approval, compaction, and finalization nodes to communicate
without relying on hidden global variables.

### Task 2 — Build the Reasoning Node

#### What I built

I implemented `reason_node()` in `agent.py`. It loads the conversation state,
adds the system prompt, initializes the configured Groq model, discovers MCP
tools, and binds their schemas to the LLM.

The reasoning node can return either:

- a tool call containing a name and arguments, or
- final answer text.

#### How it works

The node does not execute tools. It only decides what should happen next.
LangGraph routing inspects the response and sends it to tool execution,
human approval, context compaction, or finalization.

#### Why it matters

Separating reasoning from execution creates a clean agent architecture. The
model can decide whether to search, calculate, retrieve, summarize, or answer
directly without hard-coded question-specific rules.

### Task 3 — Tool Provisioning and JSON-RPC Execution

#### What I built

I implemented five Python tools in `tools.py`:

- `web_search`
- `store_document`
- `retrieve_documents`
- `calculator`
- `summarise_text`

Each tool has a JSON-compatible schema. The `execute_tool_node()` function in
`agent.py` extracts the tool name, arguments, and tool-call ID.

#### How it works

The execution node sends a `tools/call` request to the MCP client. The result
is returned as a `ToolMessage` and appended to the state. The reasoning node
then receives the result and decides whether another action is required.

#### Why it matters

External tools are more reliable for deterministic work such as arithmetic,
search, and storage. The separate execution node also makes it possible to
add tools without changing the core reasoning logic.

### Task 4 — Implement MCP

#### What I built

I created:

- `mcp_server.py`: local stdio JSON-RPC server
- `mcp_client.py`: client used by the agent

The MCP server supports:

- `initialize`
- `tools/list`
- `tools/call`

#### How it works

When the agent starts, the client launches `mcp_server.py` as a subprocess and
requests `tools/list`. The returned schemas are converted to the format
expected by the Groq tool-binding interface. When the LLM requests a tool,
the client sends a JSON-RPC `tools/call` request to the MCP server.

The deployed logs confirmed dynamic discovery of five tools:

```text
Discovered 5 tools via MCP:
['web_search', 'store_document', 'retrieve_documents',
 'calculator', 'summarise_text']
```

#### Why it matters

MCP decouples the agent from external tool implementations. The reasoning
node receives capabilities and schemas, not hard-coded Python details.

### Task 5 — Add Pre- and Post-tool Hooks

#### What I built

I implemented `pre_tool_hook()` and `post_tool_hook()` in `tools.py`.

The pre-tool hook validates:

- minimum and maximum web-search query length
- non-empty document content
- document title presence
- unsafe calculator expressions

The post-tool hook records result metadata and writes an audit entry to
`tool_audit.log`.

#### How it works

If validation fails, the MCP server returns a JSON-RPC error with code `400`
and a message explaining how the model should correct the request. The error
is added to the conversation so the model can self-correct.

#### Why it matters

LLM output is non-deterministic. Deterministic hooks provide a safety boundary
before a tool executes and create an audit trail after execution.

### Task 6 — Human-in-the-Loop Routing

#### What I built

`store_document` is treated as a high-impact action. The graph routes this
action to `hitl_approval` instead of executing it immediately.

The graph is compiled with an interrupt before the approval node. The API
also supports `auto_approve_hitl` for non-interactive deployment testing.

#### How it works

The workflow stores the pending tool name, arguments, and call ID. It pauses
and waits for approval. An approved action proceeds to `execute_tool`; a
rejected action returns to the reasoning node for replanning.

#### Why it matters

Human approval prevents an autonomous agent from persisting important
information without review. It is a practical guardrail for high-impact
actions.

### Task 7 — Context Compaction and Checkpointing

#### What I built

I implemented `compact_node()` in `agent.py`. The configuration supports:

```text
MAX_MESSAGES_BEFORE_COMPACTION=10
MESSAGES_TO_SUMMARISE=8
```

The graph also uses SQLite checkpointing with `checkpoints.sqlite` and a
conversation-specific `thread_id`.

#### How it works

When the message history exceeds the configured threshold, older messages are
summarized and replaced with a compact summary while recent messages remain
available. LangGraph writes state and graph progress to SQLite.

#### Why it matters

Long research sessions can exceed the LLM context window. Compaction controls
token usage, while checkpointing supports recovery after restarts and HITL
interruptions.

### Task 8 — Continuous Telemetry

#### What I built

I implemented telemetry in `telemetry.py`. The agent records trace lifecycle,
node spans, elapsed time, tool names, and output metadata. Valid Langfuse
credentials enable Langfuse; otherwise the application uses a local timer
tracer.

#### How it works

The execution loop is wrapped in a trace. Reasoning and tool execution are
wrapped in spans. Local logs include events such as:

```text
[TRACE START]
[SPAN] reason_node
[SPAN] execute_tool
[TRACE END]
```

#### Why it matters

Agent workflows contain multiple model and tool calls. Telemetry makes
latency, failures, and the exact workflow path observable.

### Task 9 — Evaluation and Deployment

#### What I built

I implemented evaluation in `eval.py`. It loads up to ten traces, asks Gemini
to grade error recovery from 1 to 5 when configured, and falls back to
deterministic heuristic scoring when Gemini is unavailable.

I implemented deployment in `api.py` with:

- `POST /webhook`
- `POST /webhook/async`
- `GET /status/{thread_id}`
- `GET /health`
- `GET /docs`

The application is containerized using `Dockerfile` and deployed on Render.

#### How it works

The synchronous endpoint runs the LangGraph workflow in a thread executor.
The asynchronous endpoint starts a background job and provides a polling
endpoint. Render runs the Docker container and supplies the production port.

#### Why it matters

Evaluation measures reliability, while FastAPI and Docker make the agent
available as a production-style service rather than only a local script.

---

## 3. Testing and Results

ResearchBot was tested through both local and deployed interfaces.

### Deployment Health

**Health endpoint:**
https://research-agent-igk9.onrender.com/health

**Verified response:**

```json
{"status":"healthy","service":"researchbot","version":"1.0.0"}
```

The endpoint returned HTTP 200.

### Live Agent Execution

The deployed Render logs confirmed:

- Groq model initialized as `openai/gpt-oss-120b`
- Groq requests returned HTTP 200
- Five tools discovered through MCP
- Web-search tools executed successfully
- Final answer generated
- `POST /webhook` returned HTTP 200

### Evaluation Result

The recorded `eval_report.json` contains:

- **Traces evaluated:** 10
- **Aggregate error-recovery score:** 4.5 / 5
- **Scoring mode:** heuristic fallback for the recorded run

The evaluation covered successful tool calls, pre-tool validation errors,
calculator errors, document-storage validation, and HITL rejection with
recovery.

---

## 4. Screenshots for Submission

Insert screenshots from the actual project, not illustrative internet images.
Do not include `.env` values, API keys, or access tokens.

### Figure 2 — Project Structure

Show the VS Code Explorer containing:

```text
agent.py
state.py
tools.py
mcp_client.py
mcp_server.py
api.py
eval.py
Dockerfile
requirements.txt
```

**Caption:** Project structure showing the modular agent, state, tools, MCP,
API, evaluation, and deployment components.

### Figure 3 — ResearchBot Interface

Open:

https://research-agent-igk9.onrender.com/

Show the question box and **Ask ResearchBot** button.

**Caption:** Public ResearchBot interface running on Render.

### Figure 4 — API Documentation

Open:

https://research-agent-igk9.onrender.com/docs

Show the available FastAPI endpoints.

**Caption:** Swagger documentation for the deployed ResearchBot API.

### Figure 5 — MCP Discovery and Tool Execution

Capture terminal or Render logs containing:

```text
Discovered 5 tools via MCP
[execute_tool] web_search(...)
POST /webhook HTTP/1.1 200 OK
```

**Caption:** MCP dynamically discovering tools and routing a tool call.

### Figure 6 — Health Check

Capture the HTTP 200 response from:

https://research-agent-igk9.onrender.com/health

**Caption:** Live deployment health check returning HTTP 200.

### Figure 7 — Completed Research Answer

Show the deployed interface after a successful question and generated answer.

**Caption:** ResearchBot completing an end-to-end research request.

### Figure 8 — Evaluation Report

Show `eval_report.json` or the evaluation terminal output containing ten
traces and the aggregate score of 4.5/5.

**Caption:** LLM-as-a-judge evaluation output for error recovery.

### Figure 9 — Human-in-the-Loop

If demonstrated, show the approval prompt for `store_document`.

**Caption:** Human approval before a high-impact document-storage action.

---

## 5. Challenges and Solutions

| Challenge | Solution |
|---|---|
| The LLM needs current information | Added web-search tool support |
| Tools should be dynamically discoverable | Added MCP client/server JSON-RPC |
| Model arguments may be invalid | Added pre-tool hooks and 400 errors |
| Sensitive actions need review | Added LangGraph HITL interruption |
| Long conversations consume tokens | Added context compaction |
| Workflow state must survive pauses | Added SQLite checkpointing |
| Dynamic workflows are difficult to debug | Added local tracing and Langfuse support |
| Deprecated Groq model after deployment | Updated configuration to `openai/gpt-oss-120b` |
| Windows Python alias was unavailable | Added direct Python launchers |

---

## 6. Future Improvements

- Add API authentication and rate limiting.
- Add automated unit and integration tests for every tool.
- Add citation verification and source deduplication.
- Add retry and backoff policies for external APIs.
- Replace the in-memory async job store with Redis or a database.
- Add a monitoring dashboard for production traces.
- Improve document retrieval and ranking.

---

## 7. Project Links

- **GitHub Repository:** https://github.com/sonali250401/research-agent
- **Live ResearchBot Application:** https://research-agent-igk9.onrender.com/
- **API Documentation:** https://research-agent-igk9.onrender.com/docs
- **Health Check:** https://research-agent-igk9.onrender.com/health

### Local Development

From the project directory on Windows:

```powershell
.\start_server.bat
```

Open:

- http://127.0.0.1:8080/
- http://127.0.0.1:8080/docs

These local URLs work only while the server is running on the development
computer.

---

## 8. Final Submission Checklist

- [ ] Add name and project title to the cover page.
- [ ] Include the problem statement, objectives, and architecture diagram.
- [ ] Explain Tasks 1–9 using What I built, How it works, and Why it matters.
- [ ] Insert actual project screenshots and figure captions.
- [ ] Include the evaluation result and state that heuristic fallback was used.
- [ ] Include the GitHub repository and verified Render URLs.
- [ ] Check that no API keys or credentials appear in screenshots.
- [ ] Export the report as PDF and inspect the exported file once.

---

## Conclusion

ResearchBot demonstrates a complete stateful agent architecture. It combines
LLM reasoning, JSON-RPC tools, dynamic MCP discovery, deterministic
guardrails, human approval, context management, checkpointing, telemetry,
evaluation, and FastAPI deployment. The separation between decisions and
actions makes the system modular, observable, and easier to extend than a
simple chatbot.
