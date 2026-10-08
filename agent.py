"""
Tasks 2, 6, 7, 8: LangGraph Agent — Reasoning Node + Full Graph
================================================================
Nodes
-----
  reason         → Task 2: Groq LLM decides next action
  execute_tool   → Task 3/4: dispatch tool call via MCP
  hitl_approval  → Task 6: pause for human approval
  compact        → Task 7: summarise + checkpoint
  finalize       → assemble the final answer

Edges / routing
---------------
  reason → execute_tool | hitl_approval | finalize | reason
  hitl_approval → execute_tool (approved) | reason (rejected)
  execute_tool → compact? → reason (loop)
"""

import sys
import json
import logging
import sqlite3
from typing import Literal

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

# LangGraph 1.x ships SQLite saver as a separate package
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except ImportError:
    try:
        from langgraph_checkpoint_sqlite import SqliteSaver
    except ImportError:
        from langgraph.checkpoint.memory import MemorySaver as SqliteSaver
        logging.warning("SQLite saver not found — using in-memory checkpointing (state won't survive restarts).\n"
                        "Fix: pip install langgraph-checkpoint-sqlite")

from config import cfg
from mcp_client import MCPClient
from state import AgentState
from telemetry import get_tracer

log = logging.getLogger("agent")

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are ResearchBot, an expert AI research assistant.

Your goal: Answer the user's research question thoroughly, factually, and concisely.

CURRENT YEAR & TIMELINESS:
- The current year is 2026.
- Prioritize latest 2026 breakthroughs, using 2025 where relevant for context.

RESEARCH GUIDELINES:
- Perform 2 to 4 focused `web_search` queries to gather the core facts.
- Do not search repeatedly once the main answers are established.
- When you have sufficient information, provide your final comprehensive answer in clear, markdown-formatted plain text (without calling more tools).
- Be thorough and cite source titles and URLs.
"""

MAX_TOOL_CALLS = 5

# ── Singleton MCP client (spawned once per process) ───────────────────────────
_mcp_client: MCPClient | None = None


def get_mcp_client() -> MCPClient:
    global _mcp_client
    if _mcp_client is None:
        _mcp_client = MCPClient()
        _mcp_client.list_tools()  # Task 4: dynamic discovery
    return _mcp_client


def extract_text_content(msg) -> str:
    """Extract string content from any AIMessage, list of parts, or reasoning payload."""
    if not msg:
        return ""
    content = getattr(msg, "content", "")
    if isinstance(content, list):
        text_parts = [
            p.get("text", "") if isinstance(p, dict) else str(p)
            for p in content
        ]
        content = "\n".join(t for t in text_parts if t)
    if not content and hasattr(msg, "additional_kwargs"):
        content = msg.additional_kwargs.get("reasoning_content", "") or ""
    return str(content).strip()


# ══════════════════════════════════════════════════════════════════════════════
# Task 2: Reasoning Node
# ══════════════════════════════════════════════════════════════════════════════

def reason_node(state: AgentState) -> dict:
    """
    The engine. Calls the LLM with message history and dynamically discovered MCP tools.
    Automatically caps tool execution at MAX_TOOL_CALLS and synthesizes final answer.
    """
    tracer = get_tracer()
    mcp = get_mcp_client()
    call_count = state.get("tool_call_count", 0)

    # Build message list: inject compact summary if present
    messages = []
    messages.append(SystemMessage(content=SYSTEM_PROMPT))

    if state.get("summary"):
        messages.append(
            HumanMessage(
                content=f"[Context summary from earlier conversation]\n{state['summary']}"
            )
        )

    messages.extend(state["messages"])

    # Task 2: Instantiate LLM (supports Gemini or Groq)
    base_llm = None
    if cfg.GEMINI_API_KEY and not cfg.GEMINI_API_KEY.startswith("your_") and len(cfg.GEMINI_API_KEY) > 10:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            log.info(f"[LLM] Using Gemini model: {cfg.GEMINI_MODEL}")
            base_llm = ChatGoogleGenerativeAI(
                model=cfg.GEMINI_MODEL or "gemini-1.5-flash",
                google_api_key=cfg.GEMINI_API_KEY,
                temperature=0,
            )
        except Exception as e:
            log.warning(f"Failed to initialize Gemini: {e}")

    if base_llm is None and cfg.GROQ_API_KEY and not cfg.GROQ_API_KEY.startswith("your_"):
        log.info(f"[LLM] Using Groq model: {cfg.GROQ_MODEL}")
        base_llm = ChatGroq(
            model=cfg.GROQ_MODEL,
            api_key=cfg.GROQ_API_KEY,
            temperature=0,
        )

    if base_llm is None:
        raise ValueError(
            "No valid LLM key found. Please set a valid GROQ_API_KEY or GEMINI_API_KEY in your .env file."
        )

    # If 3 or more tool messages exist or max budget reached, synthesize immediately
    num_tool_messages = sum(
        1 for m in state["messages"]
        if isinstance(m, ToolMessage) or getattr(m, "type", "") == "tool"
    )

    bound_llm = base_llm.bind_tools(mcp.tool_schemas)

    if num_tool_messages >= 3 or call_count >= 3:
        log.info(f"[reason] Research budget reached ({num_tool_messages} searches completed) — synthesizing final answer.")
        tool_findings = "\n\n".join(
            f"[{getattr(m, 'name', 'Tool')}]: {getattr(m, 'content', '')}"
            for m in state["messages"]
            if isinstance(m, ToolMessage) or getattr(m, "type", "") == "tool"
        )
        synthesis_messages = [
            SystemMessage(content=SYSTEM_PROMPT + "\n\nIMPORTANT: Provide the final detailed answer directly in markdown text. Do NOT call any tools."),
            HumanMessage(
                content=(
                    f"User question: {state['query']}\n\n"
                    f"Research findings gathered from search:\n{tool_findings}\n\n"
                    "Please write a comprehensive, detailed, and structured research report answering the user's question with citations in clean Markdown."
                )
            ),
        ]
        with tracer.span("reason_node_final_synthesis") as span:
            response = bound_llm.invoke(synthesis_messages)
            text = extract_text_content(response)
            final_msg = AIMessage(content=text if text else "Research summary assembled.")
            span.set_attribute("content_len", len(final_msg.content))
        return {
            "messages": [final_msg],
            "tool_call_count": call_count + 1,
        }

    with tracer.span("reason_node") as span:
        response = bound_llm.invoke(messages)
        span.set_attribute("tool_calls", len(getattr(response, "tool_calls", [])))
        span.set_attribute("content_len", len(response.content or ""))

    log.info(f"[reason] tool_calls={getattr(response, 'tool_calls', [])}")

    return {
        "messages": [response],
        "tool_call_count": call_count + 1,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Task 3/4: Tool Execution Node
# ══════════════════════════════════════════════════════════════════════════════

def execute_tool_node(state: AgentState) -> dict:
    """
    Dispatch the LLM's tool_call request via MCP and append the result
    back to the messages state as a ToolMessage.
    """
    tracer = get_tracer()
    mcp = get_mcp_client()

    last_ai = state["messages"][-1]
    tool_call = last_ai.tool_calls[0]
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    log.info(f"[execute_tool] {tool_name}({tool_args})")

    with tracer.span("execute_tool") as span:
        span.set_attribute("tool_name", tool_name)
        result_str = mcp.call_tool(tool_name, tool_args)
        span.set_attribute("result_len", len(result_str))

    tool_msg = ToolMessage(
        content=result_str,
        tool_call_id=tool_call["id"],
        name=tool_name,
    )

    updates: dict = {"messages": [tool_msg]}

    # Cache search results for later reference
    if tool_name == "web_search":
        try:
            updates["search_results"] = json.loads(result_str)
        except Exception:
            pass

    return updates


# ══════════════════════════════════════════════════════════════════════════════
# Task 6: HITL Approval Node
# ══════════════════════════════════════════════════════════════════════════════

def hitl_approval_node(state: AgentState) -> dict:
    """
    Pauses the graph for human review.
    LangGraph's interrupt() mechanism serialises state to the checkpoint DB.
    When resumed, `hitl_approved` will be set by the caller.
    """
    from langgraph.types import interrupt

    last_ai = state["messages"][-1]
    tool_call = last_ai.tool_calls[0]

    pending = {
        "tool_name": tool_call["name"],
        "tool_args": tool_call["args"],
        "tool_call_id": tool_call["id"],
    }

    print("\n" + "═" * 60)
    print("🔔  HUMAN-IN-THE-LOOP APPROVAL REQUIRED")
    print("═" * 60)
    print(f"  Tool     : {pending['tool_name']}")
    print(f"  Arguments: {json.dumps(pending['tool_args'], indent=4)}")
    print("═" * 60)

    # LangGraph pauses here; the outer runner must call graph.update_state()
    # with hitl_approved=True/False and then resume.
    decision = interrupt(
        {
            "prompt": "Approve or Reject this tool call?",
            "pending_action": pending,
        }
    )

    return {"pending_action": pending, "hitl_approved": decision == "Approve"}


# ══════════════════════════════════════════════════════════════════════════════
# Task 7: Context Compaction Node
# ══════════════════════════════════════════════════════════════════════════════

def compact_node(state: AgentState) -> dict:
    """
    If messages > MAX_MESSAGES_BEFORE_COMPACTION, summarise the oldest N
    messages into a single paragraph and replace them in state.
    """
    msgs = state["messages"]
    threshold = cfg.MAX_MESSAGES_BEFORE_COMPACTION
    n = cfg.MESSAGES_TO_SUMMARISE

    if len(msgs) <= threshold:
        return {}  # nothing to do

    log.info(f"[compact] Compacting {n} messages …")

    old_msgs = msgs[:n]
    keep_msgs = msgs[n:]

    # Offline summarisation call
    try:
        from groq import Groq
        client = Groq(api_key=cfg.GROQ_API_KEY)
        conversation_text = "\n".join(
            f"{m.__class__.__name__}: {getattr(m, 'content', '')[:300]}"
            for m in old_msgs
        )
        # Use fast 8b or configured model for lightweight summary
        summ_model = "llama-3.1-8b-instant" if ("120b" in cfg.GROQ_MODEL or not cfg.GROQ_MODEL) else cfg.GROQ_MODEL
        resp = client.chat.completions.create(
            model=summ_model,
            messages=[
                {
                    "role": "system",
                    "content": "Summarise the following agent conversation into one concise paragraph.",
                },
                {"role": "user", "content": conversation_text},
            ],
            max_tokens=300,
        )
        new_summary = resp.choices[0].message.content
    except Exception as e:
        log.warning(f"[compact] LLM summarization fallback: {e}")
        new_summary = f"Summary of {len(old_msgs)} previous conversation steps."

    log.info(f"[compact] Summary: {new_summary[:100]} …")

    return {
        "summary": new_summary,
        "messages": keep_msgs,  # replace full list (NOT append)
    }


# ══════════════════════════════════════════════════════════════════════════════
# Finalize Node
# ══════════════════════════════════════════════════════════════════════════════

def finalize_node(state: AgentState) -> dict:
    """Extract the agent's last plain-text response as the final answer."""
    last = state["messages"][-1]
    answer = extract_text_content(last)
    if not answer:
        for m in reversed(state["messages"]):
            if isinstance(m, AIMessage):
                candidate = extract_text_content(m)
                if candidate:
                    answer = candidate
                    break
    log.info(f"[finalize] answer_len={len(answer)}")
    return {"final_answer": answer or "Research completed."}


# ══════════════════════════════════════════════════════════════════════════════
# Routing functions (conditional edges)
# ══════════════════════════════════════════════════════════════════════════════

def route_after_reason(state: AgentState) -> Literal[
    "execute_tool", "hitl_approval", "finalize", "compact"
]:
    last = state["messages"][-1]

    # No tool call → finalise
    if not getattr(last, "tool_calls", None):
        msgs = state["messages"]
        if len(msgs) > cfg.MAX_MESSAGES_BEFORE_COMPACTION:
            return "compact"
        return "finalize"

    tool_name = last.tool_calls[0]["name"]

    # store_document is a "high-stakes" action requiring HITL
    if tool_name == "store_document":
        return "hitl_approval"

    # Compact before executing if history is long
    if len(state["messages"]) > cfg.MAX_MESSAGES_BEFORE_COMPACTION:
        return "compact"

    return "execute_tool"


def route_after_hitl(state: AgentState) -> Literal["execute_tool", "reason"]:
    if state.get("hitl_approved"):
        log.info("[HITL] Approved ✔")
        return "execute_tool"
    else:
        log.info("[HITL] Rejected ✘ — agent will re-plan")
        return "reason"


def route_after_compact(state: AgentState) -> Literal["execute_tool", "finalize"]:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "execute_tool"
    return "finalize"


# ══════════════════════════════════════════════════════════════════════════════
# Build and compile the LangGraph
# ══════════════════════════════════════════════════════════════════════════════

def build_graph():
    """Compile the LangGraph state machine with SQLite checkpointing."""

    # Task 7: SQLite checkpoint saver
    try:
        conn = sqlite3.connect(cfg.CHECKPOINT_DB_PATH, check_same_thread=False)
        checkpointer = SqliteSaver(conn)
    except Exception as e:
        log.warning(f"Could not connect to SQLite checkpoint DB: {e}. Falling back to MemorySaver.")
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()

    builder = StateGraph(AgentState)

    # Register nodes
    builder.add_node("reason", reason_node)
    builder.add_node("execute_tool", execute_tool_node)
    builder.add_node("hitl_approval", hitl_approval_node)
    builder.add_node("compact", compact_node)
    builder.add_node("finalize", finalize_node)

    # Entry
    builder.add_edge(START, "reason")

    # Conditional routing after reason
    builder.add_conditional_edges(
        "reason",
        route_after_reason,
        {
            "execute_tool": "execute_tool",
            "hitl_approval": "hitl_approval",
            "finalize": "finalize",
            "compact": "compact",
        },
    )

    # After tool execution → back to reason
    builder.add_edge("execute_tool", "reason")

    # After compaction → route based on last message
    builder.add_conditional_edges(
        "compact",
        route_after_compact,
        {"execute_tool": "execute_tool", "finalize": "finalize"},
    )

    # HITL routing
    builder.add_conditional_edges(
        "hitl_approval",
        route_after_hitl,
        {"execute_tool": "execute_tool", "reason": "reason"},
    )

    # Terminal
    builder.add_edge("finalize", END)

    # Compile with interrupt at HITL node + checkpoint saver
    graph = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["hitl_approval"],
    )
    return graph


# ══════════════════════════════════════════════════════════════════════════════
# Public runner — used by FastAPI and CLI
# ══════════════════════════════════════════════════════════════════════════════

def run_agent(query: str, thread_id: str = "default") -> str:
    """
    Execute the agent for a given query.
    Handles HITL interrupts via terminal prompt.
    Returns the final answer string.
    """
    from langchain_core.messages import HumanMessage
    from telemetry import start_trace, end_trace

    graph = build_graph()
    tracer = get_tracer()

    trace_id = start_trace(query)
    config = {"configurable": {"thread_id": thread_id}}
    initial_state: AgentState = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "search_results": [],
        "summary": None,
        "tool_call_count": 0,
        "pending_action": None,
        "hitl_approved": None,
        "final_answer": None,
        "error": None,
        "trace_id": trace_id,
    }

    with tracer.trace(query):
        # First run (may pause at hitl_approval)
        result = graph.invoke(initial_state, config=config)

        # HITL loop — handle any interrupts
        while result.get("__interrupt__"):
            interrupt_info = result["__interrupt__"][0].value
            pending = interrupt_info.get("pending_action", {})

            print(f"\n🔔  HITL: Approve '{pending.get('tool_name')}' call? [Approve/Reject]: ", end="")
            decision = input().strip()

            # Resume graph with decision
            graph.update_state(
                config,
                {"hitl_approved": decision == "Approve"},
                as_node="hitl_approval",
            )
            result = graph.invoke(None, config=config)

    end_trace(trace_id, result.get("final_answer", ""))
    return result.get("final_answer", "Agent did not produce a final answer.")
