"""
Task 1: Define the State (State Management)
============================================
The AgentState TypedDict is the single source of truth that flows
between every node in the LangGraph state machine.
"""

from typing import Annotated, Any, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    The memory object passed between all nodes.

    Fields
    ------
    messages        : Conversation history (auto-merged by LangGraph's add_messages reducer)
    query           : The user's original research question
    search_results  : Raw results returned by the search tool
    summary         : Compact summary injected when message history is too long
    tool_call_count : Running counter used to decide when to compact
    pending_action  : Dict describing an action waiting for HITL approval
    hitl_approved   : True / False / None (None = not yet reviewed)
    final_answer    : The agent's synthesised research response
    error           : Any error message from a failed tool call
    trace_id        : Langfuse trace ID for observability
    """

    messages: Annotated[list, add_messages]
    query: str
    search_results: list[dict]
    summary: Optional[str]
    tool_call_count: int
    pending_action: Optional[dict]
    hitl_approved: Optional[bool]
    final_answer: Optional[str]
    error: Optional[str]
    trace_id: Optional[str]
