"""
Task 9b: FastAPI Deployment Endpoint
======================================
Exposes the LangGraph agent as an HTTP API.

Endpoints
---------
POST /webhook       → Run agent, returns final answer (blocks until done)
POST /webhook/async → Kick off agent, returns thread_id for polling
GET  /status/{id}   → Poll async job status
GET  /health        → Liveness probe for Render / HF Spaces
GET  /              → API documentation page
"""

import asyncio
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

log = logging.getLogger("api")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# ── Application ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ResearchBot Agent API",
    description="Production LangGraph Research Agent — All 9 tasks implemented",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool for blocking LangGraph calls
executor = ThreadPoolExecutor(max_workers=4)

# In-memory job store (use Redis/DB in production)
_jobs: dict[str, dict] = {}


# ── Schemas ────────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    thread_id: Optional[str] = None
    auto_approve_hitl: bool = False  # set True for non-interactive deployments


class QueryResponse(BaseModel):
    thread_id: str
    answer: str
    tool_calls_made: int = 0


class AsyncJobResponse(BaseModel):
    thread_id: str
    status: str
    message: str


class JobStatus(BaseModel):
    thread_id: str
    status: str  # "running" | "completed" | "failed"
    answer: Optional[str] = None
    error: Optional[str] = None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ResearchBot Agent API</title>
<style>
  body { font-family: 'Segoe UI', sans-serif; background: #0f0f1a; color: #e2e8f0;
         display: flex; justify-content: center; padding: 60px 20px; }
  .card { background: #1a1a2e; border-radius: 16px; padding: 40px;
          max-width: 700px; width: 100%; box-shadow: 0 0 40px rgba(99,102,241,.3); }
  h1 { color: #818cf8; margin: 0 0 8px; font-size: 2rem; }
  .badge { display: inline-block; background: #4ade80; color: #14532d;
           border-radius: 20px; padding: 2px 12px; font-size: .8rem; margin-bottom: 24px; }
  #answer { margin-top: 20px; padding: 20px; border-radius: 8px;
            background: #0f172a; color: #cbd5e1; display: none; line-height: 1.6; }
  #answer h1, #answer h2, #answer h3 { color: #a5b4fc; margin: 20px 0 8px; }
  #answer h1:first-child, #answer h2:first-child { margin-top: 0; }
  #answer p { margin: 10px 0; }
  #answer ul, #answer ol { padding-left: 24px; }
  #answer li { margin: 5px 0; }
  #answer code { background: #1e293b; padding: 2px 5px; border-radius: 4px; color: #f9a8d4; }
  #answer pre { background: #1e293b; padding: 12px; border-radius: 6px; overflow-x: auto; }
  #answer a { color: #a5b4fc; }
  #answer table { width: 100%; border-collapse: collapse; margin: 14px 0; }
  #answer th, #answer td { text-align: left; padding: 8px; border-bottom: 1px solid #334155; }
  #answer th { color: #a5b4fc; }
  a { color: #818cf8; }
  textarea { width: 100%; min-height: 100px; box-sizing: border-box; margin-top: 12px;
             padding: 12px; border: 1px solid #334155; border-radius: 8px;
             background: #0f172a; color: #e2e8f0; font: inherit; resize: vertical; }
  button { margin-top: 12px; padding: 10px 18px; border: 0; border-radius: 8px;
           background: #6366f1; color: white; font-weight: 600; cursor: pointer; }
  button:disabled { opacity: .6; cursor: wait; }
  .error { color: #fca5a5; }
</style>
</head>
<body>
<div class="card">
  <h1>🤖 ResearchBot</h1>
  <span class="badge">● Live</span>
  <p>Ask a question and get a clear research answer.</p>
  <textarea id="question" placeholder="For example: What are the latest developments in AI?"></textarea>
  <br><br>
  <label><input id="approve" type="checkbox" checked> Automatically approve actions</label>
  <br>
  <button id="ask" onclick="askQuestion()">Ask ResearchBot</button>
  <div id="answer"></div>
</div>
<script>
function escapeHtml(value) {
  return value.replace(/[&<>"']/g, character => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[character]));
}

function inlineMarkdown(value) {
  let html = escapeHtml(value);
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  html = html.replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g,
    '<a href="$2" target="_blank" rel="noopener">$1</a>');
  html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  return html;
}

function renderMarkdown(markdown) {
  const lines = markdown.split(/\\r?\\n/);
  const output = [];
  let paragraph = [];
  let listType = null;

  function closeParagraph() {
    if (paragraph.length) {
      output.push("<p>" + paragraph.map(inlineMarkdown).join(" ") + "</p>");
      paragraph = [];
    }
  }

  function closeList() {
    if (listType) {
      output.push("</" + listType + ">");
      listType = null;
    }
  }

  for (let index = 0; index < lines.length; index++) {
    const line = lines[index].trim();
    if (!line) {
      closeParagraph();
      closeList();
      continue;
    }
    const tableSeparator = /^\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?$/;
    const isTable = line.includes("|") && index + 1 < lines.length &&
      tableSeparator.test(lines[index + 1].trim());
    if (isTable) {
      closeParagraph();
      closeList();
      const headers = line.split("|").map(cell => cell.trim()).filter(Boolean);
      const rows = [];
      index += 2;
      while (index < lines.length && lines[index].includes("|")) {
        rows.push(lines[index].split("|").map(cell => cell.trim()).filter(Boolean));
        index++;
      }
      index--;
      output.push("<table><thead><tr>" + headers.map(cell => "<th>" +
        inlineMarkdown(cell) + "</th>").join("") + "</tr></thead><tbody>" +
        rows.map(row => "<tr>" + row.map(cell => "<td>" + inlineMarkdown(cell) +
          "</td>").join("") + "</tr>").join("") + "</tbody></table>");
      continue;
    }
    const heading = line.match(/^(#{1,3})\s+(.+)$/);
    if (heading) {
      closeParagraph();
      closeList();
      output.push("<h" + heading[1].length + ">" + inlineMarkdown(heading[2]) +
        "</h" + heading[1].length + ">");
      continue;
    }
    if (/^[-*_]{3,}$/.test(line)) {
      closeParagraph();
      closeList();
      output.push("<hr>");
      continue;
    }
    const listItem = line.match(/^([-*+]|\d+\.)\s+(.+)$/);
    if (listItem) {
      closeParagraph();
      const nextListType = /^\d+\./.test(listItem[1]) ? "ol" : "ul";
      if (listType !== nextListType) {
        closeList();
        listType = nextListType;
        output.push("<" + listType + ">");
      }
      output.push("<li>" + inlineMarkdown(listItem[2]) + "</li>");
      continue;
    }
    closeList();
    paragraph.push(line);
  }
  closeParagraph();
  closeList();
  return output.join("");
}

async function askQuestion() {
  const question = document.getElementById("question").value.trim();
  const answer = document.getElementById("answer");
  const button = document.getElementById("ask");
  if (!question) {
    answer.className = "error";
    answer.textContent = "Please enter a question first.";
    answer.style.display = "block";
    return;
  }
  button.disabled = true;
  answer.className = "";
  answer.textContent = "ResearchBot is working...";
  answer.style.display = "block";
  try {
    const response = await fetch("/webhook", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        query: question,
        auto_approve_hitl: document.getElementById("approve").checked
      })
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "The request failed.");
    answer.innerHTML = renderMarkdown(data.answer || "No answer was returned.");
  } catch (error) {
    answer.className = "error";
    answer.textContent = "Error: " + error.message;
  } finally {
    button.disabled = false;
  }
}
</script>
</body>
</html>
"""


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "researchbot", "version": "1.0.0"}


@app.post("/webhook", response_model=QueryResponse)
async def run_agent_sync(request: QueryRequest):
    """
    Synchronous endpoint — blocks until the agent produces a final answer.
    HITL prompts are auto-approved when `auto_approve_hitl=True`.
    """
    thread_id = request.thread_id or str(uuid.uuid4())

    try:
        loop = asyncio.get_event_loop()
        answer = await loop.run_in_executor(
            executor,
            _run_agent_blocking,
            request.query,
            thread_id,
            request.auto_approve_hitl,
        )
        return QueryResponse(thread_id=thread_id, answer=answer)
    except Exception as e:
        log.error(f"Agent error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/webhook/async", response_model=AsyncJobResponse)
async def run_agent_async(request: QueryRequest, background_tasks: BackgroundTasks):
    """
    Async endpoint — returns immediately with a thread_id.
    Poll GET /status/{thread_id} for results.
    """
    thread_id = request.thread_id or str(uuid.uuid4())
    _jobs[thread_id] = {"status": "running", "answer": None, "error": None}

    background_tasks.add_task(
        _run_agent_background,
        request.query,
        thread_id,
        request.auto_approve_hitl,
    )

    return AsyncJobResponse(
        thread_id=thread_id,
        status="running",
        message=f"Agent started. Poll /status/{thread_id} for results.",
    )


@app.get("/status/{thread_id}", response_model=JobStatus)
async def get_status(thread_id: str):
    if thread_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = _jobs[thread_id]
    return JobStatus(thread_id=thread_id, **job)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_agent_blocking(query: str, thread_id: str, auto_approve: bool) -> str:
    """Thin wrapper that either runs interactively or auto-approves HITL."""
    if auto_approve:
        return _run_with_auto_approve(query, thread_id)
    else:
        from agent import run_agent
        return run_agent(query, thread_id)


def _run_with_auto_approve(query: str, thread_id: str) -> str:
    """Run agent, automatically approving all HITL interrupts."""
    from langchain_core.messages import HumanMessage
    from agent import build_graph
    from state import AgentState

    graph = build_graph()
    config = {"configurable": {"thread_id": thread_id}}
    initial: AgentState = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "search_results": [],
        "summary": None,
        "tool_call_count": 0,
        "pending_action": None,
        "hitl_approved": None,
        "final_answer": None,
        "error": None,
        "trace_id": None,
    }

    result = graph.invoke(initial, config=config)

    # Auto-approve all interrupts
    while result.get("__interrupt__"):
        log.info("[auto-approve] Approving HITL interrupt automatically")
        graph.update_state(config, {"hitl_approved": True}, as_node="hitl_approval")
        result = graph.invoke(None, config=config)

    return result.get("final_answer", "No answer produced.")


async def _run_agent_background(query: str, thread_id: str, auto_approve: bool):
    """Background task wrapper for async endpoint."""
    loop = asyncio.get_event_loop()
    try:
        answer = await loop.run_in_executor(
            executor, _run_agent_blocking, query, thread_id, auto_approve
        )
        _jobs[thread_id] = {"status": "completed", "answer": answer, "error": None}
    except Exception as e:
        _jobs[thread_id] = {"status": "failed", "answer": None, "error": str(e)}


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    from config import cfg

    uvicorn.run("api:app", host=cfg.HOST, port=cfg.PORT, reload=False)
