---
title: ResearchBot Agent
emoji: 🤖
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# ResearchBot — LangGraph Research Agent

A production-grade AI research assistant built with all 9 architectural tasks:

| Task | Feature |
|------|---------|
| 1 | State management with `AgentState` TypedDict |
| 2 | Groq LLM reasoning node |
| 3 | Tool provisioning & JSON-RPC execution |
| 4 | MCP (Model Context Protocol) via stdio |
| 5 | Pre/post lifecycle hooks with 400-error self-correction |
| 6 | Human-in-the-Loop approval with LangGraph interrupts |
| 7 | Context compaction + SQLite checkpointing |
| 8 | Langfuse observability (latency, tokens, payloads) |
| 9 | LLM-as-judge eval + FastAPI deployment |

## API

### Run locally

Start the API from the project directory:

```bash
.\start_server.bat
```

Open `http://127.0.0.1:8080/` or `http://localhost:8080/` in a browser.
Do not open `http://0.0.0.0:8000/`; `0.0.0.0` is the server bind address, not
a client destination.

The home page includes a question box. Type your question, leave
“Automatically approve actions” enabled for local testing, and click
“Ask ResearchBot”. You can also use the interactive API at
`http://127.0.0.1:8080/docs`.

To open the server in a separate command window that stays running, use:

```powershell
.\start_server_window.bat
```

The launcher uses the installed Python 3.11 executable directly. If you prefer
to start it manually in PowerShell, use:

```powershell
& "C:\Users\SONALI\AppData\Local\Programs\Python\Python311\python.exe" -m uvicorn api:app --host 0.0.0.0 --port 8000
```

```bash
# Synchronous run
curl -X POST https://your-space.hf.space/webhook \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the latest AI breakthroughs?", "auto_approve_hitl": true}'

# Health check
curl https://your-space.hf.space/health
```

For Render, use the Docker runtime and let Render provide `PORT`; the
container command reads that variable automatically. For Hugging Face Docker
Spaces, the default is port `7860`. The public URL is supplied by the
platform; `0.0.0.0` should only be used as the server bind address.

## Environment Variables (set in HF Spaces Secrets)

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Groq API key |
| `GEMINI_API_KEY` | Google Gemini API key |
| `TAVILY_API_KEY` | Tavily search API key |
| `LANGFUSE_PUBLIC_KEY` | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | Langfuse secret key |
| `QDRANT_URL` | Qdrant vector DB URL |
| `QDRANT_API_KEY` | Qdrant API key |
