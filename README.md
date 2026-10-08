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

## Live Deployment

ResearchBot is deployed on Render using the included Dockerfile.

- **Application:** https://research-agent-igk9.onrender.com/
- **API documentation:** https://research-agent-igk9.onrender.com/docs
- **Health check:** https://research-agent-igk9.onrender.com/health

The live health check returns HTTP 200:

```json
{"status":"healthy","service":"researchbot","version":"1.0.0"}
```

Open the application URL, enter a question, and click **Ask ResearchBot**.

## API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Browser question interface |
| `/webhook` | POST | Run one research question synchronously |
| `/webhook/async` | POST | Start a background research job |
| `/status/{thread_id}` | GET | Check a background job |
| `/health` | GET | Deployment health check |
| `/docs` | GET | Swagger API documentation |

Example request:

```bash
curl -X POST https://research-agent-igk9.onrender.com/webhook \
  -H "Content-Type: application/json" \
  -d '{"query":"What are the latest developments in AI?","auto_approve_hitl":true}'
```

## Local API

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
& "C:\Users\SONALI\AppData\Local\Programs\Python\Python311\python.exe" -m uvicorn api:app --host 0.0.0.0 --port 8080
```

For local development, open `http://127.0.0.1:8080/`. Do not open
`http://0.0.0.0`; it is a server bind address, not a browser destination.

## Evaluation

The project includes an LLM-as-a-judge evaluation script in `eval.py`.
It loads up to ten tool-audit traces, asks Gemini to score error recovery from
1 to 5, and falls back to deterministic heuristic scoring when Gemini is not
configured.

Run the evaluation locally:

```powershell
& "C:\Users\SONALI\AppData\Local\Programs\Python\Python311\python.exe" eval.py
```

The latest saved evaluation report contains:

- **Traces evaluated:** 10
- **Aggregate error-recovery score:** 4.5 / 5
- **Scoring mode:** heuristic fallback for the recorded run
- **Output file:** `eval_report.json`

The evaluation covered successful tool calls, pre-tool validation errors,
calculator errors, document-storage validation, and HITL rejection with
recovery.

## Environment Variables

Set these as environment variables locally or in the Render service settings.
Never commit `.env` or real credentials.

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Groq API key |
| `GROQ_MODEL` | Groq model; use `openai/gpt-oss-120b` |
| `GEMINI_API_KEY` | Google Gemini API key |
| `TAVILY_API_KEY` | Tavily search API key |
| `QDRANT_URL` | Qdrant server URL |
| `QDRANT_API_KEY` | Qdrant API key |
| `LANGFUSE_PUBLIC_KEY` | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | Langfuse secret key |
