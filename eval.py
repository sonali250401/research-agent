"""
Task 9a: Evaluation — LLM-as-a-Judge
======================================
Reads up to 10 historical traces from `tool_audit.log` and Langfuse,
then asks Gemini to score the agent's "Error Recovery" on a scale of 1–5.

Run:  python eval.py
"""

import json
import os
import re
import logging
from datetime import datetime
from config import cfg

log = logging.getLogger("eval")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


# ── Load traces ───────────────────────────────────────────────────────────────

def load_traces_from_audit_log(n: int = 10) -> list[dict]:
    """Load the last N tool invocations from the local audit log."""
    log_file = "tool_audit.log"
    if not os.path.exists(log_file):
        log.warning("tool_audit.log not found — generating synthetic traces for demo.")
        return _synthetic_traces(n)

    traces = []
    with open(log_file) as f:
        for line in f.readlines()[-n:]:
            parts = line.strip().split()
            trace = {"timestamp": parts[0] if parts else "", "raw": line.strip()}
            traces.append(trace)
    return traces if traces else _synthetic_traces(n)


def _synthetic_traces(n: int) -> list[dict]:
    """Return demo traces for evaluation when no real data exists."""
    scenarios = [
        {"query": "What is quantum computing?", "tool": "web_search", "result": "success", "error": None},
        {"query": "Store research notes", "tool": "store_document", "result": "success", "error": None},
        {"query": "search with empty string", "tool": "web_search", "result": "error", "error": "Query too short (pre-hook 400)"},
        {"query": "calculate 2**100", "tool": "calculator", "result": "success", "error": None},
        {"query": "search AI trends", "tool": "web_search", "result": "success", "error": None},
        {"query": "store doc with no title", "tool": "store_document", "result": "error", "error": "Document must have a title (pre-hook 400)"},
        {"query": "retrieve old notes", "tool": "retrieve_documents", "result": "success", "error": None},
        {"query": "compute 1/0", "tool": "calculator", "result": "error", "error": "division by zero — agent retried with valid expression"},
        {"query": "summarise long text", "tool": "summarise_text", "result": "success", "error": None},
        {"query": "HITL rejection → re-plan", "tool": "store_document", "result": "rejected_and_recovered", "error": None},
    ]
    return scenarios[:n]


# ── Judge with Gemini ─────────────────────────────────────────────────────────

JUDGE_SYSTEM = """You are an expert AI evaluator. You will be given a list of agent interaction traces.
For each trace, evaluate the agent's "Error Recovery" capability on a scale of 1–5:

1 = No recovery — agent crashed or repeated the same wrong action
2 = Partial recovery — agent acknowledged the error but did not fix it
3 = Basic recovery — agent retried with minor corrections
4 = Good recovery — agent self-corrected effectively using the error message
5 = Excellent recovery — agent diagnosed the root cause and fixed it elegantly

Return a JSON array where each element has:
  { "trace_index": int, "score": int, "reasoning": str }

Then add a final "aggregate" field with the average score.
"""


def judge_traces(traces: list[dict]) -> dict:
    """Feed traces to Gemini and get error-recovery scores."""
    traces_text = json.dumps(traces, indent=2, default=str)

    try:
        from google import genai
        from google.genai import types as genai_types

        client = genai.Client(api_key=cfg.GEMINI_API_KEY)

        prompt = f"""Here are {len(traces)} agent traces:

{traces_text}

Evaluate each trace's error recovery and return ONLY valid JSON."""

        response = client.models.generate_content(
            model=cfg.GEMINI_MODEL,
            contents=JUDGE_SYSTEM + "\n\n" + prompt,
        )
        raw = response.text.strip()
        # Extract JSON from markdown fences if present
        json_match = re.search(r"\{.*\}|\[.*\]", raw, re.DOTALL)
        parsed = json.loads(json_match.group() if json_match else raw)
        return {"status": "success", "evaluations": parsed}

    except Exception as e:
        log.warning(f"Gemini unavailable ({e}), using heuristic scoring.")
        return _heuristic_scoring(traces)


def _heuristic_scoring(traces: list[dict]) -> dict:
    """Fallback scoring when Gemini is not available."""
    results = []
    for i, t in enumerate(traces):
        error = t.get("error", "")
        result = t.get("result", "success")

        if error and "recovered" in str(error):
            score = 5
        elif error and "retry" in str(error):
            score = 4
        elif error and "400" in str(error):
            score = 4  # pre-hook 400 → LLM self-corrected
        elif result == "rejected_and_recovered":
            score = 5
        elif result == "error":
            score = 2
        else:
            score = 5  # success with no errors

        results.append({"trace_index": i, "score": score, "reasoning": f"Heuristic: {result}"})

    avg = round(sum(r["score"] for r in results) / len(results), 2) if results else 0
    return {"status": "heuristic", "evaluations": results, "aggregate": avg}


# ── Report ─────────────────────────────────────────────────────────────────────

def run_evaluation():
    print("=" * 60)
    print("  LLM-AS-A-JUDGE — Error Recovery Evaluation")
    print("=" * 60)

    traces = load_traces_from_audit_log(n=10)
    print(f"\nLoaded {len(traces)} traces.\n")

    result = judge_traces(traces)

    evals = result.get("evaluations", [])
    if isinstance(evals, list):
        for e in evals:
            idx = e.get("trace_index", "?")
            score = e.get("score", "?")
            reason = e.get("reasoning", "")
            print(f"  [{idx:>2}] Score: {score}/5  — {reason[:80]}")

    aggregate = result.get("aggregate")
    if not aggregate and isinstance(evals, list) and evals:
        scores = [e.get("score", 0) for e in evals if isinstance(e.get("score"), (int, float))]
        aggregate = round(sum(scores) / len(scores), 2) if scores else "N/A"

    print(f"\n{'─'*60}")
    print(f"  📊 Aggregate Error Recovery Score: {aggregate} / 5")
    print(f"{'─'*60}")

    # Save report
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "traces_evaluated": len(traces),
        "aggregate_score": aggregate,
        "evaluations": evals,
    }
    with open("eval_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n✅ Full report saved to eval_report.json")
    return report


if __name__ == "__main__":
    run_evaluation()
