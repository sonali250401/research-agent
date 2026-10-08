"""
CLI entrypoint — run the agent interactively from the terminal.
Usage:  python main.py
        python main.py "What are the latest AI breakthroughs in 2025?"
"""

import sys
import logging
import uuid

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        print("\n" + "═" * 60)
        print("  🤖  ResearchBot — LangGraph Agent (All 9 Tasks)")
        print("═" * 60)
        query = input("\nEnter your research question: ").strip()
        if not query:
            print("No query provided. Exiting.")
            sys.exit(0)

    thread_id = str(uuid.uuid4())
    print(f"\n📡 Starting agent  (thread={thread_id[:8]}…)\n")

    from agent import run_agent
    answer = run_agent(query, thread_id)

    print("\n" + "═" * 60)
    print("  ✅  FINAL ANSWER")
    print("═" * 60)
    print(answer)
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
