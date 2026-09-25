"""Interactive CLI for the Multi-Agent RAG System.

Usage:
    python cli.py
"""

from __future__ import annotations

import sys

from agents.manager_agent import ManagerAgent

EXIT_COMMANDS = {"exit", "quit"}

BANNER = (
    "Multi-Agent RAG System - ask a qualitative, quantitative, or combined "
    "question about the company. Type 'exit' or 'quit' to leave.\n"
)


def format_result(result) -> str:
    if result.clarification_needed:
        return f"[Clarification needed] {result.final_answer}"
    return result.final_answer


def run(manager: ManagerAgent, input_stream=sys.stdin, output_stream=sys.stdout) -> None:
    output_stream.write(BANNER)
    output_stream.flush()

    while True:
        output_stream.write("\n> ")
        output_stream.flush()

        line = input_stream.readline()
        if not line:  # EOF (Ctrl-D / piped input exhausted)
            break

        query = line.strip()
        if not query:
            continue
        if query.lower() in EXIT_COMMANDS:
            break

        result = manager.handle(query)
        output_stream.write(f"\n{format_result(result)}\n")
        output_stream.flush()


if __name__ == "__main__":
    run(ManagerAgent())
