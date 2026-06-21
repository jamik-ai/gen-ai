"""Оркестратор: Classifier -> Responder (tool-agent, RAG) -> Judge -> hallucination check.

Запуск:
    python pipeline.py input/test_tickets.csv output/
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

from agents.classifier import classify_ticket
from agents.judge import judge_reply
from agents.responder import run_responder
from hallucination import hallucination_report

REWORK_THRESHOLD = 0.6


def process_ticket(ticket_id: str, description: str, gold_category: str | None = None) -> dict:
    t0 = time.time()

    extraction = classify_ticket(description)

    resp = run_responder(description, extraction.category)
    reply = resp["reply"]

    judge = judge_reply(
        description=description,
        category=extraction.category,
        kb_chunks=resp["retrieved_chunks"],
        reply=reply.reply,
        cited_facts=reply.cited_facts,
    )

    reworked = False
    if judge.overall_score < REWORK_THRESHOLD:
        reworked = True
        feedback = judge.summary or "обнаружены не подтверждённые KB утверждения"
        resp2 = run_responder(description, extraction.category, feedback=feedback)
        reply2 = resp2["reply"]
        judge2 = judge_reply(
            description=description,
            category=extraction.category,
            kb_chunks=resp2["retrieved_chunks"],
            reply=reply2.reply,
            cited_facts=reply2.cited_facts,
        )
        # merge retrieved chunks/usage across both attempts for honest path accounting
        merged_chunks = {**resp["retrieved_chunks"], **resp2["retrieved_chunks"]}
        n_tool_calls = resp["n_tool_calls"] + resp2["n_tool_calls"]
        n_steps = resp["n_steps"] + resp2["n_steps"]
        usage = {
            "prompt_tokens": resp["usage"]["prompt_tokens"] + resp2["usage"]["prompt_tokens"],
            "completion_tokens": resp["usage"]["completion_tokens"] + resp2["usage"]["completion_tokens"],
            "cost_usd": round(resp["usage"]["cost_usd"] + resp2["usage"]["cost_usd"], 6),
        }
        reply, judge, retrieved_chunks = reply2, judge2, merged_chunks
    else:
        n_tool_calls, n_steps, usage = resp["n_tool_calls"], resp["n_steps"], resp["usage"]
        retrieved_chunks = resp["retrieved_chunks"]

    hreport = hallucination_report(reply.reply, reply.cited_facts, retrieved_chunks)

    elapsed = round(time.time() - t0, 2)

    result = {
        "ticket_id": ticket_id,
        "description": description,
        "gold_category": gold_category,
        "extraction": extraction.model_dump(),
        "reply": reply.model_dump(),
        "judge": judge.model_dump(),
        "hallucination": hreport,
        "retrieved_chunks": retrieved_chunks,
        "path": {
            "n_llm_calls": 2 + (2 if reworked else 0),  # classify + judge, x2 если был rework
            "n_tool_calls": n_tool_calls,
            "n_responder_steps": n_steps,
            "reworked": reworked,
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "cost_usd": usage["cost_usd"],
            "elapsed_s": elapsed,
        },
        "classification_correct": (gold_category == extraction.category) if gold_category else None,
    }
    return result


def main() -> None:
    in_csv = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("input/test_tickets.csv")
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("output")
    traces_dir = out_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    with in_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Тикетов к обработке: {len(rows)}")
    for i, row in enumerate(rows, 1):
        ticket_id = row["ticket_id"]
        print(f"[{i}/{len(rows)}] {ticket_id} ...", end=" ", flush=True)
        result = process_ticket(ticket_id, row["description"], row.get("category"))
        (traces_dir / f"{ticket_id}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2)
        )
        ok = "OK" if result["classification_correct"] else "MISS"
        print(f"pred={result['extraction']['category']} gold={row.get('category')} [{ok}] "
              f"judge={result['judge']['overall_score']:.2f} rework={result['path']['reworked']}")

    print(f"\nГотово. Трейсы -> {traces_dir}")


if __name__ == "__main__":
    main()
