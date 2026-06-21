"""Eval: считает correctness и path-метрики по трейсам в output/traces/*.json.

Запуск (после python pipeline.py):
    python eval.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

JUDGE_PASS_THRESHOLD = 0.6

ROOT = Path(__file__).resolve().parent
TRACES_DIR = ROOT / "output" / "traces"
TABLE_CSV = ROOT / "output" / "eval_table.csv"
SUMMARY_JSON = ROOT / "output" / "eval_summary.json"


def load_traces() -> list[dict]:
    traces = [json.loads(p.read_text()) for p in sorted(TRACES_DIR.glob("TCK-*.json"))]
    if not traces:
        sys.exit(f"Нет трейсов в {TRACES_DIR} — сначала запусти: python pipeline.py")
    return traces


def main() -> None:
    traces = load_traces()
    n = len(traces)

    rows = []
    for t in traces:
        rows.append({
            "ticket_id": t["ticket_id"],
            "gold_category": t["gold_category"],
            "pred_category": t["extraction"]["category"],
            "classification_correct": t["classification_correct"],
            "judge_overall_score": t["judge"]["overall_score"],
            "judge_pass": t["judge"]["overall_score"] >= JUDGE_PASS_THRESHOLD,
            "category_consistent": t["judge"]["category_consistent"],
            "ghost_number_rate": t["hallucination"]["ghost_number_rate"],
            "ghost_fact_rate": t["hallucination"]["ghost_fact_rate"],
            "no_kb_retrieved": t["hallucination"]["no_kb_retrieved"],
            "reworked": t["path"]["reworked"],
            "n_llm_calls": t["path"]["n_llm_calls"],
            "n_tool_calls": t["path"]["n_tool_calls"],
            "prompt_tokens": t["path"]["prompt_tokens"],
            "completion_tokens": t["path"]["completion_tokens"],
            "cost_usd": t["path"]["cost_usd"],
            "elapsed_s": t["path"]["elapsed_s"],
        })

    TABLE_CSV.parent.mkdir(parents=True, exist_ok=True)
    with TABLE_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    accuracy = sum(r["classification_correct"] for r in rows) / n
    judge_pass_rate = sum(r["judge_pass"] for r in rows) / n
    avg_judge_score = sum(r["judge_overall_score"] for r in rows) / n
    category_consistency_rate = sum(r["category_consistent"] for r in rows) / n
    rework_rate = sum(r["reworked"] for r in rows) / n
    total_numeric_claims = sum(t["hallucination"]["total_numeric_claims"] for t in traces)
    total_ghost_numbers = sum(len(t["hallucination"]["ghost_numbers"]) for t in traces)
    total_cited_facts = sum(t["hallucination"]["total_cited_facts"] for t in traces)
    total_ghost_facts = sum(len(t["hallucination"]["ghost_cited_facts"]) for t in traces)
    no_kb_rate = sum(r["no_kb_retrieved"] for r in rows) / n

    # confusion matrix по категориям
    confusion: dict[str, dict[str, int]] = {}
    for r in rows:
        confusion.setdefault(r["gold_category"], {}).setdefault(r["pred_category"], 0)
        confusion[r["gold_category"]][r["pred_category"]] += 1

    total_cost = sum(r["cost_usd"] for r in rows)
    total_prompt_tok = sum(r["prompt_tokens"] for r in rows)
    total_completion_tok = sum(r["completion_tokens"] for r in rows)
    avg_llm_calls = sum(r["n_llm_calls"] for r in rows) / n
    avg_tool_calls = sum(r["n_tool_calls"] for r in rows) / n
    avg_elapsed = sum(r["elapsed_s"] for r in rows) / n

    summary = {
        "n_tickets": n,
        "correctness": {
            "classification_accuracy": round(accuracy, 3),
            "judge_pass_rate": round(judge_pass_rate, 3),
            "avg_judge_overall_score": round(avg_judge_score, 3),
            "category_consistency_rate": round(category_consistency_rate, 3),
            "confusion_matrix": confusion,
        },
        "hallucination": {
            "ghost_number_rate_overall": round(total_ghost_numbers / total_numeric_claims, 3) if total_numeric_claims else 0.0,
            "total_numeric_claims": total_numeric_claims,
            "total_ghost_numbers": total_ghost_numbers,
            "ghost_fact_rate_overall": round(total_ghost_facts / total_cited_facts, 3) if total_cited_facts else 0.0,
            "total_cited_facts": total_cited_facts,
            "total_ghost_facts": total_ghost_facts,
            "no_kb_retrieved_rate": round(no_kb_rate, 3),
        },
        "path": {
            "rework_rate": round(rework_rate, 3),
            "avg_llm_calls_per_ticket": round(avg_llm_calls, 2),
            "avg_tool_calls_per_ticket": round(avg_tool_calls, 2),
            "avg_elapsed_s_per_ticket": round(avg_elapsed, 2),
            "total_prompt_tokens": total_prompt_tok,
            "total_completion_tokens": total_completion_tok,
            "total_cost_usd": round(total_cost, 4),
        },
    }

    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print(f"Тикетов: {n}")
    print(f"Classification accuracy: {accuracy:.1%}")
    print(f"Judge pass-rate (>= {JUDGE_PASS_THRESHOLD}): {judge_pass_rate:.1%}  (avg score {avg_judge_score:.2f})")
    print(f"Category consistency (judge): {category_consistency_rate:.1%}")
    print(f"Rework rate: {rework_rate:.1%}")
    print(f"Ghost numbers: {total_ghost_numbers}/{total_numeric_claims} ({summary['hallucination']['ghost_number_rate_overall']:.1%})")
    print(f"Ghost cited facts: {total_ghost_facts}/{total_cited_facts} ({summary['hallucination']['ghost_fact_rate_overall']:.1%})")
    print(f"No relevant KB retrieved: {no_kb_rate:.1%} тикетов")
    print(f"Avg LLM calls/ticket: {avg_llm_calls:.2f}, avg tool calls/ticket: {avg_tool_calls:.2f}")
    print(f"Стоимость прогона: ${total_cost:.4f} ({total_prompt_tok} prompt + {total_completion_tok} completion токенов)")
    print(f"\nConfusion matrix (gold -> pred):")
    for gold in sorted(confusion):
        print(f"  {gold}: {confusion[gold]}")
    print(f"\n-> {TABLE_CSV}\n-> {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
