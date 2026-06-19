"""
Чистая демонстрация: сколько итераций/вызовов экономит Schema-валидатор,
когда Планировщик (naive_planner, без анти-галлюцинаторного правила)
придумывает несуществующий инструмент.

Запуск:
    python demo_validator_savings.py
"""
from __future__ import annotations

import json
from pathlib import Path

from naive_planner import naive_planner
from orchestrator import run_pwc

Q = "Какой объём денежной массы М2 в России на конец 2024 года, и насколько он вырос за год?"


def main():
    out = {}

    r_no_val = run_pwc(Q, use_validator=False, planner_fn=naive_planner, max_iter=2, verbose=False)
    out["no_validator"] = {
        "answer": r_no_val.get("answer"),
        "error": r_no_val.get("error"),
        "iterations": r_no_val.get("iterations"),
        "elapsed_sec": r_no_val.get("elapsed_sec"),
        "trace_len": len(r_no_val.get("trace", [])),
        "n_worker_calls": sum(1 for t in r_no_val.get("trace", []) if t.get("kind") == "worker"),
    }

    r_val = run_pwc(Q, use_validator=True, planner_fn=naive_planner, max_iter=2, verbose=False)
    out["with_validator"] = {
        "answer": r_val.get("answer"),
        "error": r_val.get("error"),
        "iterations": r_val.get("iterations"),
        "elapsed_sec": r_val.get("elapsed_sec"),
        "trace_len": len(r_val.get("trace", [])),
        "n_worker_calls": sum(1 for t in r_val.get("trace", []) if t.get("kind") == "worker"),
    }

    for k, v in out.items():
        print(f"--- {k} ---")
        for kk, vv in v.items():
            print(f"  {kk}: {vv}")
        print()

    Path(__file__).with_name("validator_savings_demo.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
