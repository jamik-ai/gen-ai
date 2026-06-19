"""
Замер ускорения от параллельного исполнения уровня подвопросов
(ДЗ С6, часть 2).

Фиксируем ОДИН план (чтобы сравнение было честным — без вариативности
Планировщика между прогонами), затем гоняем Исполнителей по уровням:
последовательно vs через ThreadPoolExecutor.

Запуск:
    python bench_parallel.py
"""
from __future__ import annotations

import time

from orchestrator import _topological_levels, execute_level
from planner import planner


def bench(question: str, *, repeats: int = 3) -> dict:
    plan = planner(question)
    levels = _topological_levels(plan.subquestions)

    seq_times = []
    par_times = []

    for _ in range(repeats):
        t0 = time.monotonic()
        answers: dict = {}
        for level in levels:
            answers.update(execute_level(level, answers, parallel=False))
        seq_times.append(time.monotonic() - t0)

    for _ in range(repeats):
        t0 = time.monotonic()
        answers = {}
        for level in levels:
            answers.update(execute_level(level, answers, parallel=True))
        par_times.append(time.monotonic() - t0)

    seq_avg = sum(seq_times) / len(seq_times)
    par_avg = sum(par_times) / len(par_times)
    return {
        "question": question,
        "n_subquestions": len(plan.subquestions),
        "max_level_width": max(len(l) for l in levels),
        "seq_times": seq_times,
        "par_times": par_times,
        "seq_avg": seq_avg,
        "par_avg": par_avg,
        "speedup": seq_avg / par_avg if par_avg else float("inf"),
    }


if __name__ == "__main__":
    cases = [
        "Во сколько раз USD подорожал с 1 января 2022 по сегодня?",
        "Какие сейчас курсы USD, EUR и CNY к рублю? Просто перечисли все три.",
    ]
    for q in cases:
        r = bench(q)
        print(f"\n=== {q}")
        print(f"  подвопросов: {r['n_subquestions']}, макс. ширина уровня: {r['max_level_width']}")
        print(f"  последовательно: {r['seq_avg']:.2f}с  {r['seq_times']}")
        print(f"  параллельно:     {r['par_avg']:.2f}с  {r['par_times']}")
        print(f"  ускорение: {r['speedup']:.2f}x")
