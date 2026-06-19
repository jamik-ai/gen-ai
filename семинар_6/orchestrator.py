"""
Оркестратор: главный цикл Планировщик-Исполнитель-Критик.

ДЗ к семинару 6 добавляет:
- Schema-валидатор между Планировщиком и Исполнителем (validate_plan).
- Параллельное исполнение подвопросов одного уровня зависимостей
  (_topological_levels + execute_level, ThreadPoolExecutor).

Важно: max_iter защищает от бесконечного цикла, если Критик
постоянно говорит «переделай».
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from critic import critic
from llm_client import get_model, make_raw_client
from planner import planner
from schemas_pwc import Plan, SubQuestion, WorkerAnswer
from validator import validate_plan
from worker import worker


def _topological_sort(subqs: list[SubQuestion]) -> list[SubQuestion]:
    """Отсортировать подвопросы так, чтобы depends_on шли раньше."""
    by_id = {s.id: s for s in subqs}
    ordered: list[SubQuestion] = []
    visited: set[int] = set()

    def visit(node_id: int, path: list[int]):
        if node_id in visited:
            return None
        if node_id in path:
            raise ValueError(f"Цикл в depends_on: {path + [node_id]}")
        if node_id not in by_id:
            return None
        for dep in by_id[node_id].depends_on:
            visit(dep, path + [node_id])
        visited.add(node_id)
        ordered.append(by_id[node_id])

    for sq in subqs:
        visit(sq.id, [])
    return ordered


def _topological_levels(subqs: list[SubQuestion]) -> list[list[SubQuestion]]:
    """Сгруппировать подвопросы в уровни зависимостей.

    Внутри уровня подвопросы независимы друг от друга (можно исполнять
    параллельно). Между уровнями есть зависимость — уровень N+1 может
    читать ответы только уровней <= N.
    """
    by_id = {s.id: s for s in subqs}
    level_of: dict[int, int] = {}

    def depth(node_id: int, path: list[int]) -> int:
        if node_id in level_of:
            return level_of[node_id]
        if node_id in path:
            raise ValueError(f"Цикл в depends_on: {path + [node_id]}")
        if node_id not in by_id:
            return 0
        d = 0
        for dep in by_id[node_id].depends_on:
            d = max(d, depth(dep, path + [node_id]) + 1)
        level_of[node_id] = d
        return d

    for sq in subqs:
        depth(sq.id, [])

    levels: dict[int, list[SubQuestion]] = {}
    for sq in subqs:
        levels.setdefault(level_of[sq.id], []).append(sq)

    return [levels[k] for k in sorted(levels)]


def execute_level(
    level: list[SubQuestion],
    prev_answers: dict[int, WorkerAnswer],
    *,
    parallel: bool = True,
) -> dict[int, WorkerAnswer]:
    """Прогнать все подвопросы уровня (параллельно, если их больше одного)."""
    if not parallel or len(level) <= 1:
        return {sq.id: worker(sq, prev_answers) for sq in level}

    results: dict[int, WorkerAnswer] = {}
    with ThreadPoolExecutor(max_workers=len(level)) as pool:
        futures = {pool.submit(worker, sq, prev_answers): sq.id for sq in level}
        for fut, sq_id in futures.items():
            results[sq_id] = fut.result()
    return results


def _synthesize(
    question: str,
    plan: Plan,
    answers: dict[int, WorkerAnswer],
) -> str:
    """Собрать финальный ответ одним LLM-вызовом без tools."""
    parts = [f"  {i}. {answers[i].answer}" for i in sorted(answers)]
    answers_block = "\n".join(parts) or "(ответов нет)"

    client = make_raw_client()
    resp = client.chat.completions.create(
        model=get_model(),
        messages=[
            {
                "role": "system",
                "content": (
                    "Ты собираешь финальный ответ пользователю из ответов "
                    "на подвопросы. Не придумывай новых чисел, только "
                    "переформулируй и объедини то, что дано. 1-2 фразы."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Исходный вопрос: {question}\n\n"
                    f"Ответы на подвопросы:\n{answers_block}\n\n"
                    "Собери из этого короткий финальный ответ."
                ),
            },
        ],
        temperature=0.0,
    )
    content = resp.choices[0].message.content
    return (content or "").strip() or " · ".join(p.strip() for p in parts)


def run_pwc(
    question: str,
    *,
    max_iter: int = 3,
    verbose: bool = True,
    use_validator: bool = True,
    parallel: bool = True,
    max_validator_retries: int = 2,
    planner_fn=planner,
) -> dict[str, Any]:
    """Запустить цикл Планировщик-Исполнитель-Критик.

    use_validator: включает Schema-валидатор плана (домашка С6, часть 1).
    parallel: исполнять подвопросы одного уровня зависимостей через
              ThreadPoolExecutor (домашка С6, часть 2). При False —
              тот же порядок, но строго по одному (для замера ускорения).
    planner_fn: какую функцию-планировщик звать (по умолчанию — боевой
                planner из planner.py). Параметр нужен только для
                эксперимента с naive_planner в ДЗ С6, часть 1.
    """
    trace: list[dict[str, Any]] = []
    t_start = time.monotonic()

    plan = planner_fn(question)
    trace.append(
        {
            "iter": 0,
            "kind": "plan",
            "reasoning": plan.reasoning,
            "subquestions": [sq.model_dump() for sq in plan.subquestions],
        }
    )

    if use_validator:
        retries = 0
        errors = validate_plan(plan)
        while errors and retries < max_validator_retries:
            trace.append({"iter": 0, "kind": "validator_reject", "errors": errors})
            if verbose:
                print(f"\n[validator ❌] {errors}")
            plan = planner_fn(
                question, feedback=f"Инструменты не существуют: {errors}"
            )
            trace.append(
                {
                    "iter": 0,
                    "kind": "plan_after_validator",
                    "reasoning": plan.reasoning,
                    "subquestions": [sq.model_dump() for sq in plan.subquestions],
                }
            )
            errors = validate_plan(plan)
            retries += 1
        if errors:
            # Планировщик так и не выдал валидный план — не отправляем
            # Исполнителю заведомо битый план с выдуманными инструментами.
            trace.append({"iter": 0, "kind": "validator_giveup", "errors": errors})
            return {
                "answer": None,
                "error": f"валидатор не пропустил план после {retries} попыток: {errors}",
                "plan": plan,
                "answers": {},
                "trace": trace,
                "iterations": 0,
                "elapsed_sec": time.monotonic() - t_start,
            }

    if verbose:
        print(f"\n[plan] {plan.reasoning}")
        for sq in plan.subquestions:
            print(f"  {sq.id}. [{','.join(sq.expected_tools)}] {sq.question}")

    for iter_num in range(1, max_iter + 1):
        answers: dict[int, WorkerAnswer] = {}
        levels = _topological_levels(plan.subquestions)
        for level in levels:
            level_answers = execute_level(level, answers, parallel=parallel)
            answers.update(level_answers)
            for sq in level:
                ans = level_answers[sq.id]
                trace.append(
                    {
                        "iter": iter_num,
                        "kind": "worker",
                        "sq_id": sq.id,
                        "used_tools": ans.used_tools,
                        "answer": ans.answer,
                    }
                )
                if verbose:
                    print(f"  [{sq.id}] → {ans.answer}   tools={ans.used_tools}")

        verdict = critic(question, plan, answers)
        trace.append(
            {
                "iter": iter_num,
                "kind": "verdict",
                "ok": verdict.ok,
                "action": verdict.action,
                "reason": verdict.reason,
                "rework_ids": verdict.rework_ids,
            }
        )

        if verbose:
            mark = "✅" if verdict.ok else "❌"
            print(f"  [critic {mark}] {verdict.action}: {verdict.reason}")

        if verdict.ok:
            final = _synthesize(question, plan, answers)
            return {
                "answer": final,
                "plan": plan,
                "answers": answers,
                "trace": trace,
                "iterations": iter_num,
                "elapsed_sec": time.monotonic() - t_start,
            }

        if iter_num == max_iter:
            break

        if verdict.action == "replan":
            plan = planner_fn(question, feedback=verdict.reason)
            trace.append(
                {
                    "iter": iter_num,
                    "kind": "replan",
                    "reasoning": plan.reasoning,
                    "subquestions": [sq.model_dump() for sq in plan.subquestions],
                }
            )
            if verbose:
                print(f"\n[replan] {plan.reasoning}")
                for sq in plan.subquestions:
                    print(f"  {sq.id}. [{','.join(sq.expected_tools)}] {sq.question}")
        elif verdict.action == "rework":
            feedback = (
                f"Критик просит переделать подвопросы {verdict.rework_ids}: "
                f"{verdict.reason}"
            )
            plan = planner_fn(question, feedback=feedback)
            trace.append(
                {
                    "iter": iter_num,
                    "kind": "rework_replan",
                    "rework_ids": verdict.rework_ids,
                    "reasoning": plan.reasoning,
                    "subquestions": [sq.model_dump() for sq in plan.subquestions],
                }
            )
            if verbose:
                print(f"\n[rework→replan] {plan.reasoning}")
        else:
            break

    return {
        "answer": None,
        "error": f"не удалось получить вердикт 'accept' за {max_iter} итераций",
        "plan": plan,
        "answers": answers,
        "trace": trace,
        "iterations": max_iter,
        "elapsed_sec": time.monotonic() - t_start,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="+", help="Вопрос к агенту")
    ap.add_argument("--max-iter", type=int, default=3)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--no-validator", action="store_true", help="Отключить Schema-валидатор")
    ap.add_argument("--sequential", action="store_true", help="Без ThreadPoolExecutor (для замера ускорения)")
    ap.add_argument(
        "--trace", type=Path, default=None, help="Куда сохранить JSON-лог (если задан)"
    )
    args = ap.parse_args()

    q = " ".join(args.query)
    res = run_pwc(
        q,
        max_iter=args.max_iter,
        verbose=not args.quiet,
        use_validator=not args.no_validator,
        parallel=not args.sequential,
    )

    print("\n=== ВОПРОС ===")
    print(q)
    print("\n=== ОТВЕТ ===")
    print(res.get("answer") or res.get("error"))
    print(f"\n(итераций: {res.get('iterations', '?')}, время: {res.get('elapsed_sec', 0):.1f}с)")

    if args.trace:
        args.trace.write_text(
            json.dumps(
                {"query": q, **_serialize(res)},
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print(f"Трейс сохранён: {args.trace}")


def _serialize(res: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in res.items():
        if k == "plan" and v is not None:
            out[k] = v.model_dump()
        elif k == "answers":
            out[k] = {i: a.model_dump() for i, a in v.items()}
        else:
            out[k] = v
    return out


if __name__ == "__main__":
    main()
