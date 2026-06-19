"""
Eval мульти-агента: 3 вопроса, на которых одиночный агент С5 ломается.

Каждый вопрос прогоняется дважды:
  1) через одиночного агента С5 (agent_s5.run_agent)
  2) через PWC-цикл (orchestrator.run_pwc)

и сравниваются:
  - вызван ли calculate там, где нужно (для арифметических вопросов)
  - нет ли галлюцинаций инструментов
  - есть ли в ответе обязательная подстрока (must_have)

Прогон N=5 раз, считаем долю успешных прогонов. Результат пишется в eval_pwc_results.json.

Запуск:
    python eval_pwc.py           # полный прогон
    python eval_pwc.py --single  # только один прогон каждого, быстрая проверка
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_s5 import run_agent
from orchestrator import run_pwc


CASES = [
    {
        "id": "Q1",
        "query": "Во сколько раз USD подорожал с 1 января 2022 по сегодня?",
        "comment": (
            "Класс ошибки C: одиночный часто считает в уме, не зовёт calculate. "
            "PWC должен починить — Планировщик обязан добавить calculate-подвопрос."
        ),
        "expected_tools_pwc": {"get_fx_rate", "calculate"},
        "must_have_keywords": ["раз", "USD"],
        "forbid_hallucinated_tools": True,
    },
    {
        "id": "Q2",
        "query": (
            "Какая сейчас реальная ключевая ставка, если инфляцию брать "
            "по последнему доступному месяцу, а не по году?"
        ),
        "comment": (
            "Класс ошибки B: одиночный не умеет искать «последний доступный» "
            "месяц, зацикливается. PWC должен разбить на шаги."
        ),
        "expected_tools_pwc": {"get_inflation", "get_key_rate", "calculate"},
        "must_have_keywords": ["%"],
        "forbid_hallucinated_tools": True,
    },
    {
        "id": "Q3",
        "query": (
            "Какова накопленная инфляция с января 2022 по март 2026? "
            "Рассчитай как произведение всех (1 + ипц_м/100) по месяцам."
        ),
        "comment": (
            "Класс ошибки D (граница паттерна): требует get_inflation за много "
            "месяцев + большое calculate-выражение. Одиночный галлюцинирует "
            "get_cumulative_inflation; PWC обычно тоже (Планировщик может добавить "
            "выдуманный инструмент в план). Это — повод для Schema-Validator в домашке."
        ),
        "expected_tools_pwc": {"get_inflation", "calculate"},
        "must_have_keywords": ["%"],
        "forbid_hallucinated_tools": True,
    },
    {
        "id": "Q4",
        "query": (
            "Какой объём денежной массы М2 в России на конец 2024 года, "
            "и насколько он вырос за год?"
        ),
        "comment": (
            "Вопрос про М2 — данных нет ни в одном из 4 инструментов. С "
            "production planner.py (явный запрет придумывать tools) все три "
            "конфигурации честно отказываются — это и есть правильное "
            "поведение. Демонстрация того, что Schema-Validator РЕАЛЬНО "
            "ловит галлюцинации, сделана отдельно через naive_planner.py "
            "(см. отчёт, часть 1) — там planner без анти-галлюцинаторного "
            "правила придумывает get_money_supply, а валидатор перехватывает "
            "это сразу (0 итераций), тогда как без валидатора Worker+Critic "
            "тратят 2 полные итерации, прежде чем прийти к тому же честному "
            "отказу."
        ),
        "expect_refusal": True,
        "refusal_keywords": ["не могу", "нет инструмента", "недоступ", "не имею", "не предоставля"],
        "forbid_hallucinated_tools": True,
    },
    {
        "id": "Q5",
        "query": "Какие сейчас курсы USD, EUR и CNY к рублю? Просто перечисли все три.",
        "comment": (
            "Естественная параллельность: 3 независимых подвопроса "
            "(get_fx_rate × 3) на одном уровне зависимостей — кейс для "
            "замера ускорения из части 2 (bench_parallel.py: ~3.2x)."
        ),
        "expected_tools_pwc": {"get_fx_rate"},
        "must_have_keywords": ["usd", "eur", "cny"],
        "forbid_hallucinated_tools": True,
    },
    {
        "id": "Q6",
        "query": (
            "На сколько процентных пунктов изменилась реальная ключевая "
            "ставка ЦБ (ставка минус инфляция) с начала 2023 года до сегодня?"
        ),
        "comment": (
            "Реальный вопрос по макроэкономике: 4 независимых факта "
            "(ставка и инфляция на 2 даты) + 1 calculate-подвопрос с "
            "depends_on на все четыре — хороший тест на DAG глубже "
            "одного уровня и на дисциплину 'считай через calculate'."
        ),
        "expected_tools_pwc": {"get_key_rate", "get_inflation", "calculate"},
        "must_have_keywords": ["%"],
        "forbid_hallucinated_tools": True,
    },
]


VALID_TOOL_NAMES = {"get_fx_rate", "get_key_rate", "get_inflation", "calculate"}


def _check_single(case: dict, result: dict) -> dict:
    """Проверить результат одиночного прогона."""
    used = {e["call"] for e in result.get("trace", []) if "call" in e}
    ans = (result.get("answer") or "").lower()
    hallucinated = used - VALID_TOOL_NAMES

    if case.get("expect_refusal"):
        must = any(kw.lower() in ans for kw in case["refusal_keywords"])
        ok = bool(ans) and not hallucinated and must
        return {
            "ok": ok, "used_tools": sorted(used), "hallucinated": sorted(hallucinated),
            "must_have_ok": must, "arith_without_calc": False,
            "answer_preview": (result.get("answer") or "")[:180],
        }

    must = all(kw.lower() in ans for kw in case["must_have_keywords"])
    arith_without_calc = (
        case["id"] in {"Q1", "Q2", "Q3", "Q6"}
        and "calculate" not in used
        and bool(ans)
    )
    ok = bool(ans) and not hallucinated and must and not arith_without_calc
    return {
        "ok": ok,
        "used_tools": sorted(used),
        "hallucinated": sorted(hallucinated),
        "must_have_ok": must,
        "arith_without_calc": arith_without_calc,
        "answer_preview": (result.get("answer") or "")[:180],
    }


def _check_pwc(case: dict, result: dict) -> dict:
    """Проверить результат PWC-прогона."""
    used = set()
    for t in result.get("trace", []):
        if t.get("kind") == "worker":
            used.update(t.get("used_tools") or [])
    ans = (result.get("answer") or "").lower()
    hallucinated = used - VALID_TOOL_NAMES
    # Также проверим галлюцинации на этапе Планировщика (в плане expected_tools)
    plan_tools = set()
    plan = result.get("plan")
    if plan is not None:
        for sq in plan.subquestions:
            plan_tools.update(sq.expected_tools)
    plan_hallucinated = plan_tools - VALID_TOOL_NAMES

    if case.get("expect_refusal"):
        # Отказ — это PASS, если система НЕ нагаллюцинировала инструмент.
        # Ответ может быть None (валидатор не пропустил план) — тоже ок.
        ok = not hallucinated and not plan_hallucinated
        return {
            "ok": ok, "used_tools": sorted(used), "plan_tools": sorted(plan_tools),
            "hallucinated_in_workers": sorted(hallucinated),
            "hallucinated_in_plan": sorted(plan_hallucinated),
            "must_have_ok": True, "iterations": result.get("iterations", -1),
            "elapsed_sec": result.get("elapsed_sec"),
            "answer_preview": (result.get("answer") or result.get("error") or "")[:180],
        }

    must = all(kw.lower() in ans for kw in case["must_have_keywords"])
    ok = (
        bool(result.get("answer"))
        and not hallucinated
        and not plan_hallucinated
        and must
    )
    return {
        "ok": ok,
        "used_tools": sorted(used),
        "plan_tools": sorted(plan_tools),
        "hallucinated_in_workers": sorted(hallucinated),
        "hallucinated_in_plan": sorted(plan_hallucinated),
        "must_have_ok": must,
        "iterations": result.get("iterations", -1),
        "elapsed_sec": result.get("elapsed_sec"),
        "answer_preview": (result.get("answer") or "")[:180],
    }


def run_case(case: dict, *, n: int = 3) -> dict:
    single = {"runs": [], "pass": 0}
    pwc_no_val = {"runs": [], "pass": 0}
    pwc_val = {"runs": [], "pass": 0}

    for i in range(n):
        # --- Одиночный агент ---
        try:
            r1 = run_agent(case["query"], max_iter=8, verbose=False)
        except Exception as e:
            r1 = {"answer": None, "error": f"{type(e).__name__}: {e}", "trace": []}
        check1 = _check_single(case, r1)
        single["runs"].append(check1)
        single["pass"] += int(check1["ok"])

        # --- PWC без валидатора ---
        try:
            r2 = run_pwc(case["query"], max_iter=3, verbose=False, use_validator=False)
        except Exception as e:
            r2 = {"answer": None, "error": f"{type(e).__name__}: {e}",
                  "trace": [], "plan": None}
        check2 = _check_pwc(case, r2)
        pwc_no_val["runs"].append(check2)
        pwc_no_val["pass"] += int(check2["ok"])

        # --- PWC с валидатором ---
        try:
            r3 = run_pwc(case["query"], max_iter=3, verbose=False, use_validator=True)
        except Exception as e:
            r3 = {"answer": None, "error": f"{type(e).__name__}: {e}",
                  "trace": [], "plan": None}
        check3 = _check_pwc(case, r3)
        pwc_val["runs"].append(check3)
        pwc_val["pass"] += int(check3["ok"])

    return {
        "id": case["id"],
        "query": case["query"],
        "comment": case["comment"],
        "n": n,
        "single": single,
        "pwc_no_validator": pwc_no_val,
        "pwc_validator": pwc_val,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", action="store_true",
                    help="Только один прогон каждого кейса (быстро)")
    ap.add_argument("-n", type=int, default=3,
                    help="Сколько прогонов на кейс (default=3)")
    args = ap.parse_args()
    n = 1 if args.single else args.n

    print(f"Eval С6: {len(CASES)} кейсов × 3 конфигурации × {n} прогонов\n")
    results = []
    for case in CASES:
        print(f"=== {case['id']}: {case['query'][:70]}...")
        r = run_case(case, n=n)
        results.append(r)
        s = r["single"]; pn = r["pwc_no_validator"]; pv = r["pwc_validator"]
        print(f"   single: {s['pass']}/{n}    pwc(no-val): {pn['pass']}/{n}    pwc(+val): {pv['pass']}/{n}")
        for run in pn["runs"][:1]:
            if run["hallucinated_in_plan"]:
                print(f"   ⚠ pwc без валидатора: план с выдуманными инструментами: {run['hallucinated_in_plan']}")
        print()

    # Итог
    print("=" * 70)
    print("ИТОГО:")
    for r in results:
        print(f"  {r['id']}: single {r['single']['pass']}/{n}  "
              f"pwc(no-val) {r['pwc_no_validator']['pass']}/{n}  "
              f"pwc(+val) {r['pwc_validator']['pass']}/{n}  — {r['query'][:50]}")

    out = Path(__file__).parent / "eval_pwc_results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2,
                              default=str), encoding="utf-8")
    print(f"\nРезультаты: {out}")


if __name__ == "__main__":
    main()
