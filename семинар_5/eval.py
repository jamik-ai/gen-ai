"""
Eval макро-агента: 10 вопросов (4 базовых + 6 новых из ДЗ5).

Новые 6 вопросов:
  Q5, Q6  — требуют compare_periods
  Q7, Q8  — «трудные» (неоднозначность/граничный случай)
  Q9, Q10 — реальные макро-вопросы

Прогон:
    python eval.py
    python eval.py --cache --cost
    python eval.py --only 5
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import CACHE_STATS, run_agent

CASES = [
    # --- базовые 4 из стартера ---
    {
        "id": 1,
        "query": "Какая сегодня ключевая ставка ЦБ?",
        "expected_tools": ["get_key_rate"],
        "must_have": [],
        "comment": "Базовый тест — один инструмент, одно число.",
    },
    {
        "id": 2,
        "query": "Сколько стоит доллар сегодня и сколько стоил 1 января 2022?",
        "expected_tools": ["get_fx_rate"],
        "must_have": [],
        "comment": "Два вызова одного инструмента с разными аргументами.",
    },
    {
        "id": 3,
        "query": "Какая сейчас реальная ключевая ставка? (номинальная минус инфляция г/г)",
        "expected_tools": ["get_key_rate", "get_inflation", "calculate"],
        "must_have": ["%"],
        "comment": "Три разных инструмента + арифметика.",
    },
    {
        "id": 4,
        "query": "Посчитай, за сколько лет удвоится вклад 100 тыс руб при текущей ключевой ставке (формула 72).",
        "expected_tools": ["get_key_rate", "calculate"],
        "must_have": ["год"],
        "comment": "Вычисление с формулой: 72 / ставка = годы.",
    },
    # --- ДЗ5: 2 вопроса с compare_periods ---
    {
        "id": 5,
        "query": "Во сколько раз вырос курс USD с января 2022 по апрель 2026?",
        "expected_tools": ["compare_periods"],
        "must_have": ["раз"],
        "comment": "compare_periods: ratio USD за 4 года.",
    },
    {
        "id": 6,
        "query": "Как изменилась ключевая ставка с марта 2022 по март 2025? На сколько процентных пунктов?",
        "expected_tools": ["compare_periods"],
        "must_have": ["%"],
        "comment": "compare_periods: delta key_rate за 3 года.",
    },
    # --- ДЗ5: 2 «трудных» вопроса ---
    {
        "id": 7,
        "query": "Какой был индекс нищеты в России в феврале 2022?",
        "expected_tools": ["get_inflation", "get_unemployment", "calculate"],
        "must_have": ["%"],
        "comment": (
            "ТРУДНЫЙ: дата неоднозначна — агент должен взять февраль 2022. "
            "Модель может попытаться взять 24 февраля (нет такой точки) или перепутать год."
        ),
    },
    {
        "id": 8,
        "query": "Во сколько раз вырос курс CNY к рублю с 1 января 2020 по сегодня?",
        "expected_tools": ["compare_periods"],
        "must_have": ["раз"],
        "comment": (
            "ТРУДНЫЙ: CNY может отсутствовать в fallback_csv на 2020 год — "
            "агент должен справиться с ошибкой или использовать ближайшую дату."
        ),
    },
    # --- ДЗ5: 2 реальных макро-вопроса ---
    {
        "id": 9,
        "query": (
            "Какова была реальная доходность рублёвого вклада в декабре 2023? "
            "Считай как (1 + ставка/100) / (1 + инфляция/100) - 1, результат в процентах."
        ),
        "expected_tools": ["get_key_rate", "get_inflation", "calculate"],
        "must_have": ["%"],
        "comment": "Реальный вопрос: реальная доходность вклада с поправкой на инфляцию.",
    },
    {
        "id": 10,
        "query": (
            "Сколько юаней можно купить на 1000 рублей в январе 2024? "
            "Используй кросс-курс: 1000 / курс_CNY_к_рублю."
        ),
        "expected_tools": ["get_fx_rate", "calculate"],
        "must_have": ["юан"],
        "comment": "Реальный вопрос: кросс-курс через рубль.",
    },
]


def run_case(case: dict, *, use_cache: bool = False, track_cost: bool = False) -> dict:
    print(f"\n{'=' * 70}\n[Q{case['id']}] {case['query']}\n{'-' * 70}")
    res = run_agent(
        case["query"],
        max_iter=8,
        verbose=True,
        use_cache=use_cache,
        track_cost=track_cost,
    )
    used_tools = [e["call"] for e in res["trace"] if "call" in e]
    answer = res.get("answer") or ""

    tool_match = all(t in used_tools for t in case["expected_tools"])
    text_match = all(s.lower() in answer.lower() for s in case["must_have"])
    ok = bool(answer) and tool_match and text_match

    print(f"\n  tools used  : {used_tools}")
    print(f"  expected    : {case['expected_tools']}  → {'OK' if tool_match else 'MISS'}")
    print(f"  answer      : {answer[:200]}")
    print(f"  must_have   : {case['must_have']}  → {'OK' if text_match else 'MISS'}")
    print(f"  verdict     : {'PASS' if ok else 'FAIL'}")

    return {
        "id": case["id"],
        "query": case["query"],
        "ok": ok,
        "tools_used": used_tools,
        "steps": res["steps"],
        "answer": answer,
    }


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Eval макро-агента (10 вопросов)")
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--cost", action="store_true")
    ap.add_argument("--only", type=int, default=None, help="Прогнать только один кейс по id")
    a = ap.parse_args()

    if a.cache:
        CACHE_STATS["hits"] = CACHE_STATS["misses"] = 0

    cases = CASES if a.only is None else [c for c in CASES if c["id"] == a.only]
    results = [run_case(c, use_cache=a.cache, track_cost=a.cost) for c in cases]
    passed = sum(1 for r in results if r["ok"])

    print(f"\n{'=' * 70}\nИтого: {passed}/{len(results)} пройдено\n")
    print(f"{'id':<4} {'ok':<6} {'steps':<6} {'tools':<42} query")
    print("-" * 100)
    for r in results:
        mark = "PASS" if r["ok"] else "FAIL"
        tools_str = ",".join(r["tools_used"])[:40]
        print(f"{r['id']:<4} {mark:<6} {r['steps']:<6} {tools_str:<42} {r['query'][:48]}")

    if a.cache:
        h, m = CACHE_STATS["hits"], CACHE_STATS["misses"]
        print(f"\n[кэш] {h} попаданий из {h + m} обращений к инструментам.")

    out = Path(__file__).parent / "eval_results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nРезультаты: {out}")


if __name__ == "__main__":
    main()
