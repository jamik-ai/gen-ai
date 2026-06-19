"""
Schema-валидатор: проверка плана Планировщика до того, как Исполнитель
начнёт его выполнять.

Зачем: Планировщик не вызывает tools, он просто пишет имена строкой в
expected_tools. Ничто не мешает ему написать "get_cumulative_inflation"
или "get_gdp" — несуществующий инструмент. Worker такой инструмент не
найдёт (agent_s5 ограничен реальным TOOL_SCHEMAS), но узнаём мы об этом
только после нескольких потраченных LLM-вызовов. Валидатор ловит это
сразу после planner(), до Исполнителя.
"""
from __future__ import annotations

from schemas_pwc import Plan

VALID_TOOLS = {"get_fx_rate", "get_key_rate", "get_inflation", "calculate"}


def validate_plan(plan: Plan) -> list[str]:
    """Вернуть список ошибок плана (пустой список — всё ок)."""
    errors: list[str] = []
    seen_ids: set[int] = set()
    for sq in plan.subquestions:
        if sq.id in seen_ids:
            errors.append(f"подвопрос {sq.id}: дублирующийся id")
        seen_ids.add(sq.id)

        bad_tools = set(sq.expected_tools) - VALID_TOOLS
        if bad_tools:
            errors.append(
                f"подвопрос {sq.id} использует несуществующие инструменты: "
                f"{sorted(bad_tools)}"
            )

        if not sq.expected_tools:
            errors.append(f"подвопрос {sq.id} не указывает ни одного инструмента")

    valid_ids = {sq.id for sq in plan.subquestions}
    for sq in plan.subquestions:
        bad_deps = [d for d in sq.depends_on if d not in valid_ids]
        if bad_deps:
            errors.append(
                f"подвопрос {sq.id} ссылается на несуществующие depends_on: {bad_deps}"
            )

    return errors


if __name__ == "__main__":
    from schemas_pwc import SubQuestion

    bad_plan = Plan(
        reasoning="тест",
        subquestions=[
            SubQuestion(
                id=1,
                question="Какой ВВП России в 2023?",
                expected_tools=["get_gdp"],
            ),
            SubQuestion(
                id=2,
                question="Накопленная инфляция?",
                expected_tools=["get_cumulative_inflation"],
                depends_on=[1],
            ),
        ],
    )
    print(validate_plan(bad_plan))
