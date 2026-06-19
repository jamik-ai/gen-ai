"""
«Наивный» планировщик — копия planner.py БЕЗ явного запрета выдумывать
инструменты (правило 4 из planner.SYSTEM_PROMPT снято).

Нужен только для одного эксперимента в ДЗ С6 (часть 1 и Q4 в eval):
показать, что Schema-валидатор действительно ловит галлюцинации
инструментов, когда промпт недостаточно строгий — а не только когда
модель и так хорошо себя ведёт. Боевой planner.py не трогаем.
"""
from __future__ import annotations

from llm_client import get_model, make_client
from schemas_pwc import Plan

NAIVE_SYSTEM_PROMPT = """\
Ты — планировщик макроэкономического агента. Твоя задача — разложить
сложный вопрос пользователя на 1-5 простых подвопросов, каждый из
которых решается одним конкретным инструментом.

Стандартные инструменты:
- get_fx_rate(currency, on_date): курс валюты к рублю на дату.
- get_key_rate(on_date): ключевая ставка ЦБ на дату.
- get_inflation(year, month): ИПЦ г/г на конец месяца.
- calculate(expression): безопасный калькулятор.

Если для подвопроса нет подходящего инструмента из списка выше — придумай
разумное имя инструмента в стиле get_xxx, который мог бы вернуть нужные
данные, и укажи его в expected_tools.

Если подвопрос N зависит от ответа подвопроса K — поставь K в depends_on.
"""


def naive_planner(question: str, *, feedback: str | None = None) -> Plan:
    client = make_client()
    messages: list[dict] = [
        {"role": "system", "content": NAIVE_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    if feedback:
        messages.append(
            {
                "role": "user",
                "content": f"Предыдущая попытка не прошла проверку. Замечание: {feedback}",
            }
        )
    return client.chat.completions.create(
        model=get_model(),
        messages=messages,
        response_model=Plan,
        temperature=0.0,
        max_retries=2,
    )
