"""JudgeAgent: LLM-as-judge для ответа ResponderAgent."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_client import get_model, make_client
from schemas import JudgeReport

SYSTEM_PROMPT = """Ты — независимый аудитор качества ответов поддержки. Тебе дают тикет клиента, категорию,
найденные фрагменты базы знаний (KB) и финальный ответ агента поддержки.

Проверь:
1. category_consistent — соответствует ли содержание ответа заявленной категории тикета (по смыслу, не вслепую).
2. claims — разбей ответ на 2-5 ключевых фактических утверждений (особенно цифры/сроки/суммы) и для каждого
   вынеси вердикт: supported (прямо подтверждено текстом KB), weakly_supported (связано, но не дословно),
   not_supported (в KB этого нет — агент мог это придумать).
3. helpfulness (0-1) — насколько ответ практически решает вопрос клиента.
4. overall_score (0-1) — общая оценка: доля supported + 0.5*weakly_supported среди claims, с поправкой на
   category_consistent и helpfulness.

Не верь ответу агента по умолчанию — сверяй каждую цифру с текстом KB ниже. Если в KB ничего релевантного нет,
а агент всё равно называет цифры — это not_supported.
"""


def judge_reply(*, description: str, category: str, kb_chunks: dict[str, str], reply: str, cited_facts: list[str]) -> JudgeReport:
    client = make_client()
    kb_block = "\n\n".join(f"[{cid}]\n{text}" for cid, text in kb_chunks.items()) or "(KB ничего не нашла)"
    user_content = (
        f"Категория тикета: {category}\n"
        f"Текст тикета:\n{description}\n\n"
        f"Найденные фрагменты KB:\n{kb_block}\n\n"
        f"Ответ агента клиенту:\n{reply}\n\n"
        f"Факты, которые агент заявил как использованные (cited_facts):\n" + "\n".join(f"- {c}" for c in cited_facts)
    )
    return client.chat.completions.create(
        model=get_model(),
        response_model=JudgeReport,
        max_retries=3,
        temperature=0.7,  # не 0.0 — иначе судья просто соглашается с моделью (см. семинар_6/critic.py)
        max_tokens=1024,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
