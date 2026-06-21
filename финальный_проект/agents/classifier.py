"""ClassifierAgent: structured IE по тексту тикета (раунд 1 мультиагентного пайплайна)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_client import get_model, make_client
from schemas import TicketExtraction

CATEGORY_LABELS = json.loads((Path(__file__).resolve().parent.parent / "kb" / "category_labels.json").read_text())

_LABELS_BLOCK = "\n".join(
    f"- {cat}: {info['label']} — {info['note']}" for cat, info in CATEGORY_LABELS.items()
)

SYSTEM_PROMPT = f"""Ты — классификатор тикетов IT-поддержки. Текст тикета уже лемматизирован и анонимизирован
(нет имён, почти нет дат), может быть зашумлён.

Категории (определены по анализу частот слов в реальных данных, бери только из этого списка):
{_LABELS_BLOCK}

Определи:
- category — одна из пяти категорий выше.
- urgency — low/medium/high, по тону и содержанию тикета.
- key_topic — кратко суть проблемы (3-6 слов, на английском).
- estimated_resolution_days — сколько рабочих дней типично занимает решение такой проблемы (0-30).
- confidence — твоя уверенность в категории (0-1).

Категории B и C обе про карты доступа и похожи по словарю (card, visitor, floor) — отличай их по сути:
B — выпуск/процесс по карте (visitor, leaver, maternity), C — инцидент с картой (lost, restricted door, лог событий).
"""


def classify_ticket(description: str) -> TicketExtraction:
    client = make_client()
    return client.chat.completions.create(
        model=get_model(),
        response_model=TicketExtraction,
        max_retries=3,
        temperature=0.0,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Текст тикета:\n{description}"},
        ],
    )
