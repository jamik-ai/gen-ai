from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Category = Literal["CategoryA", "CategoryB", "CategoryC", "CategoryD", "CategoryE"]
Urgency = Literal["low", "medium", "high"]
Support = Literal["supported", "weakly_supported", "not_supported"]


class TicketExtraction(BaseModel):
    """Структурированный результат IE-шага (ClassifierAgent)."""

    category: Category
    urgency: Urgency
    key_topic: str = Field(description="Краткая тема тикета, 3-6 слов, на английском (как сам тикет)")
    estimated_resolution_days: int = Field(
        description="Оценка срока решения в рабочих днях по типовому SLA для этой темы"
    )
    confidence: float = Field(ge=0, le=1)

    @field_validator("estimated_resolution_days")
    @classmethod
    def resolution_days_in_range(cls, v: int) -> int:
        # Бизнес-инвариант: ни один SLA поддержки в нашей KB не превышает 30 рабочих
        # дней (самый долгий процесс — maternity-приостановка карты, но это не SLA
        # решения тикета, а статус). Отрицательный срок тоже не имеет смысла.
        if not 0 <= v <= 30:
            raise ValueError("estimated_resolution_days must be between 0 and 30")
        return v


class ResponderReply(BaseModel):
    """Финальный структурированный ответ ResponderAgent (передаётся через submit_reply)."""

    reply: str = Field(description="Текст ответа клиенту на английском")
    cited_facts: list[str] = Field(
        default_factory=list,
        description="Короткие факты/цифры из найденных KB-чанков, на которые опирается ответ",
    )
    kb_chunk_ids: list[str] = Field(default_factory=list, description="ID использованных KB-чанков")


class ClaimVerdict(BaseModel):
    claim: str
    support: Support
    comment: str = ""


class JudgeReport(BaseModel):
    """Вывод LLM-as-judge по одному тикету."""

    category_consistent: bool = Field(
        description="Согласуется ли категория тикета с темой ответа (без доступа к golden-метке)"
    )
    claims: list[ClaimVerdict] = Field(default_factory=list)
    helpfulness: float = Field(ge=0, le=1, description="Насколько ответ практически полезен клиенту")
    overall_score: float = Field(ge=0, le=1)
    summary: str = ""
