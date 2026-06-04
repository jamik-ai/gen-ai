from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

CURRENT_DATE = date.today()

ALL_ASPECTS: list[str] = ["severity", "urgency", "scope", "resolution"]


# ══════════════════════════════════════════════════════════
# Раунд 1 — Information Extraction
# ══════════════════════════════════════════════════════════

class Complaint(BaseModel):
    category: Literal["data_loss", "performance", "billing", "integration", "ux", "security", "support"]
    severity: int = Field(ge=1, le=5)
    quote: str


class Customer(BaseModel):
    ticket_id: str
    customer_name: Optional[str] = None
    plan: Literal["free", "starter", "pro", "enterprise"]
    complaints: list[Complaint]
    is_resolved: bool = False

    @field_validator("ticket_id")
    @classmethod
    def ticket_id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("ticket_id не может быть пустым")
        return v.strip()


class MatchVerdict(BaseModel):
    matched: bool
    matched_index: int = Field(default=-1, description="индекс жалобы или -1")
    reason: str = ""


# ══════════════════════════════════════════════════════════
# Раунд 2 — Аспектный анализ (фиксированные аспекты)
# ══════════════════════════════════════════════════════════

class AspectSentiment(BaseModel):
    aspect: Literal["severity", "urgency", "scope", "resolution"]
    sentiment: Literal["positive", "negative", "neutral"]
    quote: str
    confidence: float = Field(ge=0, le=1)


class CustomerSentiment(BaseModel):
    ticket_id: str
    aspects: list[AspectSentiment]


# ══════════════════════════════════════════════════════════
# Раунд 2.5 — Autodiscovery аспектов
# ══════════════════════════════════════════════════════════

class DiscoveredAspect(BaseModel):
    name: str
    description: str = Field(min_length=5)


class DiscoveredAspects(BaseModel):
    aspects: list[DiscoveredAspect] = Field(min_length=3, max_length=12)


class DynamicAspect(BaseModel):
    aspect: str
    sentiment: Literal["positive", "negative", "neutral"]
    quote: str
    confidence: float = Field(ge=0, le=1)


class DynamicCustomer(BaseModel):
    ticket_id: str
    aspects: list[DynamicAspect]


# ══════════════════════════════════════════════════════════
# Раунд 3 — Map-Reduce
# ══════════════════════════════════════════════════════════

class ChunkSummary(BaseModel):
    batch_id: str
    key_points: list[str] = Field(min_length=1, max_length=6)
    sentiment: Literal["positive", "negative", "mixed"]


class TicketsSummary(BaseModel):
    headline: str
    key_findings: list[str] = Field(min_length=2, max_length=8)
    action_items: list[str] = Field(min_length=1, max_length=8)


# ══════════════════════════════════════════════════════════
# Раунд 5 — LLM-as-judge
# ══════════════════════════════════════════════════════════

class ActionVerdict(BaseModel):
    action: str
    support: Literal["supported", "weakly_supported", "not_supported"]
    evidence: list[str] = Field(default_factory=list)
    comment: str


class JudgeReport(BaseModel):
    verdicts: list[ActionVerdict]
    overall_score: float = Field(ge=0, le=1)
    summary: str
