"""Проверка галлюцинаций: ищем числа/факты в ответе, которых нет в retrieved KB-чанках.

Паттерн взят из check_quotes()/fidelity() (ДЗ3, семинар_3): подстрочный поиск по исходному
тексту, который здесь — конкатенация retrieved-чанков (источник истины для Responder'а).

Важный нюанс этого проекта: KB написана на русском (по конвенции курса), а Responder
отвечает на английском (как сами тикеты). Из-за этого дословное сопоставление текста
не работает на переводных парафразах ("4 hours" vs "4 часа") — поэтому числа сверяются
по цифровому ядру (язык-независимо), а не по полной фразе. Это и есть единственный
надёжный сигнал в условиях двуязычного KB/ответа; ограничение явно обсуждается в отчёте.
"""
from __future__ import annotations

import re

NUMERIC_RE = re.compile(r"\$\s?\d+(?:\.\d+)?|\d+(?:\.\d+)?\s?%|\d+\s*(?:business\s+)?days?|\d+\s*hours?")
DIGIT_CORE_RE = re.compile(r"\d+(?:\.\d+)?")
DISCLAIMER_RE = re.compile(
    r"no (relevant|specific|kb|camera|matching)|not (found|available|contain)|"
    r"does not contain|nothing relevant|no information|no kb (articles|facts|information)",
    re.IGNORECASE,
)


def extract_numeric_claims(text: str) -> list[str]:
    return [m.group(0).strip() for m in NUMERIC_RE.finditer(text)]


def _strip_markdown(s: str) -> str:
    return s.replace("**", "").replace("*", "").replace("#", "")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", _strip_markdown(s).lower()).strip()


def _digit_set(text: str) -> set[str]:
    return set(DIGIT_CORE_RE.findall(text))


def is_disclaimer(fact: str) -> bool:
    """True, если fact — корректное заявление об отсутствии данных в KB, а не утверждение факта."""
    return bool(DISCLAIMER_RE.search(fact))


def ghost_numbers(reply_text: str, kb_text: str) -> list[str]:
    """Числа из ответа, цифровое ядро которых не встречается вообще ни в одном KB-чанке."""
    kb_digits = _digit_set(kb_text)
    ghosts = []
    for claim in extract_numeric_claims(reply_text):
        claim_digits = _digit_set(claim)
        if claim_digits and not (claim_digits & kb_digits):
            ghosts.append(claim)
    return ghosts


def ghost_cited_facts(cited_facts: list[str], kb_text: str) -> list[str]:
    """cited_facts, не подтверждаемые KB: либо содержат число, которого нет в KB, либо
    (для безчисловых фактов) первые слова не находятся в тексте KB. Дисклеймеры об
    отсутствии информации в KB не считаются галлюцинацией."""
    kb_norm = _norm(kb_text)
    kb_digits = _digit_set(kb_text)
    ghosts = []
    for fact in cited_facts:
        if is_disclaimer(fact):
            continue
        fact_digits = _digit_set(fact)
        if fact_digits:
            if not (fact_digits & kb_digits):
                ghosts.append(fact)
            continue
        words = _norm(fact).split()
        probe = " ".join(words[:6])
        if probe and probe not in kb_norm:
            ghosts.append(fact)
    return ghosts


def hallucination_report(reply_text: str, cited_facts: list[str], retrieved_chunks: dict[str, str]) -> dict:
    kb_text = "\n".join(retrieved_chunks.values())
    g_num = ghost_numbers(reply_text, kb_text)
    g_fact = ghost_cited_facts(cited_facts, kb_text)
    disclaimers = [f for f in cited_facts if is_disclaimer(f)]
    total_numeric = len(extract_numeric_claims(reply_text))
    checkable_facts = len(cited_facts) - len(disclaimers)
    return {
        "ghost_numbers": g_num,
        "ghost_cited_facts": g_fact,
        "disclaimer_facts": disclaimers,
        "total_numeric_claims": total_numeric,
        "ghost_number_rate": round(len(g_num) / total_numeric, 3) if total_numeric else 0.0,
        "total_cited_facts": len(cited_facts),
        "checkable_cited_facts": checkable_facts,
        "ghost_fact_rate": round(len(g_fact) / checkable_facts, 3) if checkable_facts else 0.0,
        "no_kb_retrieved": len(retrieved_chunks) == 0,
    }
