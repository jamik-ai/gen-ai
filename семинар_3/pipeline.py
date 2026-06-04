"""
pipeline.py — пайплайн анализа тикетов поддержки
==================================================
Конвейер: IE → аспектный анализ → Autodiscovery → Map-Reduce → LLM-as-judge

Запуск:
    python3 pipeline.py input/tickets.txt
    python3 pipeline.py input/tickets.txt output/
"""

from __future__ import annotations

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from llm_client import get_model, make_client
from prompts import (
    ASPECTS_SYSTEM,
    CHUNK_SYSTEM,
    CONCERN_CHECK,
    DISCOVER_SYSTEM,
    DYNAMIC_ASPECTS_SYSTEM,
    IE_SYSTEM,
    JUDGE_SYSTEM,
    REDUCE_SYSTEM,
)
from schema import (
    ALL_ASPECTS,
    ActionVerdict,
    AspectSentiment,
    ChunkSummary,
    Complaint,
    Customer,
    CustomerSentiment,
    DiscoveredAspects,
    DynamicCustomer,
    JudgeReport,
    MatchVerdict,
    TicketsSummary,
)

client = make_client()
MODEL = get_model()

BATCH_SIZE = 8

# DeepSeek pricing (deepseek-chat): $0.27/1M input, $1.10/1M output
_PRICE_INPUT  = 0.27 / 1_000_000
_PRICE_OUTPUT = 1.10 / 1_000_000


class _UsageTracker:
    def __init__(self) -> None:
        self.input_tokens  = 0
        self.output_tokens = 0
        self.calls         = 0

    def add(self, resp) -> None:
        if resp and hasattr(resp, "usage") and resp.usage:
            self.input_tokens  += getattr(resp.usage, "prompt_tokens",     0) or 0
            self.output_tokens += getattr(resp.usage, "completion_tokens", 0) or 0
        self.calls += 1

    def cost_usd(self) -> float:
        return self.input_tokens * _PRICE_INPUT + self.output_tokens * _PRICE_OUTPUT


_tracker = _UsageTracker()


def _create(**kwargs):
    """Обёртка над client.chat.completions.create с трекингом токенов."""
    result, resp = client.chat.completions.create(with_completion=True, **kwargs)
    _tracker.add(resp)
    return result


# ══════════════════════════════════════════════════════════
# Вспомогательные утилиты
# ══════════════════════════════════════════════════════════

def _split_into_batches(text: str, batch_size: int = BATCH_SIZE) -> list[tuple[str, str]]:
    """Разбить текст на пакеты тикетов. Возвращает [(batch_id, text), ...]."""
    tickets_raw = re.split(r"(?=^=== ТИКЕТ)", text, flags=re.MULTILINE)
    tickets_raw = [t.strip() for t in tickets_raw if t.strip()]
    batches = []
    for i in range(0, len(tickets_raw), batch_size):
        chunk = "\n\n".join(tickets_raw[i : i + batch_size])
        batch_id = f"Batch_{i // batch_size + 1}"
        batches.append((batch_id, chunk))
    return batches


# ══════════════════════════════════════════════════════════
# Раунд 1 — Information Extraction
# ══════════════════════════════════════════════════════════

def extract_customers(text: str) -> list[Customer]:
    batches = _split_into_batches(text)
    results: list[Customer] = []
    for _bid, chunk in batches:
        batch = _create(
            model=MODEL,
            response_model=list[Customer],
            max_retries=3,
            temperature=0.0,
            messages=[
                {"role": "system", "content": IE_SYSTEM},
                {"role": "user", "content": chunk},
            ],
        )
        results.extend(batch)
    return results


# ══════════════════════════════════════════════════════════
# Раунд 2 — Аспектный анализ (фиксированные аспекты)
# ══════════════════════════════════════════════════════════

def extract_aspects(text: str) -> list[CustomerSentiment]:
    batches = _split_into_batches(text)
    results: list[CustomerSentiment] = []
    for _bid, chunk in batches:
        batch = _create(
            model=MODEL,
            response_model=list[CustomerSentiment],
            max_retries=3,
            temperature=0.0,
            messages=[
                {"role": "system", "content": ASPECTS_SYSTEM},
                {"role": "user", "content": chunk},
            ],
        )
        results.extend(batch)
    return results


def check_quotes(
    aspects: list[CustomerSentiment],
    source_text: str,
) -> list[tuple[str, str]]:
    """Вернуть (ticket_id, ghost-цитата) для цитат не найденных в тексте."""
    t = source_text.lower()
    ghosts: list[tuple[str, str]] = []
    for r in aspects:
        for a in r.aspects:
            probe = a.quote.strip().lower()[:30]
            if probe and probe not in t:
                ghosts.append((r.ticket_id, a.quote))
    return ghosts


def build_heatmap(aspects: list[CustomerSentiment], out_path: str = "heatmap.png") -> None:
    """Тепловая карта ticket × aspect (sentiment → +1/0/-1)."""
    ids = [r.ticket_id for r in aspects]
    sent_map = {"positive": 1, "negative": -1, "neutral": 0}
    matrix = np.full((len(ids), len(ALL_ASPECTS)), np.nan)
    for i, r in enumerate(aspects):
        for a in r.aspects:
            if a.aspect in ALL_ASPECTS:
                j = ALL_ASPECTS.index(a.aspect)
                matrix[i, j] = sent_map[a.sentiment]

    fig_h = max(5, len(ids) * 0.32)
    plt.figure(figsize=(8, fig_h))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".0f",
        xticklabels=ALL_ASPECTS,
        yticklabels=ids,
        center=0,
        cmap="RdYlGn",
        cbar_kws={"label": "+1 positive / 0 neutral / -1 negative"},
        linewidths=0.3,
    )
    plt.title("Аспектная тональность по тикетам")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ══════════════════════════════════════════════════════════
# Раунд 2.5 — Autodiscovery аспектов
# ══════════════════════════════════════════════════════════

def discover_aspects(text: str) -> DiscoveredAspects:
    return _create(
        model=MODEL,
        response_model=DiscoveredAspects,
        max_retries=3,
        temperature=0.0,
        messages=[
            {"role": "system", "content": DISCOVER_SYSTEM},
            {"role": "user", "content": text},
        ],
    )


def extract_dynamic_aspects(text: str, discovered: DiscoveredAspects) -> list[DynamicCustomer]:
    aspects_list = "\n".join(
        f"- {a.name}: {a.description}" for a in discovered.aspects
    )
    system = DYNAMIC_ASPECTS_SYSTEM.format(aspects_list=aspects_list)
    batches = _split_into_batches(text)
    results: list[DynamicCustomer] = []
    for _bid, chunk in batches:
        batch = _create(
            model=MODEL,
            response_model=list[DynamicCustomer],
            max_retries=3,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": chunk},
            ],
        )
        results.extend(batch)
    return results


def compare_aspects(
    fixed: list[CustomerSentiment],
    dynamic: list[DynamicCustomer],
    discovered: DiscoveredAspects,
) -> dict:
    """Сравнить покрытие фиксированных и динамических аспектов."""
    fixed_names = set(ALL_ASPECTS)
    dynamic_names = {a.name for a in discovered.aspects}
    invented = dynamic_names - fixed_names

    fixed_neg = sum(
        1 for r in fixed for a in r.aspects if a.sentiment == "negative"
    )
    dynamic_neg = sum(
        1 for r in dynamic for a in r.aspects if a.sentiment == "negative"
    )

    return {
        "fixed_aspects": sorted(fixed_names),
        "discovered_aspects": sorted(dynamic_names),
        "invented_by_model": sorted(invented),
        "fixed_negative_signals": fixed_neg,
        "dynamic_negative_signals": dynamic_neg,
    }


# ══════════════════════════════════════════════════════════
# Раунд 3 — Map-Reduce
# ══════════════════════════════════════════════════════════

def _summarize_batch(batch_id: str, chunk: str) -> ChunkSummary:
    prompt = f"Пакет тикетов {batch_id}:\n\n{chunk}"
    result = _create(
        model=MODEL,
        response_model=ChunkSummary,
        max_retries=3,
        temperature=0.0,
        messages=[
            {"role": "system", "content": CHUNK_SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )
    result.batch_id = batch_id
    return result


def _reduce_summaries(summaries: list[ChunkSummary], reduce_prompt: str = REDUCE_SYSTEM) -> TicketsSummary:
    joined = "\n\n".join(
        f"## {s.batch_id} ({s.sentiment})\n" + "\n".join(f"- {p}" for p in s.key_points)
        for s in summaries
    )
    return _create(
        model=MODEL,
        response_model=TicketsSummary,
        max_retries=3,
        temperature=0.0,
        messages=[
            {"role": "system", "content": reduce_prompt},
            {"role": "user", "content": joined},
        ],
    )


def summarize_tickets(text: str, workers: int = 4) -> TicketsSummary:
    batches = _split_into_batches(text)
    n = len(batches)
    print(f"  [MR] MAP: {n} пакетов, до {workers} параллельно...")
    t0 = time.time()
    summaries: list[ChunkSummary | None] = [None] * n
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_summarize_batch, bid, chunk): i
            for i, (bid, chunk) in enumerate(batches)
        }
        done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            summaries[i] = fut.result()
            done += 1
            print(f"  [MR] {done}/{n} готов ({time.time() - t0:.1f}с)")
    print(f"  [MR] MAP {time.time() - t0:.1f}с → REDUCE...")
    result = _reduce_summaries([s for s in summaries if s is not None])
    print(f"  [MR] всего {time.time() - t0:.1f}с")
    return result


# ══════════════════════════════════════════════════════════
# Раунд 5 — LLM-as-judge
# ══════════════════════════════════════════════════════════

def _build_evidence_packet(customers: list[dict], summary: dict) -> str:
    parts = ["## Рекомендации (оцениваем обоснованность)"]
    for i, a in enumerate(summary.get("action_items", []), 1):
        parts.append(f"  {i}. {a}")
    parts.append("\n## Жалобы клиентов (исходные данные)")
    for c in customers:
        for complaint in c.get("complaints", []):
            parts.append(
                f"  - [{c['ticket_id']}/{complaint['category']}, sev={complaint['severity']}]"
                f" «{complaint['quote']}»"
            )
    return "\n".join(parts)


def run_judge(customers: list[dict], summary: dict) -> JudgeReport:
    evidence = _build_evidence_packet(customers, summary)
    return _create(
        model=MODEL,
        response_model=JudgeReport,
        max_retries=3,
        temperature=0.0,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": evidence},
        ],
    )


# ══════════════════════════════════════════════════════════
# Главная функция
# ══════════════════════════════════════════════════════════

def analyze(input_path: str, out_dir: str = "output") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    text = Path(input_path).read_text(encoding="utf-8")
    total_start = time.time()

    # ── Акт 1: IE ──────────────────────────────────────────
    print("→ Акт 1: извлечение тикетов (IE)...")
    customers = extract_customers(text)
    valid = len(customers)
    total_complaints = sum(len(c.complaints) for c in customers)
    print(f"   {valid} тикетов, {total_complaints} жалоб")

    customers_data = [c.model_dump(mode="json") for c in customers]
    (out / "customers.json").write_text(
        json.dumps(customers_data, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    # ── Акт 2: Аспекты (фиксированные) ────────────────────
    print("→ Акт 2: аспектный анализ (фиксированные)...")
    aspects = extract_aspects(text)
    ghosts = check_quotes(aspects, text)
    ghost_count = len(ghosts)
    total_quotes = sum(len(r.aspects) for r in aspects)
    ghost_pct = ghost_count / total_quotes * 100 if total_quotes else 0
    if ghosts:
        print(f"   ⚠ ghost-цитат: {ghost_count}/{total_quotes} ({ghost_pct:.1f}%)")
        for tid, q in ghosts[:3]:
            print(f"     {tid}: «{q[:80]}»")
    else:
        print(f"   ghost-цитат: 0/{total_quotes}")

    build_heatmap(aspects, out_path=str(out / "heatmap.png"))
    aspects_data = [a.model_dump() for a in aspects]
    (out / "aspects.json").write_text(
        json.dumps(aspects_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ── Акт 2.5: Autodiscovery ─────────────────────────────
    print("→ Акт 2.5: autodiscovery аспектов...")
    discovered = discover_aspects(text)
    print(f"   Обнаружено {len(discovered.aspects)} аспектов: "
          f"{', '.join(a.name for a in discovered.aspects)}")

    dynamic_aspects = extract_dynamic_aspects(text, discovered)
    comparison = compare_aspects(aspects, dynamic_aspects, discovered)

    autodiscovery_data = {
        "discovered": discovered.model_dump(),
        "dynamic_aspects": [d.model_dump() for d in dynamic_aspects],
        "comparison": comparison,
    }
    (out / "autodiscovery.json").write_text(
        json.dumps(autodiscovery_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    invented = comparison["invented_by_model"]
    if invented:
        print(f"   Аспекты вне фиксированного списка: {invented}")

    # ── Акт 3: Map-Reduce ──────────────────────────────────
    print("→ Акт 3: Map-Reduce резюме...")
    summary = summarize_tickets(text)
    (out / "summary.json").write_text(
        summary.model_dump_json(indent=2), encoding="utf-8"
    )

    # ── Акт 5: Судья ───────────────────────────────────────
    print("→ Акт 5: LLM-as-judge...")
    report = run_judge(customers_data, json.loads(summary.model_dump_json()))

    if report.overall_score < 0.7:
        print(f"   ⚠ overall_score={report.overall_score:.2f} < 0.7 — повторный прогон...")
        improved_reduce = REDUCE_SYSTEM + (
            "\n\nВАЖНО: каждый action_item должен напрямую опираться "
            "на конкретные жалобы клиентов из тикетов. "
            "Не рекомендуй то, о чём клиенты не жаловались."
        )
        batches = _split_into_batches(text)
        batch_summaries = [_summarize_batch(bid, chunk) for bid, chunk in batches]
        summary = _reduce_summaries(batch_summaries, reduce_prompt=improved_reduce)
        (out / "summary.json").write_text(summary.model_dump_json(indent=2), encoding="utf-8")
        report = run_judge(customers_data, json.loads(summary.model_dump_json()))
        print(f"   Новый overall_score: {report.overall_score:.2f}")

    (out / "judge_report.json").write_text(
        report.model_dump_json(indent=2), encoding="utf-8"
    )

    elapsed = time.time() - total_start

    # ── Итог ───────────────────────────────────────────────
    print("\n══════════════════ ИТОГ ══════════════════")
    print(summary.headline)
    print("\nКлючевые выводы:")
    for kf in summary.key_findings:
        print(f"  • {kf}")
    print("\nРекомендации:")
    for ai in summary.action_items:
        print(f"  → {ai}")
    cost = _tracker.cost_usd()
    print(f"\nОценка судьи:  {report.overall_score:.2f}")
    print(f"ghost-цитат:   {ghost_count}/{total_quotes} ({ghost_pct:.1f}%)")
    print(f"Время прогона: {elapsed:.1f}с")
    print(f"Токены:        {_tracker.input_tokens:,} input / {_tracker.output_tokens:,} output  ({_tracker.calls} вызовов)")
    print(f"Стоимость:     ${cost:.4f}")
    print(f"\nАртефакты в: {out}/")
    print("  customers.json  aspects.json  autodiscovery.json")
    print("  heatmap.png  summary.json  judge_report.json")


def main() -> None:
    if len(sys.argv) < 2:
        print("Использование: python3 pipeline.py <input_file> [out_dir]")
        sys.exit(1)
    analyze(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "output")


if __name__ == "__main__":
    main()
