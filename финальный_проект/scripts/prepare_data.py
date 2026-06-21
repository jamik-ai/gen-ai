"""Готовит тест-сет и анализ ключевых слов по категориям тикетов.

Запуск: python scripts/prepare_data.py
Вход:   input/raw_subset_tickets.csv  (Description, Category)
Выход:  input/test_tickets.csv            — стратифицированная выборка (ticket_id, description, category)
        output/category_keywords.json    — топ отличительных слов на категорию
"""
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "input" / "raw_subset_tickets.csv"
TEST_CSV = ROOT / "input" / "test_tickets.csv"
KEYWORDS_JSON = ROOT / "output" / "category_keywords.json"

PER_CATEGORY_TEST = 6  # 5 категорий * 6 = 30 тикетов в eval-сете
TOP_KEYWORDS = 15

# Текст уже лемматизирован, но содержит письменный "шум" (приветствия/подписи),
# который встречается во всех категориях одинаково часто и не несёт сигнала.
BOILERPLATE_STOPWORDS = {
    "hi", "hello", "dear", "please", "thanks", "thank", "regards", "best",
    "kind", "kindly", "team", "today", "would", "could", "since", "also",
    "the", "a", "an", "to", "for", "of", "in", "on", "is", "are", "be",
    "and", "or", "with", "this", "that", "it", "as", "by", "from", "at",
    "we", "you", "your", "i", "have", "has", "had", "not", "can", "will",
    "do", "does", "did", "if", "but", "so", "all", "any", "our", "us",
    "after", "before", "about", "into", "out", "up", "down", "then",
    # дни недели / месяцы — метаданные тикета, а не сигнал о теме
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
    # общая лексика тикетов поддержки, встречается в любой категории
    "sent", "help", "issue", "issues", "problem", "log", "information",
    "details", "attached", "let", "take", "working", "number", "open",
    "one", "ext", "name", "link", "report", "reports", "engineer", "senior",
}

WORD_RE = re.compile(r"[a-z]{3,}")


def load_rows() -> list[dict]:
    with RAW_CSV.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def tokenize(text: str) -> list[str]:
    return [w for w in WORD_RE.findall(text.lower()) if w not in BOILERPLATE_STOPWORDS]


def category_keywords(rows: list[dict]) -> dict[str, list[tuple[str, int]]]:
    by_cat: dict[str, Counter] = {}
    overall = Counter()
    for row in rows:
        cat = row["Category"].replace("Catetgory", "Category")  # опечатка в исходных данных (CatetgoryD)
        tokens = tokenize(row["Description"])
        by_cat.setdefault(cat, Counter()).update(tokens)
        overall.update(tokens)

    result = {}
    for cat, counts in by_cat.items():
        scored = []
        for word, cnt in counts.items():
            if cnt < 15:
                continue
            distinctiveness = cnt / overall[word]  # доля встречаемости слова именно в этой категории
            if distinctiveness < 0.3:  # отсекаем слова, равномерно размазанные по всем 5 категориям
                continue
            scored.append((word, cnt, round(distinctiveness, 3)))
        scored.sort(key=lambda x: x[1], reverse=True)  # сортируем по частоте, не по distinctiveness
        result[cat] = [(w, c) for w, c, _ in scored[:TOP_KEYWORDS]]
    return result


def stratified_test_set(rows: list[dict], seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    by_cat: dict[str, list[dict]] = {}
    for row in rows:
        cat = row["Category"].replace("Catetgory", "Category")
        by_cat.setdefault(cat, []).append(row)

    out = []
    idx = 1
    for cat in sorted(by_cat):
        sample = rng.sample(by_cat[cat], PER_CATEGORY_TEST)
        for row in sample:
            out.append({
                "ticket_id": f"TCK-{idx:04d}",
                "description": row["Description"],
                "category": cat,
            })
            idx += 1
    rng.shuffle(out)
    return out


def main() -> None:
    rows = load_rows()
    print(f"Загружено тикетов: {len(rows)}")

    keywords = category_keywords(rows)
    KEYWORDS_JSON.parent.mkdir(parents=True, exist_ok=True)
    KEYWORDS_JSON.write_text(json.dumps(keywords, ensure_ascii=False, indent=2))
    print(f"Ключевые слова по категориям -> {KEYWORDS_JSON}")
    for cat, words in sorted(keywords.items()):
        print(f"  {cat}: {', '.join(w for w, _ in words[:10])}")

    test_set = stratified_test_set(rows)
    with TEST_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ticket_id", "description", "category"])
        writer.writeheader()
        writer.writerows(test_set)
    print(f"Тест-сет ({len(test_set)} тикетов) -> {TEST_CSV}")


if __name__ == "__main__":
    main()
