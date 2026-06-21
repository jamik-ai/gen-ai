"""BM25-индекс по базе знаний kb/*.md и тул search_kb для ResponderAgent."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

KB_DIR = Path(__file__).resolve().parent / "kb"
WORD_RE = re.compile(r"[a-zа-яё0-9$%]+", re.IGNORECASE)


@dataclass
class Chunk:
    chunk_id: str
    category: str  # CategoryA..E, выводится из имени файла (первая буква до "_")
    text: str


def _tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def load_chunks() -> list[Chunk]:
    # Документы короткие (одна статья KB ~100-150 слов) — режем на чанки целыми
    # файлами, а не по абзацам: разбивка по \n\n отрывала заголовок (где обычно
    # упомянута тема и ключевые слова) от текста с цифрами SLA, и BM25 находил
    # пустой заголовочный чанк вместо чанка с фактами.
    letter_to_category = {"A": "CategoryA", "B": "CategoryB", "C": "CategoryC", "D": "CategoryD", "E": "CategoryE"}
    chunks: list[Chunk] = []
    for path in sorted(KB_DIR.glob("*.md")):
        letter = path.stem[0]
        category = letter_to_category.get(letter, "Unknown")
        text = path.read_text(encoding="utf-8").strip()
        if text:
            chunks.append(Chunk(chunk_id=path.stem, category=category, text=text))
    return chunks


class KnowledgeBase:
    def __init__(self) -> None:
        self.chunks = load_chunks()
        self._tokens = [_tokenize(c.text) for c in self.chunks]
        self._bm25 = BM25Okapi(self._tokens)

    def search(self, query: str, category: str | None = None, k: int = 3) -> list[Chunk]:
        scores = self._bm25.get_scores(_tokenize(query))
        order = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)
        results = []
        for i in order:
            chunk = self.chunks[i]
            if category and chunk.category != category:
                continue
            if scores[i] <= 0:
                continue
            results.append(chunk)
            if len(results) >= k:
                break
        return results


_KB: KnowledgeBase | None = None


def get_kb() -> KnowledgeBase:
    global _KB
    if _KB is None:
        _KB = KnowledgeBase()
    return _KB


SEARCH_KB_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_kb",
        "description": (
            "Найти релевантные фрагменты внутренней базы знаний поддержки (KB) по запросу. "
            "Используй, чтобы найти точные SLA/сроки/лимиты/контакты перед тем, как отвечать клиенту. "
            "Не придумывай цифры — бери их только из результатов этого инструмента."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Поисковый запрос на английском, по теме тикета"},
                "category": {
                    "type": "string",
                    "enum": ["CategoryA", "CategoryB", "CategoryC", "CategoryD", "CategoryE"],
                    "description": "Категория тикета — ограничивает поиск этой категорией KB",
                },
            },
            "required": ["query", "category"],
        },
    },
}


def search_kb(query: str, category: str) -> dict:
    chunks = get_kb().search(query, category=category, k=3)
    if not chunks:
        return {"results": [], "note": "Ничего релевантного не найдено для этой категории"}
    return {"results": [{"chunk_id": c.chunk_id, "text": c.text} for c in chunks]}
