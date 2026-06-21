# Финальный проект: триаж IT-тикетов поддержки (классификация + RAG-ответ)

Трек B (прикладной). Мультиагентный пайплайн: **Classifier → Responder (tool-agent, RAG по KB) → Judge
(LLM-as-judge)**, с проверкой галлюцинаций и опциональным rework-шагом при низкой оценке судьи.

## Запуск одной командой

```bash
pip install -r requirements.txt
cp .env.example .env   # заполнить LLM_BASE_URL/LLM_AUTH_TOKEN/LLM_MODEL (или OPENAI_API_KEY)

python scripts/prepare_data.py        # генерирует input/test_tickets.csv (30 тикетов) и анализ категорий
python pipeline.py input/test_tickets.csv output/   # прогоняет пайплайн, пишет output/traces/*.json
python eval.py                        # считает метрики -> output/eval_table.csv, output/eval_summary.json
```

`input/test_tickets.csv` и `output/*` уже приложены в репозитории как артефакты последнего прогона — шаги
выше можно не повторять, чтобы посмотреть результат.

## Что где лежит

```
schemas.py            # Pydantic-схемы: TicketExtraction (+field_validator), ResponderReply, JudgeReport
llm_client.py          # JSON-клиент над OpenAI-совместимым API (structured output, max_retries) + raw-клиент для tool calling
kb.py                  # BM25-индекс по kb/*.md + тул search_kb (RAG)
hallucination.py       # проверка галлюцинаций: ghost-числа и ghost-факты против retrieved KB-чанков
agents/
  classifier.py        # IE: категория, urgency, key_topic, SLA-оценка, confidence
  responder.py          # tool-calling агент: сам вызывает search_kb, пишет ответ клиенту
  judge.py               # LLM-as-judge: вердикт по каждому факту ответа + overall_score
pipeline.py             # оркестратор: classify -> respond -> judge -> [rework если score < 0.6] -> hallucination check
eval.py                 # метрики correctness (accuracy, judge pass-rate) и path (звонки/токены/стоимость)
scripts/prepare_data.py # анализ ключевых слов по категориям + стратифицированная выборка eval-сета
kb/                     # 15 синтетических markdown-документов базы знаний (3 на категорию)
input/raw_subset_tickets.csv  # исходные 3000 тикетов, 5 категорий
input/test_tickets.csv         # eval-сет (30 тикетов, сгенерирован prepare_data.py)
output/traces/<id>.json        # полный трейс на тикет: extraction, ответ, judge, hallucination, путь
output/eval_table.csv, eval_summary.json, category_keywords.json
отчёт.md
```

## Данные

`input/raw_subset_tickets.csv` — 3000 IT-support тикетов, 5 сбалансированных категорий `CategoryA..E`
(текст лемматизирован и анонимизирован, без имён и почти без дат). Реального описания категорий в данных
нет — `scripts/prepare_data.py` восстанавливает вероятный смысл категорий по частотному анализу слов
(`output/category_keywords.json`), на основе которого вручную написана **синтетическая** база знаний
`kb/` (реальной KB для этого датасета нет — это явное допущение прикладного трека, обсуждается в отчёт.md).

## Ограничения (подробнее в отчёт.md)

- RAG — только BM25 (без dense/эмбеддингов): дешевле и быстрее для масштаба проекта, но слабее на парафразах.
- KB написана на русском (конвенция курса), ответы агента — на английском (как сами тикеты): это ограничивает
  текстовую fidelity-проверку галлюцинаций до сопоставления по цифрам (язык-независимо).
- Категории — анонимные `CategoryA..E`; их смысловая расшифровка — наша гипотеза по словам, не подтверждённый факт.
