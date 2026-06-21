"""ResponderAgent: tool-calling агент (RAG через search_kb), пишет ответ клиенту."""
from __future__ import annotations

import json
import sys
from json.decoder import JSONDecodeError
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_client import get_model, make_raw_client
from kb import SEARCH_KB_SCHEMA, search_kb
from schemas import ResponderReply

SUBMIT_REPLY_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit_reply",
        "description": "Вызови, когда нашёл достаточно фактов через search_kb и готов дать финальный ответ клиенту.",
        "parameters": {
            "type": "object",
            "properties": {
                "reply": {"type": "string", "description": "Текст ответа клиенту, на английском"},
                "cited_facts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Короткие факты/цифры из найденных KB-чанков, использованные в ответе",
                },
                "kb_chunk_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["reply", "cited_facts", "kb_chunk_ids"],
        },
    },
}

TOOLS = [SEARCH_KB_SCHEMA, SUBMIT_REPLY_SCHEMA]

SYSTEM_PROMPT = """Ты — агент службы поддержки. Тебе дают тикет клиента и его категорию.
Твоя задача: через инструмент search_kb найти релевантные факты внутренней базы знаний (SLA, сроки, лимиты,
контакты) и написать короткий ответ клиенту (3-5 предложений, на английском), который ссылается на конкретные
найденные факты.

Правила:
1. ЧИСЛА (сроки, суммы, лимиты) НИКОГДА НЕ ПРИДУМЫВАЙ — бери их только из результатов search_kb.
2. Если search_kb не нашёл ничего релевантного — явно скажи клиенту, что вопрос передан специалисту, без выдумки.
3. Когда готов — вызови submit_reply: reply (текст), cited_facts (факты, которые использовал), kb_chunk_ids
   (id чанков, на которые опирался).
4. Не вызывай search_kb больше 3 раз.
"""

PRICE_IN_PER_MTOK = 0.14
PRICE_OUT_PER_MTOK = 0.28


def _exec_search(args: dict) -> dict:
    try:
        return search_kb(**args)
    except TypeError as e:
        return {"error": f"плохие аргументы для search_kb: {e}"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def run_responder(description: str, category: str, *, max_iter: int = 5, feedback: str | None = None) -> dict[str, Any]:
    client = make_raw_client()
    model = get_model()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Категория тикета: {category}\nТекст тикета:\n{description}"},
    ]
    if feedback:
        messages.append({
            "role": "user",
            "content": f"Твой предыдущий ответ оценил аудитор: {feedback}\nПерепроверь факты через search_kb и перепиши ответ.",
        })
    trace: list[dict[str, Any]] = []
    usage_log: list[dict[str, Any]] = []
    retrieved_chunks: dict[str, str] = {}

    for step in range(1, max_iter + 1):
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=TOOLS, tool_choice="auto", temperature=0.0, max_tokens=512
        )
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        u = getattr(resp, "usage", None)
        if u is not None:
            cost = u.prompt_tokens / 1e6 * PRICE_IN_PER_MTOK + u.completion_tokens / 1e6 * PRICE_OUT_PER_MTOK
            usage_log.append(
                {"step": step, "prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens,
                 "cost_usd": round(cost, 6)}
            )

        if not msg.tool_calls:
            # Модель ответила текстом, не вызвав submit_reply — считаем это финалом без citations.
            trace.append({"step": step, "kind": "final_text_no_submit"})
            reply = ResponderReply(reply=msg.content or "", cited_facts=[], kb_chunk_ids=[])
            return _result(reply, trace, usage_log, retrieved_chunks, step)

        submitted = None
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except JSONDecodeError as e:
                obs = {"error": f"битый json аргументов: {e}"}
                args = {}
            else:
                if tc.function.name == "search_kb":
                    obs = _exec_search(args)
                    for r in obs.get("results", []):
                        retrieved_chunks[r["chunk_id"]] = r["text"]
                elif tc.function.name == "submit_reply":
                    submitted = args
                    obs = {"status": "ok"}
                else:
                    obs = {"error": f"неизвестный инструмент: {tc.function.name}"}

            trace.append({"step": step, "kind": "tool_call", "call": tc.function.name, "args": args, "obs": obs})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(obs, ensure_ascii=False)})

        if submitted is not None:
            try:
                reply = ResponderReply.model_validate(submitted)
            except Exception:
                reply = ResponderReply(
                    reply=submitted.get("reply", ""),
                    cited_facts=submitted.get("cited_facts") or [],
                    kb_chunk_ids=submitted.get("kb_chunk_ids") or [],
                )
            return _result(reply, trace, usage_log, retrieved_chunks, step)

    trace.append({"step": max_iter, "kind": "max_iter_exceeded"})
    reply = ResponderReply(reply="", cited_facts=[], kb_chunk_ids=[])
    return _result(reply, trace, usage_log, retrieved_chunks, max_iter)


def _result(reply: ResponderReply, trace, usage_log, retrieved_chunks, n_steps) -> dict[str, Any]:
    return {
        "reply": reply,
        "trace": trace,
        "n_steps": n_steps,
        "n_tool_calls": sum(1 for t in trace if t.get("kind") == "tool_call"),
        "usage": {
            "prompt_tokens": sum(u["prompt_tokens"] for u in usage_log),
            "completion_tokens": sum(u["completion_tokens"] for u in usage_log),
            "cost_usd": round(sum(u["cost_usd"] for u in usage_log), 6),
        },
        "retrieved_chunks": retrieved_chunks,
    }
