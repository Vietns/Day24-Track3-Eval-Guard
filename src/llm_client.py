from __future__ import annotations

"""Small OpenAI-compatible LLM client used by the Day 18 pipeline.

Set GROQ_API_KEY in .env to use Groq, or OPENAI_API_KEY to use OpenAI.
For Groq, keep LLM_BASE_URL=https://api.groq.com/openai/v1 and use a Groq
chat model such as llama-3.3-70b-versatile.
"""

import json

from openai import OpenAI

from config import JUDGE_MODEL, LLM_API_KEY, LLM_BASE_URL


def _client() -> OpenAI:
    if not LLM_API_KEY:
        raise RuntimeError("Missing GROQ_API_KEY or OPENAI_API_KEY in .env")
    kwargs = {"api_key": LLM_API_KEY}
    if LLM_BASE_URL:
        kwargs["base_url"] = LLM_BASE_URL
    return OpenAI(**kwargs)


def chat_completion(
    system: str,
    user: str,
    *,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 500,
) -> str:
    response = _client().chat.completions.create(
        model=model or JUDGE_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def chat_json(
    system: str,
    user: str,
    *,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 500,
) -> dict | None:
    content = chat_completion(
        system,
        user,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(content[start:end + 1])
            except json.JSONDecodeError:
                return None
        return None