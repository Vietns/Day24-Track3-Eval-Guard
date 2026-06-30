from __future__ import annotations

"""Module 5: chunk enrichment pipeline."""

import os
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.llm_client import chat_json


@dataclass
class EnrichedChunk:
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def _llm_json(system: str, user: str, max_tokens: int = 400) -> dict | None:
    return chat_json(system, user, max_tokens=max_tokens)


def summarize_chunk(text: str) -> str:
    payload = _llm_json(
        "Return JSON only: {\"summary\": \"2 short Vietnamese sentences\"}.",
        text,
        max_tokens=150,
    )
    if payload and payload.get("summary"):
        return str(payload["summary"])
    sentences = _sentences(text)
    return ". ".join(s.rstrip(".") for s in sentences[:2]) + ("." if sentences else "")


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    payload = _llm_json(
        f"Return JSON only: {{\"questions\": [exactly {n_questions} Vietnamese questions answerable from the text]}}.",
        text,
        max_tokens=200,
    )
    if payload and isinstance(payload.get("questions"), list):
        return [str(q).strip() for q in payload["questions"] if str(q).strip()][:n_questions]

    lower = text.lower()
    questions: list[str] = []
    if any(term in lower for term in ["nghi", "nghá", "phép", "ngày"]):
        questions.append("Nhân viên được nghỉ phép bao nhiêu ngày?")
    if any(term in lower for term in ["mật khẩu", "mat khau", "password"]):
        questions.append("Chính sách mật khẩu yêu cầu gì?")
    if any(term in lower for term in ["thử việc", "thu viec"]):
        questions.append("Thời gian thử việc là bao lâu?")
    for sentence in _sentences(text):
        if len(questions) >= n_questions:
            break
        questions.append(f"{sentence.rstrip('.')}?")
    return questions[:n_questions]


def contextual_prepend(text: str, document_title: str = "") -> str:
    prefix = f"Trích từ tài liệu {document_title}. " if document_title else "Ngữ cảnh tài liệu nội bộ. "
    topic = extract_metadata(text).get("topic", "chính sách")
    return f"{prefix}Đoạn này nói về {topic}.\n\n{text}"


def extract_metadata(text: str) -> dict:
    payload = _llm_json(
        'Return JSON only: {"topic": "...", "entities": ["..."], "category": "hr|it|finance|policy|other", "language": "vi|en"}.',
        text,
        max_tokens=150,
    )
    if payload:
        return payload

    lower = text.lower()
    if any(term in lower for term in ["mật khẩu", "vpn", "password", "wireguard"]):
        category, topic = "it", "công nghệ thông tin"
    elif any(term in lower for term in ["lương", "thưởng", "phụ cấp"]):
        category, topic = "finance", "lương thưởng"
    elif any(term in lower for term in ["nghỉ", "thử việc", "nhân viên"]):
        category, topic = "hr", "nhân sự"
    else:
        category, topic = "policy", "chính sách nội bộ"

    entities = re.findall(r"\b[A-ZÀ-Ỵ][\wÀ-ỹ-]{2,}\b", text)
    return {"topic": topic, "entities": entities[:5], "category": category, "language": "vi"}


def _enrich_single_call(text: str, source: str) -> dict:
    payload = _llm_json(
        """Return JSON only:
{
  "summary": "...",
  "questions": ["...", "...", "..."],
  "context": "one sentence document context",
  "metadata": {"topic": "...", "entities": ["..."], "category": "hr|it|finance|policy|other", "language": "vi|en"}
}""",
        f"Source: {source}\n\n{text}",
        max_tokens=400,
    )
    if payload:
        return payload
    meta = extract_metadata(text)
    return {
        "summary": summarize_chunk(text),
        "questions": generate_hypothesis_questions(text),
        "context": f"Trích từ {source}; nội dung liên quan đến {meta.get('topic', 'chính sách nội bộ')}.",
        "metadata": meta,
    }


def enrich_chunks(chunks: list[dict], methods: list[str] | None = None) -> list[EnrichedChunk]:
    if methods is None:
        methods = ["combined"]
    use_combined = "combined" in methods
    enriched: list[EnrichedChunk] = []

    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")
        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        ))
        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm."
    print(_enrich_single_call(sample, "demo.md"))
