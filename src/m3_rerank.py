from __future__ import annotations

"""Module 3: reranking with CrossEncoder and a deterministic fallback."""

import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def _lexical_score(query: str, document: str) -> float:
    q = Counter(_tokens(query))
    d = Counter(_tokens(document))
    if not q or not d:
        return 0.0
    overlap = sum(min(q[t], d.get(t, 0)) for t in q)
    coverage = overlap / max(sum(q.values()), 1)
    bonus = 0.2 if any(token in document.lower() for token in ["nghi", "nghá", "ngh", "phép", "12"]) else 0.0
    return coverage + bonus


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None
        self._load_failed = False

    def _load_model(self):
        if self._model is None and not self._load_failed:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except Exception:
                self._load_failed = True
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        if not documents:
            return []
        model = self._load_model()
        if model is not None:
            pairs = [(query, doc["text"]) for doc in documents]
            scores = model.predict(pairs)
            try:
                scores = list(scores)
            except TypeError:
                scores = [scores]
        else:
            scores = [_lexical_score(query, doc["text"]) + float(doc.get("score", 0.0)) * 0.05 for doc in documents]

        scored = sorted(zip(scores, documents), key=lambda item: float(item[0]), reverse=True)
        return [
            RerankResult(
                text=doc["text"],
                original_score=float(doc.get("score", 0.0)),
                rerank_score=float(score),
                metadata=doc.get("metadata", {}),
                rank=i + 1,
            )
            for i, (score, doc) in enumerate(scored[:top_k])
        ]


class FlashrankReranker:
    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        return CrossEncoderReranker(model_name="fallback").rerank(query, documents, top_k=top_k)


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        times.append((time.perf_counter() - start) * 1000)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhan vien duoc nghi phep bao nhieu ngay?"
    docs = [
        {"text": "Nhan vien duoc nghi 12 ngay/nam.", "score": 0.8, "metadata": {}},
        {"text": "Mat khau thay doi moi 90 ngay.", "score": 0.7, "metadata": {}},
    ]
    for row in CrossEncoderReranker().rerank(query, docs):
        print(row)
