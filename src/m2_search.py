from __future__ import annotations

"""Module 2: hybrid search with BM25, dense-style fallback, and RRF."""

import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import COLLECTION_NAME, DENSE_TOP_K, EMBEDDING_DIM, EMBEDDING_MODEL, HYBRID_TOP_K, BM25_TOP_K, QDRANT_HOST, QDRANT_PORT


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str


def segment_vietnamese(text: str) -> str:
    try:
        from underthesea import word_tokenize
        return word_tokenize(text, format="text").replace("_", " ")
    except Exception:
        return re.sub(r"\s+", " ", text).strip()


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", segment_vietnamese(text).lower(), flags=re.UNICODE)


class BM25Search:
    def __init__(self):
        self.corpus_tokens: list[list[str]] = []
        self.documents: list[dict] = []
        self.idf: dict[str, float] = {}
        self.avgdl = 0.0

    def index(self, chunks: list[dict]) -> None:
        self.documents = chunks
        self.corpus_tokens = [_tokenize(chunk["text"]) for chunk in chunks]
        self.avgdl = sum(len(tokens) for tokens in self.corpus_tokens) / max(len(self.corpus_tokens), 1)
        doc_freq: Counter[str] = Counter()
        for tokens in self.corpus_tokens:
            doc_freq.update(set(tokens))
        n_docs = max(len(self.corpus_tokens), 1)
        self.idf = {term: math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5)) for term, freq in doc_freq.items()}

    def _score(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        if not query_tokens or not doc_tokens:
            return 0.0
        tf = Counter(doc_tokens)
        k1, b = 1.5, 0.75
        dl = len(doc_tokens)
        score = 0.0
        for term in query_tokens:
            freq = tf.get(term, 0)
            if freq == 0:
                continue
            denom = freq + k1 * (1 - b + b * dl / max(self.avgdl, 1e-9))
            score += self.idf.get(term, 0.0) * freq * (k1 + 1) / denom
        return score

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        query_tokens = _tokenize(query)
        scored = [(self._score(query_tokens, tokens), i) for i, tokens in enumerate(self.corpus_tokens)]
        scored.sort(reverse=True)
        results: list[SearchResult] = []
        for score, idx in scored[:top_k]:
            if score <= 0:
                continue
            doc = self.documents[idx]
            results.append(SearchResult(doc["text"], float(score), doc.get("metadata", {}), "bm25"))
        return results


class DenseSearch:
    def __init__(self):
        self.documents: list[dict] = []
        self.doc_vectors: list[Counter[str]] = []
        self.client = None
        self.encoder = None
        self.use_qdrant = False

    def _load_production_stack(self) -> bool:
        if self.use_qdrant:
            return True
        try:
            from qdrant_client import QdrantClient
            from sentence_transformers import SentenceTransformer

            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
            self.client.get_collections()
            self.encoder = SentenceTransformer(EMBEDDING_MODEL)
            self.use_qdrant = True
            return True
        except Exception:
            self.client = None
            self.encoder = None
            self.use_qdrant = False
            return False

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        self.documents = chunks
        self.doc_vectors = [Counter(_tokenize(chunk["text"])) for chunk in chunks]
        if not chunks or not self._load_production_stack():
            return
        try:
            from qdrant_client.models import Distance, PointStruct, VectorParams

            self.client.recreate_collection(
                collection,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
            texts = [chunk["text"] for chunk in chunks]
            vectors = self.encoder.encode(texts, show_progress_bar=False)
            points = [
                PointStruct(
                    id=i,
                    vector=vector.tolist() if hasattr(vector, "tolist") else list(vector),
                    payload={**chunk.get("metadata", {}), "text": chunk["text"]},
                )
                for i, (chunk, vector) in enumerate(zip(chunks, vectors))
            ]
            self.client.upsert(collection, points=points)
        except Exception:
            self.use_qdrant = False

    def _cosine(self, a: Counter[str], b: Counter[str]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(a[t] * b.get(t, 0) for t in a)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        return dot / max(na * nb, 1e-9)

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        if self.use_qdrant and self.client is not None and self.encoder is not None:
            try:
                query_vector = self.encoder.encode(query)
                query_vector = query_vector.tolist() if hasattr(query_vector, "tolist") else list(query_vector)
                response = self.client.query_points(collection, query=query_vector, limit=top_k)
                return [
                    SearchResult(
                        text=point.payload.get("text", ""),
                        score=float(point.score),
                        metadata={k: v for k, v in point.payload.items() if k != "text"},
                        method="dense",
                    )
                    for point in response.points
                ]
            except Exception:
                self.use_qdrant = False

        qv = Counter(_tokenize(query))
        scored = [(self._cosine(qv, vec), i) for i, vec in enumerate(self.doc_vectors)]
        scored.sort(reverse=True)
        results: list[SearchResult] = []
        for score, idx in scored[:top_k]:
            if score <= 0:
                continue
            doc = self.documents[idx]
            results.append(SearchResult(doc["text"], float(score), doc.get("metadata", {}), "dense"))
        return results


def reciprocal_rank_fusion(
    results_list: list[list[SearchResult]],
    k: int = 60,
    top_k: int = HYBRID_TOP_K,
) -> list[SearchResult]:
    fused: dict[str, dict] = {}
    for results in results_list:
        for rank, result in enumerate(results):
            if result.text not in fused:
                fused[result.text] = {"score": 0.0, "result": result}
            fused[result.text]["score"] += 1.0 / (k + rank + 1)
    ranked = sorted(fused.values(), key=lambda item: item["score"], reverse=True)
    return [
        SearchResult(item["result"].text, float(item["score"]), item["result"].metadata, "hybrid")
        for item in ranked[:top_k]
    ]


class HybridSearch:
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(segment_vietnamese("Nhan vien duoc nghi phep nam"))
