from __future__ import annotations

"""Phase A: RAGAS Production Evaluation: 50q, 3 distributions, cluster analysis."""

import json
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ANSWERS_PATH, TEST_SET_PATH

Distribution = str
DISTRIBUTIONS = ["factual", "multi_hop", "adversarial"]
METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

DIAGNOSTIC_TREE = {
    "faithfulness": ("LLM hallucinating", "Tighten system prompt, lower temperature"),
    "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
    "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
    "answer_relevancy": ("Answer does not match question", "Improve prompt template"),
}


@dataclass
class RagasResult:
    question_id: int
    distribution: Distribution
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float

    @property
    def avg_score(self) -> float:
        return (self.faithfulness + self.answer_relevancy + self.context_precision + self.context_recall) / 4

    @property
    def worst_metric(self) -> str:
        scores = {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
        }
        return min(scores, key=scores.get)


def load_test_set_50q(path: str = TEST_SET_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_answers(path: str = ANSWERS_PATH) -> list[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"answers_50q.json not found at {path}; run python setup_answers.py first")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def group_by_distribution(test_set: list[dict]) -> dict[str, list[dict]]:
    groups = {dist: [] for dist in DISTRIBUTIONS}
    for item in test_set:
        dist = item.get("distribution")
        if dist in groups:
            groups[dist].append(item)
    return groups


def _get_metric(row, name: str) -> float:
    if isinstance(row, dict):
        value = row.get(name, 0.0)
    else:
        value = getattr(row, name, 0.0)
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def run_ragas_50q(answers: list[dict]) -> list[RagasResult]:
    if not answers:
        return []
    from src.m4_eval import evaluate_ragas

    questions = [a.get("question", "") for a in answers]
    answer_texts = [a.get("answer", "") for a in answers]
    contexts = [a.get("contexts") or a.get("context") or [] for a in answers]
    contexts = [c if isinstance(c, list) else [str(c)] for c in contexts]
    ground_truths = [a.get("ground_truth", "") for a in answers]

    raw = evaluate_ragas(questions, answer_texts, contexts, ground_truths)
    per_question = raw.get("per_question", []) if isinstance(raw, dict) else []

    results: list[RagasResult] = []
    for i, answer in enumerate(answers):
        row = per_question[i] if i < len(per_question) else {}
        results.append(RagasResult(
            question_id=int(answer.get("id", answer.get("question_id", i + 1))),
            distribution=answer.get("distribution", "unknown"),
            question=answer.get("question", questions[i]),
            answer=answer.get("answer", answer_texts[i]),
            contexts=contexts[i],
            ground_truth=answer.get("ground_truth", ground_truths[i]),
            faithfulness=_get_metric(row, "faithfulness"),
            answer_relevancy=_get_metric(row, "answer_relevancy"),
            context_precision=_get_metric(row, "context_precision"),
            context_recall=_get_metric(row, "context_recall"),
        ))
    return results


def bottom_10(results: list[RagasResult]) -> list[dict]:
    output = []
    for rank, row in enumerate(sorted(results, key=lambda r: r.avg_score)[:10], start=1):
        diagnosis, fix = DIAGNOSTIC_TREE.get(row.worst_metric, ("Unknown", "Inspect retrieval and prompt"))
        output.append({
            "rank": rank,
            "question_id": row.question_id,
            "distribution": row.distribution,
            "question": row.question,
            "avg_score": round(row.avg_score, 4),
            "worst_metric": row.worst_metric,
            "diagnosis": diagnosis,
            "suggested_fix": fix,
        })
    return output


def cluster_analysis(results: list[RagasResult]) -> dict:
    matrix = {metric: {dist: 0 for dist in DISTRIBUTIONS} for metric in METRICS}
    for row in results:
        if row.worst_metric in matrix and row.distribution in matrix[row.worst_metric]:
            matrix[row.worst_metric][row.distribution] += 1

    dominant_dist = max(DISTRIBUTIONS, key=lambda d: sum(matrix[m][d] for m in METRICS)) if results else "none"
    dominant_metric = max(METRICS, key=lambda m: sum(matrix[m].values())) if results else "none"
    fix = DIAGNOSTIC_TREE.get(dominant_metric, ("", "Inspect failed examples"))[1]
    insight = f"Most failures are in {dominant_dist}; weakest metric is {dominant_metric}. Suggested next step: {fix}."
    return {
        "matrix": matrix,
        "dominant_failure_distribution": dominant_dist,
        "dominant_failure_metric": dominant_metric,
        "insight": insight,
    }


def save_phase_a_report(results: list[RagasResult], clusters: dict, path: str = "reports/ragas_50q.json") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    per_dist: dict[str, dict] = {}
    for dist in DISTRIBUTIONS:
        subset = [r for r in results if r.distribution == dist]
        if subset:
            per_dist[dist] = {
                "count": len(subset),
                **{metric: round(sum(getattr(r, metric) for r in subset) / len(subset), 4) for metric in METRICS},
                "avg_score": round(sum(r.avg_score for r in subset) / len(subset), 4),
            }
    report = {
        "total_questions": len(results),
        "per_distribution": per_dist,
        "failure_clusters": clusters,
        "bottom_10": bottom_10(results),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Phase A report saved -> {path}")


if __name__ == "__main__":
    test_set = load_test_set_50q()
    groups = group_by_distribution(test_set)
    for dist, items in groups.items():
        print(f"{dist}: {len(items)} questions")
    answers = load_answers()
    results = run_ragas_50q(answers)
    clusters = cluster_analysis(results)
    save_phase_a_report(results, clusters)