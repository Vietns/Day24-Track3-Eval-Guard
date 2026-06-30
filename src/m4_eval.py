from __future__ import annotations

"""Module 4: RAGAS-style evaluation and failure analysis."""

import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def _overlap_score(source: str, target: str) -> float:
    src, tgt = _tokens(source), _tokens(target)
    if not tgt:
        return 0.0
    return min(1.0, len(src & tgt) / len(tgt))


def _heuristic_eval(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict:
    rows: list[EvalResult] = []
    for q, a, ctx, gt in zip(questions, answers, contexts, ground_truths):
        context_text = " ".join(ctx)
        faithfulness = _overlap_score(a, context_text)
        answer_relevancy = max(_overlap_score(a, q), _overlap_score(a, gt))
        context_precision = _overlap_score(q + " " + gt, context_text)
        context_recall = _overlap_score(context_text, gt)
        rows.append(EvalResult(
            question=q,
            answer=a,
            contexts=ctx,
            ground_truth=gt,
            faithfulness=round(faithfulness, 4),
            answer_relevancy=round(answer_relevancy, 4),
            context_precision=round(context_precision, 4),
            context_recall=round(context_recall, 4),
        ))
    return {
        "faithfulness": round(mean([r.faithfulness for r in rows]) if rows else 0.0, 4),
        "answer_relevancy": round(mean([r.answer_relevancy for r in rows]) if rows else 0.0, 4),
        "context_precision": round(mean([r.context_precision for r in rows]) if rows else 0.0, 4),
        "context_recall": round(mean([r.context_recall for r in rows]) if rows else 0.0, 4),
        "per_question": rows,
    }


def evaluate_ragas(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict:
    if OPENAI_API_KEY:
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

            dataset = Dataset.from_dict({
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            })
            result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
            df = result.to_pandas()
            rows = [
                EvalResult(
                    question=row["question"],
                    answer=row["answer"],
                    contexts=row["contexts"],
                    ground_truth=row["ground_truth"],
                    faithfulness=float(row.get("faithfulness", 0.0) or 0.0),
                    answer_relevancy=float(row.get("answer_relevancy", 0.0) or 0.0),
                    context_precision=float(row.get("context_precision", 0.0) or 0.0),
                    context_recall=float(row.get("context_recall", 0.0) or 0.0),
                )
                for _, row in df.iterrows()
            ]
            return {
                "faithfulness": round(mean([r.faithfulness for r in rows]) if rows else 0.0, 4),
                "answer_relevancy": round(mean([r.answer_relevancy for r in rows]) if rows else 0.0, 4),
                "context_precision": round(mean([r.context_precision for r in rows]) if rows else 0.0, 4),
                "context_recall": round(mean([r.context_recall for r in rows]) if rows else 0.0, 4),
                "per_question": rows,
            }
        except Exception as exc:
            print(f"  RAGAS evaluation failed, using heuristic fallback: {exc}")
    return _heuristic_eval(questions, answers, contexts, ground_truths)


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating or answer not grounded", "Tighten prompt and cite retrieved context."),
        "context_recall": ("Missing relevant chunks", "Improve chunking, add BM25 terms, or increase top_k."),
        "context_precision": ("Too many irrelevant chunks", "Add reranking, metadata filters, or query rewriting."),
        "answer_relevancy": ("Answer does not match the question", "Improve answer prompt and rerank by question intent."),
    }
    scored: list[tuple[float, EvalResult, str]] = []
    for row in eval_results:
        metric_values = {
            "faithfulness": row.faithfulness,
            "answer_relevancy": row.answer_relevancy,
            "context_precision": row.context_precision,
            "context_recall": row.context_recall,
        }
        avg = sum(metric_values.values()) / 4
        worst_metric = min(metric_values, key=metric_values.get)
        scored.append((avg, row, worst_metric))
    scored.sort(key=lambda item: item[0])
    failures = []
    for avg, row, worst_metric in scored[:bottom_n]:
        diagnosis, fix = diagnostic_tree[worst_metric]
        failures.append({
            "question": row.question,
            "answer": row.answer,
            "ground_truth": row.ground_truth,
            "worst_metric": worst_metric,
            "score": round(avg, 4),
            "diagnosis": diagnosis,
            "suggested_fix": fix,
        })
    return failures


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "per_question": [
            asdict(item) if isinstance(item, EvalResult) else item
            for item in results.get("per_question", [])
        ],
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    print(f"Loaded {len(load_test_set())} test questions")
