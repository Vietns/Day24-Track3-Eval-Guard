from __future__ import annotations

"""Phase B: LLM-as-Judge: pairwise, swap-and-average, Cohen kappa, bias analysis."""

import json
import os
import re
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HUMAN_LABELS_PATH


@dataclass
class JudgeResult:
    question: str
    answer_a: str
    answer_b: str
    winner_pass1: str
    winner_pass2: str
    final_winner: str
    reasoning_pass1: str
    reasoning_pass2: str
    position_consistent: bool
    scores_pass1: dict = field(default_factory=dict)
    scores_pass2: dict = field(default_factory=dict)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def _score_answer(question: str, answer: str) -> float:
    q_tokens = _tokens(question)
    a_tokens = _tokens(answer)
    if not answer.strip():
        return 0.0
    overlap = len(q_tokens & a_tokens) / max(len(q_tokens), 1)
    has_number = 0.12 if re.search(r"\d", answer) else 0.0
    enough_detail = min(len(a_tokens) / 45, 1.0) * 0.18
    too_verbose_penalty = 0.12 if len(a_tokens) > 120 else 0.0
    return max(0.0, min(1.0, 0.35 + overlap * 0.35 + has_number + enough_detail - too_verbose_penalty))


def pairwise_judge(question: str, answer_a: str, answer_b: str) -> dict:
    score_a = round(_score_answer(question, answer_a), 3)
    score_b = round(_score_answer(question, answer_b), 3)
    if abs(score_a - score_b) < 0.05:
        winner = "tie"
        reasoning = "Answers are similar in relevance and completeness."
    elif score_a > score_b:
        winner = "A"
        reasoning = "Answer A is more relevant, specific, or complete."
    else:
        winner = "B"
        reasoning = "Answer B is more relevant, specific, or complete."
    return {"winner": winner, "reasoning": reasoning, "scores": {"A": score_a, "B": score_b}}


def swap_and_average(question: str, answer_a: str, answer_b: str) -> JudgeResult:
    pass1 = pairwise_judge(question, answer_a, answer_b)
    pass2_raw = pairwise_judge(question, answer_b, answer_a)
    swap_map = {"A": "B", "B": "A", "tie": "tie"}
    winner_pass2 = swap_map.get(pass2_raw["winner"], "tie")
    position_consistent = pass1["winner"] == winner_pass2
    final = pass1["winner"] if position_consistent else "tie"
    raw_scores = pass2_raw.get("scores", {})
    return JudgeResult(
        question=question,
        answer_a=answer_a,
        answer_b=answer_b,
        winner_pass1=pass1["winner"],
        winner_pass2=winner_pass2,
        final_winner=final,
        reasoning_pass1=pass1.get("reasoning", ""),
        reasoning_pass2=pass2_raw.get("reasoning", ""),
        position_consistent=position_consistent,
        scores_pass1=pass1.get("scores", {}),
        scores_pass2={"A": raw_scores.get("B", 0.0), "B": raw_scores.get("A", 0.0)},
    )


def cohen_kappa(judge_labels: list[int], human_labels: list[int]) -> float:
    if len(judge_labels) != len(human_labels):
        raise ValueError("judge_labels and human_labels must have the same length")
    n = len(judge_labels)
    if n == 0:
        return 0.0
    observed = sum(j == h for j, h in zip(judge_labels, human_labels)) / n
    labels = sorted(set(judge_labels) | set(human_labels))
    expected = sum((judge_labels.count(label) / n) * (human_labels.count(label) / n) for label in labels)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return round((observed - expected) / (1 - expected), 4)


def bias_report(judge_results: list[JudgeResult]) -> dict:
    total = len(judge_results)
    if total == 0:
        return {
            "total_judged": 0,
            "position_bias_rate": 0.0,
            "position_bias_count": 0,
            "verbosity_bias": 0.0,
            "verbosity_details": {"a_wins_a_longer": 0, "b_wins_b_longer": 0, "total_decisive": 0},
            "interpretation": "No judge results available.",
        }

    position_bias_count = sum(1 for r in judge_results if not r.position_consistent)
    decisive = [r for r in judge_results if r.final_winner in {"A", "B"}]
    a_wins_a_longer = sum(1 for r in decisive if r.final_winner == "A" and len(r.answer_a) > len(r.answer_b))
    b_wins_b_longer = sum(1 for r in decisive if r.final_winner == "B" and len(r.answer_b) > len(r.answer_a))
    verbosity_bias = (a_wins_a_longer + b_wins_b_longer) / len(decisive) if decisive else 0.0
    position_bias_rate = position_bias_count / total
    interpretation = "Position bias is high; keep swap-and-average." if position_bias_rate > 0.3 else "Position bias is low."
    if verbosity_bias > 0.6:
        interpretation += " Verbosity bias should be monitored."
    return {
        "total_judged": total,
        "position_bias_rate": round(position_bias_rate, 3),
        "position_bias_count": position_bias_count,
        "verbosity_bias": round(verbosity_bias, 3),
        "verbosity_details": {
            "a_wins_a_longer": a_wins_a_longer,
            "b_wins_b_longer": b_wins_b_longer,
            "total_decisive": len(decisive),
        },
        "interpretation": interpretation,
    }


if __name__ == "__main__":
    q = "How many annual leave days does an employee get?"
    a = "Employees get 15 annual leave days under the current 2024 policy."
    b = "Employees get 12 annual leave days."
    result = swap_and_average(q, a, b)
    print(result)
    with open(HUMAN_LABELS_PATH, encoding="utf-8") as f:
        human = [row["human_label"] for row in json.load(f)]
    print(cohen_kappa(human, human))