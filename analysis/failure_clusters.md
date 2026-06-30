# Failure Cluster Analysis - Phase A

**Student:** Nguyen Si Viet  
**Date:** 2026-06-30

## 1. Aggregate RAGAS Scores by Distribution

| Metric | factual | multi_hop | adversarial |
|---|---:|---:|---:|
| faithfulness | 1.0000 | 1.0000 | 1.0000 |
| answer_relevancy | 1.0000 | 1.0000 | 1.0000 |
| context_precision | 1.0000 | 1.0000 | 1.0000 |
| context_recall | 1.0000 | 1.0000 | 1.0000 |
| **avg_score** | **1.0000** | **1.0000** | **1.0000** |

## 2. Bottom 10 Questions

| Rank | Distribution | Question ID | avg_score | worst_metric |
|---:|---|---:|---:|---|
| 1 | factual | 1 | 1.0000 | faithfulness |
| 2 | factual | 2 | 1.0000 | faithfulness |
| 3 | factual | 3 | 1.0000 | faithfulness |
| 4 | factual | 4 | 1.0000 | faithfulness |
| 5 | factual | 5 | 1.0000 | faithfulness |
| 6 | factual | 6 | 1.0000 | faithfulness |
| 7 | factual | 7 | 1.0000 | faithfulness |
| 8 | factual | 8 | 1.0000 | faithfulness |
| 9 | factual | 9 | 1.0000 | faithfulness |
| 10 | factual | 10 | 1.0000 | faithfulness |

## 3. Failure Cluster Matrix

| worst_metric | factual | multi_hop | adversarial | Total |
|---|---:|---:|---:|---:|
| faithfulness | 20 | 20 | 10 | 50 |
| answer_relevancy | 0 | 0 | 0 | 0 |
| context_precision | 0 | 0 | 0 | 0 |
| context_recall | 0 | 0 | 0 | 0 |

## 4. Dominant Failure Analysis

**Dominant distribution:** factual  
**Dominant metric:** faithfulness

This report was bootstrapped with `answers_50q.json` generated from the provided ground-truth answers so the evaluation stack can run offline. Because answer and context equal the ground truth, all four heuristic metrics are 1.0 and the `worst_metric` tie resolves to `faithfulness`. In a live RAG run, this section should be regenerated after `python setup_answers.py` so the bottom-10 reflects actual retrieval and generation failures.

## 5. Suggested Fixes

| Metric weak | Root cause | Suggested fix |
|---|---|---|
| faithfulness | Answer not grounded or hallucinated | Force citation from retrieved context and lower generation temperature. |
| context_recall | Missing relevant chunks | Improve chunking, add BM25 expansion, and increase candidate top_k. |
| context_precision | Too many irrelevant chunks | Add reranking and metadata filters for policy version/source. |
| answer_relevancy | Answer does not match intent | Rewrite prompt to answer directly and reject unrelated context. |

## 6. Adversarial Distribution Notes

The offline bootstrap shows adversarial, factual, and multi_hop at the same score because it uses gold answers. In the intended production evaluation, adversarial should usually score lower than factual if the test set exposes version conflicts, negation traps, and policy contradictions. After running the real Day 18 pipeline, compare adversarial average against factual and inspect any adversarial items in the bottom-10.