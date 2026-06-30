# LLM Judge Bias Report - Phase B

**Student:** Nguyen Si Viet  
**Date:** 2026-06-30  
**Judge model:** heuristic offline judge; configured live model: llama-3.3-70b-versatile via Groq

## 1. Pairwise Judge Results

| # | Question summary | Winner | Reasoning summary |
|---:|---|---|---|
| 1 | Marriage leave | tie | Similar relevance and completeness in offline bootstrap. |
| 2 | Purchase approval | tie | Similar relevance and completeness in offline bootstrap. |
| 3 | Data classification | tie | Similar relevance and completeness in offline bootstrap. |
| 4 | Performance review | tie | Similar relevance and completeness in offline bootstrap. |
| 5 | Trial salary | tie | Similar relevance and completeness in offline bootstrap. |

## 2. Swap-and-Average Results

| # | Pass 1 Winner | Pass 2 Winner | Final | Position Consistent? |
|---:|---|---|---|---|
| 1 | tie | tie | tie | true |
| 2 | tie | tie | tie | true |
| 3 | tie | tie | tie | true |
| 4 | tie | tie | tie | true |
| 5 | tie | tie | tie | true |

**Position bias rate:** 0.0% (= 0 inconsistent / 10 total)

## 3. Cohen Kappa Analysis

**Human labels:** `human_labels_10q.json`  
**Judge labels:** `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]`

| Question # | Human Label | Judge Label | Agree? |
|---:|---:|---:|---|
| 1 | 1 | 1 | yes |
| 2 | 0 | 1 | no |
| 3 | 1 | 1 | yes |
| 4 | 1 | 1 | yes |
| 5 | 1 | 1 | yes |
| 6 | 0 | 1 | no |
| 7 | 1 | 1 | yes |
| 8 | 0 | 1 | no |
| 9 | 1 | 1 | yes |
| 10 | 0 | 1 | no |

**Cohen kappa:** 0.0  
**Interpretation:** poor/slight agreement for the offline bootstrap. A real LLM judge should be run with the configured Groq model for production scoring.

## 4. Verbosity Bias

In decisive cases (non-tie):
- A wins + A longer than B: 0 / 0 cases
- B wins + B longer than A: 0 / 0 cases
- **Verbosity bias rate:** 0.0%

The offline judge produced ties for the synthetic pairs, so verbosity bias is not observable in this bootstrap run. In production, track whether the selected answer is simply longer rather than more accurate.

## 5. Overall Notes

The implementation includes pairwise scoring, swap-and-average conversion, Cohen kappa, and bias aggregation. The current report is generated without paid LLM calls so tests and deliverables are reproducible. For production, run the judge against real answer pairs using Groq/OpenAI-compatible configuration, keep swap-and-average enabled, and alert when position bias exceeds 30% or kappa stays below 0.6.