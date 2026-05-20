# Local Eval Results Metrics

This document explains the columns produced by `scripts/openai_compatible_eval.py`.
The output CSV is intended for evaluating precomputed RAG answers with
OpenAI-compatible local chat and embedding models.

## Identity Columns

| Column | Meaning |
| --- | --- |
| `query_id` | Stable query identifier. Used to join generated answers with golden answers. |
| `query` | User question being evaluated. |
| `query_run` | Run number for repeated answers to the same query. |
| `generated_answer` | Generated answer text after citation markers are parsed/removed from answer parts. |

## Retrieval Metrics

| Column | Meaning |
| --- | --- |
| `retrieval_score_umbrela_scores` | JSON map from `passage_id` to an UMBRELA-style relevance score from 0 to 3. |
| `retrieval_score_precision_metrics` | JSON object containing `precision@K`, `AP@K`, and `MRR` derived from UMBRELA scores. Passages with UMBRELA score >= 2 are treated as relevant. |
| `retrieval_score_mean_umbrela_score` | Average UMBRELA score across retrieved passages. Higher means the retrieved context is more relevant to the query. |

UMBRELA score scale:

| Score | Meaning |
| --- | --- |
| `0` | Passage is unrelated to the query. |
| `1` | Passage is related but does not answer the query. |
| `2` | Passage contains some answer, possibly mixed with extra information. |
| `3` | Passage directly and specifically answers the query. |

Important: the script does not compute true retrieval recall because true
`Recall@K` requires a manually labeled set of all relevant/golden passages for
each query.

## Nugget Coverage Metrics

| Column | Meaning |
| --- | --- |
| `generation_score_autonugget_scores` | JSON object containing generated nuggets, vital/okay labels, support assignments, and aggregate nuggetizer scores. |
| `generation_score_mean_nugget_assignment_score` | Average support score over generated nuggets. `support=1.0`, `partial_support=0.5`, `not_support=0.0`. Higher means the answer covers more nugget facts. |
| `generation_score_vital_nuggetizer_score` | Coverage score for nuggets labeled `vital`. This is especially useful as an answer-level recall signal. |

For recall-focused analysis, `generation_score_vital_nuggetizer_score` is often
more useful than the all-nugget average because it emphasizes key information.

## Source Support And Citation Metrics

| Column | Meaning |
| --- | --- |
| `generation_score_hallucination_score` | LLM-judge source-support score from 0 to 1. Higher means the generated answer is judged to be better supported by retrieved sources. This is not the original HHEM score. |
| `generation_score_citation_scores` | JSON object with citation-level and answer-part-level support scores. |
| `generation_score_citation_f1_score` | F1 score combining citation weighted precision and answer-part weighted recall. Higher means citations better support the cited answer parts. |
| `generation_score_no_answer_score` | JSON object indicating whether the generated response attempts to answer the query, e.g. `{"query_answered": "yes"}`. |

Use `generation_score_citation_f1_score` carefully. It can be very strict when
one answer sentence combines facts from multiple passages or when citations are
attached at paragraph level instead of sentence level.

## Golden Answer Metrics

These columns are present when the row has a matching `query_id` in the golden
answer CSV.

| Column | Meaning |
| --- | --- |
| `generation_score_semantic_similarity` | Cosine similarity between generated answer embedding and expected answer embedding. Higher means closer semantic meaning. |
| `generation_score_factual_correctness_precision` | Fraction of generated factual claims entailed by the expected answer. Lower values often mean the answer includes extra claims not present in the golden answer. |
| `generation_score_factual_correctness_recall` | Fraction of expected/golden factual claims covered by the generated answer. This is the most direct answer-level recall metric. |
| `generation_score_factual_correctness_f1` | Harmonic mean of factual precision and factual recall. |
| `generation_score_expected_answer` | Golden/reference answer used for comparison. |
| `generation_score_generated_claims` | JSON list of atomic factual claims extracted from the generated answer. |
| `generation_score_expected_claims` | JSON list of atomic factual claims extracted from the expected answer. |
| `generation_score_precision_verdicts` | JSON verdicts for generated claims checked against the expected answer. |
| `generation_score_recall_verdicts` | JSON verdicts for expected claims checked against the generated answer. |

For recall-focused analysis, prioritize:

1. `generation_score_factual_correctness_recall`
2. `generation_score_vital_nuggetizer_score`
3. `generation_score_mean_nugget_assignment_score`

## Token Usage Columns

| Column | Meaning |
| --- | --- |
| `total_input_tokens` | Total prompt/input tokens reported by the local chat model for that row. |
| `total_output_tokens` | Total completion/output tokens reported by the local chat model for that row. |
| `total_tokens` | Sum of input and output tokens. |

Token counts depend on whether the OpenAI-compatible server returns usage
metadata. If the server does not return usage, these values may be `0` or
underreported.

## Interpreting Summary Statistics

When using `scripts/summarize_recall_metrics.py`, the printed statistics mean:

| Statistic | Meaning |
| --- | --- |
| `mean` | Average score across rows. Useful for overall system level. |
| `median` | Middle score after sorting. Useful for typical-case performance. |
| `min` | Worst row. Useful for finding failure cases. |
| `p25` | 25th percentile. A quarter of rows are at or below this value. |
| `p75` | 75th percentile. Three quarters of rows are at or below this value. |
| `max` | Best row. Useful for seeing the system upper bound. |

