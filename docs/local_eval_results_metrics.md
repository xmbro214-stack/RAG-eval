# Local Eval Results 指标说明

本文档说明 `scripts/openai_compatible_eval.py` 生成的
`local_eval_results.csv` 中各个字段的含义。该结果文件用于评估已经生成好的
RAG 回答，评估过程调用 OpenAI API compatible 的本地 chat 模型和 embedding
模型。

## 基础标识字段

| 字段 | 含义 |
| --- | --- |
| `query_id` | 问题的稳定 ID。用于把生成结果和 golden answer 对齐。 |
| `query` | 被评估的用户问题。 |
| `query_run` | 同一个问题的第几次运行结果。用于多次生成/一致性分析。 |
| `generated_answer` | 生成答案文本。脚本会解析引用标记后，把答案片段合并成该字段。 |

## 检索相关指标

| 字段 | 含义 |
| --- | --- |
| `retrieval_score_umbrela_scores` | JSON 字典，key 是 `passage_id`，value 是 UMBRELA 风格的相关性分数，范围 0 到 3。 |
| `retrieval_score_precision_metrics` | JSON 对象，包含基于 UMBRELA 分数计算出的 `precision@K`、`AP@K` 和 `MRR`。UMBRELA 分数 >= 2 的 passage 会被视为相关。 |
| `retrieval_score_mean_umbrela_score` | 所有 retrieved passages 的 UMBRELA 平均分。越高表示检索出来的上下文整体越相关。 |

UMBRELA 分数含义：

| 分数 | 含义 |
| --- | --- |
| `0` | passage 与 query 无关。 |
| `1` | passage 与 query 相关，但不能回答问题。 |
| `2` | passage 包含部分答案，但可能不够清晰，或者混有额外信息。 |
| `3` | passage 直接、完整、专门回答该问题。 |

注意：当前脚本没有计算严格意义上的检索召回率 `Recall@K`。真正的检索召回率
需要你为每个 query 标注“所有应该被检索到的相关/golden passages”。当前结果
中更适合作为检索质量参考的是 `retrieval_score_mean_umbrela_score`、
`precision@K`、`AP@K` 和 `MRR`。

## Nugget 覆盖指标

| 字段 | 含义 |
| --- | --- |
| `generation_score_autonugget_scores` | JSON 对象，包含自动生成的 nuggets、每个 nugget 的 vital/okay 标签、答案对 nugget 的支持情况，以及聚合分数。 |
| `generation_score_mean_nugget_assignment_score` | 所有 nuggets 的平均覆盖分。`support=1.0`，`partial_support=0.5`，`not_support=0.0`。越高表示答案覆盖了越多 nugget 信息。 |
| `generation_score_vital_nuggetizer_score` | 只针对 `vital` nuggets 的覆盖分。越高表示关键事实覆盖越好。 |

如果你关注“答案有没有答全”，`generation_score_vital_nuggetizer_score`
通常比全量 nugget 平均分更重要，因为它更强调关键点覆盖。

## 来源支持与引用指标

| 字段 | 含义 |
| --- | --- |
| `generation_score_hallucination_score` | LLM judge 给出的来源支持分，范围 0 到 1。越高表示生成答案越被检索来源支持。注意：这不是原项目默认的 HHEM 分数，而是轻量脚本里的本地 LLM judge 版本。 |
| `generation_score_citation_scores` | JSON 对象，包含 citation 级别和答案片段级别的支持分。 |
| `generation_score_citation_f1_score` | 综合 citation weighted precision 和 answer-part weighted recall 的 F1 分数。越高表示引用越能支撑对应答案片段。 |
| `generation_score_no_answer_score` | JSON 对象，表示答案是否尝试回答问题，例如 `{"query_answered": "yes"}`。 |

`generation_score_citation_f1_score` 需要谨慎解读。它对引用粒度非常敏感：
如果一个答案句子综合了多个 passage 的信息，或者引用是段落级而不是句子级，
该指标可能会被压得很低。

## Golden Answer 对比指标

这些字段只有在结果行的 `query_id` 能匹配到 golden answer CSV 时才会出现。

| 字段 | 含义 |
| --- | --- |
| `generation_score_semantic_similarity` | 生成答案和期望答案的 embedding cosine similarity。越高表示语义越接近。 |
| `generation_score_factual_correctness_precision` | 生成答案中的事实 claims，有多少能被期望答案支持。低分通常表示生成答案包含了 golden answer 中没有出现的额外事实。 |
| `generation_score_factual_correctness_recall` | 期望答案中的事实 claims，有多少被生成答案覆盖。这是最直接的“答案层面召回率”指标。 |
| `generation_score_factual_correctness_f1` | factual precision 和 factual recall 的调和平均。 |
| `generation_score_expected_answer` | 用于对比的 golden/reference answer。 |
| `generation_score_generated_claims` | 从生成答案中抽取出的原子事实 claims，JSON 列表。 |
| `generation_score_expected_claims` | 从期望答案中抽取出的原子事实 claims，JSON 列表。 |
| `generation_score_precision_verdicts` | 生成答案 claims 对照期望答案的判定结果，JSON 列表。 |
| `generation_score_recall_verdicts` | 期望答案 claims 对照生成答案的判定结果，JSON 列表。 |

如果你重点关注召回，建议优先看：

1. `generation_score_factual_correctness_recall`
2. `generation_score_vital_nuggetizer_score`
3. `generation_score_mean_nugget_assignment_score`

## Token 用量字段

| 字段 | 含义 |
| --- | --- |
| `total_input_tokens` | 本地 chat 模型为该行评估上报的输入 token 总数。 |
| `total_output_tokens` | 本地 chat 模型为该行评估上报的输出 token 总数。 |
| `total_tokens` | 输入和输出 token 总和。 |

Token 统计依赖你的 OpenAI-compatible 服务是否返回 `usage` 信息。如果服务端不返回
usage，这些值可能是 `0`，或者低于真实消耗。

## 统计汇总字段含义

使用 `scripts/summarize_recall_metrics.py` 时，会输出一些统计项：

| 统计项 | 含义 |
| --- | --- |
| `mean` | 平均值。适合看整体水平，但会受到极高/极低值影响。 |
| `median` | 中位数。把所有分数排序后取中间值，更能代表典型表现。 |
| `min` | 最低分。适合定位最差 case。 |
| `p25` | 25 分位数。表示有 25% 的样本分数小于或等于该值。 |
| `p75` | 75 分位数。表示有 75% 的样本分数小于或等于该值。 |
| `max` | 最高分。适合看系统上限。 |

