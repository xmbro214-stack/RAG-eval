# Local Eval Results 指标说明

本文档说明 `scripts/openai_compatible_eval.py` 生成的
`local_eval_results.csv` 中各个字段的含义。该结果文件用于评估已经生成好的
RAG 回答，评估过程调用 OpenAI API compatible 的本地 chat 模型和 embedding
模型。

## 快速解读建议

如果只想先看核心指标，可以按下面的顺序理解：

| 评估目标 | 优先看 | 辅助看 | 说明 |
| --- | --- | --- | --- |
| 检索排序质量 | `retrieval_score_ndcg_metrics` | `retrieval_score_mean_umbrela_score`、`precision@K`、`AP@K`、`MRR` | NDCG 会利用 UMBRELA 的 0-3 多级相关性，更适合判断高价值 passage 是否排在前面。 |
| 答案关键点召回 | `generation_score_factual_correctness_recall` | `generation_score_vital_nuggetizer_score`、`generation_score_mean_nugget_assignment_score` | 如果有专家标准答案，factual recall 是最直接的答案召回指标。 |
| 答案忠实度 | `generation_score_faithfulness_score` | `generation_score_hallucination_score`、`generation_score_citation_f1_score` | Faithfulness 不依赖严格引用标记，更适合本地企业级 RAG 评测。 |
| 语义接近程度 | `generation_score_semantic_similarity` | factual precision/recall/F1 | Semantic similarity 使用 embedding cosine similarity，只表示整体语义接近，不等价于事实完全正确。 |

本版本新增或重点更新了两个指标方向：

| 指标方向 | 相关字段 | 变化 |
| --- | --- | --- |
| NDCG@K | `retrieval_score_ndcg_metrics` | 使用 UMBRELA 的 0-3 多级相关性计算排序质量，奖励把 3 分 passage 排在更靠前的位置。 |
| Faithfulness | `generation_score_faithfulness_score`、`generation_score_faithfulness_claims`、`generation_score_faithfulness_verdicts`、`generation_score_unsupported_claims` | 将生成答案拆成 claims，再用本地 LLM judge 判断每个 claim 是否能被 retrieved context 支持。 |

另外，UMBRELA 和 nugget 相关 judge prompt 已加入 few-shot 示例；如果 golden
answer CSV 中提供了专家 `expected_answer`，脚本会把它放入 prompt，帮助 judge 更
贴近“检索块是否支撑专家标准答案”和“答案是否覆盖业务关键点”的目标。

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
| `retrieval_score_precision_metrics` | JSON 对象，包含基于 UMBRELA 分数计算出的 `precision@K`、`AP@K`、`NDCG@K` 和 `MRR`。其中 `precision@K`、`AP@K`、`MRR` 会把 UMBRELA 分数 >= 2 的 passage 视为相关。 |
| `retrieval_score_ndcg_metrics` | JSON 对象，只包含 `NDCG@K`。NDCG 会直接使用 UMBRELA 的 0-3 多级相关性分数，因此比二元 precision/MRR 更适合衡量排序质量。 |
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
`NDCG@K`、`precision@K`、`AP@K` 和 `MRR`。

如果重点关注排序质量，建议优先看 `NDCG@K`。它会奖励把 3 分 passage 排在前面，
也会惩罚把 1 分 passage 排在高位；相比 `mean_umbrela_score`，它更能反映排序
引擎是否把最有用的资料放在靠前位置。

NDCG@K 的计算方式：

1. 对前 K 个 passage 使用 UMBRELA 分数作为相关性 `rel`。
2. 计算 `DCG@K = sum((2^rel - 1) / log2(rank + 1))`，其中 `rank` 从 1 开始。
3. 将同一组 passage 按相关性从高到低排序，得到理想排序的 `IDCG@K`。
4. `NDCG@K = DCG@K / IDCG@K`。如果没有任何相关性增益，则记为 `0`。

直观理解：同样都是检索到高分 passage，排在第 1 位比排在第 5 位更值钱；3 分
passage 的收益也会明显高于 1 分 passage。

当前轻量脚本的 UMBRELA judge prompt 已加入业务 few-shot 示例。如果某个 query
在 golden answer CSV 中有专家标准答案，prompt 会把 `expected_answer` 一并提供给
judge，要求 judge 判断 retrieved chunk 是否能支撑该标准答案。这样可以减少仅凭
关键词相似导致的高分，更贴近“检索块是否真正支撑专家答案”的目标。

## Nugget 覆盖指标

| 字段 | 含义 |
| --- | --- |
| `generation_score_autonugget_scores` | JSON 对象，包含自动生成的 nuggets、每个 nugget 的 vital/okay 标签、答案对 nugget 的支持情况，以及聚合分数。 |
| `generation_score_mean_nugget_assignment_score` | 所有 nuggets 的平均覆盖分。`support=1.0`，`partial_support=0.5`，`not_support=0.0`。越高表示答案覆盖了越多 nugget 信息。 |
| `generation_score_vital_nuggetizer_score` | 只针对 `vital` nuggets 的覆盖分。越高表示关键事实覆盖越好。 |

如果你关注“答案有没有答全”，`generation_score_vital_nuggetizer_score`
通常比全量 nugget 平均分更重要，因为它更强调关键点覆盖。

当前轻量脚本的 nugget 抽取 prompt 也加入了业务 few-shot 示例。如果存在专家
`expected_answer`，nuggets 会以标准答案作为目标，并结合 retrieved context 保持
可溯源。这样生成的 nuggets 更接近业务专家认为“应该答到”的关键事实。

## 来源支持与引用指标

| 字段 | 含义 |
| --- | --- |
| `generation_score_hallucination_score` | LLM judge 给出的来源支持分，范围 0 到 1。越高表示生成答案越被检索来源支持。注意：这不是原项目默认的 HHEM 分数，而是轻量脚本里的本地 LLM judge 版本。 |
| `generation_score_faithfulness_score` | 基于 NLI 的忠实度分数。脚本先把生成答案拆成多个 claims，再判断每个 claim 是否能被完整 retrieved context 支持。分数 = 被 context entail 的 claims 数 / 总 claims 数。 |
| `generation_score_faithfulness_claims` | 从生成答案中抽取出来、用于忠实度判断的 claims，JSON 列表。 |
| `generation_score_faithfulness_verdicts` | 每个 generated claim 对照 retrieved context 的 NLI 判定结果，JSON 列表。判定值包括 `entailment`、`contradiction`、`neutral`。 |
| `generation_score_unsupported_claims` | 没有被 retrieved context 支持的 claims，包含 `neutral` 和 `contradiction`。 |
| `generation_score_citation_scores` | JSON 对象，包含 citation 级别和答案片段级别的支持分。 |
| `generation_score_citation_f1_score` | 综合 citation weighted precision 和 answer-part weighted recall 的 F1 分数。越高表示引用越能支撑对应答案片段。 |
| `generation_score_no_answer_score` | JSON 对象，表示答案是否尝试回答问题，例如 `{"query_answered": "yes"}`。 |

相比 `generation_score_citation_f1_score`，更推荐优先使用
`generation_score_faithfulness_score` 判断答案是否忠实于检索上下文。Faithfulness
不依赖严格的 `[1][2]` 引用标记，只判断生成答案的事实主张是否能在 retrieved
context 中找到依据，因此更适合引用粒度不稳定、答案会综合多段上下文的企业级 RAG
评测场景。

Faithfulness 的计算方式：

1. 从 `generated_answer` 中抽取原子事实主张 claims。
2. 把所有 retrieved passages 拼成完整 context。
3. 对每个 claim 做 NLI 判定：`entailment` 表示 context 支持该主张，`neutral`
   表示 context 不足以推出该主张，`contradiction` 表示 context 与该主张冲突。
4. `generation_score_faithfulness_score = entailment claims 数 / generated claims 总数`。

因此，`generation_score_unsupported_claims` 是排查幻觉和无依据扩写时最有用的
明细列。如果该列中出现关键业务结论，说明答案虽然可能看起来合理，但没有被本次检索
上下文支撑。

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

Factual correctness 的计算口径：

| 子指标 | 计算方式 | 主要含义 |
| --- | --- | --- |
| precision | 生成答案 claims 中，被 expected answer 支持的比例。 | 生成答案有没有加入标准答案外的额外事实。 |
| recall | expected answer claims 中，被生成答案覆盖的比例。 | 生成答案有没有答全专家标准答案中的关键事实。 |
| F1 | precision 和 recall 的调和平均。 | 同时考虑多答和漏答。 |

如果你重点关注召回，建议优先看：

1. `generation_score_factual_correctness_recall`
2. `generation_score_vital_nuggetizer_score`
3. `generation_score_mean_nugget_assignment_score`

其中 `generation_score_factual_correctness_recall` 依赖专家 `expected_answer`；
如果某些 query 没有 expected answer，就优先看 nugget 相关指标。

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

## 兼容性说明

旧版本脚本生成的 `local_eval_results.csv` 可能没有 NDCG 或 Faithfulness 相关列。
如果你需要分析这些新指标，需要用当前版本的 `scripts/openai_compatible_eval.py`
重新跑一次评估。

`retrieval_score_precision_metrics` 这个字段名保留了历史命名，但其中现在也包含
`NDCG@`。如果只想读取 NDCG，建议直接使用 `retrieval_score_ndcg_metrics`。
