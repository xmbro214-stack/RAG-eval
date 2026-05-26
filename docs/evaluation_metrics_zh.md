# RAG 评估项目中文说明

本文档说明本项目 `rag_eval_pipeline/evaluation.py` 当前产出的评估项目。评估流程以 `generated_answers.csv` 为输入，结合检索片段、生成答案、可选的标准答案 `golden_csv`，调用 OpenAI 兼容的评审模型和向量模型，输出到 `eval_result.csv`。

分数一般在 `0` 到 `1` 之间，越高越好；少数检索相关的原始分数使用 `0` 到 `3` 的整数等级。所有 LLM 评审均以 `temperature=0` 调用。

## 一、检索效果评估

### 1. UMBRELA 片段相关性评分

**输出字段**

- `retrieval_score_umbrela_scores`
- `retrieval_score_mean_umbrela_score`

**含义**

UMBRELA 评分用于判断每个召回片段是否能支撑用户问题及标准答案。它评的是检索阶段返回的 chunk 质量，而不是最终回答质量。

**评估维度**

- 片段是否与 query 主题相关。
- 片段是否能支撑标准答案中的关键事实。
- 片段是否完整、直接、实质性地回答问题。
- 在有 `expected_answer` 时，以专家标准答案为目标；没有标准答案时，以 query 意图和召回片段本身为目标。

**计算方式**

评审模型对每个 passage 给出 `0` 到 `3` 的整数分：

- `0`：与问题和标准答案无关。
- `1`：术语或主题相关，但不能支撑标准答案。
- `2`：能支撑部分标准答案，但不完整或混杂额外信息。
- `3`：能直接、充分支撑标准答案。

`retrieval_score_umbrela_scores` 保存每个 passage 的原始分数：

```text
{passage_id: score}
```

`retrieval_score_mean_umbrela_score` 为所有 passage 原始分数的平均值：

```text
mean_umbrela_score = sum(passage_scores) / passage_count
```

如果没有 passage，则为 `0.0`。

**意义**

该指标回答“系统检索出来的材料本身是否有用”。均值高说明召回内容更贴近答案依据；均值低通常说明召回失败、切片策略不合适、相似度阈值不合适，或知识库缺少相关内容。

### 2. Retrieval Precision@K

**输出字段**

- `retrieval_score_precision_metrics` 中的 `precision@`

**含义**

Precision@K 衡量前 K 个召回片段中，有多少比例被认为是相关片段。

**评估维度**

- 召回结果前排位置的相关性。
- 检索排序是否把可用证据排在前面。

**计算方式**

代码先把 UMBRELA 原始分转成二值相关性：

```text
relevant = 1 if UMBRELA_score >= 2 else 0
```

然后对配置的每个 `k` 值计算：

```text
Precision@K = 前 K 个片段中 relevant=1 的数量 / K
```

只有当 `K <= passage_count` 时才会输出该 K 的结果。

**意义**

用于判断模型最终生成答案时能否在上下文窗口的前部看到足够可靠的证据。Precision@1 尤其能反映首条召回是否命中。

### 3. Average Precision@K

**输出字段**

- `retrieval_score_precision_metrics` 中的 `AP@`

**含义**

AP@K 不只看前 K 个结果里相关片段的比例，还关注相关片段出现的位置。越早出现相关片段，分数越高。

**评估维度**

- 前 K 个片段的相关性。
- 相关片段在排序中的提前程度。

**计算方式**

对前 K 个片段进行二值相关性判断。每当第 `i` 个片段相关时，计算当前位置的 precision：

```text
Precision_i = 截至第 i 位的相关片段数 / i
```

AP@K 为所有相关位置的 `Precision_i` 平均值：

```text
AP@K = mean(Precision_i for each relevant item in top K)
```

如果前 K 个片段没有相关片段，则为 `0.0`。

**意义**

适合比较不同检索参数下的排序质量。两个配置 Precision@K 可能相同，但 AP@K 更高的配置说明相关证据出现得更早。

### 4. NDCG@K

**输出字段**

- `retrieval_score_ndcg_metrics`
- `retrieval_score_precision_metrics` 中的 `NDCG@`

**含义**

NDCG@K 使用 UMBRELA 的 `0` 到 `3` 多级相关性，衡量检索排序接近“理想排序”的程度。

**评估维度**

- 片段相关性的强弱。
- 高相关片段是否被排在更靠前位置。

**计算方式**

先计算 DCG：

```text
DCG@K = sum((2^score_i - 1) / log2(i + 1))
```

其中 `i` 是从 `1` 开始的位置，`score_i` 是对应 passage 的 UMBRELA 分数。

再计算理想排序下的 IDCG：

```text
IDCG@K = DCG@K(sorted(scores, descending=True))
```

最终：

```text
NDCG@K = DCG@K / IDCG@K
```

如果没有分数或 `IDCG@K=0`，则为 `0.0`。

**意义**

NDCG@K 比 Precision@K 更细，因为它区分“部分支持”和“充分支持”。它适合评价检索排序器是否把最高价值证据放在前面。

### 5. MRR

**输出字段**

- `retrieval_score_precision_metrics` 中的 `MRR`

**含义**

MRR 衡量第一个相关片段出现得有多早。

**评估维度**

- 首个可用证据的排名位置。

**计算方式**

同样使用二值相关性：

```text
relevant = 1 if UMBRELA_score >= 2 else 0
```

找到第一个相关片段的位置 `rank`：

```text
MRR = 1 / rank
```

如果没有相关片段，则为 `0.0`。

**意义**

MRR 对“第一条有效证据”非常敏感。它能帮助判断用户问题是否能被检索系统快速命中，尤其适合看首屏或短上下文场景。

## 二、答案覆盖度评估

### 6. AutoNugget 关键事实覆盖

**输出字段**

- `generation_score_autonugget_scores`
- `generation_score_vital_nuggetizer_score`
- `generation_score_mean_nugget_assignment_score`

**含义**

AutoNugget 将一个好答案应包含的信息拆成若干原子事实 nugget，再判断生成答案覆盖了多少。它评估答案是否“答到了点上”。

**评估维度**

- 标准答案或相关检索上下文中有哪些关键事实。
- 每个事实是必要信息 `vital`，还是补充信息 `okay`。
- 生成答案对每个事实是完全覆盖、部分覆盖，还是未覆盖。

**计算方式**

流程如下：

1. 使用 UMBRELA 分数 `>=1` 的 passage 作为候选上下文。
2. 评审模型迭代生成最多 30 个 nugget。
3. 评审模型将 nugget 标注为 `vital` 或 `okay`。
4. 按 vital 优先排序，最多保留 20 个 nugget。
5. 评审模型判断生成答案对每个 nugget 的覆盖标签：

```text
support = 1.0
partial_support = 0.5
not_support = 0.0
```

`generation_score_mean_nugget_assignment_score` 为所有保留 nugget 覆盖分的平均值：

```text
mean_nugget_assignment_score = sum(assignment_scores) / nugget_count
```

`generation_score_vital_nuggetizer_score` 为 vital nugget 覆盖分的平均值：

```text
vital_nuggetizer_score = sum(vital_assignment_scores) / vital_nugget_count
```

`generation_score_autonugget_scores` 还包含更完整的明细：

- `nuggets`：抽取出的事实点。
- `labels`：每个事实点的 `vital` 或 `okay` 标签。
- `assignments`：生成答案对每个事实点的覆盖标签。
- `assignment_scores`：覆盖标签对应的数值分。
- `nuggetizer_scores`：聚合分数。

其中 `nuggetizer_scores` 的聚合方式为：

```text
All = 所有 nugget 覆盖分平均值
All Strict = 所有 nugget 严格覆盖平均值，只有 support 计 1，其余计 0
Vital = vital nugget 覆盖分平均值
Vital Strict = vital nugget 严格覆盖平均值
Weighted = (sum(vital_scores) + 0.5 * sum(okay_scores)) / (vital_count + 0.5 * okay_count)
Weighted Strict = 同上，但只把 support 计为 1
```

如果没有对应类别，代码用 `max(count, 1)` 避免除零。

**意义**

该指标非常适合发现“答案看起来流畅但漏掉关键点”的问题。`vital_nuggetizer_score` 更关注必要事实，`mean_nugget_assignment_score` 则反映整体覆盖度。

## 三、引用与来源支撑评估

### 7. Citation Support / Citation F1

**输出字段**

- `generation_score_citation_scores`
- `generation_score_citation_f1_score`

**含义**

Citation Support 判断答案中的引用标记是否真的支撑被引用的陈述。它要求生成答案中包含形如 `[1]`、`[2]` 或 `[ID:1]` 的引用。

**评估维度**

- 每个答案片段的陈述是否被其引用 passage 支撑。
- 每个 citation 是否被正确使用。
- 答案片段与引用证据之间的一致性。

**计算方式**

评审模型对每个“答案片段-引用 passage”组合输出：

```text
full_support = 1.0
partial_support = 0.5
no_support = 0.0
```

然后分别聚合：

```text
citation_score_x = 同一 citation 被使用时的平均支撑分
part_score_x = 同一答案片段所有 citation 的平均支撑分
weighted_precision = mean(citation_score_x)
weighted_recall = mean(part_score_x)
Citation F1 = 2 * weighted_precision * weighted_recall / (weighted_precision + weighted_recall)
```

如果没有 citation 或没有 part 分数，相关值为 `0.0`。

**意义**

该指标用于衡量引用的可信度。高分说明答案不仅引用了材料，而且引用位置能支撑对应陈述；低分常见于乱引、引用不充分、引用缺失。

### 8. Faithfulness 忠实度

**输出字段**

- `generation_score_faithfulness_score`
- `generation_score_faithfulness_claims`
- `generation_score_faithfulness_verdicts`
- `generation_score_unsupported_claims`

**含义**

Faithfulness 判断生成答案中的事实声明是否能被检索上下文蕴含。它关注答案是否忠实于给定来源。

**评估维度**

- 答案中有哪些原子事实声明。
- 每条声明相对于检索上下文是蕴含、矛盾，还是中立。
- 不被上下文支撑的声明数量和内容。

**计算方式**

1. 评审模型从生成答案中抽取原子事实声明 `claims`。
2. 评审模型逐条判断 claim 相对于所有检索 passage 的关系：

```text
entailment
contradiction
neutral
```

3. 计算蕴含比例：

```text
faithfulness_score = entailment_claim_count / total_claim_count
```

没有抽取到 claim 时，分数为 `0.0`。

**意义**

该指标用于发现“脱离检索材料自行发挥”的答案。高分说明答案事实基本可由召回上下文支持；低分说明存在幻觉、误引或回答超出证据范围。

### 9. Hallucination / Source Support

**输出字段**

- `generation_score_hallucination_score`

**含义**

该字段沿用 hallucination 相关输出名，但当前实现实际是 LLM judge 给出的“来源支撑度”分数：答案越被 source passages 支撑，分数越高。

**评估维度**

- 生成答案整体是否被全部检索片段支持。
- 是否存在事实不一致、无来源支撑或矛盾内容。

**计算方式**

评审模型读取所有 source passages 和 generated answer，直接返回 `0` 到 `1` 的分数：

```text
0 = unsupported / contradicted
1 = fully supported
```

代码会把结果裁剪到 `[0.0, 1.0]` 区间。

**意义**

这是一个整体性的来源支撑评分，比 Faithfulness 更粗粒度。适合快速比较不同实验配置下的答案可信度。

## 四、标准答案对齐评估

以下指标只有当 `query_id` 在 `golden_csv` 中存在标准答案时才会输出。

### 10. Semantic Similarity 语义相似度

**输出字段**

- `generation_score_semantic_similarity`

**含义**

语义相似度衡量生成答案与专家标准答案在向量空间中的接近程度。

**评估维度**

- 答案整体语义是否接近标准答案。
- 不要求逐字一致，更关注表达含义接近。

**计算方式**

调用 embedding 模型分别对生成答案和标准答案编码，然后计算 cosine similarity：

```text
semantic_similarity = dot(answer_embedding, expected_embedding) /
  (norm(answer_embedding) * norm(expected_embedding))
```

如果任一向量范数为 `0`，返回 `0.0`。

**意义**

该指标适合捕捉措辞不同但语义相近的答案。不过它不能精确判断事实遗漏或事实错误，因此需要和 factual correctness、nugget 指标一起看。

### 11. Factual Correctness Precision

**输出字段**

- `generation_score_factual_correctness_precision`
- `generation_score_generated_claims`
- `generation_score_precision_verdicts`

**含义**

事实正确率的 precision 衡量生成答案提出的事实中，有多少能被标准答案支持。

**评估维度**

- 生成答案中的事实声明是否准确。
- 是否产生标准答案不支持的额外事实。

**计算方式**

1. 从生成答案中抽取原子事实声明 `generated_claims`。
2. 用标准答案作为 reference，判断每个 generated claim 是否为 `entailment`。
3. 计算蕴含比例：

```text
factual_correctness_precision = entailed_generated_claim_count / generated_claim_count
```

如果没有生成 claim 或没有标准答案 claim，则 precision、recall、F1 均为 `0.0`。

**意义**

Precision 高说明答案中说出来的事实大多是对的；Precision 低说明答案可能有错误事实、过度扩展或与标准答案不一致。

### 12. Factual Correctness Recall

**输出字段**

- `generation_score_factual_correctness_recall`
- `generation_score_expected_claims`
- `generation_score_recall_verdicts`

**含义**

事实正确率的 recall 衡量标准答案中的事实，有多少被生成答案覆盖。

**评估维度**

- 标准答案关键事实是否被答全。
- 是否存在遗漏。

**计算方式**

1. 从标准答案中抽取原子事实声明 `expected_claims`。
2. 用生成答案作为 reference，判断每个 expected claim 是否为 `entailment`。
3. 计算蕴含比例：

```text
factual_correctness_recall = entailed_expected_claim_count / expected_claim_count
```

**意义**

Recall 高说明答案覆盖了标准答案的大部分事实；Recall 低说明答案可能太简略、漏掉关键定义、原因、措施或判定条件。

### 13. Factual Correctness F1

**输出字段**

- `generation_score_factual_correctness_f1`

**含义**

F1 综合事实 precision 和 recall，用于同时衡量“说得准”和“说得全”。

**评估维度**

- 生成事实是否可靠。
- 标准答案事实是否覆盖完整。

**计算方式**

```text
F1 = 2 * precision * recall / (precision + recall)
```

如果 `precision + recall = 0`，则为 `0.0`。

**意义**

F1 适合做总体答案事实质量排序。相比单看 recall，它会惩罚编造或多说错说；相比单看 precision，它也会惩罚只答一小部分。

## 五、拒答与可回答性诊断

### 14. No-answer Detection

**输出字段**

- `generation_score_no_answer_score`

**含义**

该指标判断生成答案是否尝试回答了问题，不评判回答是否正确。

**评估维度**

- 答案是在回答问题，还是明确表示无法回答、信息不足。
- 只判断 answered / not answered，不判断事实质量。

**计算方式**

评审模型返回：

```text
{"query_answered": "yes"}
```

或：

```text
{"query_answered": "no"}
```

**意义**

用于区分“低分是因为答错”还是“系统拒答/无答案”。在分析检索失败、知识库缺失或保守回答策略时很有用。

## 六、召回导向综合指标

### 15. Recall Metric Avg

**输出字段**

- `recall_metric_avg`
- `recall_metric_band`

**来源**

该指标不是 `evaluation.py` 的原始输出，而是由 `scripts/summarize_recall_metrics.py` 从评估结果中二次汇总生成。

**含义**

Recall Metric Avg 是面向答案覆盖度的简化综合指标。

**评估维度**

- 标准答案事实覆盖。
- vital nugget 覆盖。
- 所有 nugget 平均覆盖。

**计算方式**

对以下三个指标取平均：

```text
generation_score_factual_correctness_recall
generation_score_vital_nuggetizer_score
generation_score_mean_nugget_assignment_score
```

```text
recall_metric_avg = mean(上述非空指标)
```

并按阈值分档：

```text
good:   >= 0.8
medium: >= 0.5 and < 0.8
low:    < 0.5
```

**意义**

适合快速筛选“覆盖不足”的样本。尤其在批量评估时，可以用它优先定位低召回 query，再回看具体 nugget、claim 和检索片段。

## 七、成本与运行诊断字段

### 16. Token 使用量

**输出字段**

- `total_input_tokens`
- `total_output_tokens`
- `total_tokens`

**含义**

记录评估过程中评审模型的 token 使用量。

**评估维度**

- 单条样本评估成本。
- 输入 prompt 规模。
- 输出解析负载。

**计算方式**

每次 chat judge 请求后，从 OpenAI 兼容接口返回的 `usage` 中累加：

```text
total_input_tokens = sum(prompt_tokens)
total_output_tokens = sum(completion_tokens)
total_tokens = total_input_tokens + total_output_tokens
```

Embedding 请求当前不计入这些字段。

**意义**

用于估算评估成本、定位异常耗时或异常长上下文样本。比较不同配置时，也可以观察评估成本是否随 page size、召回数量增加而显著上升。

## 八、建议解读方式

### 检索问题优先看

- `retrieval_score_mean_umbrela_score`
- `Precision@K`
- `NDCG@K`
- `MRR`

如果这些低，说明答案质量差可能首先来自检索材料不足或排序不佳。

### 答案漏点优先看

- `generation_score_factual_correctness_recall`
- `generation_score_vital_nuggetizer_score`
- `generation_score_mean_nugget_assignment_score`
- `recall_metric_avg`

这些指标低，通常代表答案没有覆盖标准答案或关键 nugget。

### 答案编造或证据不足优先看

- `generation_score_faithfulness_score`
- `generation_score_hallucination_score`
- `generation_score_citation_f1_score`
- `generation_score_unsupported_claims`

这些指标低，通常代表答案脱离来源、引用错误或包含未支撑事实。

### 标准答案一致性优先看

- `generation_score_semantic_similarity`
- `generation_score_factual_correctness_precision`
- `generation_score_factual_correctness_recall`
- `generation_score_factual_correctness_f1`

语义相似度看整体表达接近程度，事实 precision/recall/F1 看原子事实是否准确且完整。
