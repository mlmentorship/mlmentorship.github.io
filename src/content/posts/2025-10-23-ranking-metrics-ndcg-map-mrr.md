---
title: "Ranking metrics: NDCG, MAP, MRR"
description: "Beyond binary precision-recall: how to measure ranking quality when order matters and labels are graded."
date: "2025-10-23"
draft: false
tags: ["concepts"]
category: "concepts"
---

## One-line definition

Ranking metrics evaluate **ordered** lists of items. **NDCG** (Normalized Discounted Cumulative Gain) handles graded relevance with position discount. **MAP** (Mean Average Precision) handles binary relevance averaged over recall levels. **MRR** (Mean Reciprocal Rank) handles a single correct answer per query.

## Why it matters

Search, recommendation, retrieval, and question-answering systems produce ranked lists, not classifications. Treating these problems as classification (precision / recall / F1) ignores order. A wrong top-1 hurts more than a wrong top-10. Ranking metrics quantify "right things at the top."

For senior interviews, knowing which metric to use for which ranking problem is expected.

## NDCG. The dominant ranking metric

For a query with predicted ranking and ground-truth relevance grades $\text{rel}_1, \dots, \text{rel}_K$ (e.g., 0 = irrelevant, 1 = relevant, 2 = highly relevant):

**Discounted Cumulative Gain at $K$**:

$$
\text{DCG}_K = \sum_{i=1}^{K} \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}.
$$

The numerator rewards high-relevance items more than linearly. The denominator (the discount) penalizes putting relevant items deeper.

**NDCG** normalizes by the ideal ranking's DCG so scores live in [0, 1]:

$$
\text{NDCG}_K = \frac{\text{DCG}_K}{\text{IDCG}_K}.
$$

Average NDCG across queries to get a system-level metric.

**Why NDCG is the default**: handles graded relevance, position-discounts deeper results, normalized for cross-query comparison, has standardized $K$ (NDCG@5, @10).

## MAP. Average precision averaged over queries

For a query with binary relevance:

**Precision at position $k$**: $\text{P@k} = \frac{\text{relevant items in top } k}{k}$.

**Average Precision**:

$$
\text{AP} = \frac{1}{R} \sum_{k=1}^{K} \text{P@k} \cdot \mathbf{1}[\text{item at } k \text{ is relevant}]
$$

where $R$ is total relevant items. Averages precision over the recall levels at which relevant items appear.

**MAP** (Mean Average Precision) = average AP across queries. Used heavily in information retrieval (TREC) before NDCG took over for graded relevance.

## MRR. When there's one right answer

For each query with a single correct answer at position $r$ (or no correct in top-K):

$$
\text{RR} = \begin{cases} 1/r & \text{if a correct answer is in top-K} \\ 0 & \text{otherwise} \end{cases}
$$

**MRR** = mean of RR across queries. Used in: question answering (one correct answer per question), passage retrieval (one gold passage per query), some entity disambiguation.

## Hit rate and recall@K

**Hit rate@K** (or recall@K): fraction of queries where a relevant item appears in the top $K$. Used heavily in retrieval / candidate-generation evaluation, where the goal is "get the gold into the candidate pool" and a downstream ranker handles ordering.

| Metric | Order matters? | Graded relevance? | Multi-relevant per query? |
|--------|----------------|-------------------|--------------------------|
| **NDCG** | Yes | Yes | Yes |
| **MAP** | Yes | No (binary) | Yes |
| **MRR** | Yes | No | One per query |
| **Recall@K** | No (just need in top-K) | No | Yes |
| **Precision@K** | No | No | Yes |

## When to use which

- **Web search, e-commerce search**: NDCG@10 (graded relevance, deep results matter less).
- **Recommendations** with implicit feedback: NDCG@K with binary relevance, or hit rate@K.
- **Information retrieval academic benchmarks**: MAP (TREC tradition).
- **Question answering, fact retrieval**: MRR (one correct answer).
- **Retrieval candidate generation**: Recall@K (downstream ranker handles order).
- **Top-1 critical applications**: precision@1 or accuracy.

## Common pitfalls

- **Reporting NDCG at one $K$**: report NDCG@5, @10, @20 to show whether order or coverage matters more.
- **Comparing NDCG across systems with different relevance grading scales.** A system rated on a 0-3 scale gives different NDCG than the same system on 0-4. Standardize.
- **Treating MRR as MAP for QA.** If there can be multiple correct answers, MAP is more informative.
- **Using accuracy for ranking.** Accuracy ignores order entirely; nearly always wrong choice for ranking problems.
- **Confusing macro vs. micro averaging across queries.** Standard ranking metrics average per-query (one score per query, then mean). Analogous to macro. Don't pool TP/FP across queries.

## Related

- [Precision, recall, F1](/concepts/precision-recall-f1/). Binary classification metrics.
- [Two-tower retrieval](/concepts/two-tower-retrieval/). What gets evaluated with these metrics in production.
