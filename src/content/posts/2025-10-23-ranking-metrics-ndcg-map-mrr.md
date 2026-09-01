---
title: "Ranking metrics: NDCG, MAP, MRR"
description: "Beyond binary precision-recall: how to measure ranking quality when order matters and labels are graded."
date: "2025-10-23"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Ranking metrics evaluate **ordered** lists of items. **NDCG** (Normalized Discounted Cumulative Gain) handles graded relevance with position discount. **MAP** (Mean Average Precision) handles binary relevance averaged over recall levels. **MRR** (Mean Reciprocal Rank) handles a single correct answer per query.

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

<!-- visual:ranking-metrics-per-rank-credit -->
<figure class="learning-figure" aria-labelledby="ranking-metrics-credit-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="ranking-metrics-credit-title">Trace what each metric counts as credit while scanning from rank 1 downward.</p>
	<div class="visual-grid--two" role="group" aria-label="Worked per-rank comparison of NDCG and average precision beside a reciprocal-rank stopping example">
		<section class="visual-panel">
			<h4>Many relevant items: graded or binary?</h4>
			<p>The same four results support two different judgments.</p>
			<table class="cm-grid" aria-label="At ranks one through four, graded relevance is 3, 0, 2, 1; NDCG credit is 7.00, 0, 1.50, 0.43; MAP precision checkpoints are 1 over 1, none, 2 over 3, and 3 over 4">
				<thead><tr><th scope="col">Rank</th><th scope="col">Grade</th><th scope="col">NDCG credit</th><th scope="col">AP checkpoint</th></tr></thead>
				<tbody>
					<tr><th scope="row">1</th><td><strong>3</strong> · relevant</td><td>(2³−1) / 1 = <strong>7.00</strong></td><td><strong>1 / 1</strong></td></tr>
					<tr><th scope="row">2</th><td><strong>0</strong> · not relevant</td><td><strong>0</strong></td><td>Skip</td></tr>
					<tr><th scope="row">3</th><td><strong>2</strong> · relevant</td><td>(2²−1) / 2 = <strong>1.50</strong></td><td><strong>2 / 3</strong></td></tr>
					<tr><th scope="row">4</th><td><strong>1</strong> · relevant</td><td>(2¹−1) / 2.32 = <strong>0.43</strong></td><td><strong>3 / 4</strong></td></tr>
				</tbody>
			</table>
			<p class="cm-equation">NDCG@4 = 8.93 / 9.39 = 0.95<br>AP = (1 + 2/3 + 3/4) / 3 = 0.81</p>
		</section>
		<section class="visual-panel">
			<h4>One correct answer: stop at the first hit</h4>
			<p>MRR ignores later ranks once the answer is found.</p>
			<table class="cm-grid" aria-label="The first two ranked results are wrong, the third is correct, and reciprocal rank is one third">
				<thead><tr><th scope="col">Rank</th><th scope="col">Answer?</th><th scope="col">MRR action</th></tr></thead>
				<tbody>
					<tr><th scope="row">1</th><td>Wrong</td><td>Continue</td></tr>
					<tr><th scope="row">2</th><td>Wrong</td><td>Continue</td></tr>
					<tr><th class="cm-selected" scope="row">3</th><td class="cm-selected"><strong>Correct</strong></td><td class="cm-selected"><strong>Stop</strong></td></tr>
				</tbody>
			</table>
			<p class="cm-equation">RR = 1 / first correct rank = 1 / 3 = 0.33</p>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> scan the left table by row. NDCG uses every grade and discounts lower ranks; AP uses only relevant rows and records precision at each hit. Use the right table only when one answer matters: MRR stops at the first correct result. The calculations are an original example checked against the <a href="https://doi.org/10.1145/582415.582418">original cumulative-gain paper</a>, the <a href="https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html">IR textbook definition of MAP</a>, and the <a href="https://trec.nist.gov/pubs/trec8/papers/qa8.pdf">TREC-8 QA evaluation</a>.</figcaption>
</figure>

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
- [Learning-to-rank losses](/concepts/learning-to-rank-losses/). Surrogate objectives for ordered results.
- [Position bias and counterfactual ranking](/concepts/position-bias-counterfactual-learning-to-rank/). Evaluation from biased interaction logs.
