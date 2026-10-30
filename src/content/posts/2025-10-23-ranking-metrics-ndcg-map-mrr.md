---
title: "Ranking metrics: NDCG, MAP, MRR"
description: "Beyond binary precision-recall: how to measure ranking quality when order matters and labels are graded."
date: "2025-10-23"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Ranking metrics assign different credit to gain, coverage, precision at hits, and the first useful result.
Choose the decision target, judged universe, cutoff, relevance definition, and query weights before comparing systems.
A high score under one credit rule need not establish performance under another.

## NDCG. The dominant ranking metric

Use nonnegative graded judgments and exponential gain for this lesson:

**Discounted Cumulative Gain at $K$**:

$$
\text{DCG}_K = \sum_{i=1}^{K} \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}.
$$

For the hypothetical fully judged ranking `[3,0,2,1]` at cutoff four, DCG is $7+0+3/2+1/\log_2(5)=8.93068$.
Sort the judged grades only for the ideal reference: `[3,2,1,0]` gives IDCG $9.39279$.

**NDCG** normalizes by the ideal ranking's DCG so scores live in [0, 1]:

$$
\text{NDCG}_K = \frac{\text{DCG}_K}{\text{IDCG}_K}.
$$

The ratio is approximately $0.95080$.
It lies in $[0,1]$ under the stated nonnegative-gain and common-universe conventions when IDCG is positive.
Normalization does not make different gain scales or judging populations interchangeable.

## MAP. Average precision averaged over queries

For a query with binary relevance:

**Precision at position $k$**: $\text{P@k} = \frac{\text{relevant items in top } k}{k}$.

**Average Precision**:

$$
\text{AP} = \frac{1}{R} \sum_{k=1}^{K} \text{P@k} \cdot \mathbf{1}[\text{item at } k \text{ is relevant}]
$$

Define binary relevance as grade greater than zero; the fixture has total $R=3$.
Use the full-universe R denominator even when truncating at K here.
Then AP@4 is $(1+2/3+3/4)/3=29/36\approx0.80556$.
Alternative denominators, such as $\min(R,K)$, define different truncated metrics and must be labeled.

MAP is the declared query-weighted mean of AP; equal-query averaging and traffic-weighted averaging target different populations.

## MRR. When there's one right answer

Let r be the first relevant rank, even if multiple relevant items exist:

$$
\text{RR} = \begin{cases} 1/r & \text{if a correct answer is in top-K} \\ 0 & \text{otherwise} \end{cases}
$$

MRR averages reciprocal rank over queries.
The left fixture has first hit at rank one, so RR is one.
The separate right-hand example has first hit at rank three, so RR is $1/3$.

<!-- visual:ranking-metrics-per-rank-credit -->
<figure class="learning-figure" aria-labelledby="ranking-metrics-credit-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="ranking-metrics-credit-title">Average precision accumulates relevant hits; reciprocal rank stops at the first</p><div class="visual-argument" data-visual-argument="ranking-metrics-per-rank-credit" data-argument-kind="ledger"><p class="visual-argument-premise">Reciprocal rank stops; average precision keeps collecting hits</p><svg viewBox="0 0 360 322" role="img" aria-label="Reciprocal rank stops; average precision keeps collecting hits. Relevant ranks 2 and 4 give reciprocal rank 1/2. Average precision is ((1/2)+(2/4))/2=1/2, but its two contributions come from both hits."><title>Reciprocal rank stops; average precision keeps collecting hits</title><desc>Relevant ranks 2 and 4 give reciprocal rank 1/2. Average precision is ((1/2)+(2/4))/2=1/2, but its two contributions come from both hits.</desc><text x="20" y="28" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Relevance by rank</tspan></text><rect data-proof-token="no" x="20" y="58" width="76" height="34" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke)"></rect><text x="58" y="79" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="58" dy="0">no</tspan></text><rect data-proof-token="yes" x="100" y="58" width="76" height="34" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke)"></rect><text x="138" y="79" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="138" dy="0">yes</tspan></text><rect data-proof-token="no" x="180" y="58" width="76" height="34" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke)"></rect><text x="218" y="79" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="218" dy="0">no</tspan></text><rect data-proof-token="yes" x="260" y="58" width="76" height="34" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke)"></rect><text x="298" y="79" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="298" dy="0">yes</tspan></text><text x="20" y="126" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Reciprocal-rank contribution</tspan></text><rect data-proof-token="0" x="20" y="156" width="76" height="34" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke)"></rect><text x="58" y="177" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="58" dy="0">0</tspan></text><rect data-proof-token="1/2" x="100" y="156" width="76" height="34" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke)"></rect><text x="138" y="177" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="138" dy="0">1/2</tspan></text><rect data-proof-token="0" x="180" y="156" width="76" height="34" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke)"></rect><text x="218" y="177" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="218" dy="0">0</tspan></text><rect data-proof-token="0" x="260" y="156" width="76" height="34" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke)"></rect><text x="298" y="177" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="298" dy="0">0</tspan></text><text x="20" y="224" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Average-precision hit terms</tspan></text><rect data-proof-token="0" x="20" y="254" width="76" height="34" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke)"></rect><text x="58" y="275" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="58" dy="0">0</tspan></text><rect data-proof-token="1/2" x="100" y="254" width="76" height="34" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke)"></rect><text x="138" y="275" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="138" dy="0">1/2</tspan></text><rect data-proof-token="0" x="180" y="254" width="76" height="34" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke)"></rect><text x="218" y="275" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="218" dy="0">0</tspan></text><rect data-proof-token="2/4" x="260" y="254" width="76" height="34" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke)"></rect><text x="298" y="275" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="298" dy="0">2/4</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
	<div class="visual-grid--two" role="group" aria-label="Worked per-rank comparison of NDCG and average precision beside a reciprocal-rank stopping example">
		<section class="visual-panel">
			<h4>Many relevant items: graded or binary?</h4>
			<p>The same four results support two different judgments.</p>
			<table class="cm-grid" aria-label="At ranks one through four, graded relevance is 3, 0, 2, 1; unnormalized DCG credit is 7.00, 0, 1.50, 0.43; AP precision checkpoints are 1 over 1, none, 2 over 3, and 3 over 4">
				<thead><tr><th scope="col">Rank</th><th scope="col">Grade</th><th scope="col">DCG credit</th><th scope="col">AP checkpoint</th></tr></thead>
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
			<h4>First useful answer: stop at its rank</h4>
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
	<div class="visual-derivation-caption">the left table accumulates unnormalized DCG terms and precision at binary-relevant hits. Divide by IDCG or R under the stated conventions. The separate right example stops at its first useful answer. Original calculations follow the <a href="https://doi.org/10.1145/582415.582418">cumulative-gain paper</a>, <a href="https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html">IR textbook</a>, and <a href="https://trec.nist.gov/pubs/trec8/papers/qa8.pdf">TREC-8 QA evaluation</a>.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="ranking-metrics-per-rank-credit"><strong>Check:</strong> Which metric keeps accumulating after the first relevant item? What extra averaging step produces MAP or MRR?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="ranking-metrics-per-rank-credit">Relevant ranks 2 and 4 give reciprocal rank 1/2. Average precision is ((1/2)+(2/4))/2=1/2, but its two contributions come from both hits.</span></figcaption>
</figure>

## Hit rate and recall@K

Recall@K is retrieved relevant count divided by R; hit@K is an indicator that any relevant result was retrieved.
They coincide only under particular relevance settings, such as exactly one relevant item per query.

### Change Retrieval Depth

**New task:** Permit only two candidates to reach the next stage.

**Picture and change:** Set K to two without changing total R or the judged universe.

**Check:** The fixture has hit@2 one, recall@2 $1/3$, and full-R AP@2 $1/3$.
One successful hit does not mean complete candidate coverage.

**Use when:** Candidate-generation capacity binds and the downstream stage cannot recover omitted items.

### Change the Reward Target

**New task:** One useful answer suffices regardless of later results.

**Picture and change:** Use reciprocal first-hit rank instead of AP.

**Check:** Adding a relevant result at rank four after a first hit at rank three leaves RR at $1/3$.

**Use when:** The user stops at the first satisfactory answer; multiple valid answers remain possible.

For R zero or IDCG zero, the corresponding ratios are undefined; this lesson excludes those queries and reports their count.
Benchmark zero-value conventions are alternatives, not silent substitutions.
Declare unjudged-item handling, duplicates, score ties, and query weights before evaluation; unjudged does not automatically mean irrelevant.

<!-- learning:core-end -->
<div data-learning-extension>

| Metric | Order matters? | Graded relevance? | Multi-relevant per query? |
|--------|----------------|-------------------|--------------------------|
| **NDCG** | Yes | Yes | Yes |
| **MAP** | Yes | No (binary) | Yes |
| **MRR** | Yes | No | Yes, but only first hit earns credit |
| **Recall@K** | No (just need in top-K) | No | Yes |
| **Precision@K** | No | No | Yes |

## When to use which

- **Graded utility:** choose gains and discounts justified by the judgment/task model.
- **Multiple relevant items:** AP rewards precision at successive hits; recall measures coverage.
- **First-answer utility:** RR ignores later relevant items after the first.
- **Candidate generation:** inspect coverage at its actual capacity and query slices.

## Common pitfalls

- **Different gain scales:** grades zero-to-three and zero-to-four need common semantics before comparing their exponential-gain scores.
- **Wrong universe:** computing IDCG over only returned items can conceal missing relevant candidates.
- **Pooling queries:** equal-query, traffic-weighted, and pooled-count measurements need not agree.
- **One number:** report intended cutoffs, uncertainty, judging coverage, and important slices rather than treating one metric as universal utility.

## Related

- [Precision, recall, F1](/concepts/precision-recall-f1/). Binary classification metrics.
- [Two-tower retrieval](/concepts/two-tower-retrieval/). What gets evaluated with these metrics in production.
- [Learning-to-rank losses](/concepts/learning-to-rank-losses/). Surrogate objectives for ordered results.
- [Position bias and counterfactual ranking](/concepts/position-bias-counterfactual-learning-to-rank/). Evaluation from biased interaction logs.

</div>
