---
title: "k-means clustering"
description: "Partition n points into k clusters by minimizing within-cluster variance. Lloyd's algorithm: alternate assigning points to nearest center and recomputing centers."
date: "2025-10-26"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

K-means seeks low squared Euclidean distortion by alternating nearest-center assignment with fixed-membership mean updates:

$$
\min_{\mu_1, \dots, \mu_k,\ C} \sum_{i=1}^{n} \|x_i - \mu_{C(i)}\|^2.
$$

Each half-step minimizes the same objective while holding different variables fixed.
This optimization definition does not require equal cluster sizes or establish that the resulting groups are natural classes.

## The algorithm

Use finite points in meaningful Euclidean units, $1\le k\le n$, and an explicit initialization.
Break distance ties by a fixed center order; retain an old center if its cluster is empty in this trace.
Then repeat:

1. **Assignment**: $C(i) = \arg\min_j \|x_i - \mu_j\|^2$ for each $i$.
2. **Update**: $\mu_j = \tfrac{1}{|C^{-1}(j)|} \sum_{i \in C^{-1}(j)} x_i$ for each $j$.

For a nonempty cluster, $\sum_i\|x_i-\mu\|^2=\sum_i\|x_i-\bar x\|^2+m\|\mu-\bar x\|^2$.
The second term proves the mean minimizes fixed-membership squared distortion.
An empty cluster has no defined mean, which is why an explicit policy is needed.

In the figure, initial centers are A0 $(110,130)$ and B0 $(210,120)$.
Points $(50,80),(70,110),(90,80)$ choose A; $(230,70),(250,100),(270,70)$ choose B.
For $(90,80)$, squared distances are $2900$ versus $16000$.
Each group's initial assigned SSE is $11000$, giving total $22000$.
Updates produce A1 $(70,90)$ and B1 $(250,80)$.
Each group then contributes $500+400+500=1400$, so total SSE drops to $2800$.
<!-- visual:kmeans-assign-then-update -->
<figure class="learning-figure" aria-labelledby="kmeans-iteration-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="kmeans-iteration-title">What stays fixed in each half of a Lloyd iteration?</p>
	<div class="visual-grid--two" style="grid-template-columns: 1fr;" role="group" aria-label="Nearest-centroid assignment followed by centroid mean update">
		<section class="visual-panel plot-panel">
			<svg viewBox="0 0 340 250" role="img" aria-labelledby="kmeans-assign-title kmeans-assign-desc">
				<title id="kmeans-assign-title">Assignment step with two fixed initial centroids</title>
				<desc id="kmeans-assign-desc">Three circular observations at data coordinates 50 comma 80, 70 comma 110, and 90 comma 80 are assigned to initial centroid A at 110 comma 130. Three square observations at 230 comma 70, 250 comma 100, and 270 comma 70 are assigned to initial centroid B at 210 comma 120. For the circular observation at 90 comma 80, squared distance to A is 2900 and squared distance to B is 16000, so it joins A. Centroids do not move during this step.</desc>
				<rect class="viz-plot-bg" x="10" y="26" width="320" height="170" rx="5"></rect>
				<text class="viz-axis-label" x="18" y="18">1 · ASSIGN · HOLD CENTROIDS FIXED</text>
				<path d="M90 120L110 70M90 120L210 80" style="fill:none;stroke:var(--viz-edge);stroke-width:1.5;stroke-dasharray:5 4"></path>
				<text class="viz-label" x="74" y="88">d² to A = 2,900</text>
				<text class="viz-label" x="190" y="112">d² to B = 16,000</text>
				<circle class="viz-operating-point" cx="50" cy="120" r="6"></circle>
				<circle class="viz-operating-point" cx="70" cy="90" r="6"></circle>
				<circle class="viz-operating-point" cx="90" cy="120" r="6"></circle>
				<rect class="viz-node viz-node--focus" x="224" y="124" width="12" height="12" rx="1"></rect>
				<rect class="viz-node viz-node--focus" x="244" y="94" width="12" height="12" rx="1"></rect>
				<rect class="viz-node viz-node--focus" x="264" y="124" width="12" height="12" rx="1"></rect>
				<path d="M102 62L118 78M118 62L102 78" style="fill:none;stroke:var(--viz-state-stroke);stroke-width:3"></path>
				<path d="M202 72L218 88M218 72L202 88" style="fill:none;stroke:var(--viz-state-stroke);stroke-width:3"></path>
				<text class="viz-callout" x="102" y="50" text-anchor="end">A₀ (110, 130)</text>
				<text class="viz-callout" x="218" y="63">B₀ (210, 120)</text>
				<text class="viz-label" x="18" y="216">○ joins nearest A₀ · □ joins nearest B₀</text>
				<text class="viz-label" x="18" y="236">Only membership changes.</text>
			</svg>
		</section>
		<section class="visual-panel plot-panel">
			<svg viewBox="0 0 340 250" role="img" aria-labelledby="kmeans-update-title kmeans-update-desc">
				<title id="kmeans-update-title">Update step moves each centroid to its assigned points' mean</title>
				<desc id="kmeans-update-desc">The same three circles and three squares retain their assignments. Initial centroid A moves from 110 comma 130 to the circular points' arithmetic mean at 70 comma 90. Initial centroid B moves from 210 comma 120 to the square points' arithmetic mean at 250 comma 80. Labelled arrows connect each old center to its new mean. Assignments do not change during this step.</desc>
				<defs><marker id="kmeans-move-head" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path class="viz-arrow-forward" d="M0 0L7 3.5L0 7Z"></path></marker></defs>
				<rect class="viz-plot-bg" x="10" y="26" width="320" height="170" rx="5"></rect>
				<text class="viz-axis-label" x="18" y="18">2 · UPDATE · HOLD ASSIGNMENTS FIXED</text>
				<circle class="viz-operating-point" cx="50" cy="120" r="6"></circle>
				<circle class="viz-operating-point" cx="70" cy="90" r="6"></circle>
				<circle class="viz-operating-point" cx="90" cy="120" r="6"></circle>
				<rect class="viz-node viz-node--focus" x="224" y="124" width="12" height="12" rx="1"></rect>
				<rect class="viz-node viz-node--focus" x="244" y="94" width="12" height="12" rx="1"></rect>
				<rect class="viz-node viz-node--focus" x="264" y="124" width="12" height="12" rx="1"></rect>
				<path d="M102 62L118 78M118 62L102 78M202 72L218 88M218 72L202 88" style="fill:none;stroke:var(--viz-edge);stroke-width:2;stroke-dasharray:3 3"></path>
				<path d="M105 76L77 103M216 84L243 113" style="fill:none;stroke:var(--viz-state-stroke);stroke-width:2;marker-end:url(#kmeans-move-head)"></path>
				<path d="M62 102L78 118M78 102L62 118" style="fill:none;stroke:var(--viz-state-stroke);stroke-width:3"></path>
				<path d="M242 112L258 128M258 112L242 128" style="fill:none;stroke:var(--viz-state-stroke);stroke-width:3"></path>
				<text class="viz-callout" x="82" y="145">A₁ = (70, 90)</text>
				<text class="viz-callout" x="258" y="151" text-anchor="end">B₁ = (250, 80)</text>
				<text class="viz-label" x="18" y="216">A₁ = ((50+70+90)/3, (80+110+80)/3)</text>
				<text class="viz-label" x="18" y="236">B₁ = ((230+250+270)/3, (70+100+70)/3)</text>
			</svg>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> assignment changes memberships while centers stay fixed. Updating then holds membership fixed and moves each center to its mean. The illustrated SSE decreases from 22,000 to 2,800. Original construction follows <a href="https://doi.org/10.1109/TIT.1982.1056489">Lloyd's paper</a> and the <a href="https://scikit-learn.org/stable/modules/clustering.html#k-means">scikit-learn guide</a>. <span class="visual-reader-check" data-reader-check="kmeans-assign-then-update"><strong>Check:</strong> What is held fixed while A moves to the mean of its three assigned samples?</span></figcaption>
</figure>

Dense assignment costs $O(nkd)$ per iteration; mean updates add $O(nd)$.
Standard exact updates with stable tie/empty-cluster policies reach a fixed point on finite data, not a guaranteed global optimum.
Floating-point tolerances and iteration limits add numerical stopping policies.

### Give Observations Unequal Weight

**New task:** Double the weight of $(90,80)$ in A's fixed membership.

**Picture and change:** Minimize $\sum_i a_i\|x_i-\mu\|^2$ with positive weights, using $\mu=\sum_i a_ix_i/\sum_i a_i$.
Positive per-observation weights leave fixed-center nearest assignment unchanged.

**Check:** A becomes $((50+70+180)/4,(80+110+160)/4)=(75,87.5)$.

**Use when:** Rows represent justified frequencies or importance in the distortion objective.

### Update from an Arriving Point

**New task:** After the reference update, assign $(110,70)$ to A, whose count is three.

**Picture and change:** Increment the count to four and use $\mu_{new}=\mu_{old}+(x-\mu_{old})/4$.

**Check:** A moves from $(70,90)$ to $(80,85)$ while B remains unchanged.

**Use when:** A stream or mini-batch budget prevents full reassignment and mean recomputation.
An incremental step does not inherit full-data Lloyd objective nonincrease.

Adding outlier $(1000,80)$ to B's fixed membership moves its mean to $(437.5,80)$.
Squared distance strongly rewards accommodating outliers, even when the resulting center is not typical.

<!-- learning:core-end -->
<div data-learning-extension>

## k-means++ initialization

Random initialization sometimes converges to bad local minima. **k-means++** [(Arthur & Vassilvitskii, 2007)](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf):

1. Pick $\mu_1$ uniformly at random.
2. For $j = 2, \dots, k$: pick $\mu_j$ from data with probability proportional to $\min_{j' < j} \|x - \mu_{j'}\|^2$. Points far from existing centers are more likely to be picked.

The $O(\log k)$ expected approximation guarantee concerns seeded distortion under the theorem's conditions, not global-optimum recovery or runtime.

## Choosing $k$

There is no universally right answer. Heuristics:

- **Elbow method:** plot within-cluster SSE against k. Inspect diminishing returns; an elbow need not be distinct or objectively best.
- **Silhouette score**: average similarity of each point to its own cluster vs. nearest other cluster; pick $k$ maximizing it.
- **Gap statistic**: compare to clustering on uniform random data of the same shape.
- **Domain knowledge**: often the right answer.

Choose geometry, downstream usefulness, and stability criteria before selecting among these heuristics.

## Assumptions

Euclidean units and squared distortion define the task; standardizing features changes that task rather than automatically repairing it.
Spherical mixture interpretations require additional probabilistic assumptions, not equal-size constraints in the objective.
Replacing squared distance with absolute coordinate differences yields coordinate medians, not the same mean update.

## Mini-batch k-means

[Mini-batch updates](https://research.google/pubs/web-scale-k-means-clustering/) trade full reassignment work for stochastic updates; evaluate quality, count policy, and workload-specific cost.

## When k-means is and isn't appropriate

Use it when squared-distance compression or grouping matches the task, including vector-quantization codebooks.
Nonconvex shape discovery, categorical semantics, and alternative similarity objectives need separately justified methods.

## Common pitfalls

Cluster IDs are permutation-invariant; compare partitions or aligned centers, not raw labels across runs.
Alternating minimization alone does not make this algorithm an arbitrary mixture model's EM procedure.

</div>
