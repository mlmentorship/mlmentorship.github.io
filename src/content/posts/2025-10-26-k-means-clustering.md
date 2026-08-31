---
title: "k-means clustering"
description: "Partition n points into k clusters by minimizing within-cluster variance. Lloyd's algorithm: alternate assigning points to nearest center and recomputing centers."
date: "2025-10-26"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

**k-means** partitions $n$ data points into $k$ clusters by minimizing the sum of squared distances from each point to its cluster's centroid:

$$
\min_{\mu_1, \dots, \mu_k,\ C} \sum_{i=1}^{n} \|x_i - \mu_{C(i)}\|^2.
$$

**Lloyd's algorithm** (1957) alternates: (a) assign each point to its nearest centroid; (b) recompute each centroid as the mean of its assigned points. Iterate to convergence.

k-means is the canonical clustering algorithm: $O(n k d)$ per iteration, easy to implement, and a good first attempt at any unsupervised structure problem. It is used directly (customer segmentation, vector-quantization codebooks) and as a building block (initialization for GMMs, anchor selection for object detection, sub-sampling for retrieval index training).

## The algorithm

Initialize $k$ centroids (random points or **k-means++** for stability). Then repeat:

1. **Assignment**: $C(i) = \arg\min_j \|x_i - \mu_j\|^2$ for each $i$.
2. **Update**: $\mu_j = \tfrac{1}{|C^{-1}(j)|} \sum_{i \in C^{-1}(j)} x_i$ for each $j$.

<!-- visual:kmeans-assign-then-update -->
<figure class="learning-figure" aria-labelledby="kmeans-iteration-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="kmeans-iteration-title">What stays fixed in each half of a Lloyd iteration?</p>
	<div class="visual-grid--two" role="group" aria-label="Nearest-centroid assignment followed by centroid mean update">
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
	<figcaption><strong>Read it this way:</strong> in the left panel, compare distances while A₀ and B₀ stay fixed; circles and squares encode the resulting memberships without relying on color. In the right panel, freeze those memberships and move each × to the arithmetic mean of its three points. Alternating these two objective-nonincreasing steps reaches a fixed point, but the result can still be only a local minimum. Original construction checked against <a href="https://doi.org/10.1109/TIT.1982.1056489">Lloyd's least-squares quantization paper</a> and the <a href="https://scikit-learn.org/stable/modules/clustering.html#k-means">scikit-learn k-means guide</a>.</figcaption>
</figure>

Stop when assignments don't change (or change below a threshold). Convergence is guaranteed in finite steps but only to a local minimum. K-means is **not convex**.

## k-means++ initialization

Random initialization sometimes converges to bad local minima. **k-means++** [(Arthur & Vassilvitskii, 2007)](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf):

1. Pick $\mu_1$ uniformly at random.
2. For $j = 2, \dots, k$: pick $\mu_j$ from data with probability proportional to $\min_{j' < j} \|x - \mu_{j'}\|^2$. Points far from existing centers are more likely to be picked.

Gives an $O(\log k)$-approximation to the optimum in expectation. Used as default initialization in scikit-learn and most modern implementations.

## Choosing $k$

There is no universally right answer. Heuristics:

- **Elbow method**: plot total within-cluster sum of squares vs $k$; look for the "elbow" where adding more clusters stops paying off.
- **Silhouette score**: average similarity of each point to its own cluster vs. nearest other cluster; pick $k$ maximizing it.
- **Gap statistic**: compare to clustering on uniform random data of the same shape.
- **Domain knowledge**: often the right answer.

For very large data, run k-means with a few values of $k$ in parallel and pick the one with the best silhouette on a subsample.

## Assumptions

k-means implicitly assumes:

- **Spherical, equal-variance clusters** (penalizes non-spherical clusters by splitting them).
- **Equal cluster sizes** (large clusters are sometimes split, small ones merged).
- **Euclidean distance is meaningful** in the input space.

When clusters are elongated, of unequal density, or non-convex, k-means fails. Try GMMs (allows ellipsoidal clusters), DBSCAN (density-based, arbitrary shapes), or spectral clustering.

## Mini-batch k-means

For huge $n$, full assignment is expensive. **Mini-batch k-means** [(Sculley, 2010)](https://research.google/pubs/web-scale-k-means-clustering/): each iteration uses a small random subset to update centroids. Trades accuracy for huge speedup.

## When k-means is and isn't appropriate

| Setting | Verdict |
|---------|---------|
| Roughly spherical clusters in low-d | Excellent |
| Vector quantization codebook | Excellent (k = codebook size) |
| Image segmentation by color | Good (k = number of colors) |
| Customer segmentation on raw features | Try, but standardize first |
| Text or sparse data | Use spherical k-means (cosine distance) |
| Non-convex shapes (moons, rings) | Fails; use DBSCAN or spectral |
| Large $k$ (thousands) | Use HNSW-accelerated variants |

## Common pitfalls

- **Forgetting to standardize features.** k-means uses Euclidean distance; features with larger scale dominate.
- **Random initialization without k-means++.** Different runs give different clusterings; report consensus or use k-means++.
- **Treating cluster IDs as meaningful.** Cluster 0 in one run may be cluster 7 in the next; permutation invariance.
- **Using k-means on categorical data without thought.** Use k-modes, k-prototypes, or one-hot + k-means with care.
- **Reporting one clustering as "the answer."** Run multiple times, take the best by within-cluster SS.
