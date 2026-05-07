---
title: "k-means clustering"
description: "Partition n points into k clusters by minimizing within-cluster variance. Lloyd's algorithm: alternate assigning points to nearest center and recomputing centers."
date: "2026-05-07"
draft: false
tags: ["reference"]
category: "reference"
---

## One-line definition

**k-means** partitions $n$ data points into $k$ clusters by minimizing the sum of squared distances from each point to its cluster's centroid:

$$
\min_{\mu_1, \dots, \mu_k,\ C} \sum_{i=1}^{n} \|x_i - \mu_{C(i)}\|^2.
$$

**Lloyd's algorithm** (1957) alternates: (a) assign each point to its nearest centroid; (b) recompute each centroid as the mean of its assigned points. Iterate to convergence.

## Why it matters

k-means is the canonical clustering algorithm: $O(n k d)$ per iteration, easy to implement, and a good first attempt at any unsupervised structure problem. It is used directly (customer segmentation, vector-quantization codebooks) and as a building block (initialization for GMMs, anchor selection for object detection, sub-sampling for retrieval index training).

## The algorithm

Initialize $k$ centroids (random points or **k-means++** for stability). Then repeat:

1. **Assignment**: $C(i) = \arg\min_j \|x_i - \mu_j\|^2$ for each $i$.
2. **Update**: $\mu_j = \tfrac{1}{|C^{-1}(j)|} \sum_{i \in C^{-1}(j)} x_i$ for each $j$.

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
