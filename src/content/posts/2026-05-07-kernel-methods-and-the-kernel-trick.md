---
title: "Kernel methods and the kernel trick"
description: "Compute inner products in a high-dimensional feature space without ever materializing the features. The mathematical move that lets a linear classifier draw nonlinear boundaries."
date: "2026-05-07"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A **kernel** $k(x, x') = \langle \phi(x), \phi(x') \rangle$ computes the inner product of two points in a feature space defined by $\phi$, without explicitly evaluating $\phi$. Algorithms expressible in terms of inner products (SVM, Gaussian processes, kernel PCA, kernel ridge regression) can swap dot products for $k$ and operate implicitly in feature space.

Linear models are limited to linear decision boundaries. The classical fix was to engineer nonlinear features and run a linear model on them. Kernel methods give you that move for free: any positive-definite kernel implicitly defines a (potentially infinite-dimensional) feature space, and the algorithm only ever touches inner products.

This was the dominant nonlinear ML approach from roughly 1995 to 2012. Then deep learning replaced it for almost every supervised task. Kernels remain central to Gaussian processes, attention (the softmax of $Q K^\top$ is a kernelized similarity), and certain interpretability and theoretical-analysis tools (NTK, kernel ridge regression as a baseline).

## The trick

Suppose your algorithm only reads data through inner products $\langle x_i, x_j \rangle$. Substitute $k(x_i, x_j)$. You are now running the algorithm in the feature space defined by $\phi$, where $k(x_i, x_j) = \langle \phi(x_i), \phi(x_j) \rangle$, without touching $\phi$.

Concrete example: polynomial kernel of degree 2 in $\mathbb{R}^d$:

$$
k(x, x') = (x^\top x' + 1)^2.
$$

Expand it. You will find this equals $\langle \phi(x), \phi(x') \rangle$ where $\phi$ maps to a $\binom{d+2}{2}$-dimensional space of monomials up to degree 2. Computing $\phi$ explicitly is $O(d^2)$ memory and compute; computing $k$ is $O(d)$.

For the **RBF kernel** $k(x, x') = \exp(-\|x - x'\|^2 / 2\sigma^2)$, the implicit feature space is infinite-dimensional. You cannot compute $\phi$ explicitly even in principle.

## The Gram matrix

For a dataset of $N$ points, the kernel defines an $N \times N$ **Gram matrix** $K_{ij} = k(x_i, x_j)$. Many kernel algorithms reduce to operations on $K$:

- **Kernel ridge regression**: $\hat{f}(x) = \sum_i \alpha_i k(x_i, x)$ where $\alpha = (K + \lambda I)^{-1} y$.
- **Kernel PCA**: eigendecompose the centered $K$.
- **Gaussian processes**: use $K + \sigma^2 I$ as the covariance.
- **SVM**: dual formulation depends only on $K$ and labels.

The Gram matrix shape is the bottleneck: $O(N^2)$ memory, $O(N^3)$ to invert. Limits naive kernel methods to roughly $N \le 10^5$.

## What makes a valid kernel

$k$ must be **positive semi-definite**: for any finite set of points, the Gram matrix $K$ is PSD ($v^\top K v \ge 0$ for all $v$). Equivalently, $k(x, x') = \langle \phi(x), \phi(x') \rangle$ for some $\phi$ into some inner-product space (Mercer's theorem).

Common kernels:

- **Linear**: $k(x, x') = x^\top x'$. The trivial case.
- **Polynomial**: $k(x, x') = (x^\top x' + c)^d$.
- **RBF / Gaussian**: $k(x, x') = \exp(-\gamma \|x - x'\|^2)$. The default.
- **Laplacian**: $k(x, x') = \exp(-\gamma \|x - x'\|_1)$.
- **String kernels**: count shared substrings.
- **Graph kernels**: count shared subgraphs.

Combinations of valid kernels (sums, products, scalings) are valid kernels. Standard recipe for engineering domain-specific kernels.

## Kernel trick in attention

A bilinear attention score $s(q, k) = q^\top k / \sqrt{d}$ is the linear kernel. Linear attention papers replace this with feature-map kernels $\phi(q)^\top \phi(k)$ for cheaper computation; a softmax-attention variant can be approximated by random Fourier features for the RBF kernel.

## Why deep learning won

Kernels make a strong assumption: the right similarity function is fixed in advance. Deep learning learns the feature representation jointly with the task. For high-dimensional structured data (images, text, audio), learned representations beat hand-picked kernels by huge margins.

Kernels remain useful when:

- Data is small (Gaussian processes).
- The kernel encodes domain knowledge (string kernels in computational biology).
- Theoretical analysis is the goal (NTK, infinite-width networks).

## Common pitfalls

- **Confusing the kernel trick with the kernel matrix**. The trick is the substitution; the matrix is the data structure.
- **Using RBF without scaling**. Standardize or whiten features first; the bandwidth $\gamma$ is sensitive to feature scale.
- **Treating kernel methods as scalable**. The $O(N^2)$ Gram matrix kills naive applications above $\sim 10^5$ points. Approximations (Nyström, random Fourier features, inducing points for GPs) exist.
- **Conflating "kernel" in the SVM sense with "kernel" in the convolution sense**. Two unrelated meanings of the same word.

## Related

- [SVM and the kernel trick](/concepts/svm-and-kernels/).
- [Gaussian processes](/concepts/gaussian-processes/).
- [The attention mechanism](/concepts/attention-mechanism/).
