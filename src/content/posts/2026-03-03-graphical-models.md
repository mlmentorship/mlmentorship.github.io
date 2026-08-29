---
title: "Probabilistic graphical models"
description: "Express joint distributions as graphs whose structure encodes conditional independence. Bayesian networks (directed) and Markov random fields (undirected)."
date: "2026-03-03"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A **probabilistic graphical model** (PGM) is a representation of a joint probability distribution as a graph whose nodes are random variables and whose edges encode dependencies. The graph structure determines a factorization of the joint and a set of conditional independence relations.

PGMs were the dominant framework for probabilistic ML from the 1990s through the early 2010s. Many modern probabilistic methods. VAEs, latent-variable diffusion, message passing in transformers (loosely). Descend from PGM ideas. Knowing PGMs gives you the right conceptual vocabulary for any latent-variable model: independence, factorization, marginalization, conditioning.

## Two main families

### Bayesian networks (directed acyclic graphs)

Each node has a conditional distribution given its parents. The joint factorizes as

$$
p(x_1, \dots, x_n) = \prod_{i=1}^{n} p(x_i \mid \mathrm{parents}(x_i)).
$$

Examples: naive Bayes (one parent class node, leaf observation nodes), HMM, Bayesian linear regression, hierarchical Bayesian models.

Encoded independence: each node is conditionally independent of its non-descendants given its parents (local Markov property).

### Markov random fields (undirected graphs)

The joint factorizes over **cliques** $C \subseteq \text{nodes}$:

$$
p(x_1, \dots, x_n) = \frac{1}{Z} \prod_C \psi_C(x_C),
$$

with potential functions $\psi_C \ge 0$ and partition function $Z = \sum_x \prod_C \psi_C(x_C)$.

Examples: image MRFs (pairwise potentials between neighboring pixels), CRFs (conditional random fields, discriminative MRFs), Boltzmann machines.

Encoded independence: $X_A \perp X_B \mid X_C$ if $C$ separates $A$ from $B$ in the graph.

## d-separation (Bayesian networks)

A path between two nodes is **blocked** by a set $C$ if either:

- A non-collider on the path is in $C$, or
- A collider (node with two incoming arrows on the path) and none of its descendants are in $C$.

Two nodes are **d-separated** by $C$ if every path between them is blocked. d-separation $\Rightarrow$ conditional independence given $C$ (in the model).

This formalism explains the famous explaining-away phenomenon: conditioning on a common effect makes its causes correlated.

## Inference tasks

For a graphical model, the standard tasks are:

- **Marginal**: compute $p(x_i)$ or $p(x_S)$ for a subset $S$.
- **Conditional**: compute $p(x_S \mid x_O)$ given observations.
- **MAP**: find $\arg\max_x p(x \mid x_O)$.

Exact methods:

- **Variable elimination**: marginalize out variables one by one, exploiting factorization.
- **Belief propagation / sum-product**: message passing on tree-structured graphs (or cluster graphs / junction trees for general graphs).
- **Junction tree algorithm**: exact inference in any graph by clustering into a tree of cliques.

For graphs with high tree-width, exact inference is exponential. Approximate methods:

- **MCMC** (Gibbs, Metropolis-Hastings).
- **Variational inference** (mean-field, structured, neural).
- **Expectation propagation**.

## Special cases that became their own fields

| Graphical model | Modern name |
|----------------|-------------|
| Latent variable Bayesian network | VAE (with neural conditional distributions) |
| Linear-Gaussian state space | Kalman filter |
| Discrete latent chain | HMM |
| Conditional MRF | CRF |
| Boltzmann machine | RBM, deep belief net (historical) |
| Topic model (Bayesian doc-topic) | LDA |
| Naive Bayes | Naive Bayes (still used) |

## Relevance in 2026

PGM as a framework is less central than it was in 2010, replaced by neural networks for most practical inference. But graphical-model thinking persists in:

- Diffusion models (Markov chain over noise levels).
- VAEs (latent → observation Bayesian network).
- Probabilistic programming (Pyro, Stan, NumPyro).
- Causal inference (DAGs are the language).
- Structured prediction with CRFs in some NLP pipelines.

## Common pitfalls

- **Confusing causation with d-separation.** PGMs model dependencies; causation requires additional assumptions (intervention, do-calculus).
- **Treating the joint distribution as fully specified by the graph alone.** The graph only specifies *structure*; the conditional distributions are separate.
- **Forgetting that exact inference is intractable for general MRFs.** Tree-width matters.
- **Reading missing edges as independence.** They imply *conditional* independence given the rest, not marginal independence.

## Related

- [Markov chains](/concepts/markov-chains/). Simplest sequential PGM.
- [Bayes' rule and the posterior](/concepts/bayes-rule-and-posterior/). Foundation for PGM inference.
