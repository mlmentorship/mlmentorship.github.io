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

<!-- visual:d-separation-conditioning-switch -->
<figure class="learning-figure" aria-labelledby="d-separation-conditioning-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="d-separation-conditioning-title">Predict whether conditioning blocks or opens a path</p>
	<div class="visual-scroll">
		<svg viewBox="0 0 360 388" role="img" aria-labelledby="d-separation-svg-title d-separation-svg-desc">
			<title id="d-separation-svg-title">Conditioning closes a chain path but opens a collider path</title>
			<desc id="d-separation-svg-desc">Four directed three-node graphs compare a chain A to M to B with a collider A to C from B. In the unconditioned chain, the path from A to B is active, so A and B may be dependent. Observing middle node M blocks the chain, so A and B are conditionally independent. In the unconditioned collider, the path is blocked, so A and B are independent. Observing common effect C opens the collider path, so its causes A and B become conditionally dependent through explaining away. Observed nodes have a double outline and the word observed; blocked paths have a central cross and dashed outer edges.</desc>
			<rect class="viz-plot-bg" x="8" y="8" width="344" height="178" rx="5"></rect>
			<text class="viz-axis-label" x="18" y="29">CHAIN · A → M → B</text>
			<text class="viz-label" x="18" y="53">Do not observe M</text>
			<circle class="viz-node" cx="142" cy="50" r="18"></circle>
			<circle class="viz-node" cx="220" cy="50" r="18"></circle>
			<circle class="viz-node" cx="298" cy="50" r="18"></circle>
			<text class="viz-node-label" x="142" y="55">A</text>
			<text class="viz-node-label" x="220" y="55">M</text>
			<text class="viz-node-label" x="298" y="55">B</text>
			<path d="M160 50H202M238 50H280" style="fill:none;stroke:var(--viz-edge);stroke-width:2"></path>
			<path d="M195 46L202 50L195 54ZM273 46L280 50L273 54Z" style="fill:var(--viz-edge)"></path>
			<text class="viz-callout" x="142" y="81" text-anchor="middle">ACTIVE PATH</text>
			<text class="viz-label" x="271" y="81" text-anchor="middle">A and B may depend</text>
			<path class="viz-gridline" d="M18 96H342"></path>
			<text class="viz-label" x="18" y="125">Observe M</text>
			<circle class="viz-node" cx="142" cy="122" r="18"></circle>
			<circle class="viz-node viz-node--focus" cx="220" cy="122" r="21"></circle>
			<circle class="viz-node" cx="220" cy="122" r="16"></circle>
			<circle class="viz-node" cx="298" cy="122" r="18"></circle>
			<text class="viz-node-label" x="142" y="127">A</text>
			<text class="viz-node-label" x="220" y="127">M</text>
			<text class="viz-node-label" x="298" y="127">B</text>
			<path d="M160 122H199M241 122H280" style="fill:none;stroke:var(--viz-edge);stroke-width:1.5;stroke-dasharray:5 4"></path>
			<path d="M192 118L199 122L192 126ZM273 118L280 122L273 126Z" style="fill:var(--viz-edge)"></path>
			<text class="viz-node-value" x="220" y="158">OBSERVED</text>
			<text class="viz-callout" x="142" y="175" text-anchor="middle">BLOCKED</text>
			<text class="viz-label" x="271" y="175" text-anchor="middle">A ⟂ B | M</text>
			<rect class="viz-plot-bg" x="8" y="202" width="344" height="178" rx="5"></rect>
			<text class="viz-axis-label" x="18" y="223">COLLIDER · A → C ← B</text>
			<text class="viz-label" x="18" y="247">Do not observe C</text>
			<circle class="viz-node" cx="142" cy="244" r="18"></circle>
			<circle class="viz-node" cx="220" cy="244" r="18"></circle>
			<circle class="viz-node" cx="298" cy="244" r="18"></circle>
			<text class="viz-node-label" x="142" y="249">A</text>
			<text class="viz-node-label" x="220" y="249">C</text>
			<text class="viz-node-label" x="298" y="249">B</text>
			<path d="M160 244H202M280 244H238" style="fill:none;stroke:var(--viz-edge);stroke-width:1.5;stroke-dasharray:5 4"></path>
			<path d="M195 240L202 244L195 248ZM245 240L238 244L245 248Z" style="fill:var(--viz-edge)"></path>
			<path d="M214 234L226 254M226 234L214 254" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:2"></path>
			<text class="viz-callout" x="151" y="275" text-anchor="middle">BLOCKED AT C</text>
			<text class="viz-label" x="275" y="275" text-anchor="middle">A ⟂ B</text>
			<path class="viz-gridline" d="M18 290H342"></path>
			<text class="viz-label" x="18" y="319">Observe C</text>
			<circle class="viz-node" cx="142" cy="316" r="18"></circle>
			<circle class="viz-node viz-node--focus" cx="220" cy="316" r="21"></circle>
			<circle class="viz-node" cx="220" cy="316" r="16"></circle>
			<circle class="viz-node" cx="298" cy="316" r="18"></circle>
			<text class="viz-node-label" x="142" y="321">A</text>
			<text class="viz-node-label" x="220" y="321">C</text>
			<text class="viz-node-label" x="298" y="321">B</text>
			<path d="M160 316H199M280 316H241" style="fill:none;stroke:var(--viz-edge);stroke-width:2"></path>
			<path d="M192 312L199 316L192 320ZM248 312L241 316L248 320Z" style="fill:var(--viz-edge)"></path>
			<text class="viz-node-value" x="220" y="352">OBSERVED</text>
			<text class="viz-callout" x="140" y="369" text-anchor="middle">PATH OPENS</text>
			<text class="viz-label" x="275" y="369" text-anchor="middle">A and B depend | C</text>
		</svg>
	</div>
	<figcaption><strong>Read it this way:</strong> compare the middle node in each pair. Observing a non-collider blocks the chain, but a collider starts blocked and observing its common effect opens the path. If two independent causes can produce the same effect, learning that the effect occurred makes evidence for one cause evidence against the other: explaining away. The graph encodes the independence pattern; it does not by itself assert causation. Original schematic checked against <a href="https://probml.github.io/pml-book/book1.html">Murphy (2022)</a> and <a href="https://dl.acm.org/doi/10.1016/0890-5401%2890%2990060-T">Geiger, Verma, and Pearl (1990)</a>.</figcaption>
</figure>

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
