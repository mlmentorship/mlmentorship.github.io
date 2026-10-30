---
title: "Probabilistic graphical models"
description: "Express joint distributions as graphs whose structure encodes conditional independence. Bayesian networks (directed) and Markov random fields (undirected)."
date: "2026-03-03"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A probabilistic graphical model pairs a graph with factors defining a joint probability distribution.
Nodes represent variables, not necessarily causal mechanisms.
The graph constrains independence; probability tables or densities supply the numerical parameters.

Ask whether observing a middle node blocks or opens information between two endpoints.
Our teaching fixture uses independent fair binary variables $A,B$ and their common effect $C=A\operatorname{OR}B$.

## Two main families

### Bayesian networks (directed acyclic graphs)

Each node has a conditional distribution given its parents. The joint factorizes as

$$
p(x_1, \dots, x_n) = \prod_{i=1}^{n} p(x_i \mid \mathrm{parents}(x_i)).
$$

The local Markov property makes each node independent of its non-descendants given its parents.
For our collider, $p(a,b,c)=p(a)p(b)p(c\mid a,b)$.
The last factor is one when $c=a\operatorname{OR}b$, zero otherwise.

### Markov random fields (undirected graphs)

The joint factorizes over **cliques** $C \subseteq \text{nodes}$:

$$
p(x_1, \dots, x_n) = \frac{1}{Z} \prod_C \psi_C(x_C),
$$

Potentials $\psi_C$ are nonnegative; require finite, positive $Z$.
For continuous variables, replace the sum defining $Z$ with an integral.
An undirected separator blocks every path between the sets whose independence it establishes.
Strict positivity is required for the general equivalence between undirected Markov properties and clique factorization.

## d-separation (Bayesian networks)

A path is blocked by an observation set $O$ if either:

- A non-collider on that path is observed.
- A collider and all its descendants are unobserved.

At a collider, both path arrows point toward the middle node.
All paths must be blocked for d-separation to imply conditional independence.
Chains $A\to M\to B$ and forks $A\leftarrow M\to B$ both close when $M$ is observed.

An observed collider or its descendant instead opens that path.
An open path permits association; it does not guarantee association for every choice of parameters.

<!-- visual:d-separation-conditioning-switch -->
<figure class="learning-figure" aria-labelledby="d-separation-conditioning-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="d-separation-conditioning-title">Conditioning can close a chain while opening a collider</p><div class="visual-argument" data-visual-argument="d-separation-conditioning-switch" data-argument-kind="matrix"><p class="visual-argument-premise">Conditioning can close one path and open another</p><svg viewBox="0 0 360 142" role="img" aria-label="Conditioning can close one path and open another. A conditioned chain middle blocks its path. Conditioning on a collider can open the path between its parents."><title>Conditioning can close one path and open another</title><desc>A conditioned chain middle blocks its path. Conditioning on a collider can open the path between its parents.</desc><text x="154.24" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="154.24" dy="0">uncond.</tspan></text><text x="278.08" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="278.08" dy="0">cond.</tspan></text><text x="80.32" y="54" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="80.32" dy="0">chain</tspan></text><rect data-proof-cell="0,0" x="92.32" y="36" width="120.84" height="44" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke)"></rect><text x="152.74" y="63.5" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="152.74" dy="0">open</tspan></text><path d="M96.32 76L209.16 40" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><rect data-proof-cell="0,1" x="216.16" y="36" width="120.84" height="44" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke)"></rect><text x="276.58" y="63.5" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="276.58" dy="0">closed</tspan></text><text x="80.32" y="101" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="80.32" dy="0">collider</tspan></text><rect data-proof-cell="1,0" x="92.32" y="83" width="120.84" height="44" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke)"></rect><text x="152.74" y="110.5" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="152.74" dy="0">closed</tspan></text><rect data-proof-cell="1,1" x="216.16" y="83" width="120.84" height="44" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke)"></rect><text x="276.58" y="110.5" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="276.58" dy="0">open</tspan></text><path d="M220.16 123L333 87" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
	<div class="visual-scroll">
		<svg viewBox="0 0 360 388" role="img" aria-labelledby="d-separation-svg-title d-separation-svg-desc">
			<title id="d-separation-svg-title">Conditioning closes a chain path but opens a collider path</title>
			<desc id="d-separation-svg-desc">Four three-node graphs compare a chain with a collider. Observing the middle of the chain blocks its path. An unobserved collider blocks its path; observing the collider opens it and permits association. Open paths do not force dependence for every parameterization. Observed nodes have double outlines; blocked paths have a cross and dashed edges.</desc>
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
			<text class="viz-label" x="275" y="369" text-anchor="middle">A and B may depend | C</text>
		</svg>
	</div>
	<table><caption>OR collider: independent fair inputs</caption><thead><tr><th>A</th><th>B</th><th>C</th><th>Joint mass</th><th>Mass given C=1</th></tr></thead><tbody><tr><td>0</td><td>0</td><td>0</td><td>1/4</td><td>0</td></tr><tr><td>0</td><td>1</td><td>1</td><td>1/4</td><td>1/3</td></tr><tr><td>1</td><td>0</td><td>1</td><td>1/4</td><td>1/3</td></tr><tr><td>1</td><td>1</td><td>1</td><td>1/4</td><td>1/3</td></tr></tbody></table>
	<div class="visual-derivation-caption">observing the chain's middle blocks its path; observing the collider opens its path. The OR table gives an actual dependence: once C=1 is known, evidence for B reduces the probability of A. The arrows alone do not guarantee that numerical result or establish causation. The original schematic retains its <a href="https://probml.github.io/pml-book/book1.html">Murphy (2022)</a> and <a href="https://dl.acm.org/doi/10.1016/0890-5401%2890%2990060-T">Geiger-Verma-Pearl (1990)</a> provenance.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="d-separation-conditioning-switch"><strong>Check:</strong> Why does observing the middle node block the chain but open the collider? Which conclusion is only a possibility?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="d-separation-conditioning-switch">A conditioned chain middle blocks its path. Conditioning on a collider can open the path between its parents.</span></figcaption>
</figure>

### Normalize the collider

The four input pairs each have probability $1/4$.
Observing $C=1$ excludes only $(0,0)$, so $p(C=1)=3/4$.
The three remaining pairs each receive conditional mass $1/3$:

$$
p(A=1\mid C=1)=\frac{1/2}{3/4}=\frac23.
$$

Further observing $B=1$ leaves two equally likely pairs, giving $p(A=1\mid C=1,B=1)=1/2$.
Observing $B=0$ instead forces $A=1$.
This is explaining away within the specified model, not an observational proof of causation.

### Observe a non-collider

**New task:** use $A\to M\to B$, with fair $A$ and binary copying probabilities $0.8$, then $0.9$.

**Picture and change:** replace the common effect with a middle transition.
Before observing $M$,

$$p(B=1\mid A=1)=0.8(0.9)+0.2(0.1)=0.74.$$

**Check:** given $M=1$, the probability of $B=1$ is $0.9$ regardless of $A$.
Conditioning blocks this path rather than opening it.

**Use when:** the observed middle node is a non-collider on the relevant path.

### Observe a collider's descendant

**New task:** add $C\to D$ with deterministic $D=C$ and observe $D=1$.

**Picture and change:** evidence arrives downstream instead of at $C$ directly.

**Check:** the same three OR cases survive, so $p(A=1\mid D=1)=2/3$.

**Use when:** selection or measurement depends on a common effect's descendant.

### An active path need not produce association

Keep the collider graph, but set $p(C=1\mid A,B)=1/2$ for every input pair.
Observing $C$ then leaves $A,B$ independent.
This parameterization is Markov to the graph but not faithful: it has extra independence beyond graphical separation.

<!-- learning:core-end -->

<div data-learning-extension>

## Inference tasks

Marginalization sums out variables; conditioning renormalizes evidence-compatible cases; joint MAP finds the most probable assignment.
Separate marginal modes need not form the joint MAP assignment.

Variable elimination exploits factors; exact cost depends on variable domains and induced treewidth.
Sum-product is exact on trees; junction trees cluster variables for exact general-graph inference.
Large cliques can make that impractical.
MCMC, variational inference, and expectation propagation instead require approximation or convergence checks.

## Special cases that became their own fields

| Graphical model | Modern name |
|----------------|-------------|
| Latent variable Bayesian network | VAE with approximate amortized inference |
| Linear-Gaussian state space | Kalman filter |
| Discrete latent chain | HMM |
| Conditional MRF | CRF |
| Boltzmann machine | RBM, deep belief net (historical) |
| Topic model (Bayesian doc-topic) | LDA |
| Naive Bayes | Naive Bayes (still used) |

## Relevance in 2026

Neural conditional distributions can parameterize a graphical model; they do not generally replace inference.
Probabilistic programming, state-space models, and structured prediction still require explicit distributions and conditioning operations.

## Common pitfalls

Does a missing $A$-$B$ edge always imply marginal independence?

<details><summary>Answer</summary>

No: a chain can connect them through another variable.
Use the graph-specific separation rule and actual conditioning set.
Causal interpretations additionally require a causal model and identifying assumptions.

</details>

</div>

## Related

- [Markov chains](/concepts/markov-chains/). Simplest sequential PGM.
- [Bayes' rule and the posterior](/concepts/bayes-rule-and-posterior/). Foundation for PGM inference.
