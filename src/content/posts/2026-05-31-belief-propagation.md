---
title: "Belief propagation (message passing)"
description: "Belief propagation computes graphical-model marginals through local messages. Sum-product is exact on trees and approximate on graphs with cycles."
date: "2026-05-31"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Belief propagation is a **message-passing** algorithm for computing marginal distributions in a graphical model. Nodes exchange local "messages" summarizing the evidence from their part of the graph; the **sum-product** variant computes marginals, the **max-product** variant computes the most probable configuration. It is **exact on trees** and an approximation (**loopy BP**) on graphs with cycles.

Inference (computing $p(x_i \mid \text{evidence})$) is the central operation in probabilistic models, and the naive sum over all configurations is exponential. Belief propagation is *the* algorithm that exploits the graph's factorization to make it tractable. It's the engine behind:

- **HMM forward-backward** and **CRF** training/decoding (these are BP on a chain).
- Decoding **LDPC / turbo codes** (loopy BP, the reason your phone's error correction works).
- General **factor-graph** inference in vision, sensor fusion, and probabilistic programming.

Interviewers use it to test whether you understand that **inference cost is governed by graph structure**, not just by the number of variables.

## The intuition

A graphical model factorizes a joint distribution into local factors:

$$
p(\mathbf{x}) = \frac{1}{Z}\prod_{a} \psi_a(\mathbf{x}_a).
$$

To get a marginal $p(x_i)$ you must sum out every other variable, which is exponential in general. BP avoids the blow-up by noticing that on a **tree**, the sum **distributes**: you can push summations inside products and reuse partial sums. Each "message" is exactly one of those reusable partial sums, flowing along an edge.

## The sum-product algorithm (factor graphs)

Two message types alternate until they reach the root / converge:

**Variable → factor** (multiply incoming messages from other factors):

$$
\mu_{x \to a}(x) = \prod_{b \in N(x)\setminus a} \mu_{b \to x}(x).
$$

**Factor → variable** (multiply the factor by incoming messages, then sum out the other variables):

$$
\mu_{a \to x}(x) = \sum_{\mathbf{x}_a \setminus x} \psi_a(\mathbf{x}_a) \prod_{y \in N(a)\setminus x} \mu_{y \to a}(y).
$$

The marginal at a variable is the product of all incoming messages (normalized):

$$
p(x_i) \propto \prod_{a \in N(x_i)} \mu_{a \to x_i}(x_i).
$$

<!-- visual:belief-propagation-branch-summary -->
<figure class="learning-figure" aria-labelledby="belief-propagation-branch-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="belief-propagation-branch-title">How does a whole branch become one local message?</p>
	<div class="visual-grid--two" role="group" aria-label="Two-step binary belief-propagation example">
		<section class="visual-panel plot-panel">
			<svg viewBox="0 0 300 230" role="img" aria-labelledby="bp-collapse-title bp-collapse-desc">
				<title id="bp-collapse-title">A factor sums out its branch variable to produce a message about x</title>
				<desc id="bp-collapse-desc">Variable y sends factor a the two-entry function 0.2, 0.8. Factor a combines each possible x with both possible y values and sums y out. The outgoing message is another two-entry function of x: 0.26, 0.68.</desc>
				<rect class="viz-plot-bg" x="8" y="25" width="284" height="195" rx="5"></rect>
				<text class="viz-axis-label" x="12" y="16">1 · COLLAPSE ONE BRANCH</text>
				<circle class="viz-node viz-node--input" cx="43" cy="70" r="22"></circle>
				<text class="viz-node-label" x="43" y="74" text-anchor="middle">y</text>
				<rect class="viz-node" x="126" y="48" width="44" height="44" rx="3"></rect>
				<text class="viz-node-label" x="148" y="74" text-anchor="middle">ψₐ</text>
				<circle class="viz-node viz-node--output" cx="255" cy="70" r="22"></circle>
				<text class="viz-node-label" x="255" y="74" text-anchor="middle">x</text>
				<path d="M65 70H126M170 70H233" style="fill:none;stroke:var(--viz-edge);stroke-width:1.5"></path>
				<path d="M119 66L126 70L119 74ZM226 66L233 70L226 74Z" style="fill:var(--viz-edge)"></path>
				<text class="viz-label" x="95" y="55" text-anchor="middle">μᵧ→ₐ(y)</text>
				<text class="viz-callout" x="95" y="91" text-anchor="middle">[0.20, 0.80]</text>
				<text class="viz-label" x="201" y="55" text-anchor="middle">μₐ→ₓ(x)</text>
				<text class="viz-callout" x="201" y="91" text-anchor="middle">[0.26, 0.68]</text>
				<text class="viz-axis-label" x="18" y="122">SUM OUT y; KEEP x AS THE INDEX</text>
				<text class="viz-callout" x="18" y="148">x = 0: 0.9 × 0.2 + 0.1 × 0.8 = 0.26</text>
				<text class="viz-callout" x="18" y="174">x = 1: 0.2 × 0.2 + 0.8 × 0.8 = 0.68</text>
				<text class="viz-label" x="18" y="203">The branch is now a function of x only.</text>
			</svg>
		</section>
		<section class="visual-panel plot-panel">
			<svg viewBox="0 0 300 230" role="img" aria-labelledby="bp-combine-title bp-combine-desc">
				<title id="bp-combine-title">The target variable multiplies incoming branch messages and normalizes</title>
				<desc id="bp-combine-desc">At variable x, the left message 0.26, 0.68 and right message 0.50, 0.25 are multiplied entry by entry. The unnormalized products 0.13, 0.17 sum to 0.30 and normalize to the belief 0.43, 0.57.</desc>
				<rect class="viz-plot-bg" x="8" y="25" width="284" height="195" rx="5"></rect>
				<text class="viz-axis-label" x="12" y="16">2 · COMBINE AT THE TARGET</text>
				<rect class="viz-node" x="20" y="48" width="42" height="42" rx="3"></rect>
				<text class="viz-node-label" x="41" y="73" text-anchor="middle">ψₐ</text>
				<circle class="viz-node viz-node--output" cx="150" cy="69" r="24"></circle>
				<text class="viz-node-label" x="150" y="66" text-anchor="middle">x</text>
				<text class="viz-node-value" x="150" y="80" text-anchor="middle">belief</text>
				<rect class="viz-node" x="238" y="48" width="42" height="42" rx="3"></rect>
				<text class="viz-node-label" x="259" y="73" text-anchor="middle">ψᵦ</text>
				<path d="M62 69H126M238 69H174" style="fill:none;stroke:var(--viz-edge);stroke-width:1.5"></path>
				<path d="M119 65L126 69L119 73ZM181 65L174 69L181 73Z" style="fill:var(--viz-edge)"></path>
				<text class="viz-callout" x="94" y="101" text-anchor="middle">[0.26, 0.68]</text>
				<text class="viz-callout" x="206" y="101" text-anchor="middle">[0.50, 0.25]</text>
				<text class="viz-axis-label" x="18" y="127">MULTIPLY MATCHING ENTRIES</text>
				<text class="viz-callout" x="18" y="151">x = 0: 0.26 × 0.50 = 0.13</text>
				<text class="viz-callout" x="18" y="176">x = 1: 0.68 × 0.25 = 0.17</text>
				<text class="viz-label" x="18" y="203">Normalize [0.13, 0.17] → p(x) = [0.43, 0.57]</text>
			</svg>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> follow one branch toward <var>x</var>: the factor multiplies its local compatibility by the incoming evidence and sums out <var>y</var>, leaving a two-entry function of <var>x</var>. At <var>x</var>, multiply matching entries from every branch, then normalize once to read the marginal. A message summarizes a branch; it is not itself a normalized probability.</figcaption>
</figure>

Replace the inner $\sum$ with a $\max$ and you get **max-product** (a.k.a. max-sum in log space), which finds the MAP configuration, the general version of **Viterbi**.

## Exact vs approximate

| Graph structure | BP behavior | Cost |
| --- | --- | --- |
| **Tree / chain** | Exact marginals in two passes (leaves→root→leaves) | $O(\text{edges} \cdot \lvert V \rvert^2)$ |
| **General graph** | **Loopy BP**: iterate messages until (hopefully) convergence; approximate | per-iteration linear in edges |
| **Treewidth-$k$ graph** | Exact via the **junction-tree** algorithm | exponential in treewidth $k$ |

The deep fact: **exact inference is exponential in the graph's treewidth**, not its size. A chain has treewidth 1 (cheap); a fully connected grid has high treewidth (hard). That's why we fall back to loopy BP, variational inference, or sampling on dense graphs.

## Loopy BP

Run the same message updates on a graph with cycles, iterating until messages stop changing. There's **no guarantee of convergence or correctness**, yet it works remarkably well in practice: it's how modern error-correcting codes are decoded and is closely connected to variational (Bethe free energy) approximations.

## What an interviewer expects you to say

1. State that BP computes **marginals by passing local messages** and exploits the **factorization** of the joint to avoid the exponential sum.
2. Distinguish **sum-product (marginals)** from **max-product (MAP / Viterbi)**.
3. Know it is **exact on trees**, and that **HMM forward-backward and CRF inference are special cases** of sum-product on a chain.
4. State the key complexity insight: exact inference is **exponential in treewidth**; use **junction tree** for exactness or **loopy BP / variational / sampling** otherwise.
5. Bonus: loopy BP has no convergence guarantee but powers LDPC/turbo decoding.

## Common confusions

- **"BP always gives the right answer."** Only on trees (and junction trees). On loopy graphs it's a heuristic approximation.
- **"Sum-product and max-product are different algorithms."** Same message structure; one sums to marginalize, the other maxes to find the mode.
- **"Inference is hard because there are many variables."** It's hard because of **treewidth / connectivity**. A million-variable chain is easy; a 50-variable dense graph can be infeasible.
- **"Forward-backward is unrelated to BP."** It *is* sum-product BP on a chain. Viterbi is max-product BP on a chain.
- **"Messages are probabilities."** They're unnormalized factors (functions of a variable); you normalize only at the end to read off a marginal.

---

*Related: [Graphical models](/concepts/graphical-models/), [Forward-backward and Viterbi](/concepts/forward-backward-and-viterbi/), [Conditional random fields](/concepts/conditional-random-fields/), [Hidden Markov models](/concepts/hidden-markov-models/), [Expectation-maximization](/concepts/expectation-maximization/).*
