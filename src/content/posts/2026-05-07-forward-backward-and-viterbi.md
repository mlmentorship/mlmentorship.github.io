---
title: "Forward-backward and Viterbi: dynamic programming on chains"
description: "Sum and max over exponentially many paths in linear time. Forward-backward computes posteriors over hidden states; Viterbi finds the most likely state sequence. The same idea, two semirings."
date: "2026-05-07"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

For a hidden Markov model with $T$ timesteps and $K$ states, **forward-backward** computes $P(z_t \mid x_{1:T})$ for every $t$ in $O(T K^2)$ time. **Viterbi** computes $\arg\max_{z_{1:T}} P(z_{1:T} \mid x_{1:T})$ in the same time. Brute force would be $O(K^T)$.

These are the canonical examples of dynamic programming on sequences. Speech recognition (HMM-based decoding), part-of-speech tagging, gene-finding, conditional random fields for sequence labeling, and any structured-prediction task with a chain factor graph relies on one or both. The duality between sum (forward-backward) and max (Viterbi) is the textbook example of how the same DP works in two different semirings.

Even in modern deep-learning sequence models, Viterbi shows up: CTC decoding for speech, beam search as an approximation when the state space is too large, CRF layers on top of BERT for NER.

## The setup

A hidden Markov model has:

- States $z_t \in \{1, \dots, K\}$.
- Observations $x_t$.
- Initial $\pi_k = P(z_1 = k)$.
- Transitions $A_{ij} = P(z_{t+1} = j \mid z_t = i)$.
- Emissions $B_k(x) = P(x \mid z_t = k)$.

Joint probability of a hidden path and observation sequence:

$$
P(z_{1:T}, x_{1:T}) = \pi_{z_1} B_{z_1}(x_1) \prod_{t=2}^{T} A_{z_{t-1}, z_t} B_{z_t}(x_t).
$$

There are $K^T$ possible paths. Both algorithms factor through a $T \times K$ DP table.

<!-- visual:forward-backward-viterbi-chain-objectives -->
<figure class="visual-container" aria-label="Forward-backward and Viterbi objectives compared on the same state chain">
	<div class="visual-grid--two" role="group" aria-label="The same four-timestep, three-state chain used for forward-backward state marginals and Viterbi path decoding">
		<section class="visual-panel plot-panel">
			<svg viewBox="0 0 300 230" role="img" aria-labelledby="fb-chain-title fb-chain-desc">
				<title id="fb-chain-title">Forward-backward combines summaries from both sides at one state</title>
				<desc id="fb-chain-desc">A three-state chain runs across four timesteps. At state B in timestep three, a right-pointing alpha arrow summarizes every prefix ending at B and a left-pointing beta arrow summarizes every suffix beginning at B. Their product, normalized across states at timestep three, gives the posterior marginal for state B at that timestep. It does not select a complete path.</desc>
				<rect class="viz-plot-bg" x="8" y="25" width="284" height="195" rx="5"></rect>
				<text class="viz-axis-label" x="12" y="16">FORWARD-BACKWARD · ONE TIME MARGINAL</text>
				<text class="viz-axis-label" x="38" y="43" text-anchor="middle">t=1</text>
				<text class="viz-axis-label" x="113" y="43" text-anchor="middle">t=2</text>
				<text class="viz-axis-label" x="188" y="43" text-anchor="middle">t=3</text>
				<text class="viz-axis-label" x="263" y="43" text-anchor="middle">t=4</text>
				<path d="M50 68H101M125 68H176M200 68H251M50 108H101M125 108H176M200 108H251M50 148H101M125 148H176M200 148H251" style="fill:none;stroke:var(--viz-edge);stroke-width:1.2"></path>
				<path d="M50 68L101 108M50 108L101 68M50 108L101 148M50 148L101 108M125 68L176 108M125 108L176 68M125 108L176 148M125 148L176 108M200 68L251 108M200 108L251 68M200 108L251 148M200 148L251 108" style="fill:none;stroke:var(--viz-edge);stroke-width:0.8;stroke-dasharray:3 3"></path>
				<g class="viz-node"><circle cx="38" cy="68" r="12"></circle><circle cx="38" cy="108" r="12"></circle><circle cx="38" cy="148" r="12"></circle><circle cx="113" cy="68" r="12"></circle><circle cx="113" cy="108" r="12"></circle><circle cx="113" cy="148" r="12"></circle><circle cx="188" cy="68" r="12"></circle><circle cx="188" cy="148" r="12"></circle><circle cx="263" cy="68" r="12"></circle><circle cx="263" cy="108" r="12"></circle><circle cx="263" cy="148" r="12"></circle></g>
				<circle class="viz-node viz-node--focus" cx="188" cy="108" r="15"></circle>
				<circle cx="188" cy="108" r="11" style="fill:none;stroke:var(--viz-edge);stroke-width:1"></circle>
				<g class="viz-node-label" text-anchor="middle"><text x="38" y="72">A</text><text x="38" y="112">B</text><text x="38" y="152">C</text><text x="113" y="72">A</text><text x="113" y="112">B</text><text x="113" y="152">C</text><text x="188" y="72">A</text><text x="188" y="112">B</text><text x="188" y="152">C</text><text x="263" y="72">A</text><text x="263" y="112">B</text><text x="263" y="152">C</text></g>
				<path d="M68 181H174M232 181H202" style="fill:none;stroke:var(--viz-edge);stroke-width:2"></path>
				<path d="M167 177L174 181L167 185M209 177L202 181L209 185" style="fill:none;stroke:var(--viz-edge);stroke-width:2"></path>
				<text class="viz-callout" x="121" y="174" text-anchor="middle">alpha: all prefixes</text>
				<text class="viz-callout" x="238" y="174" text-anchor="middle">beta: all suffixes</text>
				<text class="viz-label" x="150" y="207" text-anchor="middle">normalize alpha_3(B) x beta_3(B) across A, B, C</text>
			</svg>
		</section>
		<section class="visual-panel plot-panel">
			<svg viewBox="0 0 300 230" role="img" aria-labelledby="viterbi-chain-title viterbi-chain-desc">
				<title id="viterbi-chain-title">Viterbi preserves one best predecessor per state and backtracks a complete path</title>
				<desc id="viterbi-chain-desc">The same three-state chain runs across four timesteps. Thin solid and dashed lines show candidate transitions. Thick arrowed segments connect A at timestep one, B at timestep two, B at timestep three, and C at timestep four. Backtracking the stored predecessor pointers from the best final state recovers this one globally consistent path.</desc>
				<rect class="viz-plot-bg" x="8" y="25" width="284" height="195" rx="5"></rect>
				<text class="viz-axis-label" x="12" y="16">VITERBI · ONE COMPLETE PATH</text>
				<text class="viz-axis-label" x="38" y="43" text-anchor="middle">t=1</text>
				<text class="viz-axis-label" x="113" y="43" text-anchor="middle">t=2</text>
				<text class="viz-axis-label" x="188" y="43" text-anchor="middle">t=3</text>
				<text class="viz-axis-label" x="263" y="43" text-anchor="middle">t=4</text>
				<path d="M50 68H101M125 68H176M200 68H251M50 108H101M125 108H176M200 108H251M50 148H101M125 148H176M200 148H251" style="fill:none;stroke:var(--viz-edge);stroke-width:0.8"></path>
				<path d="M50 68L101 108M50 108L101 68M50 108L101 148M50 148L101 108M125 68L176 108M125 108L176 68M125 108L176 148M125 148L176 108M200 68L251 108M200 108L251 68M200 108L251 148M200 148L251 108" style="fill:none;stroke:var(--viz-edge);stroke-width:0.8;stroke-dasharray:3 3"></path>
				<path d="M50 68L101 108H176L251 148" style="fill:none;stroke:var(--viz-focus);stroke-width:4"></path>
				<path d="M94 102L101 108L92 110M169 103L176 108L169 113M242 141L251 148L240 149" style="fill:none;stroke:var(--viz-focus);stroke-width:2.5"></path>
				<g class="viz-node"><circle cx="38" cy="108" r="12"></circle><circle cx="38" cy="148" r="12"></circle><circle cx="113" cy="68" r="12"></circle><circle cx="113" cy="148" r="12"></circle><circle cx="188" cy="68" r="12"></circle><circle cx="188" cy="148" r="12"></circle><circle cx="263" cy="68" r="12"></circle><circle cx="263" cy="108" r="12"></circle></g>
				<g class="viz-node viz-node--focus"><circle cx="38" cy="68" r="14"></circle><circle cx="113" cy="108" r="14"></circle><circle cx="188" cy="108" r="14"></circle><circle cx="263" cy="148" r="14"></circle></g>
				<g class="viz-node-label" text-anchor="middle"><text x="38" y="72">A</text><text x="38" y="112">B</text><text x="38" y="152">C</text><text x="113" y="72">A</text><text x="113" y="112">B</text><text x="113" y="152">C</text><text x="188" y="72">A</text><text x="188" y="112">B</text><text x="188" y="152">C</text><text x="263" y="72">A</text><text x="263" y="112">B</text><text x="263" y="152">C</text></g>
				<path d="M263 174H49" style="fill:none;stroke:var(--viz-edge);stroke-width:2"></path>
				<path d="M56 170L49 174L56 178" style="fill:none;stroke:var(--viz-edge);stroke-width:2"></path>
				<text class="viz-callout" x="156" y="190" text-anchor="middle">backtrack stored psi pointers</text>
				<text class="viz-label" x="150" y="207" text-anchor="middle">decoded sequence: A -> B -> B -> C</text>
			</svg>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> use the same state lattice twice. Forward-backward sums every prefix arriving at a chosen state and every suffix leaving it, then combines those two summaries to answer “which state at this time?” Viterbi replaces each sum with a max, stores the winning predecessor, and backtracks to answer “which single path?” The thick path and arrowheads carry the distinction without relying on color. Original schematic checked against <a href="https://doi.org/10.1109/5.18626">Rabiner (1989)</a> and <a href="https://doi.org/10.1109/TIT.1967.1054010">Viterbi (1967)</a>.</figcaption>
</figure>

## Forward algorithm

Define the **forward variable**

$$
\alpha_t(k) = P(x_{1:t}, z_t = k).
$$

Recurrence:

$$
\alpha_1(k) = \pi_k B_k(x_1), \qquad \alpha_t(k) = B_k(x_t) \sum_{i} \alpha_{t-1}(i) A_{i k}.
$$

The total likelihood is $P(x_{1:T}) = \sum_k \alpha_T(k)$. Computing the full table is $O(T K^2)$.

## Backward algorithm

The mirror image:

$$
\beta_t(k) = P(x_{t+1:T} \mid z_t = k),
$$

with $\beta_T(k) = 1$ and

$$
\beta_t(k) = \sum_i A_{k i} B_i(x_{t+1}) \beta_{t+1}(i).
$$

## Posterior over states (forward-backward)

$$
P(z_t = k \mid x_{1:T}) = \frac{\alpha_t(k) \beta_t(k)}{\sum_i \alpha_t(i) \beta_t(i)}.
$$

This is the per-timestep posterior used in EM training of HMMs and in any system that needs marginal beliefs over hidden states.

## Viterbi: the max version

Replace the sum in the forward recurrence with a max:

$$
\delta_t(k) = \max_i \delta_{t-1}(i) A_{i k} \cdot B_k(x_t),
$$

with $\delta_1(k) = \pi_k B_k(x_1)$. Track the argmax to reconstruct the path:

$$
\psi_t(k) = \arg\max_i \delta_{t-1}(i) A_{i k}.
$$

After the forward pass, the most likely path is recovered by backtracking from $z_T^* = \arg\max_k \delta_T(k)$ through $\psi$. Same $O(T K^2)$ cost.

## Sum vs max: the semiring view

Both algorithms have the same shape; only the operations differ:

| Algorithm | "Add" | "Multiply" |
|---|---|---|
| Forward | $+$ | $\times$ |
| Viterbi | $\max$ | $\times$ (or $+$ in log space) |

Both work because the operations form a semiring (associativity, distributivity). The same DP framework computes max-marginals (Viterbi), sum-marginals (forward-backward), counts (probability of inputs), expectations (segment-level expected counts), and gradients of any of the above.

## In log space (always)

Multiplying many small probabilities underflows in float32. Always work in log space:

$$
\log \alpha_t(k) = \log B_k(x_t) + \mathrm{logsumexp}_i \big(\log \alpha_{t-1}(i) + \log A_{i k}\big).
$$

The `logsumexp` trick (subtract the max before exponentiating) keeps everything stable.

## Modern uses

- **CRF decoding for NER**: BERT produces per-token logits; a CRF layer with a learned transition matrix runs Viterbi at inference and forward-backward at training.
- **CTC decoding for speech**: a sum-product algorithm over alignments. Different state structure but the same DP machinery.
- **Beam search as approximate Viterbi**: when $K$ is too large for full DP (e.g. autoregressive language models with vocab 100k), beam search keeps only the top-$k$ partial paths at each step.

## Common pitfalls

- **Working in probability space instead of log space.** Numerical underflow guaranteed beyond $T \approx 50$.
- **Forgetting to renormalize when doing forward-backward in float32.** Some implementations renormalize $\alpha_t$ at each step and accumulate the log-normalizer separately.
- **Confusing the per-timestep argmax of forward-backward with Viterbi.** They are different: Viterbi gives the most likely full sequence; per-timestep argmax gives the sequence of most likely states, which can be infeasible (i.e., have zero probability under the model).

## Related

- [Hidden Markov models](/concepts/hidden-markov-models/).
- [Expectation-Maximization](/concepts/expectation-maximization/).
- [Decoding strategies](/concepts/decoding-strategies/).
