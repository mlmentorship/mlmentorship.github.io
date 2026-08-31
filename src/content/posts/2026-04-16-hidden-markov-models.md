---
title: "Hidden Markov models"
description: "A latent Markov chain emits observations through a per-state distribution. Forward-backward, Viterbi, Baum-Welch. The classical sequence model toolkit."
date: "2026-04-16"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A **Hidden Markov Model** is a latent-variable sequence model with: (a) a discrete latent state $z_t$ evolving as a first-order Markov chain with transition matrix $A$, and (b) per-state emission distributions producing observations $x_t \mid z_t$.

HMMs were the dominant sequence model from the 1970s through the early 2010s for speech recognition, part-of-speech tagging, gene finding, and many time-series problems. They have largely been displaced by neural sequence models (RNNs, transformers) for tasks with abundant data, but remain useful for:

- **Small-data sequence labeling**.
- **Settings with strong domain structure** (gene finding still uses HMMs).
- **Online filtering** with limited compute.
- **As a teaching example** of latent-variable inference.

The three classical HMM problems. Likelihood, decoding, learning. And their solutions (forward, Viterbi, Baum-Welch) are core probabilistic ML.

## The model

- Discrete latent states $z_t \in \{1, \dots, K\}$.
- Initial distribution $\pi_k = p(z_1 = k)$.
- Transition matrix $A_{ij} = p(z_{t+1} = j \mid z_t = i)$.
- Emission distributions $p(x_t \mid z_t = k; \phi_k)$. Typically Gaussian or categorical.

The joint:

$$
p(x_{1:T}, z_{1:T}) = \pi_{z_1} p(x_1 \mid z_1) \prod_{t=2}^{T} A_{z_{t-1}, z_t} p(x_t \mid z_t).
$$

**Learning objective:** trace how temporal dependence travels through the hidden-state chain while each observation is generated only from its same-time state.

<!-- visual:hmm-hidden-chain-local-emissions -->
<figure class="learning-figure plot-panel" aria-labelledby="hmm-factorization-title">
	<p class="visual-kicker">Hidden dynamics, local evidence</p>
	<p class="visual-title" id="hmm-factorization-title">Only the latent chain carries memory across time.</p>
	<svg viewBox="0 0 360 360" role="img" aria-labelledby="hmm-factorization-svg-title hmm-factorization-svg-desc">
		<title id="hmm-factorization-svg-title">A four-timestep hidden Markov model factorization</title>
		<desc id="hmm-factorization-svg-desc">Four circular hidden states z one through z four form a directed horizontal chain. The initial distribution pi points to z one, and each transition arrow is labelled A. Each hidden state points down through an emission arrow labelled p of x t given z t to one rounded rectangular observation x at the same timestep. There are no arrows between observations. State z two has a double ring: conditioning on it separates the past variables z one, x one, and x two from the future variables z three, z four, x three, and x four. A factor strip spells out the initial and emission factor at time one followed by the product of one transition and one local emission at later times.</desc>
		<defs>
			<marker id="hmm-factorization-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" style="fill:var(--viz-edge)"></path></marker>
		</defs>
		<rect class="viz-plot-bg" x="6" y="6" width="348" height="344" rx="6"></rect>
		<text class="viz-axis-label" x="60" y="30" text-anchor="middle">t=1</text>
		<text class="viz-axis-label" x="140" y="30" text-anchor="middle">t=2</text>
		<text class="viz-axis-label" x="220" y="30" text-anchor="middle">t=3</text>
		<text class="viz-axis-label" x="300" y="30" text-anchor="middle">t=4</text>
		<text class="viz-axis-label" x="14" y="70">HIDDEN</text>
		<rect class="viz-node viz-node--input" x="12" y="45" width="32" height="28" rx="4"></rect>
		<text class="viz-callout" x="28" y="64" text-anchor="middle">π</text>
		<path d="M44 59H47M73 59H127M153 59H207M233 59H287" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#hmm-factorization-arrow)"></path>
		<text class="viz-label" x="100" y="52" text-anchor="middle">A</text>
		<text class="viz-label" x="180" y="52" text-anchor="middle">A</text>
		<text class="viz-label" x="260" y="52" text-anchor="middle">A</text>
		<g class="viz-node viz-node--focus">
			<circle cx="60" cy="59" r="13"></circle><circle cx="140" cy="59" r="16"></circle><circle cx="220" cy="59" r="13"></circle><circle cx="300" cy="59" r="13"></circle>
		</g>
		<circle cx="140" cy="59" r="12" style="fill:none;stroke:var(--viz-edge);stroke-width:1.2"></circle>
		<g class="viz-node-label">
			<text x="60" y="64">z₁</text><text x="140" y="64">z₂</text><text x="220" y="64">z₃</text><text x="300" y="64">z₄</text>
		</g>
		<text class="viz-axis-label" x="14" y="144">OBS.</text>
		<path d="M60 73V118M140 76V118M220 73V118M300 73V118" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#hmm-factorization-arrow)"></path>
		<g class="viz-label" text-anchor="middle">
			<text x="60" y="94">p(x₁|z₁)</text><text x="140" y="94">p(x₂|z₂)</text><text x="220" y="94">p(x₃|z₃)</text><text x="300" y="94">p(x₄|z₄)</text>
		</g>
		<g class="viz-node viz-node--input">
			<rect x="42" y="119" width="36" height="30" rx="5"></rect><rect x="122" y="119" width="36" height="30" rx="5"></rect><rect x="202" y="119" width="36" height="30" rx="5"></rect><rect x="282" y="119" width="36" height="30" rx="5"></rect>
		</g>
		<g class="viz-node-label">
			<text x="60" y="139">x₁</text><text x="140" y="139">x₂</text><text x="220" y="139">x₃</text><text x="300" y="139">x₄</text>
		</g>
		<path d="M92 168H268" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.5;stroke-dasharray:5 4"></path>
		<text class="viz-callout" x="180" y="187" text-anchor="middle">no x-to-x edges</text>
		<text class="viz-label" x="180" y="204" text-anchor="middle">observations depend across time only through z₁ → z₂ → z₃ → z₄</text>
		<rect class="viz-node" x="18" y="223" width="324" height="52" rx="5"></rect>
		<text class="viz-callout" x="180" y="243" text-anchor="middle">Condition on the double-ringed z₂</text>
		<text class="viz-label" x="180" y="262" text-anchor="middle">{z₁, x₁, x₂} ⟂ {z₃, z₄, x₃, x₄} | z₂</text>
		<text class="viz-axis-label" x="18" y="301">READ THE JOINT ONE LOCAL FACTOR AT A TIME</text>
		<text class="viz-callout" x="180" y="324" text-anchor="middle">π(z₁) p(x₁|z₁) × ∏ [p(z<tspan baseline-shift="sub" font-size="8">t</tspan>|z<tspan baseline-shift="sub" font-size="8">t−1</tspan>) p(x<tspan baseline-shift="sub" font-size="8">t</tspan>|z<tspan baseline-shift="sub" font-size="8">t</tspan>)]</text>
		<text class="viz-label" x="180" y="342" text-anchor="middle">initial state + one emission, then transition + emission per step</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> follow the top row to sample the hidden trajectory, then move down once at each timestep to emit the observation. There is no direct arrow from one <var>x</var><sub>t</sub> to the next: distant observations are related because their hidden causes are related. Condition on the double-ringed <var>z</var><sub>2</sub>, and that state blocks the path between past and future. The graph therefore reads directly as the joint's initial factor followed by one transition and one emission per timestep. Original schematic checked against <a href="https://doi.org/10.1109/5.18626">Rabiner's HMM tutorial</a> and the <a href="https://hmmlearn.readthedocs.io/en/stable/tutorial.html">hmmlearn model description</a>.</figcaption>
</figure>

## The three classical problems

### 1. Likelihood: forward algorithm

Compute $p(x_{1:T})$ by marginalizing over $z_{1:T}$. Naive sum is $O(K^T)$. The **forward algorithm** uses dynamic programming:

$$
\alpha_t(k) = p(x_{1:t}, z_t = k) = p(x_t \mid z_t = k) \sum_j A_{j, k} \alpha_{t-1}(j).
$$

Complexity: $O(T K^2)$. Final likelihood: $\sum_k \alpha_T(k)$.

### 2. Decoding: Viterbi algorithm

Find the most likely sequence $z_{1:T}^*$. Same DP structure but replace sum with max:

$$
\delta_t(k) = \max_j A_{j, k} \delta_{t-1}(j) \cdot p(x_t \mid z_t = k).
$$

Backtrack from $\arg\max_k \delta_T(k)$. Complexity: $O(T K^2)$.

### 3. Learning: Baum-Welch (EM)

Estimate $\pi, A, \phi$ from observations alone. **E-step**: compute posterior over latents using forward-backward. **M-step**: weighted MLE on transitions and emissions. This is EM applied to HMMs; converges to local optimum of the log-likelihood.

## Forward-backward

The **forward variable** $\alpha_t(k) = p(x_{1:t}, z_t = k)$ and **backward variable** $\beta_t(k) = p(x_{t+1:T} \mid z_t = k)$ together give:

- Posterior over single state: $p(z_t = k \mid x_{1:T}) = \alpha_t(k) \beta_t(k) / p(x_{1:T})$.
- Posterior over consecutive pair: needed for the EM transition update.

Forward-backward is the HMM analog of message passing on a chain. Exact in $O(T K^2)$.

## Connection to other models

| Model | Relation to HMM |
|-------|----------------|
| Mixture of Gaussians | HMM with $T = 1$ |
| Linear-Gaussian state space (Kalman filter) | Continuous-state HMM |
| CRF | Discriminative HMM (model $p(z \mid x)$ directly) |
| Linear-chain RNN | Neural generalization with continuous latents |
| Transformer | Replaces Markov assumption with attention over full sequence |

## When to use HMMs in 2026

| Setting | HMM vs. alternatives |
|---------|---------------------|
| Phoneme alignment in TTS / ASR forced alignment | HMM still standard |
| Bioinformatics (gene finding, profile HMMs) | HMMs dominant |
| Small-data sequence labeling | HMM or CRF baseline |
| Modern NLP (NER, POS) | Transformers win |
| Speech recognition (end-to-end) | RNN-T or transformer encoder + CTC |

## Common pitfalls

- **Numerical underflow.** $\alpha_t(k)$ shrinks geometrically; use log-space or scaling.
- **EM local optima.** Multiple random restarts; initialize emission means with k-means.
- **Treating HMMs as state-of-the-art for general sequence tasks.** They are not for any task with abundant data.
- **Confusing first-order Markov with the model's expressiveness.** The latent is first-order Markov; the *observations* can have arbitrary long-range structure mediated by latents (which is why HMMs work at all).

## Related

- [Markov chains](/concepts/markov-chains/). The latent dynamics.
- [Expectation-Maximization](/concepts/expectation-maximization/). Baum-Welch is EM for HMMs.
