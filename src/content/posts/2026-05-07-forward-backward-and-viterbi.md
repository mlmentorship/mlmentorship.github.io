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
