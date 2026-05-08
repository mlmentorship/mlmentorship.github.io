---
title: "Hidden Markov models"
description: "A latent Markov chain emits observations through a per-state distribution. Forward-backward, Viterbi, Baum-Welch. The classical sequence model toolkit."
date: "2026-04-16"
draft: false
tags: ["concepts"]
category: "concepts"
---

## One-line definition

A **Hidden Markov Model** is a latent-variable sequence model with: (a) a discrete latent state $z_t$ evolving as a first-order Markov chain with transition matrix $A$, and (b) per-state emission distributions producing observations $x_t \mid z_t$.

## Why it matters

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
