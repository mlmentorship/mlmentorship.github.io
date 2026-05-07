---
title: "Markov chains"
description: "Stochastic processes where the future depends only on the present, not the past. Foundation of HMMs, MCMC, and many sequence models."
date: "2025-11-27"
draft: false
tags: ["reference"]
category: "reference"
---

## One-line definition

A **Markov chain** is a sequence of random variables $X_0, X_1, X_2, \dots$ such that

$$
p(X_{t+1} \mid X_t, X_{t-1}, \dots, X_0) = p(X_{t+1} \mid X_t).
$$

The conditional distribution of the future given the present is independent of the past. For finite state spaces, the dynamics are summarized by a **transition matrix** $P$ where $P_{ij} = p(X_{t+1} = j \mid X_t = i)$.

## Why it matters

Markov chains underlie:

- Hidden Markov models (HMMs) for speech, biology, finance.
- Markov chain Monte Carlo (MCMC) for Bayesian inference (Metropolis-Hastings, Gibbs).
- PageRank (random walk on the web graph).
- N-gram language models.
- Reinforcement learning (Markov decision processes).
- Diffusion models (forward and reverse Markov processes over noise levels).

The Markov property is the cleanest assumption that makes long-range stochastic systems tractable.

## Stationary distribution

A distribution $\pi$ is **stationary** for $P$ if $\pi P = \pi$ (treating $\pi$ as a row vector). It's a left eigenvector of $P$ with eigenvalue 1.

For an **irreducible** (any state reachable from any other) and **aperiodic** chain, the stationary distribution exists, is unique, and the chain converges to it from any starting state:

$$
p(X_t \mid X_0) \to \pi \quad \text{as } t \to \infty.
$$

The rate of convergence is governed by the second-largest eigenvalue of $P$ (mixing rate).

## Detailed balance and reversibility

A chain is **reversible** if there is a $\pi$ such that

$$
\pi_i P_{ij} = \pi_j P_{ji} \quad \text{for all } i, j.
$$

Detailed balance implies $\pi$ is stationary (sum both sides over $i$). MCMC algorithms (Metropolis-Hastings) construct chains satisfying detailed balance with respect to a target distribution. This is the trick that makes them sample from posteriors.

## Common cases in ML

| Use case | What is the Markov chain |
|----------|-------------------------|
| HMM | Hidden state evolves as Markov chain |
| MCMC | Sampler defines a chain with target $\pi$ as stationary |
| PageRank | Random walk on web graph; $\pi$ = page rank vector |
| Diffusion model | Sequence of noise levels $X_0 \to X_1 \to \dots \to X_T$ (Gaussian) |
| MDP / RL | State transitions given action |
| Language model | Each token depends on previous (long-context Markov chain) |

## Higher-order chains

A chain where $X_{t+1}$ depends on the last $k$ states ($k$-th order Markov) can be re-cast as first-order on the state space of $k$-tuples. Trigram language models are 2nd-order Markov over tokens, equivalent to first-order over bigrams.

## Common pitfalls

- **Assuming Markov when data has long-range dependence.** Often a useful approximation but check by holding out structure.
- **Non-converging MCMC.** A chain not yet at stationarity gives biased samples; use multiple chains and convergence diagnostics ($\hat R$, ESS).
- **Confusing transition matrix conventions.** Some texts use $P_{ij} = p(j \to i)$; others $p(i \to j)$. Check whether $P$ acts on rows or columns.
- **Mistaking the stationary distribution for the marginal of $X_t$.** Marginal at finite $t$ depends on initial distribution; stationary is the limit.
