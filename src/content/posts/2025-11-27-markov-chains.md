---
title: "Markov chains"
description: "Stochastic processes where the future depends only on the present, not the past. Foundation of HMMs, MCMC, and many sequence models."
date: "2025-11-27"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A **Markov chain** is a sequence of random variables $X_0, X_1, X_2, \dots$ such that

$$
p(X_{t+1} \mid X_t, X_{t-1}, \dots, X_0) = p(X_{t+1} \mid X_t).
$$

The conditional distribution of the future given the present is independent of the past. For finite state spaces, the dynamics are summarized by a **transition matrix** $P$ where $P_{ij} = p(X_{t+1} = j \mid X_t = i)$.

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

For a finite reversible chain, the asymptotic rate of convergence is governed by the second-largest eigenvalue magnitude of $P$ (equivalently, its absolute spectral gap from 1 controls mixing).

## Detailed balance and reversibility

A chain is **reversible** if there is a $\pi$ such that

$$
\pi_i P_{ij} = \pi_j P_{ji} \quad \text{for all } i, j.
$$

Detailed balance implies $\pi$ is stationary (sum both sides over $i$). Metropolis-Hastings constructs a chain satisfying detailed balance with respect to a target distribution. This is the trick that makes its long-run samples follow the target.

<!-- visual:markov-chain-stationary-flow-ledger -->
<figure class="learning-figure" aria-labelledby="markov-stationary-title" aria-describedby="markov-stationary-description"><p class="visual-kicker">Learning objective</p><p class="visual-title" id="markov-stationary-title">A Markov chain moves probability mass until one split maps back to itself.</p><p id="markov-stationary-description">Mass moves toward 2/3 in A and 1/3 in B; at stationarity the two cross-flows balance.</p><svg class="markov-stationary-visual" viewBox="0 0 900 420" role="img" aria-labelledby="markov-stationary-svg-title markov-stationary-svg-description" width="900" height="420"><title id="markov-stationary-svg-title">Two-state Markov chain probability mass converges to a stationary distribution</title><desc id="markov-stationary-svg-description">Four stacked bars show the marginal distribution moving from all A to point eight A, then point seven two A, then the stationary two-thirds A and one-third B split. A state diagram beside it shows equal detailed-balance flows of two fifteenths in both directions.</desc><defs><marker id="markov-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" fill="var(--viz-focus-stroke)" /></marker></defs><text x="40" y="42" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="20" font-weight="700">Repeated transitions move the marginal</text><text x="40" y="66" fill="var(--c-muted)" font-family="Lato, sans-serif" font-size="13">row vector update: mu(t+1) = mu(t) P</text><text x="88" y="108" fill="var(--c-muted)" font-family="Lato, sans-serif" font-size="12">time</text><text x="188" y="108" fill="var(--viz-input-stroke)" font-family="Lato, sans-serif" font-size="12" font-weight="700">state A mass</text><text x="440" y="108" fill="var(--viz-output-stroke)" font-family="Lato, sans-serif" font-size="12" font-weight="700">state B mass</text><text x="40" y="145" fill="var(--c-text-soft)" font-family="Lato, sans-serif" font-size="13">mu0</text><rect x="90" y="126" width="360" height="28" rx="4" fill="var(--viz-input-bg)" stroke="var(--viz-input-stroke)" /><text x="266" y="145" text-anchor="middle" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="13" font-weight="700">1.000 A</text><text x="40" y="195" fill="var(--c-text-soft)" font-family="Lato, sans-serif" font-size="13">mu1</text><rect x="90" y="176" width="288" height="28" rx="4" fill="var(--viz-input-bg)" stroke="var(--viz-input-stroke)" /><rect x="378" y="176" width="72" height="28" rx="4" fill="var(--viz-output-bg)" stroke="var(--viz-output-stroke)" /><text x="234" y="195" text-anchor="middle" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="13" font-weight="700">0.800</text><text x="414" y="195" text-anchor="middle" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="13" font-weight="700">0.200</text><text x="40" y="245" fill="var(--c-text-soft)" font-family="Lato, sans-serif" font-size="13">mu2</text><rect x="90" y="226" width="259" height="28" rx="4" fill="var(--viz-input-bg)" stroke="var(--viz-input-stroke)" /><rect x="349" y="226" width="101" height="28" rx="4" fill="var(--viz-output-bg)" stroke="var(--viz-output-stroke)" /><text x="219" y="245" text-anchor="middle" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="13" font-weight="700">0.720</text><text x="400" y="245" text-anchor="middle" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="13" font-weight="700">0.280</text><text x="40" y="305" fill="var(--c-text-soft)" font-family="Lato, sans-serif" font-size="13">pi</text><rect x="90" y="286" width="240" height="30" rx="4" fill="var(--viz-focus-bg)" stroke="var(--viz-focus-stroke)" stroke-width="2" /><rect x="330" y="286" width="120" height="30" rx="4" fill="var(--viz-output-bg)" stroke="var(--viz-output-stroke)" stroke-width="2" /><text x="210" y="306" text-anchor="middle" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="14" font-weight="700">2/3 A</text><text x="390" y="306" text-anchor="middle" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="14" font-weight="700">1/3 B</text><text x="90" y="346" fill="var(--viz-focus-stroke)" font-family="Lato, sans-serif" font-size="13" font-weight="700">pi P = pi: the split stops changing</text><line x1="462" y1="302" x2="528" y2="302" stroke="var(--viz-focus-stroke)" stroke-width="2.5" marker-end="url(#markov-arrow)" /><text x="570" y="42" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="20" font-weight="700">At stationarity, flows balance</text><text x="570" y="66" fill="var(--c-muted)" font-family="Lato, sans-serif" font-size="13">transition matrix P = [[0.8, 0.2], [0.4, 0.6]]</text><circle cx="640" cy="205" r="48" fill="var(--viz-input-bg)" stroke="var(--viz-input-stroke)" stroke-width="2" /><circle cx="790" cy="205" r="48" fill="var(--viz-output-bg)" stroke="var(--viz-output-stroke)" stroke-width="2" /><text x="640" y="200" text-anchor="middle" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="22" font-weight="700">A</text><text x="640" y="222" text-anchor="middle" fill="var(--c-text-soft)" font-family="Lato, sans-serif" font-size="13">piA = 2/3</text><text x="790" y="200" text-anchor="middle" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="22" font-weight="700">B</text><text x="790" y="222" text-anchor="middle" fill="var(--c-text-soft)" font-family="Lato, sans-serif" font-size="13">piB = 1/3</text><path d="M682 181 C714 146 748 146 779 181" fill="none" stroke="var(--viz-focus-stroke)" stroke-width="3" marker-end="url(#markov-arrow)" /><path d="M748 230 C718 268 682 268 651 230" fill="none" stroke="var(--viz-focus-stroke)" stroke-width="3" marker-end="url(#markov-arrow)" /><text x="715" y="139" text-anchor="middle" fill="var(--viz-focus-stroke)" font-family="Lato, sans-serif" font-size="13" font-weight="700">A to B: (2/3)(0.2) = 2/15</text><text x="715" y="290" text-anchor="middle" fill="var(--viz-focus-stroke)" font-family="Lato, sans-serif" font-size="13" font-weight="700">B to A: (1/3)(0.4) = 2/15</text><path d="M600 253 C566 318 648 364 704 320" fill="none" stroke="var(--c-rule)" stroke-width="2" stroke-dasharray="5 5" /><text x="650" y="372" text-anchor="middle" fill="var(--c-muted)" font-family="Lato, sans-serif" font-size="12">self-loops keep remaining mass in place</text></svg><figcaption><strong>Read it this way:</strong> the bars are the distribution before each transition, not rows of the transition matrix. Repeated multiplication moves mass toward the stationary split. At pi, the flow from A to B exactly equals the flow from B to A, so the next step has nothing net to move. Original schematic checked against <a href="https://pages.uoregon.edu/dlevin/MARKOV/">Levin and Peres</a> and the <a href="https://data140.org/textbook/content/chapter-11/balance-and-detailed-balance/">Berkeley Data 140 textbook</a>.</figcaption></figure>

## Common cases in ML

| Use case | What is the Markov chain |
|----------|-------------------------|
| HMM | Hidden state evolves as Markov chain |
| MCMC | Sampler defines a chain with target $\pi$ as stationary |
| PageRank | Random walk on web graph; $\pi$ = page rank vector |
| Diffusion model | Sequence of noise levels $X_0 \to X_1 \to \dots \to X_T$ (Gaussian) |
| MDP / RL | State transitions given action |
| Language model | Autoregressive generation is Markov when the state contains the full retained prefix or context window; tokens alone are not generally first-order Markov |

## Higher-order chains

A chain where $X_{t+1}$ depends on the last $k$ states ($k$-th order Markov) can be re-cast as first-order on the state space of $k$-tuples. Trigram language models are 2nd-order Markov over tokens, equivalent to first-order over bigrams.

## Common pitfalls

- **Assuming Markov when data has long-range dependence.** Often a useful approximation but check by holding out structure.
- **Non-converging MCMC.** A chain not yet at stationarity gives biased samples; use multiple chains and convergence diagnostics ($\hat R$, ESS).
- **Confusing transition matrix conventions.** Some texts use $P_{ij} = p(j \to i)$; others $p(i \to j)$. Check whether $P$ acts on rows or columns.
- **Mistaking the stationary distribution for the marginal of $X_t$.** Marginal at finite $t$ depends on initial distribution; stationary is the limit.
