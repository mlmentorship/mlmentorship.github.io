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
<figure class="learning-figure" aria-labelledby="markov-stationary-title" aria-describedby="markov-stationary-description">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="markov-stationary-title">When does a changing marginal become stationary?</p>
	<p id="markov-stationary-description">For a two-state chain, each row of P gives the next-state probabilities from the current state. Starting entirely in A, repeated row-vector multiplication approaches two-thirds in A and one-third in B. At that distribution, another multiplication makes no change, and probability flow from A to B equals flow from B to A.</p>
	<p class="cm-equation">rows = current state · columns = next state · P = [A: 0.8, 0.2; B: 0.4, 0.6]</p>
	<table class="cm-grid" aria-label="Successive marginals for a two-state Markov chain approaching its stationary distribution">
		<thead>
			<tr><th scope="col">Distribution</th><th scope="col">Pr(A)</th><th scope="col">Pr(B)</th></tr>
		</thead>
		<tbody>
			<tr><th scope="row">μ<sub>0</sub> (start)</th><td><strong>1.000</strong></td><td><strong>0.000</strong></td></tr>
			<tr><th scope="row">μ<sub>1</sub> = μ<sub>0</sub>P</th><td>0.800</td><td>0.200</td></tr>
			<tr><th scope="row">μ<sub>2</sub> = μ<sub>1</sub>P</th><td>0.720</td><td>0.280</td></tr>
			<tr><th scope="row">μ<sub>3</sub> = μ<sub>2</sub>P</th><td>0.688</td><td>0.312</td></tr>
			<tr><th scope="row">π and πP</th><td class="cm-selected"><strong>2/3</strong>unchanged</td><td class="cm-selected"><strong>1/3</strong>unchanged</td></tr>
		</tbody>
	</table>
	<p class="cm-equation">detailed balance: A→B = (2/3)(0.2) = 2/15 · B→A = (1/3)(0.4) = 2/15</p>
	<figcaption><strong>Read it this way:</strong> scan the rows downward: μ<sub>t</sub> is the finite-time marginal and still remembers the all-A start, while π is the fixed point that another transition leaves unchanged. Then read the final line: equal pairwise flow proves detailed balance for this example, which is stronger than stationarity. The numeric labels, row order, and equal-flow equation carry the lesson without color. Original ledger checked against <a href="https://pages.uoregon.edu/dlevin/MARKOV/">Levin and Peres</a> and the <a href="https://data140.org/textbook/content/chapter-11/balance-and-detailed-balance/">Berkeley Data 140 textbook</a>.</figcaption>
</figure>

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
