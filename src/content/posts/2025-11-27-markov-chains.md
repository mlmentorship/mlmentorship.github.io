---
title: "Markov chains"
description: "Stochastic processes where the future depends only on the present, not the past. Foundation of HMMs, MCMC, and many sequence models."
date: "2025-11-27"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A Markov state retains the information needed to specify the next-state law without consulting older history.
For finite states with row-stochastic transition matrix $P$, row-vector marginals evolve as $\mu_{t+1}=\mu_tP$.

$$
p(X_{t+1} \mid X_t, X_{t-1}, \dots, X_0) = p(X_{t+1} \mid X_t).
$$

Use $P_{ij}=P(X_{t+1}=j\mid X_t=i)$: each row is nonnegative and sums to one.
For the illustrated teaching chain,

$$
P=\begin{bmatrix}0.8&0.2\\0.4&0.6\end{bmatrix},\qquad\mu_0=(1,0).
$$

Right multiplication gives $\mu_1=(0.8,0.2)$, $\mu_2=(0.72,0.28)$, and $\mu_3=(0.688,0.312)$.
Every update preserves total mass, but these finite-time distributions still depend on the initial state.

## Stationary distribution

A stationary distribution satisfies $\pi P=\pi$, a fixed-point condition rather than a promise of convergence.
Every finite row-stochastic matrix has at least one stationary distribution.
Finite irreducibility makes it unique; adding aperiodicity ensures convergence from every initial distribution:

$$
p(X_t \mid X_0) \to \pi \quad \text{as } t \to \infty.
$$

For the fixture, $0.2\pi_A=0.4\pi_B$ and $\pi_A+\pi_B=1$ give $\pi=(2/3,1/3)$.
Writing $a_t=\mu_t(A)$ yields $a_{t+1}=0.4+0.4a_t$.
Thus $a_t-2/3=(1/3)(0.4)^t$ for this start, explicitly showing contraction.
Nontrivial eigenvalue magnitudes govern asymptotic contraction for finite irreducible reversible chains.
Bounds also depend on the starting distribution; a positive absolute spectral gap excludes periodic oscillation.

## Detailed balance and reversibility

A chain is **reversible** if there is a $\pi$ such that

$$
\pi_i P_{ij} = \pi_j P_{ji} \quad \text{for all } i, j.
$$

Summing the balance equality over $i$ gives $\sum_i\pi_iP_{ij}=\pi_j\sum_iP_{ji}=\pi_j$.
Detailed balance therefore implies stationarity, but stationarity does not imply detailed balance.
The fixture's two cross-flows both equal $(2/3)(1/5)=2/15$.
Metropolis-Hastings can construct a target-invariant reversible chain; valid convergence still requires additional conditions.

<!-- visual:markov-chain-stationary-flow-ledger -->
<figure class="learning-figure" aria-labelledby="markov-stationary-title" aria-describedby="markov-stationary-description"><p class="visual-kicker">Learning objective</p><p class="visual-title" id="markov-stationary-title">A Markov chain moves probability mass until one split maps back to itself.</p><p id="markov-stationary-description">Mass moves toward 2/3 in A and 1/3 in B; at stationarity the two cross-flows balance.</p><svg class="markov-stationary-visual" viewBox="0 0 900 420" role="img" aria-labelledby="markov-stationary-svg-title markov-stationary-svg-description" width="900" height="420"><title id="markov-stationary-svg-title">Two-state Markov chain probability mass converges to a stationary distribution</title><desc id="markov-stationary-svg-description">Four stacked bars show the marginal distribution moving from all A to point eight A, then point seven two A, then the stationary two-thirds A and one-third B split. A state diagram beside it shows equal detailed-balance flows of two fifteenths in both directions.</desc><defs><marker id="markov-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" fill="var(--viz-focus-stroke)" /></marker></defs><text x="40" y="42" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="20" font-weight="700">Repeated transitions move the marginal</text><text x="40" y="66" fill="var(--c-muted)" font-family="Lato, sans-serif" font-size="13">row vector update: mu(t+1) = mu(t) P</text><text x="88" y="108" fill="var(--c-muted)" font-family="Lato, sans-serif" font-size="12">time</text><text x="188" y="108" fill="var(--viz-input-stroke)" font-family="Lato, sans-serif" font-size="12" font-weight="700">state A mass</text><text x="440" y="108" fill="var(--viz-output-stroke)" font-family="Lato, sans-serif" font-size="12" font-weight="700">state B mass</text><text x="40" y="145" fill="var(--c-text-soft)" font-family="Lato, sans-serif" font-size="13">mu0</text><rect x="90" y="126" width="360" height="28" rx="4" fill="var(--viz-input-bg)" stroke="var(--viz-input-stroke)" /><text x="266" y="145" text-anchor="middle" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="13" font-weight="700">1.000 A</text><text x="40" y="195" fill="var(--c-text-soft)" font-family="Lato, sans-serif" font-size="13">mu1</text><rect x="90" y="176" width="288" height="28" rx="4" fill="var(--viz-input-bg)" stroke="var(--viz-input-stroke)" /><rect x="378" y="176" width="72" height="28" rx="4" fill="var(--viz-output-bg)" stroke="var(--viz-output-stroke)" /><text x="234" y="195" text-anchor="middle" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="13" font-weight="700">0.800</text><text x="414" y="195" text-anchor="middle" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="13" font-weight="700">0.200</text><text x="40" y="245" fill="var(--c-text-soft)" font-family="Lato, sans-serif" font-size="13">mu2</text><rect x="90" y="226" width="259" height="28" rx="4" fill="var(--viz-input-bg)" stroke="var(--viz-input-stroke)" /><rect x="349" y="226" width="101" height="28" rx="4" fill="var(--viz-output-bg)" stroke="var(--viz-output-stroke)" /><text x="219" y="245" text-anchor="middle" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="13" font-weight="700">0.720</text><text x="400" y="245" text-anchor="middle" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="13" font-weight="700">0.280</text><text x="40" y="305" fill="var(--c-text-soft)" font-family="Lato, sans-serif" font-size="13">pi</text><rect x="90" y="286" width="240" height="30" rx="4" fill="var(--viz-focus-bg)" stroke="var(--viz-focus-stroke)" stroke-width="2" /><rect x="330" y="286" width="120" height="30" rx="4" fill="var(--viz-output-bg)" stroke="var(--viz-output-stroke)" stroke-width="2" /><text x="210" y="306" text-anchor="middle" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="14" font-weight="700">2/3 A</text><text x="390" y="306" text-anchor="middle" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="14" font-weight="700">1/3 B</text><text x="90" y="346" fill="var(--viz-focus-stroke)" font-family="Lato, sans-serif" font-size="13" font-weight="700">pi P = pi: the split stops changing</text><line x1="462" y1="302" x2="528" y2="302" stroke="var(--viz-focus-stroke)" stroke-width="2.5" marker-end="url(#markov-arrow)" /><text x="570" y="42" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="20" font-weight="700">At stationarity, flows balance</text><text x="570" y="66" fill="var(--c-muted)" font-family="Lato, sans-serif" font-size="13">transition matrix P = [[0.8, 0.2], [0.4, 0.6]]</text><circle cx="640" cy="205" r="48" fill="var(--viz-input-bg)" stroke="var(--viz-input-stroke)" stroke-width="2" /><circle cx="790" cy="205" r="48" fill="var(--viz-output-bg)" stroke="var(--viz-output-stroke)" stroke-width="2" /><text x="640" y="200" text-anchor="middle" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="22" font-weight="700">A</text><text x="640" y="222" text-anchor="middle" fill="var(--c-text-soft)" font-family="Lato, sans-serif" font-size="13">piA = 2/3</text><text x="790" y="200" text-anchor="middle" fill="var(--c-text)" font-family="Lato, sans-serif" font-size="22" font-weight="700">B</text><text x="790" y="222" text-anchor="middle" fill="var(--c-text-soft)" font-family="Lato, sans-serif" font-size="13">piB = 1/3</text><path d="M682 181 C714 146 748 146 779 181" fill="none" stroke="var(--viz-focus-stroke)" stroke-width="3" marker-end="url(#markov-arrow)" /><path d="M748 230 C718 268 682 268 651 230" fill="none" stroke="var(--viz-focus-stroke)" stroke-width="3" marker-end="url(#markov-arrow)" /><text x="715" y="139" text-anchor="middle" fill="var(--viz-focus-stroke)" font-family="Lato, sans-serif" font-size="13" font-weight="700">A to B: (2/3)(0.2) = 2/15</text><text x="715" y="290" text-anchor="middle" fill="var(--viz-focus-stroke)" font-family="Lato, sans-serif" font-size="13" font-weight="700">B to A: (1/3)(0.4) = 2/15</text><path d="M600 253 C566 318 648 364 704 320" fill="none" stroke="var(--c-rule)" stroke-width="2" stroke-dasharray="5 5" /><text x="650" y="372" text-anchor="middle" fill="var(--c-muted)" font-family="Lato, sans-serif" font-size="12">self-loops keep remaining mass in place</text></svg>
<figcaption><strong>Read it this way:</strong> the bars are the distribution before each transition, not rows of the transition matrix. Repeated multiplication moves mass toward the stationary split. At pi, equal cross-flows leave both state masses unchanged. Original schematic checked against <a href="https://pages.uoregon.edu/dlevin/MARKOV/">Levin and Peres</a> and the <a href="https://data140.org/textbook/content/chapter-11/balance-and-detailed-balance/">Berkeley Data 140 textbook</a>.</figcaption>
</figure>

## Change the Requirement

### Remove Holding Probabilities

**New task:** Replace the fixture matrix with $\left[\begin{smallmatrix}0&1\\1&0\end{smallmatrix}\right]$.

**Picture and change:** Mass alternates between states instead of contracting.

**Check:** $(1/2,1/2)$ is the unique stationary distribution, but starting at A gives alternating point masses forever.

**Use when:** Transition structure can create periodicity; uniqueness alone is insufficient for marginal convergence.

## Higher-order chains

**New task:** Let the next-state probability depend on the previous state as well as the current state.

**Picture and change:** Use ordered pairs as states, preserving the history that changes transition probabilities.

**Check:** From `(A,B)`, move to `(B,A)` with probability $0.9$ and `(B,B)` with $0.1$.
From `(B,B)`, let those probabilities be $0.1$ and $0.9$ instead; a current-token-only B cannot represent both rows.

**Use when:** A specified older observation still changes the next-state law.
A $k$-order token chain becomes first-order over $k$-tuples, potentially increasing the state space substantially.

<!-- learning:core-end -->
<div data-learning-extension>

## Common cases in ML

| Use case | Chosen state or transition |
|----------|----------------------------|
| HMM | Hidden state, not generally the observed symbol alone |
| MCMC | Sampler state with a declared invariant target |
| PageRank | Random walk with specified graph and teleportation conventions |
| Diffusion | Noise-level-dependent transitions; include time when needed |
| MDP/RL | Environment state and action-conditioned transition |
| Language model | Full retained prefix/context, not generally one token |

## Common pitfalls

- **Insufficient state.** Test whether older observed information still changes the specified conditional law.
- **Premature sampler claims.** Multiple chains, effective sample size, and convergence diagnostics identify concerns, not universal proof of convergence.
- **Mixed conventions.** Row-stochastic and column-stochastic matrices act on different sides of their marginal vectors.
- **Stationarity versus limits.** A fixed point need not attract every start; infinite-state chains need additional recurrence conditions.

## Reconstruct

<details>
<summary>Must stationary flow balance separately on every edge?</summary>

No. Consider three states, each staying with probability one-half or moving clockwise with probability one-half.
Uniform mass is stationary, yet clockwise pair flow is $1/6$ and reverse flow is zero.
Net flow into each state balances without reversibility.

</details>

</div>
