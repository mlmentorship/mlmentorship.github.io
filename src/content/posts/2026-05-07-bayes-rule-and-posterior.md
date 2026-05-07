---
title: "Bayes' rule and the posterior"
description: "How to update beliefs given evidence: posterior ∝ likelihood × prior. The foundation of Bayesian inference, naive Bayes, and probabilistic graphical models."
date: "2026-05-07"
draft: false
tags: ["reference"]
category: "reference"
---

## One-line definition

For random variables $\theta$ (parameters / hypothesis) and $D$ (data / evidence):

$$
p(\theta \mid D) = \frac{p(D \mid \theta)\, p(\theta)}{p(D)} \propto p(D \mid \theta)\, p(\theta).
$$

The **posterior** $p(\theta \mid D)$ is proportional to the **likelihood** $p(D \mid \theta)$ times the **prior** $p(\theta)$.

## Why it matters

Bayes' rule is the only mathematically consistent way to update probabilistic beliefs given new evidence. It underlies probabilistic ML (Gaussian processes, Bayesian deep learning), classification (naive Bayes), generative models (latent variable inference), and many engineering systems (Kalman filtering, sensor fusion).

The connection to MLE: the posterior peak (MAP estimate) collapses to MLE under a uniform prior. So MLE is a special case of Bayesian inference with no prior beliefs.

## The four pieces

| Piece | Name | What it is |
|-------|------|-----------|
| $p(\theta)$ | Prior | Beliefs about $\theta$ before seeing data |
| $p(D \mid \theta)$ | Likelihood | How probable the data is under each hypothesis |
| $p(\theta \mid D)$ | Posterior | Updated beliefs after seeing data |
| $p(D)$ | Evidence / marginal likelihood | Normalizing constant; $\int p(D \mid \theta) p(\theta)\, d\theta$ |

The evidence is often intractable (high-dimensional integral). For point estimates and many decisions you can ignore it.

## The classic example

A medical test is 99% accurate: $p(\text{pos} \mid \text{disease}) = 0.99$ and $p(\text{neg} \mid \text{healthy}) = 0.99$. The disease has prevalence $p(\text{disease}) = 0.001$. A random person tests positive. What is $p(\text{disease} \mid \text{pos})$?

$$
p(\text{disease} \mid \text{pos}) = \frac{0.99 \times 0.001}{0.99 \times 0.001 + 0.01 \times 0.999} \approx 0.09.
$$

Despite the 99% test accuracy, ~91% of positive results are false. The posterior depends crucially on the prior.

## Conjugate priors

A prior is **conjugate** to a likelihood if the resulting posterior is in the same family (so updating stays in closed form).

| Likelihood | Conjugate prior | Posterior family |
|-----------|-----------------|------------------|
| Bernoulli/binomial | Beta | Beta |
| Categorical/multinomial | Dirichlet | Dirichlet |
| Gaussian (mean, known $\sigma^2$) | Gaussian | Gaussian |
| Gaussian (precision) | Gamma | Gamma |
| Poisson | Gamma | Gamma |

Used in: Thompson sampling for bandits (Beta-Bernoulli), online recsys updates, conjugate Gibbs samplers.

## Approximate inference (when conjugacy fails)

Modern Bayesian deep learning rarely has closed-form posteriors. Standard approximations:

- **Laplace approximation**: Gaussian centered at MAP with covariance from the Hessian.
- **Variational inference**: optimize a parametric family $q_\phi(\theta)$ to minimize $\mathrm{KL}(q_\phi \| p(\theta \mid D))$.
- **MCMC** (Metropolis-Hastings, HMC, NUTS): draw samples from the posterior asymptotically.
- **Stochastic-gradient Langevin / SGHMC**: scale to large data via mini-batches.

## Common pitfalls

- **Confusing likelihood with posterior.** $p(D \mid \theta)$ is not a probability distribution over $\theta$; it does not integrate to 1 over $\theta$.
- **Ignoring the prior in low-data regimes.** With small $n$, the posterior is dominated by the prior.
- **Reporting MAP without uncertainty.** A posterior contains more than its mode; the spread is often the more useful information.
- **Improper priors.** Some "uniform" priors over unbounded parameter spaces don't integrate; the posterior may still be proper (or may not be).
