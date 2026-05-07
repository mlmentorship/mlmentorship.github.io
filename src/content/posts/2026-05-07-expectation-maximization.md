---
title: "Expectation-Maximization (EM)"
description: "Iterate between estimating latent variables given parameters (E-step) and updating parameters given latents (M-step). The standard tool for latent-variable MLE when the latents are unobserved."
date: "2026-05-07"
draft: false
tags: ["reference"]
category: "reference"
---

## One-line definition

**Expectation-Maximization** ([Dempster, Laird & Rubin, 1977](https://www.jstor.org/stable/2984875)) is an iterative algorithm for finding MLE / MAP parameter estimates in latent-variable models. Each iteration alternates:

1. **E-step**: compute the posterior $q(z) = p(z \mid x; \theta_t)$ over latents.
2. **M-step**: update $\theta_{t+1} = \arg\max_\theta \mathbb{E}_{q(z)}[\log p(x, z; \theta)]$.

EM monotonically increases the log-likelihood until convergence to a local optimum.

## Why it matters

When you have a latent variable model and the latents are unobserved, direct MLE involves a marginalization $\log p(x; \theta) = \log \sum_z p(x, z; \theta)$ that is usually intractable. EM avoids this by alternately filling in expected values of $z$ and optimizing $\theta$ on the completed data.

EM underlies:

- Gaussian mixture model (GMM) fitting.
- Hidden Markov model (HMM) parameter estimation (Baum-Welch is EM).
- Probabilistic PCA, factor analysis, ICA.
- LDA topic models.
- Many missing-data imputation methods.

## The two steps

For a model with observed $x$, latent $z$, parameters $\theta$:

### E-step

Given current parameters $\theta_t$, compute the posterior over latents:

$$
q_t(z) = p(z \mid x; \theta_t).
$$

For models with conjugate or finite latent structure, this is closed-form (GMM: posterior responsibilities; HMM: forward-backward).

### M-step

Update $\theta$ to maximize the **expected complete-data log-likelihood**:

$$
\theta_{t+1} = \arg\max_\theta \mathbb{E}_{z \sim q_t}[\log p(x, z; \theta)].
$$

Often this expectation reduces to weighted MLE over the data with the latents replaced by their expected values.

## Why it works

EM maximizes a lower bound on $\log p(x; \theta)$:

$$
\log p(x; \theta) = \mathbb{E}_q[\log p(x, z; \theta)] - \mathbb{E}_q[\log q(z)] + \mathrm{KL}(q \| p_\theta(z \mid x)).
$$

The first two terms are the **ELBO** (evidence lower bound). The KL is non-negative. EM:

- E-step sets $q = p_\theta(z \mid x)$ → KL = 0 → ELBO = $\log p(x; \theta)$.
- M-step maximizes ELBO over $\theta$ holding $q$ fixed → never decreases $\log p(x; \theta)$.

So $\log p(x; \theta_{t+1}) \ge \log p(x; \theta_t)$. Convergence to a local optimum is guaranteed.

## Canonical example: GMM

A GMM has $K$ Gaussian components with mixture weights $\pi_k$, means $\mu_k$, covariances $\Sigma_k$. Latent $z_i \in \{1, \dots, K\}$ assigns each point to a component.

**E-step**: posterior responsibility of component $k$ for point $i$:

$$
r_{ik} = \frac{\pi_k \mathcal{N}(x_i; \mu_k, \Sigma_k)}{\sum_j \pi_j \mathcal{N}(x_i; \mu_j, \Sigma_j)}.
$$

**M-step**: weighted MLE updates:

$$
\mu_k = \frac{\sum_i r_{ik} x_i}{\sum_i r_{ik}}, \quad \pi_k = \frac{1}{n} \sum_i r_{ik}, \quad \Sigma_k = \frac{\sum_i r_{ik} (x_i - \mu_k)(x_i - \mu_k)^\top}{\sum_i r_{ik}}.
$$

Iterate until log-likelihood stabilizes.

## Variants

- **Hard-assignment EM** (k-means): replace soft responsibilities with hard 0/1 assignments. k-means is EM for a GMM with shared identity covariance and $\pi_k = 1/K$.
- **Stochastic EM**: sample $z_i$ instead of computing expected values; useful for large $z$.
- **Variational EM**: replace exact posterior with a variational approximation; modern incarnation is the VAE.
- **MAP-EM**: include a prior $p(\theta)$; M-step maximizes $\log p(\theta) + \mathbb{E}_q[\log p(x, z; \theta)]$.

## Limitations

- **Local optima**: EM converges to a local maximum; results depend on initialization. Multiple random restarts are standard.
- **Slow near optimum**: linear convergence; gets sluggish near a flat region.
- **Requires known model structure**: number of components, latent dimensions, etc.

## Common pitfalls

- **Initializing GMM means at the same point.** All components collapse to the same Gaussian; initialize by k-means++ or random data points.
- **Singular covariances.** A component centered on a single point gets $\Sigma_k \to 0$; log-likelihood diverges. Add regularization $\Sigma_k + \varepsilon I$.
- **Comparing log-likelihood across runs without checking convergence.** EM monotonically increases it within a run, but different inits give different local optima.
- **Confusing EM with k-means.** k-means is hard-assignment EM with restricted GMM; EM gives soft assignments and arbitrary covariances.

## Related

- [Gaussian mixture models](/reference/gaussian-mixture-models/). The canonical EM application.
- [Hidden Markov models](/reference/hidden-markov-models/). Sequential model trained by EM (Baum-Welch).
