---
title: "Discrete gradient estimators"
description: "How to get gradients through a sampling step over discrete variables, where the reparameterization trick doesn't apply. Covers the score-function (REINFORCE) estimator, the straight-through estimator, and Gumbel-Softmax."
date: "2026-05-31"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Discrete gradient estimators approximate $\nabla_\theta \mathbb{E}_{z \sim p_\theta}[f(z)]$ when $z$ is **discrete**, the case where you cannot reparameterize the sample as a smooth function of $\theta$ and noise. The three you must know: **REINFORCE** (score function), **Gumbel-Softmax** (continuous relaxation), and the **straight-through estimator**.

The [reparameterization trick](/questions/reparameterization-trick/) handles continuous latents (Gaussian VAEs). But many models sample **discrete** objects: categorical latents, hard attention, tokens, architecture choices, RL actions. You can't push a gradient through `argmax` or a categorical sample, so you need an estimator. This is the deep-DL follow-up to "explain the reparameterization trick," and it underpins RLHF (which uses the score-function estimator) and discrete latent-variable models.

## The core problem

We want $\nabla_\theta \mathbb{E}_{z \sim p_\theta(z)}[f(z)]$. The expectation is a sum over discrete $z$; the sampling operation is non-differentiable. The two families of solutions trade **bias** for **variance**.

## 1. Score-function estimator (REINFORCE / likelihood ratio)

Use the log-derivative identity $\nabla_\theta p_\theta(z) = p_\theta(z)\, \nabla_\theta \log p_\theta(z)$:

$$
\nabla_\theta \mathbb{E}_{z}[f(z)] = \mathbb{E}_{z \sim p_\theta}\big[\, f(z)\, \nabla_\theta \log p_\theta(z)\,\big].
$$

**Unbiased**, requires only that you can sample $z$ and evaluate $\log p_\theta(z)$; $f$ can be a black box (non-differentiable, even an environment reward).

The catch is **high variance**. Mitigations:

- **Baselines / control variates**: subtract a baseline $b$ that doesn't depend on $z$: $(f(z) - b)\nabla_\theta \log p_\theta(z)$. Still unbiased (since $\mathbb{E}[\nabla_\theta \log p_\theta] = 0$), lower variance. The value-function baseline in actor-critic is exactly this.
- More samples, advantage normalization, etc.

This estimator **is** policy-gradient RL. REINFORCE, A2C, and PPO are all score-function estimators with progressively better variance control.

## 2. Gumbel-Softmax (Concrete distribution)

Relax the discrete sample into a continuous one you *can* reparameterize. The **Gumbel-Max trick** says a categorical sample equals

$$
z = \operatorname*{arg\,max}_i \big(\log \pi_i + g_i\big), \qquad g_i \sim \text{Gumbel}(0,1).
$$

Replace the non-differentiable `argmax` with a temperature-$\tau$ **softmax**:

$$
y_i = \frac{\exp((\log \pi_i + g_i)/\tau)}{\sum_j \exp((\log \pi_j + g_j)/\tau)}.
$$

Now $y$ is a differentiable, reparameterized sample (a point on the simplex). As $\tau \to 0$, $y$ approaches a one-hot vector but the gradient variance blows up; as $\tau$ grows, samples are smooth but biased toward uniform. You **anneal** $\tau$ downward during training. **Low variance, biased.**

## 3. Straight-through estimator (STE)

Forward pass: use the **hard** discrete value (e.g. `argmax`, or a threshold). Backward pass: pretend the operation was the identity (or the softmax), and pass the gradient straight through.

$$
\text{forward: } z = \text{one\_hot}(\arg\max), \qquad \text{backward: } \frac{\partial z}{\partial \text{logits}} \approx \frac{\partial \,\text{softmax}}{\partial \text{logits}}.
$$

**Straight-Through Gumbel-Softmax** combines both: hard one-hot forward, soft Gumbel-Softmax gradient backward, so the rest of the network sees a genuine discrete sample. STE is **biased** (the backward op isn't the true derivative) but cheap and empirically effective; it is the workhorse behind **VQ-VAE** codebook training and binarized/quantized networks.

## The bias-variance tradeoff

| Estimator | Bias | Variance | Needs differentiable $f$? | Typical use |
| --- | --- | --- | --- | --- |
| **Score function (REINFORCE)** | Unbiased | High | No | RL, RLHF, black-box reward |
| **Gumbel-Softmax** | Biased ($\tau>0$) | Low | Yes | Discrete latents (categorical VAE) |
| **Straight-through** | Biased | Low | Yes (via surrogate) | VQ-VAE, quantization, hard attention |

The dividing question: **can you differentiate $f$?** If not (an environment, a metric, a sampled-then-scored pipeline), you're forced onto the score-function estimator. If you can, the relaxation methods give far lower variance.

## What an interviewer expects you to say

1. State *why* reparameterization fails for discrete $z$ (you can't write a discrete sample as a smooth function of noise and $\theta$).
2. Give the **score-function estimator** with the $\nabla \log p$ identity, that it's **unbiased but high-variance**, and that **baselines** reduce variance without adding bias.
3. Explain **Gumbel-Softmax** as the reparameterizable relaxation with a temperature you anneal (**biased, low variance**).
4. Describe the **straight-through estimator** (hard forward, soft/identity backward) and that it trains **VQ-VAE** and quantized nets.
5. Connect to practice: **RLHF uses score-function (PPO)** because text is discrete and a 50K-way Gumbel-Softmax is impractical; **DPO** sidesteps sampling entirely.

## Common confusions

- **"You can just backprop through argmax."** Its gradient is zero almost everywhere; that's the whole problem.
- **"REINFORCE is biased."** It's unbiased; its issue is variance. Baselines fix variance, not bias.
- **"Gumbel-Softmax is exact."** It's biased for any $\tau > 0$; only the $\tau \to 0$ limit is exact, and there the gradient is uselessly high-variance.
- **"Straight-through has a principled gradient."** It doesn't; it's a useful heuristic (the backward op deliberately mismatches the forward op).
- **"These are RL-only / VAE-only tricks."** They're general: hard attention, neural architecture search, discrete communication, and quantization all use them.

---

*Related: [Explain the reparameterization trick](/questions/reparameterization-trick/), [Policy gradient](/concepts/policy-gradient/), [PPO](/concepts/ppo/), [Variational autoencoders](/concepts/variational-autoencoders/), [Quantization](/concepts/quantization/).*
