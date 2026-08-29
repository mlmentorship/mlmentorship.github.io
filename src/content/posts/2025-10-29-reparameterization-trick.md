---
title: "Explain the reparameterization trick"
description: "How VAEs propagate gradients through a sampling step. The senior answer explains the why (you can't differentiate through a sample) and the how (move the randomness outside the parameters)."
date: "2025-10-29"
draft: false
tags: ["questions"]
category: "questions"
---


> *Asked in: ML breadth and generative-model interviews.*

A standard generative-model question. The L4 candidate states the formula. The L6 candidate explains why naive sampling breaks gradients and how the trick fixes it.

## The problem

Suppose you want to train a model that samples a latent variable `z` from a distribution `q_phi(z | x)` and uses `z` to reconstruct `x`. Loss `L(theta, phi)` depends on `z`, which is a *sample*.

To train, you need `dL / dphi`. But `z = sample(q_phi)` is a stochastic operation; the gradient through a sample isn't well-defined in general.

## The trick

Reparameterize the sample:

```
z = mu_phi(x) + sigma_phi(x) * epsilon,  where epsilon ~ N(0, I)
```

The randomness now comes from `epsilon`, which doesn't depend on `phi`. The sample `z` is now a deterministic function of `(phi, epsilon)`. The gradient `dL / dphi` is computed straightforwardly via chain rule.

In short: instead of "sample `z` from `q_phi`," do "sample `epsilon` from a fixed distribution, then transform deterministically."

## Lower-variance gradients

This is the foundational trick for variational autoencoders (VAE). Without it, you'd have to use REINFORCE / score-function gradients, which have much higher variance and need many samples to be useful.

## What an L5 answer sounds like

> "When you sample `z = sample(q_phi(z | x))` and use `z` in a downstream loss, you can't backprop through the sampling because the operation is stochastic.
>
> The trick: rewrite the sample as a deterministic function of (parameters, noise), where the noise comes from a fixed distribution.
>
> For a Gaussian: `z = mu(x) + sigma(x) * epsilon`, `epsilon ~ N(0, I)`. Now `z` is differentiable w.r.t. `mu` and `sigma`, which are computed by the encoder network. Gradients flow through normally.
>
> Used in VAEs, in some RL algorithms (Gumbel-softmax for discrete actions is the discrete analog), in normalizing flows, in continuous-control policy gradient methods."

This is L5. Mechanism explained, examples given.

## What an L6 answer adds

> "...some additional points:
>
> **It only works for distributions you can reparameterize.** Gaussian, Laplace, exponential, uniform: easy. Discrete distributions: hard (requires Gumbel-softmax with continuous relaxation). General distributions: requires implicit differentiation tricks.
>
> **Variance is much lower than score-function gradients.** For a 1D Gaussian, reparameterization gradient variance scales with the Jacobian of the network; score-function variance scales with `1 / sigma^2` of the sample, which can be huge. Empirically, reparameterization needs 1-10 samples to estimate a useful gradient; REINFORCE often needs 1000+.
>
> **For discrete latents (e.g., latent-variable models with categorical z), Gumbel-softmax / concrete distribution** is the standard relaxation. Use a continuous relaxation that's differentiable; anneal the temperature toward zero to make samples nearly discrete. Trade-off: differentiable but biased.
>
> **In LLM-RL (RLHF, DPO), reparameterization isn't used** because text generation is discrete and Gumbel-softmax over a vocabulary of 50K tokens is impractical. RLHF uses score-function gradients (PPO), accepting the variance cost. DPO sidesteps this entirely with an analytical objective that doesn't need sampling at all."

## Tells that get you a strong-hire vote

- You explain **why** sampling breaks gradients before stating the trick.
- You give the **Gaussian formula** explicitly.
- You bring up **Gumbel-softmax** for discrete latents.
- You compare **variance** to score-function gradients.
- You mention **DPO sidesteps this** for LLM RL.

## Tells that get you down-leveled

- Stating the formula without explanation.
- Confusion about why naive sampling doesn't work.
- No knowledge of discrete-relaxation alternatives.
- Treating reparameterization as a VAE-only trick (it's broader).

## Common follow-up

"How does this apply to RL?"

The L6 answer:

> "Two cases. For *continuous-action policies* (e.g., robotic control with a Gaussian policy), reparameterize the action sample and backprop through the value function (this is the basic trick behind DDPG and SAC). For *discrete-action policies* (e.g., RL on discrete decisions), reparameterization doesn't directly apply; you use score-function gradients (REINFORCE, A2C, PPO) and accept the variance cost, sometimes mitigated by control variates and baselines. The 'continuous vs discrete' choice often dominates the algorithm choice in modern RL."

---

*Related: [RLHF and DPO](/concepts/rlhf-and-dpo/), [cross-entropy and softmax](/concepts/cross-entropy-softmax/), and [Bayesian versus frequentist](/questions/bayesian-vs-frequentist/).*
