---
title: "Policy gradient methods"
description: "Directly optimize the policy by following the gradient of expected return. REINFORCE, actor-critic, and the foundation of modern RL."
date: "2025-11-20"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

**Policy gradient** methods directly parametrize a policy $\pi_\theta(a \mid s)$ and optimize $\theta$ by ascending the gradient of expected return:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot R_t \right].
$$

This is the **policy gradient theorem** [(Sutton et al., 1999)](https://papers.nips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html).

Policy gradient methods are the foundation of modern continuous-control RL (SAC, PPO), of large-scale RL (AlphaGo's policy net, OpenAI Five), and of LLM alignment (RLHF uses PPO). They are essential whenever:

- The action space is continuous (no easy $\max_a$).
- A stochastic policy is desirable (exploration, multi-modal optimal policies).
- You can write a clean, differentiable policy parametrization.

## REINFORCE

The simplest policy gradient [(Williams, 1992)](https://link.springer.com/article/10.1007/BF00992696):

$$
\nabla_\theta J = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[ \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot G_t \right]
$$

where $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$ is the return-to-go from step $t$.

Sample a trajectory by running the policy. For each step, compute $\log \pi_\theta(a_t \mid s_t)$ and weight by the actual return. Backprop. Take a gradient step.

**Pros**: extremely general. Works for any differentiable policy.
**Cons**: high variance. Return $G_t$ is a noisy estimate of expected return; gradient estimator dominates by random rollout luck.

## Variance reduction: baselines

Subtract any state-dependent baseline $b(s_t)$ from $G_t$. Does not change the expected gradient (so no bias) but can drastically reduce variance:

$$
\nabla_\theta J = \mathbb{E}\!\left[ \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot (G_t - b(s_t)) \right].
$$

A common baseline is the value function $V(s_t)$, leading to actor-critic.
It is not generally the minimum-variance baseline.
For one conditional score-function term, let $u=\nabla_\theta\log\pi_\theta(a\mid s)$.
When the denominator is nonzero, the scalar baseline minimizing the estimator's total conditional variance is

$$
b^*(s)=\frac{\mathbb{E}[G\lVert u\rVert^2\mid s]}{\mathbb{E}[\lVert u\rVert^2\mid s]}.
$$

This equals $V(s)$ only under additional conditions on the score-norm weighting.
Treat the baseline as constant when differentiating the policy term.

**Predict before revealing:** a one-step policy chooses action 1 with probability 0.8, receives reward equal to its action, and has logit score `action - 0.8`.
Must baseline $V=0.8$ have less variance than baseline 0.2?

<details><summary>Compare the mean and variance separately</summary>

No. With baseline 0.8, the two gradient estimates are 0.04 and 0.64, with probabilities 0.8 and 0.2.
Their mean is 0.16 and variance is 0.0576.
With baseline 0.2, both actions give gradient 0.16, so variance is zero.
Both baselines are action-independent and unbiased; that does not make their variances equal.

</details>

<!-- visual:policy-gradient-advantage-sign -->
<figure class="learning-figure editorial-illustration" aria-labelledby="policy-gradient-advantage-sign-edition-title"><span class="visual-fragment-anchor" id="policy-gradient-sign-title" aria-hidden="true"></span><p class="visual-title" id="policy-gradient-advantage-sign-edition-title">A positive return still suppresses an action when it falls below baseline</p><svg viewBox="0 0 360 631" role="img" aria-labelledby="policy-gradient-advantage-sign-edition-svg-title policy-gradient-advantage-sign-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="policy-gradient-advantage-sign-edition-svg-title">A positive return can still suppress the sampled action</title><desc id="policy-gradient-advantage-sign-edition-svg-desc">A declared three-action softmax starts with zero logits and uniform probabilities. After sampling A with action-independent baseline three, returns five, three, one give advantages two, zero, minus two. Exact logit gradient-ascent steps of size .1 produce normalized probabilities; A increases, stays fixed for this sample, or decreases despite positive absolute return.</desc><text x="180" y="22" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Toy 3-action softmax: initial logits (0,0,0)</tspan></text><text x="180" y="60" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Sampled A; baseline b=3; ascent step eta=0.1</tspan></text><text x="18" y="111" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="18" dy="0">Return 5 - baseline 3 = +2</tspan></text><g data-policy-case="2" data-action="0" data-probability="0.37915245309398876"><rect x="30" y="138" width="113.74573592819662" height="36" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect></g><text x="86.8728679640983" y="161" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="86.8728679640983" dy="0">A 37.9%</tspan></text><g data-policy-case="2" data-action="1" data-probability="0.3104237734530056"><rect x="143.7457359281966" y="138" width="93.12713203590168" height="36" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect></g><text x="190.30930194614746" y="161" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="190.30930194614746" dy="0">B 31.0%</tspan></text><g data-policy-case="2" data-action="2" data-probability="0.3104237734530056"><rect x="236.8728679640983" y="138" width="93.12713203590168" height="36" rx="3" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.3"></rect></g><text x="283.43643398204915" y="161" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="283.43643398204915" dy="0">C 31.0%</tspan></text><text x="180" y="207" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">A rises above 1/3; B and C renormalize.</tspan></text><text x="18" y="265" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="18" dy="0">Return 3 - baseline 3 = 0</tspan></text><g data-policy-case="0" data-action="0" data-probability="0.3333333333333333"><rect x="30" y="292" width="100" height="36" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect></g><text x="80" y="315" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="80" dy="0">A 33.3%</tspan></text><g data-policy-case="0" data-action="1" data-probability="0.3333333333333333"><rect x="130" y="292" width="100" height="36" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect></g><text x="180" y="315" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">B 33.3%</tspan></text><g data-policy-case="0" data-action="2" data-probability="0.3333333333333333"><rect x="230" y="292" width="100" height="36" rx="3" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.3"></rect></g><text x="280" y="315" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="280" dy="0">C 33.3%</tspan></text><text x="180" y="361" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">This sample contributes zero, not the whole</tspan><tspan x="180" dy="19">batch.</tspan></text><text x="18" y="419" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="18" dy="0">Return 1 - baseline 3 = -2</tspan></text><g data-policy-case="-2" data-action="0" data-probability="0.2904607870701907"><rect x="30" y="446" width="87.1382361210572" height="36" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect></g><text x="73.56911806052861" y="469" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="73.56911806052861" dy="0">A 29.0%</tspan></text><g data-policy-case="-2" data-action="1" data-probability="0.3547696064649047"><rect x="117.1382361210572" y="446" width="106.4308819394714" height="36" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect></g><text x="170.3536770907929" y="469" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="170.3536770907929" dy="0">B 35.5%</tspan></text><g data-policy-case="-2" data-action="2" data-probability="0.3547696064649047"><rect x="223.5691180605286" y="446" width="106.4308819394714" height="36" rx="3" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.3"></rect></g><text x="276.7845590302643" y="469" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="276.7845590302643" dy="0">C 35.5%</tspan></text><text x="180" y="515" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Return is positive, yet A falls below 1/3.</tspan></text><text x="180" y="599" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">logit step = eta * advantage * (one_hot(A) - p)</tspan></text></svg><figcaption><strong>Read it this way:</strong> Subtract the baseline before following the sign. Every stacked bar sums to one after the exact softmax step. Zero means this sample contributes nothing, not that an entire batch cannot update. <span class="visual-reader-check" data-reader-check="policy-gradient-advantage-sign"><strong>Check:</strong> Why can a positive absolute return still suppress the sampled action if it is below the state baseline?</span></figcaption></figure>

## Actor-critic

Two networks:

- **Actor** $\pi_\theta(a \mid s)$. The policy.
- **Critic** $V_\phi(s)$. Value function estimator.

Use the critic as the baseline. Define the **advantage** $A(s, a) = Q(s, a) - V(s) \approx G_t - V_\phi(s_t)$. Update:

$$
\nabla_\theta J = \mathbb{E}\!\left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot A_t \right].
$$

Critic is trained by TD: $V_\phi(s_t) \leftarrow V_\phi(s_t) + \alpha [r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)]$.

This is the basic **A2C** / **A3C** algorithm.

## Generalized Advantage Estimation (GAE)

Instead of one-step or full-return advantages, use a weighted sum:

$$
A_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t).
$$

$\lambda \in [0, 1]$ trades bias for variance. $\lambda = 0$ gives one-step TD; $\lambda = 1$ gives the full-return form, with terminal or truncation handling stated separately. A value such as 0.95 is a tuning choice, not a universal PPO requirement.

## Trust regions and PPO

Naive policy gradient updates can drastically change the policy in one step, leading to collapse. **Trust Region Policy Optimization** (TRPO; [Schulman et al., 2015](https://arxiv.org/abs/1502.05477)) constrains the KL divergence between old and new policies. **PPO** [(Schulman et al., 2017)](https://arxiv.org/abs/1707.06347) replaces the constraint with a clipped surrogate objective:

$$
L^{\text{PPO}}(\theta) = \mathbb{E}\!\left[ \min\big( r_t(\theta) A_t,\; \mathrm{clip}(r_t(\theta), 1 - \varepsilon, 1 + \varepsilon) A_t \big) \right]
$$

with $r_t(\theta) = \pi_\theta(a_t \mid s_t) / \pi_{\theta_\text{old}}(a_t \mid s_t)$ and $\varepsilon = 0.2$ typical.

PPO is used in control and some RLHF pipelines.
Its clipped objective removes the incentive for certain sampled-action changes beyond the clipping boundary; it is not a hard constraint on policy ratios or KL.

**Predict before revealing:** old action probability is 0.1, advantage is 1, and clipping parameter is 0.2.
Can new probability 0.9 have the same clipped objective as new probability 0.12?

<details><summary>Inspect the objective, not an imaginary constraint</summary>

Yes. The ratios are 9 and 1.2, but both clipped surrogate values are 1.2.
A flat objective beyond the boundary does not make those policy parameters infeasible.
Monitor ratios and KL; optional early stopping is a separate implementation safeguard, not a guarantee supplied by clipping.

</details>

## Off-policy actor-critic

For sample efficiency in continuous control, use a replay buffer with importance sampling correction or Q-function critics. **DDPG**, **TD3**, **SAC** are the standard off-policy actor-critic algorithms. SAC adds a maximum-entropy bonus that encourages exploration.

## Common pitfalls

- **High variance.** Test an action-independent baseline. Advantage normalization changes finite-batch estimates and scale; evaluate it rather than assuming every baseline or normalization must help.
- **Insufficient exploration.** An entropy bonus can help when appropriate; deterministic policies are not automatically a failure. For a minimized loss, $-\beta H(\pi_\theta)$ rewards entropy.
- **Reusing stale samples.** PPO uses rollout-policy probabilities in its importance ratios, but clipping does not bound staleness or make arbitrary off-policy reuse valid. Collect fresh rollouts and inspect drift.
- **Mismatch between rollout and learning.** Record the behavior-policy version and log probabilities. Ratios require matching support and do not by themselves guarantee a safe update.
- **Treating policy gradient as universally low-variance.** It isn't. Q-learning often has lower variance for discrete-action problems.

## Related

- [Q-learning](/concepts/q-learning/). Value-based alternative.
- [RLHF and DPO](/concepts/rlhf-and-dpo/). PPO applied to LLM alignment.
- [PPO](/concepts/ppo/). Algorithm-level details.
