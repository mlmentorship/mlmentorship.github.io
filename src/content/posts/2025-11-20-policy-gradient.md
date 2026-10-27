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
<figure class="learning-figure" aria-labelledby="policy-gradient-sign-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="policy-gradient-sign-title">How does the sign of advantage change the sampled action’s probability?</p>
	<div class="visual-grid--two" role="group" aria-label="One policy-gradient update rule followed by positive, zero, and negative advantage cases">
		<section class="visual-panel">
			<h4>SAMPLED ACTION · COMPARE WITH BASELINE</h4>
			<p><strong>Observed</strong><br /><var>G</var><sub>t</sub> is the return-to-go after sampling action <var>a</var><sub>t</sub> in state <var>s</var><sub>t</sub>.</p>
			<p><strong>Expected</strong><br /><var>b</var>(<var>s</var><sub>t</sub>) is an action-independent baseline, commonly <var>V</var>(<var>s</var><sub>t</sub>).</p>
			<p><strong>Weight</strong><br /><var>A</var><sub>t</sub> = <var>G</var><sub>t</sub> − <var>b</var>(<var>s</var><sub>t</sub>).</p>
		</section>
		<section class="visual-panel">
			<h4>+ · BETTER THAN EXPECTED</h4>
			<p><strong>Advantage</strong><br /><var>A</var><sub>t</sub> &gt; 0.</p>
			<p><strong>Gradient-ascent update</strong><br />Move along the score direction: increase log π(<var>a</var><sub>t</sub> | <var>s</var><sub>t</sub>) and reinforce the sampled action.</p>
		</section>
		<section class="visual-panel">
			<h4>0 · MATCHED EXPECTATION</h4>
			<p><strong>Advantage</strong><br /><var>A</var><sub>t</sub> = 0.</p>
			<p><strong>Gradient-ascent update</strong><br />This sample contributes zero to the policy gradient.</p>
		</section>
		<section class="visual-panel">
			<h4>− · WORSE THAN EXPECTED</h4>
			<p><strong>Advantage</strong><br /><var>A</var><sub>t</sub> &lt; 0.</p>
			<p><strong>Gradient-ascent update</strong><br />Reverse the score direction: decrease log π(<var>a</var><sub>t</sub> | <var>s</var><sub>t</sub>) and suppress the sampled action.</p>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> start with the sampled action, subtract what the policy expected in that state, then follow the sign. Positive advantage reinforces that action; negative advantage reverses the score direction; zero advantage supplies no update. Because policy probabilities are normalized, changing the sampled action can also redistribute probability among other actions. The action-independent baseline changes variance, not the expected policy gradient. Original schematic checked against <a href="https://papers.nips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html">Sutton et al.'s policy-gradient theorem</a> and <a href="https://doi.org/10.1007/BF00992696">Williams's REINFORCE analysis</a>.</figcaption>
</figure>

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
