---
title: "Actor-critic methods"
description: "Policy gradient with a learned value baseline. The actor picks actions; the critic estimates how good they were. The architecture under PPO, A3C, SAC, and most modern RL."
date: "2026-05-07"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

An **actor-critic** algorithm trains two networks jointly: an **actor** $\pi_\theta(a \mid s)$ that chooses actions and a **critic** $V_\phi(s)$ (or $Q_\phi(s, a)$) that estimates expected return. The actor is updated with a policy gradient using the critic's estimate as a baseline; the critic is updated to fit observed returns.

Pure policy gradient (REINFORCE) is unbiased but high-variance. Pure value-based methods (Q-learning, DQN) are sample-efficient but only support discrete actions and struggle with stochastic optimal policies. Actor-critic combines both: low-variance gradient estimates from the critic, direct policy parameterization from the actor.

Almost every modern continuous-control RL algorithm is actor-critic: PPO, A2C/A3C, SAC, TD3, DDPG. RLHF for LLMs is actor-critic.

## The two updates

### Actor (policy gradient)

$$
\nabla_\theta J(\theta) = \mathbb{E}\left[ \nabla_\theta \log \pi_\theta(a \mid s) \cdot \hat{A}(s, a) \right],
$$

where $\hat{A}$ is an advantage estimate from the critic. See [GAE](/concepts/advantage-estimation-and-gae/) for the standard recipe.

### Critic (value regression)

$$
\mathcal{L}_\phi = \mathbb{E}\left[ (V_\phi(s_t) - \hat{R}_t)^2 \right],
$$

where $\hat{R}_t$ is the return target (TD($\lambda$) target, or the GAE-derived $\hat{A}_t + V_\phi(s_t)$).

Both gradients flow on the same data. Many implementations share most of the backbone between actor and critic and split off two heads at the end.

<!-- visual:actor-critic-two-objective-flow -->
<figure class="learning-figure visual-wide plot-panel" aria-labelledby="actor-critic-visual-title">
	<p class="visual-kicker">Spatial intuition</p>
	<p class="visual-title" id="actor-critic-visual-title">One state representation feeds two objectives, but the policy loss cannot update the critic through the advantage.</p>
	<div class="visual-scroll">
		<svg viewBox="0 0 760 390" role="img" aria-labelledby="actor-critic-svg-title actor-critic-svg-desc">
			<title id="actor-critic-svg-title">Actor-critic forward and gradient paths</title>
			<desc id="actor-critic-svg-desc">A state passes through shared features and splits into an actor head above and a critic head below. The actor head produces a log action probability for the policy loss. The critic produces a value used by the value loss and by a detached advantage estimate. Solid arrows show forward loss construction. Dashed arrows show that each loss updates its head and the shared features. A stop bar blocks the policy-loss gradient from crossing the advantage into the critic.</desc>
			<defs>
				<marker id="arrow-forward" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="viz-arrow-forward" d="M0,0 L8,4 L0,8 Z"></path></marker>
				<marker id="arrow-backward" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="viz-arrow-backward" d="M0,0 L8,4 L0,8 Z"></path></marker>
			</defs>
			<text class="viz-axis-label" x="24" y="22">SOLID: construct outputs and losses</text>
			<path class="viz-forward" d="M130 175 H180"></path>
			<path class="viz-forward" d="M300 175 C330 175 325 85 355 85"></path>
			<path class="viz-forward" d="M300 175 C330 175 325 265 355 265"></path>
			<path class="viz-forward" d="M465 85 H500"></path>
			<path class="viz-forward" d="M570 85 H638"></path>
			<path class="viz-forward" d="M465 265 H500"></path>
			<path class="viz-forward" d="M570 265 H638"></path>
			<path class="viz-forward" d="M535 230 C535 205 565 198 590 193"></path>
			<path class="viz-forward" d="M535 335 C535 305 575 250 610 215"></path>
			<path class="viz-forward" d="M600 355 C635 350 667 320 677 306"></path>
			<path class="viz-forward" d="M630 160 C646 143 657 126 665 116"></path>
			<rect class="viz-node viz-node--input" x="30" y="145" width="100" height="60" rx="12"></rect>
			<text class="viz-node-label" x="80" y="170">State</text>
			<text class="viz-node-value" x="80" y="188">s<tspan baseline-shift="sub" font-size="8">t</tspan></text>
			<rect class="viz-node viz-node--focus" x="180" y="145" width="120" height="60" rx="12"></rect>
			<text class="viz-node-label" x="240" y="169">Shared features</text>
			<text class="viz-node-value" x="240" y="188">h(s<tspan baseline-shift="sub" font-size="8">t</tspan>)</text>
			<rect class="viz-node viz-node--input" x="355" y="55" width="110" height="60" rx="12"></rect>
			<text class="viz-node-label" x="410" y="79">Actor head</text>
			<text class="viz-node-value" x="410" y="98">policy parameters θ</text>
			<rect class="viz-node" x="355" y="235" width="110" height="60" rx="12"></rect>
			<text class="viz-node-label" x="410" y="259">Critic head</text>
			<text class="viz-node-value" x="410" y="278">value parameters φ</text>
			<circle class="viz-node viz-node--input" cx="535" cy="85" r="35"></circle>
			<text class="viz-node-label" x="535" y="80">log π<tspan baseline-shift="sub" font-size="9">θ</tspan></text>
			<text class="viz-node-value" x="535" y="98">(a<tspan baseline-shift="sub" font-size="8">t</tspan> | s<tspan baseline-shift="sub" font-size="8">t</tspan>)</text>
			<circle class="viz-node" cx="535" cy="265" r="35"></circle>
			<text class="viz-node-label" x="535" y="260">V<tspan baseline-shift="sub" font-size="9">φ</tspan></text>
			<text class="viz-node-value" x="535" y="278">(s<tspan baseline-shift="sub" font-size="8">t</tspan>)</text>
			<rect class="viz-node viz-node--focus" x="590" y="160" width="80" height="55" rx="10"></rect>
			<text class="viz-node-label" x="630" y="182">Â<tspan baseline-shift="sub" font-size="9">t</tspan></text>
			<text class="viz-node-value" x="630" y="200">fixed signal</text>
			<rect class="viz-node viz-node--output" x="470" y="335" width="130" height="40" rx="10"></rect>
			<text class="viz-node-label" x="535" y="360">return target R̂<tspan baseline-shift="sub" font-size="9">t</tspan></text>
			<circle class="viz-node viz-node--output" cx="685" cy="85" r="42"></circle>
			<text class="viz-node-label" x="685" y="80">policy loss</text>
			<text class="viz-node-value" x="685" y="98">−log π · Â</text>
			<circle class="viz-node viz-node--output" cx="685" cy="265" r="42"></circle>
			<text class="viz-node-label" x="685" y="260">value loss</text>
			<text class="viz-node-value" x="685" y="278">(V − R̂)²</text>
			<text class="viz-axis-label" x="24" y="325">DASHED: backpropagate parameter updates</text>
			<path class="viz-backward" d="M668 46 C605 15 465 22 438 54"></path>
			<path class="viz-backward" d="M380 55 C335 30 320 115 291 145"></path>
			<path class="viz-backward" d="M668 304 C605 330 468 326 438 296"></path>
			<path class="viz-backward" d="M380 295 C335 320 320 235 291 205"></path>
			<path class="viz-backward" d="M661 119 L648 136"></path>
			<line x1="641" y1="128" x2="655" y2="143" stroke="var(--viz-warning-stroke)" stroke-width="4"></line>
			<text class="viz-gradient-label" x="610" y="133">stop gradient</text>
			<text class="viz-edge-label" x="520" y="28">policy loss updates θ and shared features</text>
			<text class="viz-edge-label" x="520" y="327">value loss updates φ and shared features</text>
		</svg>
	</div>
	<figcaption><strong>Read it this way:</strong> follow solid arrows to build both losses from the same rollout. Then follow dashed arrows backward: the policy loss updates the actor and shared trunk through $\log \pi_\theta$, while the stop bar prevents that loss from using $\hat{A}_t$ to update the critic; the value loss trains the critic separately.</figcaption>
</figure>

## The taxonomy

### A2C / A3C

Synchronous (A2C) or asynchronous (A3C) advantage actor-critic. Multiple environments collect rollouts in parallel; updates use $n$-step returns. Simple, robust, the historical baseline ([Mnih et al., 2016](https://arxiv.org/abs/1602.01783)).

### PPO

Adds a **clipped surrogate objective** to bound how much the new policy can move from the old one. Allows multiple epochs of update on the same rollout. The default for most practical RL today, including RLHF.

### DDPG

Off-policy deterministic actor-critic for continuous control ([Lillicrap et al., 2016](https://arxiv.org/abs/1509.02971)). The actor outputs a deterministic action; the critic is $Q_\phi(s, a)$; gradient flows from $Q$ back through the actor via the chain rule. Notoriously brittle.

### TD3

DDPG plus three fixes ([Fujimoto et al., 2018](https://arxiv.org/abs/1802.09477)): twin Q-networks (take the min to reduce overestimation), delayed policy updates, target policy smoothing. Much more stable than DDPG.

### SAC

Soft Actor-Critic ([Haarnoja et al., 2018](https://arxiv.org/abs/1801.01290)). Adds an entropy bonus to the reward, learning a maximum-entropy policy. Sample-efficient and robust; the standard for off-policy continuous control.

## On-policy vs off-policy

- **On-policy** (A2C, PPO): the data must come from the current policy. Rollouts are discarded after a few updates.
- **Off-policy** (DDPG, TD3, SAC): the data can come from any policy, stored in a replay buffer. Importance sampling or deterministic gradients handle the off-policyness.

Off-policy is more sample efficient (data can be reused) but harder to stabilize. On-policy is simpler and more reliable, at the cost of needing fresh rollouts.

## How RLHF fits

RLHF is just PPO (an actor-critic algorithm) with:

- The actor initialized from a pretrained LLM.
- The critic estimating value from the same model.
- The reward coming from a learned reward model trained on human preferences.

Modern alternatives (DPO, IPO, KTO) bypass the actor-critic step entirely; they reformulate the optimization as a supervised loss on preference pairs. Faster, simpler, and the new dominant approach in 2024 to 2026.

## Common pitfalls

- **Forgetting to detach the value estimate when computing the policy gradient.** The advantage flows into the actor; the actor gradient should not flow through the critic.
- **Sharing too much of the backbone**. With one body and two heads, the value loss can dominate the gradient and hurt the policy. Tune the loss weight or use separate networks for hard tasks.
- **Off-policy updates without correction**. PPO assumes near-on-policy data; running it with a deep replay buffer breaks the clipping bound.
- **Using PPO defaults blindly**. PPO has a dozen hyperparameters (clip range, GAE lambda, value-loss coef, entropy coef, learning rate, batch size, epochs per rollout). The interactions matter.

## Related

- [Policy gradient methods](/concepts/policy-gradient/).
- [Proximal Policy Optimization](/concepts/ppo/).
- [Advantage estimation and GAE](/concepts/advantage-estimation-and-gae/).
