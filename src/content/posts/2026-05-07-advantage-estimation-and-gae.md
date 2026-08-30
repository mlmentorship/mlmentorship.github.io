---
title: "Advantage estimation and GAE"
description: "Policy gradients need a low-variance estimate of how much better an action was than average. GAE is the standard answer: an exponentially weighted blend of n-step returns."
date: "2026-05-07"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

The **advantage** $A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$ measures how much better action $a$ is than the policy's average. **Generalized Advantage Estimation** (GAE, [Schulman et al., 2016](https://arxiv.org/abs/1506.02438)) estimates it as an exponentially weighted average of $n$-step TD residuals, controlled by a single parameter $\lambda$.

Policy gradient methods optimize $\nabla_\theta J(\theta) = \mathbb{E}[\nabla \log \pi_\theta(a \mid s) \cdot \Psi]$. The choice of $\Psi$ controls the bias-variance tradeoff:

- $\Psi = R$ (full return): unbiased, high variance.
- $\Psi = Q^\pi(s, a)$: lower variance but needs an action-value estimator.
- $\Psi = A^\pi(s, a)$: same expectation as $Q$ but with the baseline subtracted, lower variance.

Substituting an estimator for $A$ introduces bias. GAE makes this tradeoff explicit and tunable. It is the default advantage estimator in PPO, the most widely deployed RL algorithm.

## The mechanism

Define the **TD residual** at step $t$:

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t).
$$

The $n$-step advantage estimate is

$$
\hat{A}_t^{(n)} = \sum_{l=0}^{n-1} \gamma^l \delta_{t+l}.
$$

GAE blends all $n$-step estimates with exponential weight $\lambda$:

$$
\hat{A}_t^{\text{GAE}}(\gamma, \lambda) = \sum_{l=0}^{\infty} (\gamma \lambda)^l \, \delta_{t+l}.
$$

In code this collapses to a backward recursion:

$$
\hat{A}_t = \delta_t + \gamma \lambda \hat{A}_{t+1}.
$$

A single backward pass over the trajectory.

<!-- visual:gae-residual-weight-trace -->
<figure class="learning-figure visual-wide plot-panel" aria-labelledby="gae-residual-visual-title">
	<p class="visual-kicker">Temporal credit assignment</p>
	<p class="visual-title" id="gae-residual-visual-title">One advantage estimate gathers every later TD residual, one extra $\gamma\lambda$ factor per step.</p>
	<div class="visual-scroll">
		<svg viewBox="0 0 760 350" role="img" aria-labelledby="gae-residual-svg-title gae-residual-svg-desc">
			<title id="gae-residual-svg-title">GAE as a weighted residual trace and reverse scan</title>
			<desc id="gae-residual-svg-desc">In a four-step finite rollout, delta t contributes to advantage t with weight one, delta t plus one with gamma lambda, delta t plus two with gamma lambda squared, and delta t plus three with gamma lambda cubed. Below, dashed arrows run from the rollout boundary toward time t, showing the equivalent backward recursion: at each step, add the local delta to gamma lambda times the accumulated value from the right.</desc>
			<defs>
				<marker id="arrow-forward" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="viz-arrow-forward" d="M0,0 L8,4 L0,8 Z"></path></marker>
				<marker id="arrow-backward" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="viz-arrow-backward" d="M0,0 L8,4 L0,8 Z"></path></marker>
			</defs>
			<text class="viz-axis-label" x="24" y="24">UNROLLED VIEW: CONTRIBUTIONS TO Â<tspan baseline-shift="sub" font-size="8">t</tspan></text>
			<path class="viz-axis" d="M100 66 H660"></path>
			<path class="viz-forward" d="M100 66 H254"></path>
			<path class="viz-forward" d="M280 66 H434"></path>
			<path class="viz-forward" d="M460 66 H614"></path>
			<circle class="viz-node viz-node--focus" cx="100" cy="66" r="31"></circle>
			<circle class="viz-node" cx="280" cy="66" r="31"></circle>
			<circle class="viz-node" cx="460" cy="66" r="31"></circle>
			<circle class="viz-node" cx="640" cy="66" r="31"></circle>
			<text class="viz-node-label" x="100" y="71">δ<tspan baseline-shift="sub" font-size="9">t</tspan></text>
			<text class="viz-node-label" x="280" y="71">δ<tspan baseline-shift="sub" font-size="9">t+1</tspan></text>
			<text class="viz-node-label" x="460" y="71">δ<tspan baseline-shift="sub" font-size="9">t+2</tspan></text>
			<text class="viz-node-label" x="640" y="71">δ<tspan baseline-shift="sub" font-size="9">t+3</tspan></text>
			<text class="viz-label" x="100" y="116" text-anchor="middle">weight 1</text>
			<text class="viz-label" x="280" y="116" text-anchor="middle">weight γλ</text>
			<text class="viz-label" x="460" y="116" text-anchor="middle">weight (γλ)²</text>
			<text class="viz-label" x="640" y="116" text-anchor="middle">weight (γλ)³</text>
			<path class="viz-gridline" d="M100 126 V145 M280 126 V145 M460 126 V145 M640 126 V145"></path>
			<rect class="viz-node viz-node--output" x="90" y="145" width="580" height="48" rx="10"></rect>
			<text class="viz-node-label" x="380" y="174">Â<tspan baseline-shift="sub" font-size="9">t</tspan> = δ<tspan baseline-shift="sub" font-size="9">t</tspan> + γλ δ<tspan baseline-shift="sub" font-size="9">t+1</tspan> + (γλ)² δ<tspan baseline-shift="sub" font-size="9">t+2</tspan> + (γλ)³ δ<tspan baseline-shift="sub" font-size="9">t+3</tspan></text>
			<text class="viz-axis-label" x="24" y="230">COMPUTATION VIEW: SCAN FROM THE ROLLOUT BOUNDARY TOWARD t</text>
			<text class="viz-callout" x="380" y="252" text-anchor="middle">at each step: accumulated value ← local δ + γλ × value from the right</text>
			<path class="viz-backward" d="M585 295 H505"></path>
			<path class="viz-backward" d="M405 295 H325"></path>
			<path class="viz-backward" d="M225 295 H145"></path>
			<rect class="viz-node viz-node--focus" x="45" y="270" width="100" height="50" rx="9"></rect>
			<rect class="viz-node" x="225" y="270" width="100" height="50" rx="9"></rect>
			<rect class="viz-node" x="405" y="270" width="100" height="50" rx="9"></rect>
			<rect class="viz-node" x="585" y="270" width="130" height="50" rx="9"></rect>
			<text class="viz-node-label" x="95" y="291">Â<tspan baseline-shift="sub" font-size="9">t</tspan></text>
			<text class="viz-node-value" x="95" y="307">final estimate</text>
			<text class="viz-node-label" x="275" y="291">Â<tspan baseline-shift="sub" font-size="9">t+1</tspan></text>
			<text class="viz-node-value" x="275" y="307">carry left</text>
			<text class="viz-node-label" x="455" y="291">Â<tspan baseline-shift="sub" font-size="9">t+2</tspan></text>
			<text class="viz-node-value" x="455" y="307">carry left</text>
			<text class="viz-node-label" x="650" y="291">Â<tspan baseline-shift="sub" font-size="9">t+3</tspan> = δ<tspan baseline-shift="sub" font-size="9">t+3</tspan></text>
			<text class="viz-node-value" x="650" y="307">start at boundary</text>
			<text class="viz-gradient-label" x="545" y="339">dashed arrows: recursion direction</text>
		</svg>
	</div>
	<figcaption><strong>Read it this way:</strong> start at the right edge of the rollout and scan left. Each step keeps its own TD residual at full strength, then adds $\gamma\lambda$ times the value already accumulated from later steps. Unrolling that recursion gives weights $1, \gamma\lambda, (\gamma\lambda)^2, \ldots$ for $\delta_t, \delta_{t+1}, \delta_{t+2}, \ldots$.</figcaption>
</figure>

## The two knobs

- **$\gamma$ (discount)**: how much future reward matters. Part of the problem definition; usually 0.99 for episodic, 0.95 to 0.999 for continuing tasks.
- **$\lambda$ (GAE)**: bias-variance dial.
  - $\lambda = 0$ recovers the 1-step TD residual: low variance, biased by $V$ errors.
  - $\lambda = 1$ recovers the full Monte Carlo return minus $V(s_t)$: unbiased, high variance.
  - $\lambda = 0.95$ to $0.97$ is the standard for PPO.

## Why subtracting a baseline reduces variance

For any function $b(s)$ depending only on state, $\mathbb{E}_\pi[\nabla \log \pi(a \mid s) \cdot b(s)] = 0$. So the gradient estimator

$$
\nabla \log \pi(a \mid s) \cdot (Q^\pi(s, a) - b(s))
$$

has the same expectation but lower variance, when $b$ correlates with $Q^\pi$. The optimal baseline is exactly $V^\pi$, hence the advantage formulation.

## How it is used in PPO

PPO trains an actor and a value critic jointly. At each rollout:

1. Run the policy for $T$ steps in $N$ parallel environments. Collect transitions.
2. Run the value network on every observed state to get $V(s_t)$.
3. Compute $\delta_t$ and then $\hat{A}_t$ via the GAE recursion.
4. Compute returns as $\hat{R}_t = \hat{A}_t + V(s_t)$ for the value-function regression target.
5. Normalize advantages (subtract mean, divide by std) per batch. Important for training stability.
6. Train policy with the clipped objective using $\hat{A}_t$, train value network on $\hat{R}_t$.

## Common pitfalls

- **Forgetting to bootstrap on truncation.** When an episode is cut off mid-trajectory (not because of termination), $\delta_T$ should use $V(s_T)$ as the bootstrap. Conflating truncation with termination is a frequent bug.
- **Not normalizing advantages.** PPO almost always benefits from per-batch advantage normalization.
- **Using $\lambda = 1$ with a large $\gamma$ on long horizons.** Variance explodes.
- **Applying GAE in off-policy settings without correction.** GAE assumes on-policy data. With a replay buffer and importance sampling, V-trace or Retrace is the corrected version.

## Related

- [Proximal Policy Optimization](/concepts/ppo/).
- [Policy gradient methods](/concepts/policy-gradient/).
- [Actor-critic methods](/concepts/actor-critic-methods/).
