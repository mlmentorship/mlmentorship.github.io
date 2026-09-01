---
title: "SGD with momentum"
description: "Add a moving average of past gradients to the update. Smoother trajectories, faster convergence in narrow valleys, and the foundation of Adam's first moment."
date: "2026-04-18"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

SGD with momentum maintains a velocity $v_t = \beta v_{t-1} + g_t$ (an exponential moving average of past gradients) and updates parameters with $\theta_{t+1} = \theta_t - \eta v_t$ instead of with the raw gradient. Typical $\beta = 0.9$.

Vanilla SGD bounces around in narrow loss valleys: gradients perpendicular to the valley axis cancel slowly, gradients along the axis are small. Momentum accumulates the consistent along-axis component while perpendicular components average to zero.

<!-- visual:momentum-alternating-components -->
<figure class="learning-figure" aria-labelledby="momentum-components-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="momentum-components-title">What does momentum do to consistent and alternating gradient components?</p>
	<div class="visual-grid--two" role="group" aria-label="Raw SGD and momentum trajectories on the same coordinate scale">
		<section class="visual-panel plot-panel">
			<svg viewBox="0 0 300 235" role="img" aria-labelledby="momentum-raw-title momentum-raw-desc">
				<title id="momentum-raw-title">Raw SGD follows alternating descent directions</title>
				<desc id="momentum-raw-desc">On a common coordinate scale, four raw SGD updates alternate between the vectors one comma two and one comma negative two. The path zigzags from zero comma zero through one comma two, two comma zero, three comma two, and four comma zero.</desc>
				<rect class="viz-plot-bg" x="30" y="28" width="244" height="152" rx="5"></rect>
				<text class="viz-axis-label" x="152" y="17" text-anchor="middle">RAW SGD · SAME SCALE</text>
				<path d="M40 130H262M40 42V174" class="viz-axis"></path>
				<path d="M40 66H262M40 98H262M40 130H262M40 162H262" class="viz-gridline"></path>
				<text class="viz-label" x="25" y="70" text-anchor="end">+2</text>
				<text class="viz-label" x="25" y="134" text-anchor="end">0</text>
				<text class="viz-label" x="25" y="166" text-anchor="end">−1</text>
				<text class="viz-label" x="40" y="194" text-anchor="middle">0</text>
				<text class="viz-label" x="94" y="194" text-anchor="middle">2</text>
				<text class="viz-label" x="148" y="194" text-anchor="middle">4</text>
				<text class="viz-label" x="202" y="194" text-anchor="middle">6</text>
				<text class="viz-label" x="256" y="194" text-anchor="middle">8</text>
				<path d="M40 130L67 66L94 130L121 66L148 130" style="fill:none;stroke:var(--viz-edge);stroke-width:3;stroke-dasharray:6 4;stroke-linejoin:round"></path>
				<g style="fill:var(--viz-neutral-bg);stroke:var(--viz-edge);stroke-width:2"><circle cx="40" cy="130" r="4"></circle><circle cx="67" cy="66" r="4"></circle><circle cx="94" cy="130" r="4"></circle><circle cx="121" cy="66" r="4"></circle><circle cx="148" cy="130" r="4"></circle></g>
				<text class="viz-callout" x="40" y="219">Δθ = (1, +2), (1, −2), …</text>
				<text class="viz-label" x="164" y="115">ends at x = 4</text>
			</svg>
		</section>
		<section class="visual-panel plot-panel">
			<svg viewBox="0 0 300 235" role="img" aria-labelledby="momentum-smoothed-title momentum-smoothed-desc">
				<title id="momentum-smoothed-title">Momentum amplifies consistent progress relative to oscillation</title>
				<desc id="momentum-smoothed-desc">For beta one half after the momentum buffer reaches its steady alternating regime, the update vectors alternate between two comma four thirds and two comma negative four thirds. On the same scale as raw SGD, four updates trace zero comma zero, two comma four thirds, four comma zero, six comma four thirds, and eight comma zero.</desc>
				<rect class="viz-plot-bg" x="30" y="28" width="244" height="152" rx="5"></rect>
				<text class="viz-axis-label" x="152" y="17" text-anchor="middle">MOMENTUM · β=0.5 · WARMED UP</text>
				<path d="M40 130H262M40 42V174" class="viz-axis"></path>
				<path d="M40 66H262M40 98H262M40 130H262M40 162H262" class="viz-gridline"></path>
				<text class="viz-label" x="25" y="70" text-anchor="end">+2</text>
				<text class="viz-label" x="25" y="134" text-anchor="end">0</text>
				<text class="viz-label" x="25" y="166" text-anchor="end">−1</text>
				<text class="viz-label" x="40" y="194" text-anchor="middle">0</text>
				<text class="viz-label" x="94" y="194" text-anchor="middle">2</text>
				<text class="viz-label" x="148" y="194" text-anchor="middle">4</text>
				<text class="viz-label" x="202" y="194" text-anchor="middle">6</text>
				<text class="viz-label" x="256" y="194" text-anchor="middle">8</text>
				<path d="M40 130L94 87.33L148 130L202 87.33L256 130" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:4;stroke-linejoin:round"></path>
				<g style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"><rect x="36" y="126" width="8" height="8"></rect><rect x="90" y="83.33" width="8" height="8"></rect><rect x="144" y="126" width="8" height="8"></rect><rect x="198" y="83.33" width="8" height="8"></rect><rect x="252" y="126" width="8" height="8"></rect></g>
				<text class="viz-callout" x="40" y="219">Δθ = (2, +4/3), (2, −4/3), …</text>
				<text class="viz-label" x="190" y="148">ends at x = 8</text>
			</svg>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> use the same axes in both panels. For alternating descent directions <code>d_t = -g_t = (1, ±2)</code>, raw SGD moves forward by 1 while crossing the valley by 2. With <code>u_t = 0.5u_(t-1) + d_t</code> after warm-up, the alternating steady-state updates are <code>(2, ±4/3)</code>: the consistent component doubles, while the cross-valley-to-forward ratio falls from 2 to 2/3. Momentum makes faster, less zigzagging progress, but the larger forward step is also why its learning rate must be tuned. This original trace was checked against <a href="https://doi.org/10.1016/0041-5553(64)90137-5">Polyak (1964)</a>, <a href="https://proceedings.mlr.press/v28/sutskever13.html">Sutskever et al. (2013)</a>, and the <a href="https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html">PyTorch SGD definition</a>.</figcaption>
</figure>

Empirically, momentum is essential for SGD to be competitive with adaptive optimizers on most problems. SGD without momentum is rarely used in modern training. Adam's first-moment estimate $m_t$ is essentially momentum, which is why Adam inherits this benefit.

## Two formulations

### Classical momentum (Polyak, 1964)

$$
v_t = \beta v_{t-1} + g_t \\
\theta_{t+1} = \theta_t - \eta v_t
$$

Effective LR for a constant gradient: $\eta / (1 - \beta)$. With $\beta = 0.9$, that's $10\eta$, so reducing $\beta$ effectively reduces the step size.

### Nesterov momentum (Nesterov, 1983)

Compute the gradient at the *look-ahead* position $\theta_t - \eta \beta v_{t-1}$ instead of at $\theta_t$. Updates:

$$
v_t = \beta v_{t-1} + \nabla L(\theta_t - \eta \beta v_{t-1}) \\
\theta_{t+1} = \theta_t - \eta v_t
$$

In practice, only marginally better than classical momentum on most workloads. Used in some vision training (e.g., ResNet original paper).

## Picking $\beta$

| Workload | $\beta$ |
|----------|---------|
| ResNet / CNN training | 0.9 |
| Reinforcement learning policy nets | 0.9 |
| Very noisy gradients (RL, contrastive) | 0.95 or 0.99 |
| Small batch with high noise | lower (0.5–0.8) to track recent gradients |

$\beta$ controls the effective averaging window: $1/(1-\beta)$ steps. $\beta = 0.9$ averages over ~10 steps; $\beta = 0.99$ averages over ~100.

## When SGD+momentum vs. Adam

| Situation | Default |
|-----------|--------|
| Vision (CNN, ViT) classification with strong regularization | SGD + momentum + cosine LR |
| Transformers (NLP, LLM training) | Adam / AdamW |
| Small datasets, fine-tuning | Adam (less hyperparameter tuning needed) |
| Sparse gradients (recsys embeddings) | Adam (per-parameter adaptive LR) |
| Reinforcement learning | Adam (default in PPO / DQN implementations) |

SGD+momentum often generalizes slightly better than Adam at convergence (sharp vs. flat minima discussion); Adam converges faster initially. For LLMs, Adam wins because the gradient distribution across parameters is highly non-uniform.

## Common pitfalls

- **Forgetting that momentum scales the effective LR.** Switching from $\beta = 0.9$ to $\beta = 0.99$ effectively multiplies the LR by 10.
- **Initializing $v_0 = 0$ without bias correction.** Early steps have biased velocity; for SGD this is usually fine, but Adam explicitly bias-corrects.
- **Mixing momentum across LR changes.** When the LR jumps, momentum carries old-LR-scaled velocities. Some implementations zero momentum at LR transitions.
- **Using SGD without momentum.** Almost always strictly worse; pick momentum or use Adam.
