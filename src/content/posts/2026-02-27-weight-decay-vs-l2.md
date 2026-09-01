---
title: "Weight decay vs. L2 regularization"
description: "L2 adds ½λ‖θ‖² to the loss; weight decay shrinks θ multiplicatively at each step. They are equivalent under SGD but not under Adam. Which is why AdamW exists."
date: "2026-02-27"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

L2 regularization adds a penalty $\frac{\lambda}{2} \|\theta\|^2$ to the loss; weight decay multiplies parameters by $(1 - \eta \lambda)$ at each step. Under vanilla SGD they are mathematically equivalent. Under adaptive optimizers like Adam they are **not**. And the difference is large enough that AdamW [(Loshchilov & Hutter, 2019)](https://arxiv.org/abs/1711.05101) is now the default for transformer training.

## The two formulations

### L2 (penalty added to loss)

$$
L_\text{total}(\theta) = L_\text{data}(\theta) + \tfrac{\lambda}{2} \|\theta\|^2
$$

Gradient: $\nabla L_\text{total} = \nabla L_\text{data} + \lambda \theta$.
Update under SGD: $\theta_{t+1} = \theta_t - \eta (\nabla L_\text{data} + \lambda \theta_t)$
$\quad\quad\quad\quad\quad\quad\quad\,= (1 - \eta \lambda) \theta_t - \eta \nabla L_\text{data}$.

### Weight decay (multiplicative shrink)

$$
\theta_{t+1} = (1 - \eta \lambda) \theta_t - \eta \nabla L_\text{data}
$$

Same expression. Under **vanilla SGD**, the two are identical.

## Why they diverge under Adam

Adam scales the gradient per parameter using its moment history. With L2, the penalty is added before those moments are updated:

$$
g_t = \nabla L_\text{data}(\theta_t) + \lambda \theta_t,
\qquad
\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat m_t(g_{1:t})}{\sqrt{\hat v_t(g_{1:t})} + \varepsilon}.
$$

The penalty therefore enters both moment histories and inherits Adam's coordinate-wise scaling. In the fixed-preconditioner view, its contribution is proportional to $\lambda\theta_t / \sqrt{\hat v_t}$: larger for coordinates with a small denominator and smaller for coordinates with a large denominator. Regularization is now coupled to gradient history.

<!-- visual:weight-decay-adaptive-paths -->
<figure class="learning-figure" aria-labelledby="weight-decay-paths-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="weight-decay-paths-title">Which update path makes shrinkage depend on Adam's gradient history?</p>
	<div class="visual-grid--two" role="group" aria-label="Two ordered update paths. In Adam with L2, the penalty joins the data gradient before both moment tracking and adaptive scaling. In AdamW, only the data gradient enters Adam while weight decay follows a separate path directly to the parameter.">
		<section class="visual-panel">
			<h4>ADAM + L2 - COUPLED PATH</h4>
			<p><strong>1. Join:</strong> data gradient + λθ</p>
			<p aria-hidden="true">&darr;</p>
			<p><strong>2. Track:</strong> both terms enter m and v</p>
			<p aria-hidden="true">&darr;</p>
			<p><strong>3. Scale by coordinate:</strong> divide through Adam's adaptive denominator</p>
			<p><strong>Result:</strong> the shrinkage rate inherits each coordinate's gradient history.</p>
		</section>
		<section class="visual-panel">
			<h4>ADAMW - DECOUPLED PATH</h4>
			<p><strong>1. Adapt:</strong> data gradient alone enters m and v</p>
			<p aria-hidden="true">&darr;</p>
			<p><strong>2. Update:</strong> apply Adam's coordinate-wise data step</p>
			<p><strong>Separate path:</strong> subtract ηλθ directly from the parameter</p>
			<p><strong>Result:</strong> every decayed coordinate receives the same fractional shrinkage, ηλ.</p>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> Follow each panel from top to bottom. L2 enters before Adam's moment tracking and adaptive scaling, so history changes its effect. AdamW routes decay around that machinery, making shrinkage independent of the adaptive denominator. Original schematic checked against <a href="https://arxiv.org/abs/1711.05101">Loshchilov and Hutter (2019)</a> and the <a href="https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html">PyTorch AdamW algorithm</a>.</figcaption>
</figure>

**AdamW** decouples them: apply Adam to the data loss only, and then shrink the parameters multiplicatively as a separate step:

$$
\theta_{t+1} = (1 - \eta \lambda) \theta_t - \eta \cdot \frac{\hat m_t}{\sqrt{\hat v_t + \varepsilon}}
$$

The shrink term has no $1/\sqrt{v_t}$ scaling. This recovers the SGD-equivalent behavior.

## Empirical impact

Loshchilov & Hutter (2019) and many follow-up benchmarks show AdamW generalizes meaningfully better than Adam-with-L2 across vision and NLP. The exact gain depends on the task; on transformer LLM training the gap is large enough that essentially all modern training uses AdamW.

## What to skip

Common practice: do not decay biases, LayerNorm parameters, or embeddings. These are 1D parameters with different statistical roles, and decaying them often hurts. Standard implementations construct two parameter groups: `{decay: linear weights, conv kernels}` and `{no decay: biases, norms, embeddings}`.

## Common pitfalls

- **Using `Adam` with `weight_decay > 0`** in PyTorch. This applies L2-as-gradient, *not* AdamW. Use `AdamW` explicitly.
- **Decaying bias and LayerNorm parameters.** Hurts performance; exclude them via parameter groups.
- **Picking $\lambda$ from a CNN recipe.** $\lambda = 1\text{e-}4$ for ResNets; $\lambda = 0.1$ for AdamW transformer pretraining (with the no-decay carve-out). Different scale, different rule of thumb.
- **Forgetting that decay scales with LR.** Effective shrink per step is $\eta \lambda$. Halving LR halves effective decay; you may need to compensate.

## Related

- [Adam and AdamW](/concepts/adam-and-adamw/). For the optimizer-side derivation.
- [Regularization](/concepts/regularization/). Broader survey of regularization techniques.
