---
title: "Why does Adam sometimes generalize worse than SGD?"
description: "Adam usually trains faster but in some settings finds sharper minima with worse generalization. The senior answer names the regimes where this happens and the modern fixes."
date: "2026-04-29"
draft: false
tags: ["questions"]
category: "questions"
---


> *Asked in: ML breadth, especially in CV-leaning or research-track interviews.*

A nuanced question. The L4 candidate doesn't know there's a difference. The L6 candidate explains the implicit-regularization argument, the regimes where SGD wins, and how AdamW + careful schedules close most of the gap.

## What an L4 answer sounds like

> "Adam adapts the learning rate per parameter, so it converges faster than SGD but might not generalize as well."

True at a slogan level, no mechanism. You've heard the fact, not the explanation.

## What an L5 answer sounds like

> "There are two main explanations:
>
> 1. **Sharp vs flat minima.** SGD's per-mini-batch noise biases it toward flat minima, which empirically generalize better than sharp ones. Adam's adaptive scaling reduces this implicit noise, so it tends to converge to sharper minima with similar training loss but worse test performance.
>
> 2. **Weight decay coupling.** Original Adam couples weight decay with the adaptive learning rate, effectively reducing weight decay on parameters with large gradient magnitudes (the wrong direction). AdamW decouples weight decay and largely fixes this.
>
> The empirical picture in 2026:
> - **Vision (CNNs)**: SGD with momentum sometimes beats AdamW on classic ImageNet-style problems. The gap is small and getting smaller.
> - **NLP (transformers)**: AdamW is essentially always better. SGD doesn't even train large transformers well.
> - **LLMs**: AdamW is the default at all scales. No serious team uses SGD for LLM pretraining."

This is L5. You've named the two mechanisms and given the regime breakdown.

## What an L6 answer sounds like

> "...subtler points:
>
> **The 'sharp minima generalize worse' argument is contested.** Multiple papers [(Dinh et al. 2017)](https://arxiv.org/abs/1703.04933) showed that sharpness can be reparameterized away, undermining the simplest version of the argument. The empirical generalization gap exists; the precise reason is still debated. Don't bet a strong answer on the sharp/flat framing alone.
>
> **AdamW closes most of the gap with SGD on vision.** The [Loshchilov & Hutter 2017](https://arxiv.org/abs/1711.05101) paper showed AdamW + cosine schedule + careful warmup matches or beats SGD on most benchmarks. The 'SGD generalizes better' folk wisdom predates AdamW and is largely outdated.
>
> **For very large batch training, neither classic Adam nor SGD work well.** LAMB (layer-wise adaptive momentum) was designed for very large batch transformer training and outperforms both at batch size 16K+.
>
> **Optimizer choice rarely matters at scale.** With enough data, the optimizer choice is dominated by data quality, model architecture, and learning rate schedule. Don't over-tune the optimizer at the expense of the schedule."

<!-- visual:optimizer-sharpness-reparameterization -->
<figure class="learning-figure plot-panel" aria-labelledby="optimizer-sharpness-visual-title">
	<p class="visual-kicker">The sharpness caveat</p>
	<p class="visual-title" id="optimizer-sharpness-visual-title">One predictor can look flat or sharp under different parameterizations.</p>
	<div class="visual-scroll">
		<svg viewBox="0 0 360 326" role="img" aria-labelledby="optimizer-sharpness-svg-title optimizer-sharpness-svg-desc">
			<title id="optimizer-sharpness-svg-title">Equivalent predictors with broad and narrow parameter-space loss profiles</title>
			<desc id="optimizer-sharpness-svg-desc">Two illustrative loss plots are stacked vertically. Parameterization A has a broad U-shaped profile and parameterization B has a narrow U-shaped profile. Both minima have the same loss, and a labeled equivalence between the plots states that the represented function, predictions, and generalization are unchanged. The comparison shows that coordinate sharpness alone cannot explain generalization.</desc>
			<rect class="viz-plot-bg" x="12" y="12" width="336" height="124" rx="3"></rect>
			<text class="viz-axis-label" x="28" y="32">PARAMETERIZATION A · BROAD PROFILE</text>
			<path class="viz-axis" d="M42 44V116H332"></path>
			<path class="viz-roc-curve" d="M62 62 C98 62 111 110 180 110 C249 110 262 62 312 62"></path>
			<path class="viz-baseline" d="M42 110H332"></path>
			<circle class="viz-operating-point" cx="180" cy="110" r="4"></circle>
			<text class="viz-label" x="49" y="57">loss</text>
			<text class="viz-label" x="327" y="130" text-anchor="end">parameter perturbation Δθ</text>
			<text class="viz-callout" x="190" y="103">same minimum L*</text>
			<path class="viz-axis" d="M180 144V174"></path>
			<path class="viz-axis" d="M174 168L180 174L186 168"></path>
			<text class="viz-callout" x="190" y="153">reparameterize</text>
			<text class="viz-label" x="190" y="169">same function and predictions</text>
			<rect class="viz-plot-bg" x="12" y="184" width="336" height="124" rx="3"></rect>
			<text class="viz-axis-label" x="28" y="204">PARAMETERIZATION B · NARROW PROFILE</text>
			<path class="viz-axis" d="M42 216V288H332"></path>
			<path class="viz-roc-curve" d="M116 234 C143 234 147 282 180 282 C213 282 217 234 244 234"></path>
			<path class="viz-baseline" d="M42 282H332"></path>
			<circle class="viz-operating-point" cx="180" cy="282" r="4"></circle>
			<text class="viz-label" x="49" y="229">loss</text>
			<text class="viz-label" x="327" y="302" text-anchor="end">parameter perturbation Δφ</text>
			<text class="viz-callout" x="190" y="275">same minimum L*</text>
			<text class="viz-callout" x="180" y="322" text-anchor="middle">apparent sharpness changed; model behavior did not</text>
		</svg>
	</div>
	<figcaption><strong>Read it this way:</strong> the lower valley looks sharper, yet it represents the same predictor as the broad valley above. Therefore an optimizer gap may be empirical, but coordinate sharpness alone is not a parameterization-invariant explanation for it. Concept after <a href="https://arxiv.org/abs/1703.04933">Dinh et al. (2017)</a>; curves are illustrative, not measured.</figcaption>
</figure>

## Tells that get you a strong-hire vote

- You name **flat-vs-sharp minima** but acknowledge it's contested.
- You distinguish **AdamW from Adam** and name the weight-decay coupling fix.
- You give the **regime breakdown**: SGD competitive on CNNs, AdamW dominant on transformers/LLMs.
- You mention **LAMB** for very large batch training.

## Tells that get you down-leveled

- Asserting "SGD always generalizes better."
- Confusing Adam with AdamW.
- No mention of weight decay coupling.
- Suggesting SGD for LLM training.

## Common follow-up

"What learning rate schedule do you use with AdamW?"

The L6 answer:

> "For transformers: linear warmup over the first 1-5% of steps to peak LR, then cosine decay to ~10% of peak over the rest of training. Warmup is critical: Adam's second moment v is unreliable in the first few hundred steps, and full LR causes divergence. For very long training (LLMs trained for trillions of tokens), some recipes use constant LR after a brief warmup, then a final decay phase ('infinite' LR schedule). The exact shape matters less than having warmup and not over-decaying."

---

*Related: [Adam, AdamW, and modern optimizer choices](/concepts/adam-and-adamw/), [How do you choose a learning rate?](/questions/how-to-choose-learning-rate/), [Regularization](/concepts/regularization/).*
