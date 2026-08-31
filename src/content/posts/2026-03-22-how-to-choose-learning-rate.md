---
title: "How do you choose a learning rate?"
description: "The right answer is a procedure, not a number. The wrong answers are 'use the default' and 'try a few values.'"
date: "2026-03-22"
draft: false
tags: ["questions"]
category: "questions"
---


> *Asked in: ML breadth at every level.*

The right answer is a procedure, not a number. The wrong answers are "use the framework default" and "try a few values."

## What an L4 answer sounds like

> "I'd start with 1e-3 for Adam or 0.1 for SGD, then try a few values and pick the one that works best on validation."

Right number ranges, no method. You've copy-pasted from tutorials.

## What an L5 answer sounds like

> "Learning rate is the single most impactful hyperparameter, so I'd be deliberate about it. My standard procedure:
>
> **For a new model on a new dataset**: run an LR range test. Train for 200-500 steps with LR linearly increasing from 1e-7 to 1e-1. Plot loss vs LR on a log axis. The right LR is roughly in the steepest descent region, with a safety margin (1/3 of the LR where loss starts blowing up).
>
> **For SGD with momentum**: typically 0.01-0.1 with cosine annealing or step decay.
>
> **For Adam / AdamW**: typically 1e-4 to 1e-3 for most tasks; lower (1e-5 to 5e-5) for fine-tuning a pretrained model.
>
> **Schedule matters more than initial value**. A linear warmup over the first ~5% of steps followed by cosine decay to ~10% of the peak is the standard transformer recipe. Without warmup, large transformers diverge in the first hundred steps; the warm-up gives Adam's moment estimates time to stabilize.
>
> **For LLM pretraining**: peak LRs are surprisingly small (1e-4 to 6e-4) at standard scales, with decay to ~10% of peak. The Chinchilla-style training recipes use cosine decay over the full training horizon."

This is L5. You've named the LR range test, the schedule structure, the per-optimizer ranges, and the LLM-specific recipe.

<!-- visual:learning-rate-range-to-schedule -->
<figure class="learning-figure" aria-labelledby="learning-rate-procedure-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="learning-rate-procedure-title">Turn a range test into a peak learning rate, then schedule around it</p>
	<div class="visual-grid--two" role="group" aria-label="Two-stage learning-rate selection procedure">
		<section class="visual-panel plot-panel">
			<svg viewBox="0 0 300 238" role="img" aria-labelledby="lr-range-title lr-range-desc">
				<title id="lr-range-title">Read a learning-rate range test from too small through useful descent to instability</title>
				<desc id="lr-range-desc">An original qualitative plot shows training loss against learning rate on a logarithmic horizontal axis. At very small rates the loss is nearly flat. It then falls, with a hatched candidate band around the sustained steep descent. Farther right the curve bottoms and turns sharply upward in an instability region marked with crosses. The selected candidate is before the minimum and blow-up, not at either one.</desc>
				<defs>
					<pattern id="lr-candidate-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
						<line x1="0" y1="0" x2="0" y2="6" style="stroke:var(--viz-focus-stroke);stroke-width:2"></line>
					</pattern>
				</defs>
				<rect class="viz-plot-bg" x="38" y="28" width="248" height="158" rx="4"></rect>
				<rect x="125" y="28" width="64" height="158" style="fill:url(#lr-candidate-hatch);opacity:.24"></rect>
				<path class="viz-gridline" d="M38 81H286M38 133H286M100 28V186M162 28V186M224 28V186"></path>
				<path class="viz-axis" d="M38 28V186H286"></path>
				<path class="viz-pr-curve" d="M45 61C76 62 94 60 112 70C134 83 151 112 171 139C187 160 205 169 218 160C235 149 246 103 259 57C267 35 275 30 282 34"></path>
				<circle class="viz-operating-point" cx="158" cy="121" r="5"></circle>
				<path class="viz-operating-guide" d="M158 121V186"></path>
				<path d="M252 47L262 57M262 47L252 57M267 31L277 41M277 31L267 41" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:2"></path>
				<text class="viz-axis-label" x="42" y="18">1 · SWEEP LR UP; WATCH LOSS</text>
				<text class="viz-label" x="69" y="48" text-anchor="middle">too small</text>
				<text class="viz-callout" x="157" y="102" text-anchor="middle">candidate</text>
				<text class="viz-callout" x="157" y="115" text-anchor="middle">steep descent</text>
				<text class="viz-label" x="247" y="82" text-anchor="middle">unstable</text>
				<text class="viz-label" x="38" y="204">10⁻⁷</text>
				<text class="viz-label" x="151" y="204">10⁻⁴</text>
				<text class="viz-label" x="268" y="204">10⁻¹</text>
				<text class="viz-axis-label" x="162" y="228" text-anchor="middle">learning rate (log scale) →</text>
				<text class="viz-axis-label" transform="translate(14 132) rotate(-90)">training loss</text>
			</svg>
		</section>
		<section class="visual-panel plot-panel">
			<svg viewBox="0 0 300 238" role="img" aria-labelledby="lr-schedule-title lr-schedule-desc">
				<title id="lr-schedule-title">Use the selected candidate as the peak of a warmup and decay schedule</title>
				<desc id="lr-schedule-desc">A learning-rate schedule begins low, rises linearly during a short warmup to the candidate peak selected by the range test, then follows a curved decay over the remaining training steps to a lower final rate. A vertical dotted guide marks the end of warmup, and the peak has a double-ring marker. The figure emphasizes that the selected value is a peak, not a constant rate used for every step.</desc>
				<rect class="viz-plot-bg" x="38" y="28" width="248" height="158" rx="4"></rect>
				<path class="viz-gridline" d="M38 81H286M38 133H286M100 28V186M162 28V186M224 28V186"></path>
				<path class="viz-axis" d="M38 28V186H286"></path>
				<path d="M45 174L89 58C132 60 174 70 211 92C241 110 263 133 282 154" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2.5"></path>
				<path class="viz-operating-guide" d="M89 42V186"></path>
				<circle class="viz-node viz-node--focus" cx="89" cy="58" r="8"></circle>
				<circle class="viz-node" cx="89" cy="58" r="4"></circle>
				<text class="viz-axis-label" x="42" y="18">2 · USE IT AS THE PEAK</text>
				<text class="viz-callout" x="65" y="104" text-anchor="middle">warmup</text>
				<text class="viz-callout" x="89" y="43" text-anchor="middle">selected peak</text>
				<text class="viz-callout" x="205" y="78" text-anchor="middle">decay</text>
				<text class="viz-label" x="57" y="204" text-anchor="middle">start</text>
				<text class="viz-label" x="89" y="204" text-anchor="middle">warmup end</text>
				<text class="viz-label" x="274" y="204" text-anchor="middle">finish</text>
				<text class="viz-axis-label" x="162" y="228" text-anchor="middle">training step →</text>
				<text class="viz-axis-label" transform="translate(14 132) rotate(-90)">learning rate</text>
			</svg>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> sweep from left to right on the first plot. Ignore the flat region, then choose inside the sustained descent with margin before the loss turns upward; the minimum and blow-up are warning boundaries, not targets. In the second plot, that candidate becomes a peak reached after warmup and reduced during decay. Re-run the test when the model, optimizer, data, batch, or training regime changes, and confirm the candidate with a short controlled run. Original qualitative schematic informed by <a href="https://arxiv.org/abs/1506.01186">Smith (2017)</a> and the <a href="https://docs.fast.ai/callback.schedule.html#lrfind">fastai learning-rate finder documentation</a>.</figcaption>
</figure>

## What an L6 answer sounds like

> "...and a few practical things I've learned:
>
> **The right LR depends on batch size.** The linear scaling rule (LR proportional to batch size) holds approximately for SGD but not for Adam. For Adam, the LR-batch relationship is roughly square-root. Doubling batch size, you'd want roughly 1.4&times; the LR for Adam.
>
> **Warmup matters more for adaptive optimizers.** Adam's per-parameter step size depends on the running variance estimate, which is unreliable for the first few hundred steps. Without warmup, you can get huge effective steps that destabilize training. SGD doesn't have this issue but benefits from warmup for other reasons (the gradient direction is noisy at the start).
>
> **For fine-tuning vs from-scratch, the LR is dramatically different.** Pretrained models have well-conditioned weights; large LRs will destroy them. The standard recipe is 5-100&times; smaller than from-scratch.
>
> **For LLM SFT and RLHF, even smaller.** Standard SFT recipes use LRs in the 1e-6 to 5e-5 range. RLHF/DPO is even more delicate; some papers use 1e-7. The right LR here is empirically determined and varies a lot by base model.
>
> **Cyclical LR or one-cycle policy** can sometimes beat cosine decay, especially for shorter training runs. Worth trying if cosine isn't giving great results.
>
> **The LR is downstream of the loss landscape.** A lot of what looks like 'wrong LR' is actually 'wrong initialization' or 'bad loss surface' or 'gradient pathology elsewhere.' If you find yourself needing very strange LRs to make training work, the LR isn't the issue."

This is L6. You're past the recipe and into the operational reality.

## The tells that get you a strong-hire vote

- You describe **a procedure** (LR range test) for picking the LR, not a guess.
- You distinguish **SGD vs Adam** ranges and **from-scratch vs fine-tuning** ranges.
- You bring up **the schedule** (warmup + decay) as separately important.
- You mention **LR-batch-size scaling**: especially the Adam-specific square-root rule.
- You acknowledge that **the LR is often downstream of other issues**.

## The tells that get you down-leveled

- Single number answer ("1e-3").
- No mention of warmup for transformers.
- "Just use the default in the framework", no calibration.
- Same LR for from-scratch and fine-tuning.
- No mention of schedule.

---

*Related: [How would you debug a model that's not learning?](/questions/debug-model-not-learning/), [Walk me through the bias-variance tradeoff](/questions/bias-variance-tradeoff/).*
