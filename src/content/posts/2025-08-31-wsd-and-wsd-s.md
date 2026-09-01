---
title: "WSD and WSD-S learning rate schedules"
description: "Warmup-Stable-Decay keeps the learning rate flat before a final decay. WSD-S adds single-path decay-and-return checkpoints when the final token budget is uncertain."
date: "2025-08-31"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

**WSD** (Warmup-Stable-Decay) is a three-phase schedule: warm up to a peak LR, hold at that peak for most of training, then decay sharply at the end.

**WSD-S** (Warmup-Stable-Decay-Simplified) replaces WSD's separate cooldown branches with **periodic short decay-and-return cycles** in one continuous run, producing cooled intermediate checkpoints without fixing the final training horizon.

Both schedules differ from cosine decay in one critical way: the schedule is not parameterized by total training horizon. You can decide to keep going at any point.

Cosine decay (the dominant default for pretraining circa 2022) requires knowing the total training horizon $T$ upfront, because the curve $\eta(t) = \tfrac{1}{2}(1 + \cos(\pi t/T)) \cdot \eta_{\max}$ depends on $T$ explicitly. If you decide to extend training past $T$, you have to re-parameterize the schedule and either restart the cosine or splice in something new.

WSD removes the dependency. You hold the LR flat for as long as you want and decide to cool down whenever you stop. To produce an intermediate cooled model with WSD, you branch from the hot stable run, decay the copy, and then resume the unchanged hot branch. WSD-S removes that rollback: after each brief decay, it raises the LR and continues from the cooled checkpoint's weights. This is what enabled Marin's reactive ("Tootsie Roll") pretraining strategy, where they extended the 8B run from a planned 4T tokens to an actual 12.7T tokens across multiple unplanned data mixture changes.

<!-- visual:wsd-checkpoint-lineage -->
<figure class="learning-figure plot-panel" aria-labelledby="wsd-lineage-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="wsd-lineage-title">Which weights continue after an intermediate cooldown?</p>
	<svg viewBox="0 0 360 378" role="img" aria-labelledby="wsd-lineage-svg-title wsd-lineage-svg-desc">
		<title id="wsd-lineage-svg-title">Checkpoint lineage under WSD and WSD-S</title>
		<desc id="wsd-lineage-svg-desc">Two normalized learning-rate traces compare checkpoint lineage. In WSD, a dashed cooldown branch creates a cooled copy while the hot main branch remains at peak learning rate and continues toward a later final decay. In WSD-S, one path decays to cooled checkpoint one, returns the learning rate to peak while retaining those weights, then repeats for checkpoint two before the final decay. Labels, square cooled checkpoints, a circular hot checkpoint, and solid versus dashed paths carry the distinction without color.</desc>
		<text class="viz-axis-label" x="16" y="21">WSD · KEEP A HOT MAIN BRANCH</text>
		<rect class="viz-plot-bg" x="40" y="34" width="304" height="132" rx="3"></rect>
		<path class="viz-gridline" d="M40 68H344M40 142H344"></path><path class="viz-axis" d="M40 30V166H348"></path>
		<path d="M48 142L76 68H316L340 142" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:3;stroke-linecap:round;stroke-linejoin:round"></path>
		<path d="M178 68C190 91 201 119 216 142" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:3;stroke-dasharray:6 4;stroke-linecap:round"></path>
		<circle cx="178" cy="68" r="5" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:2.5"></circle><rect x="211" y="137" width="10" height="10" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2.5"></rect>
		<text class="viz-callout" x="112" y="57">hot main branch continues</text><text class="viz-label" x="224" y="132">cooled copy</text><text class="viz-label" x="306" y="157">final</text>
		<text class="viz-label" x="34" y="72" text-anchor="end">peak</text><text class="viz-label" x="34" y="146" text-anchor="end">low</text>
		<text class="viz-axis-label" x="16" y="207">WSD-S · CONTINUE THE COOLED WEIGHTS</text>
		<rect class="viz-plot-bg" x="40" y="220" width="304" height="132" rx="3"></rect>
		<path class="viz-gridline" d="M40 254H344M40 328H344"></path><path class="viz-axis" d="M40 216V352H348"></path>
		<path d="M48 328L76 254H145L168 328" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:3;stroke-linecap:round;stroke-linejoin:round"></path><path d="M168 328L178 254" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:3;stroke-dasharray:3 4"></path>
		<path d="M178 254H244L268 328" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:3;stroke-linecap:round;stroke-linejoin:round"></path><path d="M268 328L278 254" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:3;stroke-dasharray:3 4"></path><path d="M278 254H316L340 328" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:3;stroke-linecap:round;stroke-linejoin:round"></path>
		<rect x="163" y="323" width="10" height="10" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2.5"></rect><rect x="263" y="323" width="10" height="10" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2.5"></rect>
		<text class="viz-label" x="168" y="345" text-anchor="middle">checkpoint 1</text><text class="viz-label" x="268" y="345" text-anchor="middle">checkpoint 2</text><text class="viz-callout" x="224" y="243" text-anchor="middle">same weights · LR back to peak</text>
		<text class="viz-label" x="34" y="258" text-anchor="end">peak</text><text class="viz-label" x="34" y="332" text-anchor="end">low</text><text class="viz-axis-label" x="180" y="372" text-anchor="middle">training progress →</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> in WSD, the dashed probe cools a copy while training can continue from the circular checkpoint on the hot main branch. In WSD-S, each square is on the only path: keep that cooled checkpoint's weights, raise only its learning rate back to the peak, and continue. The final downward segment is the terminal cooldown. This original schematic is checked against the <a href="https://arxiv.org/abs/2404.06395">MiniCPM WSD method</a> and the <a href="https://arxiv.org/abs/2410.05192">WSD-S definition</a>.</figcaption>
</figure>

## The mechanism

### WSD

Three phases, with $W$ = warmup steps, $S$ = stable steps, $D$ = decay steps:

$$
\eta(t) = \begin{cases}
\eta_{\max} \cdot t / W & t \le W \quad \text{(warmup)} \\
\eta_{\max} & W < t \le W + S \quad \text{(stable)} \\
\eta_{\min} + (\eta_{\max} - \eta_{\min}) \cdot f((t - W - S) / D) & t > W + S \quad \text{(decay)}
\end{cases}
$$

The decay function $f$ is typically linear or 1-sqrt. The decay phase is usually short: 10-20% of total steps.

The key property: the model can be considered "trained" at any point during the stable phase by initiating a decay. There is no fixed end.

### WSD-S

WSD-S uses one path with periodic decay-and-return cycles:

```
warmup -> stable -> short decay -> return to peak -> stable -> short decay -> return to peak -> ... -> final decay
```

Each short decay cycle lowers the LR by some factor (e.g., 10x) and yields a cooled checkpoint for evaluation. Training then continues from those weights at the peak LR. In the schedule defined by [Wen et al. (2024)](https://arxiv.org/abs/2410.05192), the LR returns directly to the peak outside each decay interval; a gradual rewarm is not a required WSD-S phase. This gives a "what does the model look like cooled down right now?" signal without ending the run or rolling back to a separate hot checkpoint.

In Marin's 8B run, the cycle was: every 20K steps, decay over 2K steps (so ~10% of steps spent decayed). The rest was at peak LR.

## When to use each

| Situation | Schedule |
|---|---|
| Fixed total budget known upfront, single planned run | Cosine |
| Reactive pretraining, may extend the run | WSD |
| Reactive pretraining, want cooled intermediate checkpoints in one continuous run | WSD-S |
| Exploratory training where you want checkpoints that are individually deployable | WSD or WSD-S |

For SFT or fine-tuning, the standard remains cosine decay over the planned epochs. WSD and WSD-S are pretraining-specific.

## Empirical findings worth knowing

- WSD and cosine give comparable final loss when both use the same total compute and final LR. The advantage of WSD is operational, not numerical.
- WSD-S decay cycles produce a "river and hill" decomposition of the loss curve: the river is the underlying trend, the hill is the variance from being at high LR. Cooling temporarily reveals the river. This is a useful diagnostic on its own.
- When you finally do the long final decay in a WSD or WSD-S run, mixing in higher-quality data during the cooldown gives a meaningful boost. Marin and Olmo 2 both report this. The cooldown is also the right time to introduce small fractions of FLAN-style instruction data to improve few-shot performance.

## Common pitfalls

- **Choosing too high a peak LR.** Because WSD spends almost all of training at the peak, instability that would have been masked by cosine's quick descent is exposed. Marin used $1.0 \times 10^{-3}$ for the 8B run, lower than the DCLM paper's recommended $2.0 \times 10^{-3}$ which they found unstable.
- **Forgetting to use z-loss.** During deep WSD or WSD-S cooldowns, the `lm_head` can slowly explode. See the [z-loss reference](/concepts/z-loss/).
- **Comparing WSD-S decay-cycle losses to cosine end-of-training losses.** WSD-S decay cycles show the model partway through training; cosine end-of-training losses show the final model. The numbers are not directly comparable.

## What an interviewer expects you to say

If asked about WSD or WSD-S:

1. Frame the motivation: cosine requires knowing $T$ upfront, WSD doesn't.
2. Describe the three phases of WSD (warmup, stable at peak, decay at end).
3. Describe WSD-S as a single path with periodic decay checkpoints followed by a return to peak LR, continuing from the cooled weights.
4. Note that final loss is comparable to cosine; the advantage is operational flexibility.
5. Bonus: mention the mid-training data mix change (e.g., adding HQ data during cooldown) that WSD enables.

## Further reading

- [Hu et al., 2024 (MiniCPM)](https://arxiv.org/abs/2404.06395) introduced WSD.
- [Wen et al., 2024](https://arxiv.org/abs/2410.05192) introduced WSD-S and provided the river-and-hill loss decomposition.
- [Marin 8B retrospective](https://marin.readthedocs.io/en/latest/reports/marin-8b-retro/) for a full case study of WSD-S used in production.
- [Cosine decay reference](/concepts/learning-rate-schedules/) for the default it replaces.
