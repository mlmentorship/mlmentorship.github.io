---
title: "WSD and WSD-S learning rate schedules"
description: "Warmup-Stable-Decay keeps the learning rate flat before a final decay. WSD-S adds decay-and-rewarm probes when the final token budget is uncertain."
date: "2025-08-31"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

**WSD** (Warmup-Stable-Decay) is a three-phase schedule: warm up to a peak LR, hold at that peak for most of training, then decay sharply at the end.

**WSD-S** (Warmup-Stable-Decay-Stable) extends WSD with **periodic short decay-and-rewarm cycles** during the stable phase, used to probe model performance without committing to a final cooldown.

Both schedules differ from cosine decay in one critical way: the schedule is not parameterized by total training horizon. You can decide to keep going at any point.

Cosine decay (the dominant default for pretraining circa 2022) requires knowing the total training horizon $T$ upfront, because the curve $\eta(t) = \tfrac{1}{2}(1 + \cos(\pi t/T)) \cdot \eta_{\max}$ depends on $T$ explicitly. If you decide to extend training past $T$, you have to re-parameterize the schedule and either restart the cosine or splice in something new.

WSD removes the dependency. You hold the LR flat for as long as you want and decide to cool down whenever you stop. WSD-S goes further: it lets you sample model quality at intermediate points by doing brief decay cycles, then rewarming. This is what enabled Marin's reactive ("Tootsie Roll") pretraining strategy, where they extended the 8B run from a planned 4T tokens to an actual 12.7T tokens across multiple unplanned data mixture changes.

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

Same structure, but the stable phase contains periodic decay-and-rewarm cycles:

```
warmup -> stable -> short decay -> rewarm -> stable -> short decay -> rewarm -> ... -> final decay
```

Each short decay cycle decays the LR by some factor (e.g., 10x), holds briefly, then rewarms back to peak. During the decay portion, you can run evals to get a "what would this model look like if I cooled down right now" signal without paying the full cost of a final cooldown.

In Marin's 8B run, the cycle was: every 20K steps, decay over 2K steps (so ~10% of steps spent decayed). The rest was at peak LR.

## When to use each

| Situation | Schedule |
|---|---|
| Fixed total budget known upfront, single planned run | Cosine |
| Reactive pretraining, may extend the run | WSD |
| Reactive pretraining, want to probe quality mid-run without final cooldown | WSD-S |
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
3. Describe the WSD-S extension as periodic decay-and-rewarm probes during the stable phase.
4. Note that final loss is comparable to cosine; the advantage is operational flexibility.
5. Bonus: mention the mid-training data mix change (e.g., adding HQ data during cooldown) that WSD enables.

## Further reading

- [Hu et al., 2024 (MiniCPM)](https://arxiv.org/abs/2404.06395) introduced WSD.
- [Wen et al., 2024](https://arxiv.org/abs/2410.05192) introduced WSD-S and provided the river-and-hill loss decomposition.
- [Marin 8B retrospective](https://marin.readthedocs.io/en/latest/reports/marin-8b-retro/) for a full case study of WSD-S used in production.
- [Cosine decay reference](/concepts/learning-rate-schedules/) for the default it replaces.
