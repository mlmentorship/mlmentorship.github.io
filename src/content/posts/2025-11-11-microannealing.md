---
title: "Microannealing and midtraining"
description: "A short cooldown applied to a mostly-trained checkpoint with a small fraction of candidate data mixed in. The standard mid-training probe for whether a new dataset is worth including."
date: "2025-11-11"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

**Microannealing** is a short, low-cost cooldown applied to a mostly-trained checkpoint, with a small fraction (typically 15-30%) of a candidate dataset mixed into the otherwise-normal pretraining mix. The downstream task performance and per-domain loss of the resulting model are compared against a control microannealing run on 100% of the normal mix.

**Midtraining** is the broader umbrella: any technique that intervenes between the bulk of pretraining and the final cooldown, including microannealing, data mixture changes, and curriculum adjustments.

Data-mix decisions during pretraining are almost impossible to evaluate cheaply. You can't fully retrain to test each candidate, and small-scale ablations don't transfer because data effects compound over training and only become visible at low LR. Microannealing gives you a real signal at less than 1% of full-run cost, by exploiting the fact that cooldown is the regime where data choices actually matter most.

This is the technique behind decisions like "include FineMath in the mix," "drop Wikipedia," or "upweight code data," in modern open releases (Olmo 2, Llama 3, Marin 8B).

## The procedure

1. Take a checkpoint that's already trained on most of your token budget, ideally one near the end of the stable phase of a WSD or WSD-S schedule. (See the [WSD reference](/concepts/wsd-and-wsd-s/).)
2. Define a candidate data mix: typically 70% of your normal pretraining distribution and 30% of the data source you want to evaluate. (Olmo 2 uses 50/50; Marin found 70/30 worked better in their setting.)
3. Run a short cooldown on this mix, on the order of 1-10B tokens, decaying LR from the current operating point down to the planned final LR.
4. Run a parallel control: same starting checkpoint, same cooldown shape, same number of tokens, but using 100% of your normal mix.
5. Evaluate both on downstream tasks (MMLU, HellaSwag, GSM8K, etc.), not just on per-domain perplexity.
6. Decide based on task performance, not loss.

<!-- visual:microannealing-controlled-fork -->
<figure class="learning-figure plot-panel" aria-labelledby="microannealing-fork-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="microannealing-fork-title">Hold the cooldown fixed, change only the data mix, and let downstream tasks veto a misleading loss win.</p>
	<svg viewBox="0 0 360 520" role="img" aria-labelledby="microannealing-fork-svg-title microannealing-fork-svg-desc">
		<title id="microannealing-fork-svg-title">A controlled microannealing fork with conflicting evaluation signals</title>
		<desc id="microannealing-fork-svg-desc">One mostly trained checkpoint forks into two disposable cooldown runs. The control uses one hundred percent of the normal pretraining mix. The candidate example uses seventy percent pretraining data and thirty percent high-quality data. Both runs keep the starting weights, learning-rate decay, token budget, and evaluation suite identical. In the Marin high-quality-only experiment, the candidate improved matched-domain validation loss but worsened downstream task performance relative to control, so the data mix was rejected despite its loss win.</desc>
		<defs>
			<marker id="microannealing-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path class="viz-arrow-forward" d="M0,0 L7,3.5 L0,7 Z"></path></marker>
			<marker id="microannealing-focus-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" style="fill:var(--viz-focus-stroke)"></path></marker>
		</defs>
		<text class="viz-axis-label" x="18" y="22">ONE LATE CHECKPOINT · TWO MATCHED COOLDOWNS</text>
		<rect class="viz-node viz-node--input" x="64" y="38" width="232" height="52" rx="4"></rect>
		<text class="viz-callout" x="180" y="58" text-anchor="middle">mostly trained checkpoint</text>
		<text class="viz-node-value" x="180" y="78">same weights and optimizer state</text>
		<path d="M180 90V112M180 112H91V132M180 112H269V132" style="fill:none;stroke:var(--viz-edge);stroke-width:2;stroke-dasharray:5 3;marker-end:url(#microannealing-arrow)"></path>
		<rect class="viz-node" x="18" y="136" width="146" height="72" rx="4"></rect>
		<text class="viz-axis-label" x="91" y="157" text-anchor="middle">CONTROL</text>
		<text class="viz-callout" x="91" y="179" text-anchor="middle">100% normal mix</text>
		<text class="viz-node-value" x="91" y="197">reference cooldown</text>
		<rect class="viz-node viz-node--focus" x="196" y="136" width="146" height="72" rx="4"></rect>
		<text class="viz-axis-label" x="269" y="157" text-anchor="middle">CANDIDATE EXAMPLE</text>
		<text class="viz-callout" x="269" y="179" text-anchor="middle">70% PT + 30% HQ</text>
		<text class="viz-node-value" x="269" y="197">one changed factor</text>
		<path d="M91 208V230H180M269 208V230H180V246" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2;stroke-dasharray:5 3;marker-end:url(#microannealing-focus-arrow)"></path>
		<rect class="viz-node" x="28" y="250" width="304" height="72" rx="4" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke)"></rect>
		<text class="viz-axis-label" x="180" y="270" text-anchor="middle">LOCKED IN BOTH RUNS</text>
		<text class="viz-node-value" x="180" y="291">same LR decay · same token budget</text>
		<text class="viz-node-value" x="180" y="309">same evaluation suite</text>
		<path d="M180 322V342" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#microannealing-arrow)"></path>
		<text class="viz-axis-label" x="18" y="363">MARIN HQ-ONLY RESULT RELATIVE TO CONTROL</text>
		<rect class="viz-node" x="18" y="377" width="324" height="92" rx="4"></rect>
		<text class="viz-label" x="30" y="400">MATCHED-DOMAIN LOSS</text>
		<text class="viz-callout" x="330" y="400" text-anchor="end">lower · candidate looks better</text>
		<path class="viz-gridline" d="M30 414H330"></path>
		<text class="viz-label" x="30" y="439">DOWNSTREAM TASKS</text>
		<text class="viz-callout" x="330" y="439" text-anchor="end">worse · candidate loses</text>
		<path d="M180 469V487" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#microannealing-arrow)"></path>
		<rect x="72" y="490" width="216" height="27" rx="4" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:2"></rect>
		<text class="viz-callout" x="180" y="508" text-anchor="middle">reject the HQ-only mix</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> fork both runs from the same late checkpoint and lock the schedule, token budget, and eval suite; the data mixture should be the only changed factor. Then read both result rows. In Marin's HQ-only probes, validation loss on similar HQ domains improved while downstream tasks regressed, so loss alone would have selected the wrong mix. This is an original schematic checked against <a href="https://github.com/marin-community/marin/issues/820">Marin's experiment record</a>.</figcaption>
</figure>

## What the experiments actually showed (Marin)

Three findings from [GH#784](https://github.com/marin-community/marin/issues/784) and [GH#820](https://github.com/marin-community/marin/issues/820), worth memorizing because they are counterintuitive:

1. **Naive HQ oversampling helped HQ eval loss but hurt task performance.** Sources like ArXiv, Wikipedia, and peS2o improved loss on Paloma's HQ subsets but degraded MMLU, HellaSwag, etc. The reason is that HQ web data lacks the few-shot-learning-inducing structure (multiple-choice Q&A, instruction-like patterns) that broad web text contains. By replacing 30% of the pretraining mix with HQ data, you lose that structure faster than the HQ data can compensate.
2. **Adding FLAN to the HQ mix recovered task performance.** A 70% PT / 15% FLAN / 15% HQ mix beat both 100% PT and 70% PT / 30% HQ. FLAN reintroduces the format diversity that HQ alone removes.
3. **70% PT / 30% FLAN alone underperformed 100% PT.** FLAN is repetitive enough that pure FLAN substitution hurts more than it helps. The lesson is that FLAN is a multiplier on HQ data, not a substitute for the broader web.

The general principle: microannealing measures format diversity as much as it measures quality. Plan your mix and your interpretation accordingly.

## Common pitfalls

- **Evaluating only on perplexity.** Per-domain loss almost always improves when you mix in similar data, even when downstream tasks regress. Always measure tasks.
- **Microannealing too early in training.** If you do this before the model is mostly trained, the cooldown isn't yet a meaningful proxy for the final model's behavior. Wait until you're past the stable phase.
- **Microannealing too short.** The cooldown needs to be long enough for the model to actually adapt to the new mix. 1-2B tokens is usually too short for meaningful task signal; 5-10B is more reliable.
- **Skipping the control run.** Without a 100% PT control on the same checkpoint, you can't separate the effect of the candidate data from the effect of the cooldown itself. The control is non-negotiable.

## When to use it vs. full ablations

| Situation | Use |
|---|---|
| Deciding whether to include a candidate dataset in your final cooldown mix | Microannealing |
| Comparing two pretraining recipes from scratch | Full ablation at small scale |
| Tuning the mixing ratio of a known-good dataset | Microannealing |
| Comparing optimizers, architectures, or LR schedules | Full ablation, microannealing is not a substitute |

Microannealing is specifically a data-evaluation tool. It's not a general substitute for ablation studies on architecture or optimization choices.

## What an interviewer expects you to say

If asked "how would you decide whether to include this dataset in your pretraining mix?":

1. State that the L5 answer (run a small ablation from scratch) doesn't transfer well, because data effects compound and become most visible at low LR late in training.
2. Describe microannealing: take a near-final checkpoint, run a short cooldown with the candidate data mixed in, compare against a control on the same checkpoint with 100% of the normal mix.
3. Specify that you evaluate on downstream tasks, not per-domain perplexity, because the two often disagree.
4. Bonus: note the FLAN finding, that format diversity matters separately from data quality.

## Further reading

- [Marin 8B retrospective](https://marin.readthedocs.io/en/latest/reports/marin-8b-retro/), "Interlude: Microannealing" section.
- [Marin GH#784](https://github.com/marin-community/marin/issues/784) (high-quality data annealing experiments).
- [Marin GH#820](https://github.com/marin-community/marin/issues/820) (evaluating HQ datasets for cooldown).
- [Olmo 2 technical report](https://arxiv.org/abs/2501.00656) for the original "midtraining" framing.
- [DBRX blog post](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm) for an industry-side description of the same technique.
