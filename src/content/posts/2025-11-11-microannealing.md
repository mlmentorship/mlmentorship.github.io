---
title: "Microannealing and midtraining"
description: "Compare matched late-checkpoint continuations to test a data-mixture change, without confusing domain loss, task gains or full-training evidence."
date: "2025-11-11"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Microannealing tests a data-mixture change through short continuations from the same mostly trained state.
It measures a local intervention, not every possible full-training recipe.
Midtraining is broader: it includes changes to data, curriculum or training between larger training stages.

Use a hypothetical 1000-token continuation to compare base data with a 70% base, 30% candidate mixture.
The stated decision rule requires task score at least as high as control.
All numbers below are invented teaching values, not Marin measurements or recommended training budgets.

<!-- visual:microannealing-controlled-fork -->
<figure class="learning-figure plot-panel" aria-labelledby="microannealing-fork-title">
	<p class="visual-kicker">Hypothetical controlled comparison</p>
	<p class="visual-title" id="microannealing-fork-title">A better domain loss cannot override a failed preservation check</p><div class="visual-argument" data-visual-argument="microannealing-controlled-fork" data-argument-kind="matrix"><p class="visual-argument-premise">A domain gain cannot waive a preservation failure</p><svg viewBox="0 0 360 95" role="img" aria-label="A domain gain cannot waive a preservation failure. The probe improves domain loss but fails the declared preservation check. Promotion requires both the gain and preservation conditions."><title>A domain gain cannot waive a preservation failure</title><desc>The probe improves domain loss but fails the declared preservation check. Promotion requires both the gain and preservation conditions.</desc><text x="112.25" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="112.25" dy="0">domain</tspan></text><text x="203.35" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="203.35" dy="0">preserve</tspan></text><text x="294.45000000000005" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="294.45000000000005" dy="0">promote</tspan></text><text x="54.69999999999999" y="54" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="54.69999999999999" dy="0">probe</tspan></text><rect data-proof-cell="0,0" x="66.69999999999999" y="36" width="88.10000000000001" height="44" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke)"></rect><text x="110.75" y="63.5" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="110.75" dy="0">better</tspan></text><rect data-proof-cell="0,1" x="157.8" y="36" width="88.10000000000001" height="44" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke)"></rect><text x="201.85000000000002" y="63.5" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="201.85000000000002" dy="0">FAIL</tspan></text><path d="M161.8 76L241.90000000000003 40" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><rect data-proof-cell="0,2" x="248.9" y="36" width="88.10000000000001" height="44" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke)"></rect><text x="292.95" y="63.5" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="292.95" dy="0">hold</tspan></text><path d="M252.9 76L333 40" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
	<svg viewBox="0 0 360 520" role="img" aria-labelledby="microannealing-fork-svg-title microannealing-fork-svg-desc">
		<title id="microannealing-fork-svg-title">A controlled microannealing fork with conflicting evaluation signals</title>
		<desc id="microannealing-fork-svg-desc">Hypothetical teaching data, not Marin measurements. One saved checkpoint forks into matched 1000-token cooldowns. Control uses 1000 base tokens and obtains domain loss 2.0 and task score 0.60. Candidate uses 700 base and 300 candidate tokens and obtains loss 1.8 and task score 0.58. The candidate fails the prespecified task rule despite its lower domain loss.</desc>
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
		<text class="viz-callout" x="269" y="179" text-anchor="middle">70% base + 30% HQ</text>
		<text class="viz-node-value" x="269" y="197">one changed factor</text>
		<path d="M91 208V230H180M269 208V230H180V246" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2;stroke-dasharray:5 3;marker-end:url(#microannealing-focus-arrow)"></path>
		<rect class="viz-node" x="28" y="250" width="304" height="72" rx="4" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke)"></rect>
		<text class="viz-axis-label" x="180" y="270" text-anchor="middle">LOCKED IN BOTH RUNS</text>
		<text class="viz-node-value" x="180" y="291">same LR decay · 1000 tokens each</text>
		<text class="viz-node-value" x="180" y="309">same evaluation suite</text>
		<path d="M180 322V342" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#microannealing-arrow)"></path>
		<text class="viz-axis-label" x="18" y="363">HYPOTHETICAL RESULTS · CONTROL → CANDIDATE</text>
		<rect class="viz-node" x="18" y="377" width="324" height="92" rx="4"></rect>
		<text class="viz-label" x="30" y="400">MATCHED-DOMAIN LOSS</text>
		<text class="viz-callout" x="330" y="400" text-anchor="end">2.0 → 1.8</text>
		<path class="viz-gridline" d="M30 414H330"></path>
		<text class="viz-label" x="30" y="439">DOWNSTREAM TASKS</text>
		<text class="viz-callout" x="330" y="439" text-anchor="end">0.60 → 0.58</text>
		<path d="M180 469V487" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#microannealing-arrow)"></path>
		<rect x="72" y="490" width="216" height="27" rx="4" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:2"></rect>
		<text class="viz-callout" x="180" y="508" text-anchor="middle">reject under the task rule</text>
	</svg>
	<div class="visual-derivation-caption">both arms receive the same continuation budget. Compare both evaluation rows with control. Lower domain loss does not rescue a failed task criterion. These values are hypothetical.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="microannealing-controlled-fork"><strong>Check:</strong> Why is the candidate rejected when its matched-domain loss improves from 2.0 to 1.8?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="microannealing-controlled-fork">The probe improves domain loss but fails the declared preservation check. Promotion requires both the gain and preservation conditions.</span></figcaption>
</figure>

## The procedure

Freeze the comparison contract before running either arm:

- Starting state: model, optimizer, scheduler, random state and checkpoint digests.
- Data: versions, deduplication rules, mixture sampling, token counts and evaluation exclusion.
- Continuation: schedule, batch, precision, execution conditions and measured compute.
- Evaluation: suite versions, repetitions, metrics and decision criteria.

Change the mixture while holding the continuation design fixed.
Equal tokens do not automatically imply equal compute; record both.
Include failed runs under a stated rule, not only successful checkpoints.
For schedule context, see [WSD and WSD-S](/concepts/wsd-and-wsd-s/).

## Read all three hypothetical arms

| Arm | Base / HQ / instruction tokens | Domain loss | Task score |
| --- | --- | --- | --- |
| Control | $1000/0/0$ | $2.0$ | $0.60$ |
| Candidate | $700/300/0$ | $1.8$ | $0.58$ |
| Mixed candidate | $700/150/150$ | $1.9$ | $0.62$ |

The candidate fails the task rule. The mixed candidate is promising, but needs independent confirmation after selection.
Comparing either cooldown with the uncooled starting checkpoint would confound mixture and continuation effects.

## Change the experimental question

### 1. Change the substitution ratio

**New task:** choose between 15% and 30% candidate data.

**Picture and change:** start both arms from the same state with the same total continuation budget.
Specify the ratio comparison before evaluation and reserve confirmation evidence.

**Check:** 1000 tokens contain 150 versus 300 candidate tokens.
Increasing total tokens would introduce another changed factor.

**Use when:** the source appears useful but its substitution rate is uncertain.

### 2. Choose a from-scratch recipe

**New task:** decide whether the candidate belongs in pretraining from initialization.

**Picture and change:** use the late probe as limited evidence, then design an appropriately scaled full-training comparison.
Checkpoint and horizon interactions may change the result.

**Check:** a cooldown win from one late state does not establish a win from random initialization.

**Use when:** the actual decision exceeds the probe's starting state or continuation regime.

## Matched budgets do not prevent leakage

A candidate can improve an evaluation contaminated by its added source.
Deduplicate and check evaluation overlap; equal token counts do not repair leakage.

## Try explaining the mixed candidate

Does HQ plus instruction data winning prove that instruction format caused the gain?

<details>
<summary>Check your answer</summary>

No. Content, coverage and format changed together.
Use matched alternatives that distinguish those explanations, then confirm on independent evidence.

</details>

<!-- learning:core-end -->

<div data-learning-extension>

## What the experiments actually showed (Marin)

The [Marin 8B retrospective](https://marin.readthedocs.io/en/latest/reports/marin-8b-retro/#interlude-microannealing) reports three local findings:

1. Oversampling HQ sources improved HQ-domain loss but degraded general task performance.
2. Within its experiment budget, 70% base / 15% FLAN / 15% HQ gave its best microannealing results.
3. A 70% base / 30% FLAN mix underperformed the base-only control.

The authors propose lost few-shot-friendly structure as an explanation, not an isolated causal result.
The retrospective points to [issue 784](https://github.com/marin-community/marin/issues/784) and [issue 820](https://github.com/marin-community/marin/issues/820).
Its ratios are experiment-specific, not universal recommendations.

## Common pitfalls

Probe cost, useful horizon and predictive transfer depend on model, checkpoint, source and evaluation.
No fixed token budget or fraction guarantees a reliable signal.
Repeated searches on one evaluation suite also create selection bias; reserve confirmation evidence.

## When to use it vs. full ablations

| Situation | Use |
|---|---|
| Deciding whether to include a candidate dataset in your final cooldown mix | Microannealing |
| Comparing two pretraining recipes from scratch | Full ablation at small scale |
| Tuning the mixing ratio of a known-good dataset | Microannealing |
| Comparing optimizers, architectures, or LR schedules | Full ablation, microannealing is not a substitute |

Changing architecture or optimizer introduces a different experimental question.
The label microannealing does not make such comparisons controlled.

## What an interviewer expects you to say

Specify the intervention, matched control, decision metric and scope of the conclusion.
Explain why the tested checkpoint and horizon are relevant to the actual data decision.

## Further reading

- [Marin 8B retrospective](https://marin.readthedocs.io/en/latest/reports/marin-8b-retro/), "Interlude: Microannealing" section.
- [Marin GH#784](https://github.com/marin-community/marin/issues/784) (high-quality data annealing experiments).
- [Marin GH#820](https://github.com/marin-community/marin/issues/820) (evaluating HQ datasets for cooldown).
- [Olmo 2 technical report](https://arxiv.org/abs/2501.00656) for the original "midtraining" framing.
- [DBRX blog post](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm) for an industry-side description of the same technique.

</div>
