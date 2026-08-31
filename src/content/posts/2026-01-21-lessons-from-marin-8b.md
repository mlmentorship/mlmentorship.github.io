---
title: "Lessons from Marin 8B: what an open pretraining log actually teaches you"
description: "Marin trained the first open-source 8B model to beat Llama 3.1 8B and published every mistake. The transferable lessons aren't about TPUs. They're about how to run pretraining like a science."
date: "2026-01-21"
draft: false
tags: ["guides"]
category: "guides"
---

**The most useful pretraining writeup of the last year is the [Marin 8B retrospective](https://marin.readthedocs.io/en/latest/reports/marin-8b-retro/), and it's useful precisely because it isn't sanitized.** Marin is Stanford CRFM's open lab. Every experiment is a preregistered GitHub issue, every run is a reproducible PR, and the retrospective walks through the mistakes alongside the wins. The team trained the first open-source 8B base model to beat Llama 3.1 8B on 14 of 19 standard benchmarks, and they did it reactively, changing the data mix, optimizer, and schedule mid-run.

If you read only one thing on practical pretraining in 2026, read that retrospective. The rest of this post is the six transferable lessons from it that show up in interviews and design docs:

1. Reactive pretraining beats the master plan, if you instrument well enough to react.
2. Z-loss is a regularizer on logit scale, not a stability hack. (See [reference](/concepts/z-loss/).)
3. "High-quality data" without format diversity hurts downstream tasks.
4. Perplexity is mostly a measurement of preprocessing, not capability.
5. Microannealing is the right way to evaluate a candidate dataset. (See [reference](/concepts/microannealing/).)
6. SFT degrades base capabilities, and the fix is to mix pretraining data back in.

The technical mechanisms behind a few of these are heavy enough that they live in their own reference pages: [z-loss](/concepts/z-loss/), [WSD and WSD-S schedules](/concepts/wsd-and-wsd-s/), and [microannealing](/concepts/microannealing/). The essay below is the narrative.

## 1. Reactive pretraining beats the master plan

Marin's internal name for their process was the "Tootsie Roll": keep training, keep folding in new data and techniques as they appear, don't pretend you knew the right recipe upfront. The 8B run had five named phases (Kestrel, Ocelot, Jellyfish, Phoenix, Starling), none planned in advance. Phase transitions involved changing the data mix, rewarming the learning rate from a finished cooldown, and even fixing rotary embedding hyperparameters that had been wrong since step zero.

The transferable lesson is that the decision to keep training and fold in changes is itself a hyperparameter. Models that look done at 4T tokens often have a lot more headroom if you're willing to rewarm and change the mix. Marin's Phoenix phase rewarmed from a finished cooldown back to peak LR on a new data mixture, saw essentially no loss spike, and continued to roughly 12.7T tokens.

The prerequisite is observability. You can only react if you can see. Marin had per-domain eval losses, per-layer norm tracking, and a checkpoint cadence that let them roll back. Without those, the same reactive style would have produced a worse model, not a better one. This is also why their schedule choice matters: WSD-S (see [reference](/concepts/wsd-and-wsd-s/)) is designed for reactive pretraining, because you can probe the model's quality with short cooldowns without committing to a final schedule.

<!-- visual:reactive-pretraining-evidence-loop -->
<figure class="learning-figure plot-panel" aria-labelledby="reactive-pretraining-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="reactive-pretraining-title">What makes reactive pretraining controlled rather than improvised?</p>
	<svg viewBox="0 0 360 560" role="img" aria-labelledby="reactive-pretraining-svg-title reactive-pretraining-svg-desc">
		<title id="reactive-pretraining-svg-title">A checkpoint, probe, evidence, and decision loop for reactive pretraining</title>
		<desc id="reactive-pretraining-svg-desc">A solid mainline reaches a recoverable checkpoint. Two dashed side branches start from that same checkpoint: a control cooldown with the current mix and a candidate cooldown changing one factor. Their per-domain losses, downstream task evaluations, and parameter norms feed one evidence bundle. A decision gate can continue unchanged, change one factor in the next phase, or restore the checkpoint. Only continue or an evidence-backed change advances the solid mainline; rollback loops back to the saved checkpoint.</desc>
		<defs>
			<marker id="reactive-loop-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" style="fill:var(--viz-edge)"></path></marker>
			<marker id="reactive-loop-focus-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" style="fill:var(--viz-focus-stroke)"></path></marker>
			<marker id="reactive-loop-warning-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" style="fill:var(--viz-warning-stroke)"></path></marker>
		</defs>
		<rect class="viz-plot-bg" x="8" y="8" width="344" height="544" rx="5"></rect>
		<text class="viz-axis-label" x="22" y="30">SOLID = EXPENSIVE MAINLINE · DASHED = DISPOSABLE PROBE</text>
		<rect class="viz-node viz-node--input" x="77" y="45" width="206" height="44" rx="4"></rect>
		<text class="viz-callout" x="180" y="63" text-anchor="middle">1 · recoverable checkpoint</text>
		<text class="viz-node-value" x="180" y="80">saved weights + optimizer + data state</text>
		<path d="M180 89V111" style="fill:none;stroke:var(--viz-edge);stroke-width:2.5;marker-end:url(#reactive-loop-arrow)"></path>
		<rect class="viz-node" x="45" y="114" width="270" height="47" rx="4"></rect>
		<text class="viz-callout" x="180" y="132" text-anchor="middle">2 · name one question</text>
		<text class="viz-node-value" x="180" y="150">new data mix? deeper cooldown? z-loss?</text>
		<path d="M180 161V181M180 181H92V197M180 181H268V197" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2;stroke-dasharray:5 3;marker-end:url(#reactive-loop-focus-arrow)"></path>
		<rect class="viz-node" x="20" y="200" width="144" height="58" rx="4"></rect>
		<text class="viz-axis-label" x="92" y="218" text-anchor="middle">CONTROL PROBE</text>
		<text class="viz-label" x="92" y="237" text-anchor="middle">same checkpoint</text>
		<text class="viz-label" x="92" y="251" text-anchor="middle">current recipe</text>
		<rect class="viz-node viz-node--focus" x="196" y="200" width="144" height="58" rx="4"></rect>
		<text class="viz-axis-label" x="268" y="218" text-anchor="middle">CANDIDATE PROBE</text>
		<text class="viz-label" x="268" y="237" text-anchor="middle">same checkpoint</text>
		<text class="viz-label" x="268" y="251" text-anchor="middle">change one factor</text>
		<path d="M92 258V280H180M268 258V280H180V297" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2;stroke-dasharray:5 3;marker-end:url(#reactive-loop-focus-arrow)"></path>
		<rect class="viz-node" x="40" y="300" width="280" height="68" rx="4" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke)"></rect>
		<text class="viz-callout" x="180" y="319" text-anchor="middle">3 · compare an evidence bundle</text>
		<text class="viz-node-value" x="180" y="339">per-domain loss · downstream tasks</text>
		<text class="viz-node-value" x="180" y="356">parameter norms · formatting checks</text>
		<path d="M180 368V390" style="fill:none;stroke:var(--viz-edge);stroke-width:2.5;marker-end:url(#reactive-loop-arrow)"></path>
		<path class="viz-node viz-node--focus" d="M180 393L274 430L180 467L86 430Z"></path>
		<text class="viz-callout" x="180" y="425" text-anchor="middle">4 · evidence supports</text>
		<text class="viz-node-value" x="180" y="442">a mainline change?</text>
		<path d="M86 430H42V487" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#reactive-loop-arrow)"></path>
		<path d="M180 467V487" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2.5;marker-end:url(#reactive-loop-focus-arrow)"></path>
		<path d="M274 430H329V67H286" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:2;stroke-dasharray:6 4;marker-end:url(#reactive-loop-warning-arrow)"></path>
		<text class="viz-label" x="306" y="421">unsafe</text>
		<text class="viz-label" x="306" y="437">or worse</text>
		<rect class="viz-node" x="13" y="490" width="116" height="47" rx="4"></rect>
		<text class="viz-callout" x="71" y="509" text-anchor="middle">continue</text>
		<text class="viz-node-value" x="71" y="526">current recipe</text>
		<rect class="viz-node viz-node--output" x="139" y="490" width="154" height="47" rx="4"></rect>
		<text class="viz-callout" x="216" y="509" text-anchor="middle">advance mainline</text>
		<text class="viz-node-value" x="216" y="526">with one justified change</text>
		<text class="viz-axis-label" x="327" y="291" text-anchor="middle" transform="rotate(90 327 291)">ROLL BACK TO CHECKPOINT</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> preserve the solid mainline at a recoverable checkpoint, then branch short control and candidate probes from exactly the same state. Compare more than aggregate loss: domain losses, task evaluations, norms, and formatting checks must agree on the decision. Only evidence returns to the mainline; a worse or unsafe result follows the dashed rollback path. Marin's named phases were unplanned, but its checkpoints, microannealing runs, and diagnostics made adaptation inspectable rather than arbitrary. This is an original synthesis checked against the <a href="https://marin.readthedocs.io/en/latest/reports/marin-8b-retro/">Marin 8B retrospective</a> and the <a href="https://arxiv.org/abs/2410.05192">WSD-S analysis</a>.</figcaption>
</figure>

## 2. Z-loss is a regularizer on logit scale, not a stability hack

Standard advice says "use z-loss if you see logit blowup." Marin's actual finding is sharper: z-loss is the only regularizer pressuring the logit scale, so you need it whenever the rest of your training pressure relaxes.

The evidence came from a deep cooldown ("Raccoon") where they decayed LR from 1.7e-3 to 1.7e-5 to improve SFT-ability. Training loss started slowly creeping up at the deepest end. Resetting the optimizer didn't help. Removing weight decay didn't help. Eventually they tracked per-layer norms and found the `lm_head` was exploding. A z-loss penalty of 1e-4 on the final logits fixed it cleanly, and z-loss is now a Marin default.

The mechanism (full version on the [z-loss reference page](/concepts/z-loss/)) is that layer norms are typically excluded from weight decay. At very low LR with no other regularization, the final layer norm and `lm_head` can drift in pathological ways even when nothing is technically diverging. If you're doing any kind of long deep cooldown, turn z-loss on by default.

## 3. "High-quality data" without format diversity hurts downstream tasks

This is the most counterintuitive finding in the retrospective. Marin's microannealing experiments (short cooldowns with 70% pretraining mix and 30% candidate data) showed:

- Naively oversampling "high quality" sources like ArXiv, Wikipedia, and peS2o **improved loss on HQ eval sets** but **degraded downstream task performance**.
- A mix of 70% pretraining / 15% FLAN / 15% HQ beat both 100% PT and 70% PT / 30% HQ on tasks.
- 70% PT / 30% FLAN alone underperformed 100% PT.

The diagnosis is that "high quality" web data lacks the few-shot-learning-inducing structure (multiple choice Q&A, instruction-like patterns) that broad web text contains. FLAN reintroduces that structure. HQ alone removes it. The real signal is format diversity, not "quality" in the academic sense.

If you've ever gotten worse benchmarks after switching to a "cleaner" dataset, this is probably why. Eval performance lives in formats the model has seen during training, and academic-clean text is a narrower distribution of formats than the open web.

## 4. Perplexity is mostly a measurement of preprocessing, not capability

Marin saw Paloma `c4en` eval loss rise during the first cooldown and fall during the second. The architecture and cooldown structure were unchanged. Some Paloma domains contained trailing spaces, and longer cooldowns amplified the mismatch. The resulting loss spike did not measure capability.

Don't trust a single perplexity number. If your eval loss moved a lot after a data change, the first question is whether the formatting of training and eval data still matches, not whether the model got better or worse.

## 5. Microannealing is the right way to evaluate a candidate dataset

Marin, OLMo 2, and Llama 3 test candidate data with a mostly trained checkpoint. Run a short cooldown with a small candidate-data fraction, then compare it with a control that uses the normal mix. This costs less than 1% of a full run and measures downstream task impact as well as per-domain loss.

The full procedure, including how to set the mixing fraction and the common failure modes, is on the [microannealing reference page](/concepts/microannealing/). The interview-ready summary is that "run an ablation" is the L5 answer and "run a microannealing study at the late-training low-LR regime where data choices actually matter" is the L6 answer.

## 6. SFT degrades base capabilities, and the fix is to mix pretraining data back in

Marin 8B Instruct loses ground on MMLU compared to the base model, the same pattern Olmo 2 reported. The mitigation, documented in [GH#702](https://github.com/marin-community/marin/issues/702), is to mix pretraining data into SFT as a literal fraction of the SFT batch. Not as an L2-style regularizer, as actual data.

This matters for anyone fine-tuning open weights for a vertical. The default "SFT on instruction data only" recipe is the one that produces the "but the base model knew this!" complaint from product. The fix is well-known internally at frontier labs and basically never written down. Marin wrote it down. If you're doing SFT in 2026 and you aren't mixing in some fraction of the original pretraining distribution, that's a free win to recover base-model performance.

## A required asterisk on every base-model number

Marin's writeup is unusually candid about evaluation contamination: "all these results come with an asterisk. The benchmarks can be found in DCLM, Dolmino, Nemotron-CC, and others. Llama 3 is likewise contaminated." This is the right energy for talking about benchmark numbers in interviews. "Beats Llama 3.1 on MMLU" is a fact about a pair of contaminated eval setups, not a fact about underlying capability. Saying so out loud, without using it as an excuse to dismiss evaluation entirely, is a senior-IC marker.

## The meta-lesson

Most of the lessons above weren't invented by Marin. The z-loss and `lm_head` failure mode, the HQ-data-degrades-tasks finding, the SFT-eats-base-knowledge problem: frontier labs have known versions of these for years. What's new is that someone wrote them down with the receipts. GitHub issues, WandB runs, full training scripts, and a retrospective that names the mistakes.

If you want to learn pretraining as a non-frontier-lab person in 2026, the actual syllabus is short: read the [Marin 8B](https://marin.readthedocs.io/en/latest/reports/marin-8b-retro/) and [32B](https://marin.readthedocs.io/en/latest/reports/marin-32b-retro/) retrospectives end to end, then read the [OPT-175B logbook](https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/chronicles/OPT175B_Logbook.pdf) for the texture of what a real loss spike looks like. That's a tighter and more practical syllabus than any course I've seen, and it's free.

---

*Related reference pages: [z-loss](/concepts/z-loss/), [WSD and WSD-S schedules](/concepts/wsd-and-wsd-s/), [microannealing and midtraining](/concepts/microannealing/).*
