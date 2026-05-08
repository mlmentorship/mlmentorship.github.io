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

Marin saw eval loss on Paloma's `c4en` jump up during their first cooldown and jump down during their second. Same model architecture, same cooldown structure, opposite directions on the same eval. The cause was structural preprocessing mismatches: in Phase 1 they discovered some Paloma domains had texts ending in trailing space characters, and the deeper the cooldown, the more the model "disliked" those trailing spaces, which showed up as a domain-specific loss spike that had nothing to do with capability.

Don't trust a single perplexity number. If your eval loss moved a lot after a data change, the first question is whether the formatting of training and eval data still matches, not whether the model got better or worse.

## 5. Microannealing is the right way to evaluate a candidate dataset

The Marin (and Olmo 2, and Llama 3) recipe for "is this dataset worth including?" is to take a checkpoint that's already mostly trained, run a short cooldown with a small fraction of the candidate data, and compare against a control on 100% of the normal mix. Cost: less than 1% of a full run. Signal: real downstream task impact, not just per-domain loss.

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
