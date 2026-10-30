---
title: "Fine-tuning vs prompting: the deep version"
description: "Past the basic decision tree. The senior answer covers SFT, LoRA, DPO, continued pretraining, and the operational trade-offs each introduces."
date: "2025-06-27"
draft: false
tags: ["questions"]
category: "questions"
---


> *Asked in: LLM-team interviews; deeper variant of the standard question.*

Choose a fine-tuning method and a defensible training plan once a model-adaptation requirement has been established.

## Problem

**Context.** A team has a base language model, task examples, and a behavior gap that requires evaluation before adaptation.
The [intervention-choice question](/questions/fine-tune-vs-prompt-vs-rag/) addresses whether fine-tuning is needed at all.

**Task.** Specify the training objective, data construction, trainable parameters, evaluation, and release plan for an appropriate fine-tuning method.
Compare alternatives and explain what evidence would change the choice.

**Constraints.** Respect compute, data rights, privacy, and serving requirements.
Separate behavior learning from changing factual knowledge, guard against leakage and regressions, and do not presume adaptation always helps.

## What the L4 candidate misses

L4: "Fine-tune the model on examples." Misses that "fine-tune" covers different training objectives (continued pretraining, SFT, preference optimization) and a separate choice about which parameters to update (full fine-tuning or a parameter-efficient method such as LoRA).

## What an L5 answer sounds like

> "There are three common training objectives, plus a separate choice about update scope:
>
> 1. **Continued pretraining**: continue the next-token-prediction objective on a domain corpus. Use case: shifting the model's underlying knowledge or vocabulary (e.g., medical, legal, code in a niche language). Expensive, requires lots of unlabeled data.
>
> 2. **Supervised fine-tuning (SFT)**: train on (prompt, ideal-response) pairs. Use case: teaching the model the format, style, or specific behaviors. Standard for instruction tuning. Moderate cost, moderate data needs (10K-1M pairs).
>
> 3. **Preference optimization (DPO, IPO, RLHF)**: train on pairwise preferences (A > B). Use case: shaping nuanced behavior that's hard to specify in SFT pairs (helpfulness, conciseness, tone). DPO has largely displaced RLHF in 2026 due to simpler training.
>
> 4. **Parameter-efficient methods (LoRA, QLoRA, prefix tuning)** are not a fourth objective. They change *how* you update a model: train a small set of additional parameters while freezing the base model. Use case: many specialized models on the same base, or training under tight memory constraints. The quality trade-off is task-dependent, while the memory and serving benefits can be large.
>
> The decision tree:
> - Need new knowledge → continued pretraining + SFT, or RAG (usually RAG is cheaper).
> - Need consistent format / behavior → SFT.
> - Need nuanced quality (helpful, on-tone) → SFT then DPO.
> - Need many task-specific variants of the same base → LoRA.
> - Need just a behavior tweak → prompting first, fine-tuning only if prompting can't get there."

<!-- visual:fine-tuning-two-decisions -->
<figure class="learning-figure editorial-illustration" aria-labelledby="fine-tuning-two-decisions-edition-title"><span class="visual-fragment-anchor" id="fine-tuning-two-decisions-title" aria-hidden="true"></span><p class="visual-title" id="fine-tuning-two-decisions-edition-title">Choose a training signal and an update scope as separate decisions</p><div class="visual-argument" data-visual-argument="fine-tuning-two-decisions" data-argument-kind="matrix"><p class="visual-argument-premise">Training signal and update scope form two independent axes</p><svg viewBox="0 0 360 142" role="img" aria-label="Training signal and update scope form two independent axes. Supervised or preference signals can update all parameters or a parameter-efficient subset. Choosing one axis does not choose the other."><title>Training signal and update scope form two independent axes</title><desc>Supervised or preference signals can update all parameters or a parameter-efficient subset. Choosing one axis does not choose the other.</desc><text x="141.43" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="141.43" dy="0">full scope</tspan></text><text x="273.81" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="273.81" dy="0">low-rank</tspan></text><text x="63.24000000000001" y="54" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="63.24000000000001" dy="0">labels</tspan></text><rect data-proof-cell="0,0" x="75.24000000000001" y="36" width="129.38" height="44" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke)"></rect><text x="139.93" y="63.5" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="139.93" dy="0">SFT full</tspan></text><rect data-proof-cell="0,1" x="207.62" y="36" width="129.38" height="44" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke)"></rect><text x="272.31" y="63.5" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="272.31" dy="0">SFT LoRA</tspan></text><text x="63.24000000000001" y="101" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="63.24000000000001" dy="0">pairs</tspan></text><rect data-proof-cell="1,0" x="75.24000000000001" y="83" width="129.38" height="44" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke)"></rect><text x="139.93" y="110.5" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="139.93" dy="0">pref full</tspan></text><rect data-proof-cell="1,1" x="207.62" y="83" width="129.38" height="44" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke)"></rect><text x="272.31" y="110.5" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="272.31" dy="0">pref LoRA</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content"><svg viewBox="0 0 360 680" role="img" aria-labelledby="fine-tuning-two-decisions-edition-svg-title fine-tuning-two-decisions-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="fine-tuning-two-decisions-edition-svg-title">Choose a training signal and an update scope, not one item from a mixed list</title><desc id="fine-tuning-two-decisions-edition-svg-desc">Three objective rows use domain text, prompt/ideal-response pairs, and preference triples. Each crosses full base-weight updates and LoRA adapter updates with the base frozen. The SFT-plus-LoRA cell is highlighted. The rows are alternatives, not a mandatory sequence.</desc><text x="180" y="22" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Objective and update scope are independent</tspan><tspan x="180" dy="19">choices</tspan></text><text x="150" y="74" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="150" dy="0">FULL</tspan></text><text x="280" y="74" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="280" dy="0">LoRA</tspan></text><text x="18" y="105" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="18" dy="0">1. Domain text</tspan></text><text x="18" y="134" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:start"><tspan x="18" dy="0">next-token loss</tspan></text><g data-editorial-node><rect x="92" y="153" width="116" height="69" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="150" y="183" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="150" dy="0">base W</tspan><tspan x="150" dy="19">TRAINABLE</tspan></text></g><g data-editorial-node><rect x="222" y="153" width="116" height="69" rx="3" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.3"></rect><text x="280" y="173.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="280" dy="0">W FROZEN</tspan><tspan x="280" dy="19">adapter A,B</tspan><tspan x="280" dy="19">TRAINABLE</tspan></text></g><text x="18" y="261" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="18" dy="0">2. Prompt + ideal response</tspan></text><text x="18" y="290" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:start"><tspan x="18" dy="0">SFT loss</tspan></text><g data-editorial-node><rect x="92" y="309" width="116" height="69" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="150" y="339" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="150" dy="0">base W</tspan><tspan x="150" dy="19">TRAINABLE</tspan></text></g><g data-editorial-node><rect x="222" y="309" width="116" height="69" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="280" y="329.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="280" dy="0">W FROZEN</tspan><tspan x="280" dy="19">adapter A,B</tspan><tspan x="280" dy="19">TRAINABLE</tspan></text></g><text x="18" y="417" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="18" dy="0">3. Prompt + preferred / rejected</tspan></text><text x="18" y="446" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:start"><tspan x="18" dy="0">preference loss</tspan></text><g data-editorial-node><rect x="92" y="465" width="116" height="69" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="150" y="495" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="150" dy="0">base W</tspan><tspan x="150" dy="19">TRAINABLE</tspan></text></g><g data-editorial-node><rect x="222" y="465" width="116" height="69" rx="3" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.3"></rect><text x="280" y="485.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="280" dy="0">W FROZEN</tspan><tspan x="280" dy="19">adapter A,B</tspan><tspan x="280" dy="19">TRAINABLE</tspan></text></g><text x="180" y="606" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Highlighted cell: SFT objective + LoRA</tspan><tspan x="180" dy="19">scope</tspan></text><text x="180" y="644" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Rows are alternatives, not a required</tspan><tspan x="180" dy="19">three-stage sequence.</tspan></text></svg><div class="visual-derivation-caption">SFT tells you which loss and examples train the model. LoRA tells you where trainable updates live. Their highlighted intersection is one valid combination.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="fine-tuning-two-decisions"><strong>Check:</strong> Can preference optimization use LoRA? Why are those not competing alternatives?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="fine-tuning-two-decisions">Supervised or preference signals can update all parameters or a parameter-efficient subset. Choosing one axis does not choose the other.</span></figcaption></figure>

This is L5. You've named the objectives and update scope, given use cases, and a decision tree.

## What an L6 answer adds

> "...operational realities:
>
> **Fine-tuning has lock-in cost.** You're tied to the base model checkpoint. When the provider deprecates it (typical 12-18 months for hosted models), you re-tune. When a new model family launches, you re-tune to evaluate. Plan for this in your roadmap; teams that don't get caught flat-footed when their tuned model goes EOL.
>
> **Data quality dominates fine-tuning quality.** A small (~10K) carefully-curated SFT dataset typically beats a large (~1M) noisy one. The team's time spent on data is the highest-leverage activity. Synthetic data (LLM-generated) is increasingly common and works well when filtered by a strong judge.
>
> **DPO is operationally simpler than RLHF**. No reward model, no PPO. Pairwise preference data is enough. Most teams in 2026 default to DPO; RLHF is reserved for the most polished products at the largest labs.
>
> **LoRA quality is close to full fine-tuning** for most adaptation tasks. Memory savings are substantial: you can train LoRA adapters on a single GPU for models that need 8 GPUs to full-fine-tune. Multiple LoRAs can be served from one base model with adapter swapping at request time, useful for multi-tenant systems.
>
> **Continued pretraining is rarely worth it** unless you have a domain corpus measured in tens of billions of tokens and a budget. Most teams better-served by RAG + SFT for domain adaptation.
>
> **The most common fine-tuning mistake**: training on a small dataset that doesn't reflect the production input distribution. The model overfits to your eval examples and underperforms on real traffic. Mitigate by sampling SFT data from production logs (with privacy controls) when possible."

## Tells that get you a strong-hire vote

- You separate **the three training objectives** (continued pretraining, SFT, preference optimization) from **parameter-efficient update methods**.
- You bring up **DPO over RLHF** for operational simplicity.
- You discuss **LoRA's serving advantages** (multi-tenant, memory).
- You mention **lock-in cost** of fine-tuning.
- You insist that **data quality dominates data quantity**.

## Tells that get you down-leveled

- Treating "fine-tuning" as one thing.
- Recommending continued pretraining as a default.
- No knowledge of LoRA / QLoRA.
- No awareness of DPO.
- Suggesting RLHF when DPO is sufficient (or vice versa, recommending DPO when the task needs on-policy training).

## Common follow-up

"When would you use full fine-tuning over LoRA?"

The L6 answer:

> "When the adaptation requires changing the model substantially: cross-language adaptation, a major behavioral shift, or when the LoRA quality gap is unacceptable for the use case. Also when serving infrastructure can't support adapter-based serving (some inference stacks don't have efficient LoRA support, though most do in 2026). For most fine-tuning tasks in 2026, LoRA is the default and full fine-tuning is the exception."

---

*Related: [When would you fine-tune vs prompt vs RAG?](/questions/fine-tune-vs-prompt-vs-rag/), [RLHF and DPO](/concepts/rlhf-and-dpo/), [Mixed precision training](/concepts/mixed-precision-training/).*
