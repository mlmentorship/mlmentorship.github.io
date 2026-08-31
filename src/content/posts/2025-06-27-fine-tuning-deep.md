---
title: "Fine-tuning vs prompting: the deep version"
description: "Past the basic decision tree. The senior answer covers SFT, LoRA, DPO, continued pretraining, and the operational trade-offs each introduces."
date: "2025-06-27"
draft: false
tags: ["questions"]
category: "questions"
---


> *Asked in: LLM-team interviews; deeper variant of the standard question.*

The basic question ([fine-tune vs prompt vs RAG](/questions/fine-tune-vs-prompt-vs-rag/)) covers when to choose each. The deep version drills into *which kind* of fine-tuning, why, and what each costs operationally.

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
<figure class="learning-figure" aria-labelledby="fine-tuning-two-decisions-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="fine-tuning-two-decisions-title">Which training signal do you need, and which parameters should it update?</p>
	<div class="visual-grid--two" role="group" aria-label="Two separate fine-tuning decisions: first select a training objective, then select full or parameter-efficient updates">
		<section class="visual-panel">
			<h4>1 · CHOOSE THE OBJECTIVE</h4>
			<p><strong>Domain text → continued pretraining</strong><br />Learn next-token prediction from unlabeled text to adapt domain language and representations.</p>
			<p><strong>Ideal responses → SFT</strong><br />Learn from prompt-response pairs to make target tasks and response patterns more reliable.</p>
			<p><strong>Chosen vs rejected responses → preference optimization</strong><br />Learn relative preferences with DPO, IPO, or an RLHF pipeline.</p>
			<p><strong>Possible sequence, not a required checklist:</strong><br />continued pretraining → SFT → preference optimization.</p>
		</section>
		<section class="visual-panel">
			<h4>2 · CHOOSE THE UPDATE SCOPE</h4>
			<p><strong>Full fine-tuning</strong><br />Make the base weights trainable and produce a new model checkpoint.</p>
			<p><strong>PEFT, such as LoRA</strong><br />Freeze the base weights and train a small adapter; keep the base model plus an adapter artifact.</p>
			<p><strong>This choice is orthogonal to the objective.</strong><br />LoRA describes where trainable updates live. It does not specify whether the data is domain text, ideal responses, or preference pairs.</p>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> read top to bottom on the left to choose what the data should teach, then move right to choose how broadly parameters should change. SFT and DPO name learning objectives; LoRA names an update mechanism, so they are not peer alternatives.</figcaption>
</figure>

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
