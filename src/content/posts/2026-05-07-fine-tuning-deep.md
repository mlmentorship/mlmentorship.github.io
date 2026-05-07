---
title: "Fine-tuning vs prompting: the deep version"
description: "Past the basic decision tree. The senior answer covers SFT, LoRA, DPO, continued pretraining, and the operational trade-offs each introduces."
date: "2026-05-07"
draft: false
tags: ["interviews"]
category: "interviews"
---


> *Asked in: LLM-team interviews; deeper variant of the standard question.*

The basic question ([fine-tune vs prompt vs RAG](/interviews/fine-tune-vs-prompt-vs-rag/)) covers when to choose each. The deep version drills into *which kind* of fine-tuning, why, and what each costs operationally.

## What the L4 candidate misses

L4: "Fine-tune the model on examples." Misses that "fine-tune" covers at least four very different operations (SFT, LoRA, DPO, continued pretraining) with different costs, data requirements, and effects.

## What an L5 answer sounds like

> "There are four common modes of fine-tuning, ordered roughly by cost and intrusiveness:
>
> 1. **Continued pretraining**: continue the next-token-prediction objective on a domain corpus. Use case: shifting the model's underlying knowledge or vocabulary (e.g., medical, legal, code in a niche language). Expensive, requires lots of unlabeled data.
>
> 2. **Supervised fine-tuning (SFT)**: train on (prompt, ideal-response) pairs. Use case: teaching the model the format, style, or specific behaviors. Standard for instruction tuning. Moderate cost, moderate data needs (10K-1M pairs).
>
> 3. **Preference optimization (DPO, IPO, RLHF)**: train on pairwise preferences (A > B). Use case: shaping nuanced behavior that's hard to specify in SFT pairs (helpfulness, conciseness, tone). DPO has largely displaced RLHF in 2026 due to simpler training.
>
> 4. **Parameter-efficient (LoRA, QLoRA, prefix tuning)**: train a small set of additional parameters while freezing the base model. Use case: many specialized models on the same base, or training under tight memory constraints. Quality is usually slightly below full fine-tuning but the operational benefits are large.
>
> The decision tree:
> - Need new knowledge → continued pretraining + SFT, or RAG (usually RAG is cheaper).
> - Need consistent format / behavior → SFT.
> - Need nuanced quality (helpful, on-tone) → SFT then DPO.
> - Need many task-specific variants of the same base → LoRA.
> - Need just a behavior tweak → prompting first, fine-tuning only if prompting can't get there."

This is L5. You've named the modes, given use cases, and a decision tree.

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

- You name **all four modes** (continued pretraining, SFT, preference optimization, parameter-efficient).
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

*Related: [When would you fine-tune vs prompt vs RAG?](/interviews/fine-tune-vs-prompt-vs-rag/), [RLHF and DPO](/reference/rlhf-and-dpo/), [Mixed precision training](/reference/mixed-precision-training/).*
