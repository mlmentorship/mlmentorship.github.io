---
title: "Negative sampling strategies: what actually matters"
description: "Choice of negatives often matters more than choice of model. The senior answer ranks the strategies (in-batch, hard, BM25-mined, model-mined) and explains the trade-offs."
date: "2026-04-17"
draft: false
tags: ["questions"]
category: "questions"
---


> *Asked in: recsys, retrieval, and embedding-training interviews.*

The L4 candidate samples random items as negatives. The L6 candidate explains why hard negatives dominate quality and how to mine them without breaking training.

## Why negatives matter

In contrastive training (two-tower retrieval, embedding learning), the model sees one positive (the query and its true match) and N negatives per training example. The model learns to push positive scores up and negative scores down. The choice of negatives determines what the model learns to *distinguish*.

Random negatives are easy to push apart; the model trivially scores them low and learns little. Hard negatives, ones that look like positives but aren't, force the model to learn fine-grained distinctions.

## What an L4 answer sounds like

> "Sample random items from the corpus as negatives."

Right baseline, missing the most important quality lever. You've trained one retrieval model, the textbook way.

## What an L5 answer sounds like

> "Several strategies, in order of typical effectiveness:
>
> 1. **In-batch negatives**: for each (query, positive) in a batch, treat all other positives in the batch as negatives. Free, parallelizable. Good baseline.
>
> 2. **Random negatives from corpus**: sample uniformly. Cheap but easy; model learns coarse distinctions.
>
> 3. **BM25-mined hard negatives**: for each query, retrieve top-K candidates with BM25, treat non-relevant ones as hard negatives. They have lexical overlap but aren't the answer; force the model to learn semantic precision.
>
> 4. **Model-mined hard negatives**: use the current model (or an earlier checkpoint) to retrieve candidates; non-positive top hits are hard negatives. Requires periodic re-mining as the model improves.
>
> 5. **Curriculum**: start with easy (random) negatives, progressively add harder ones.
>
> The biggest single quality lever is moving from random to BM25-mined hard negatives. Subsequent gains from model-mined and curriculum are smaller."

This is L5. Five strategies, ranked by impact.

## What an L6 answer adds

> "...practical things:
>
> **Too-hard negatives break training.** If the negatives are *actual* positives that happen not to be labeled (false negatives), gradients pull the model in conflicting directions. Symptoms: training loss plateaus or diverges. Mitigation: filter mined negatives by label coverage, or use a margin loss that's robust to label noise.
>
> **Negative count per positive matters.** More negatives per positive (large batch contrastive, MoCo-style queue, large negative sample) consistently improves quality up to a saturation point. Engineering effort to enable larger negative pools (gradient accumulation, queue-based negatives) usually pays off.
>
> **Distillation from a stronger model into a two-tower** can replace negative mining for some use cases. Train the two-tower to mimic a cross-encoder's scores on (query, candidate) pairs. The cross-encoder implicitly handles the hard-negative problem.
>
> **For LLM embedding training**, the modern recipe (E5, BGE, NV-Embed) uses a mix: in-batch negatives + hard mined negatives + a contrastive loss + sometimes a knowledge-distillation loss from a teacher cross-encoder. The exact weighting matters less than having all three sources.
>
> **Domain matters a lot.** Code retrieval, legal retrieval, and conversational retrieval each have different 'hard' patterns. Mine domain-specific hard negatives; don't expect general-purpose techniques to transfer cleanly."

## Tells that get you a strong-hire vote

- You name **at least four strategies** and rank them by typical impact.
- You bring up **BM25-mined hard negatives** as the highest-leverage step.
- You mention **false-negative leakage** as the failure mode of aggressive mining.
- You discuss **distillation from cross-encoders** as an alternative.
- You acknowledge **larger negative pools** as a separate quality lever.

## Tells that get you down-leveled

- "Random negatives" with no further detail.
- Suggesting in-batch negatives are the goal rather than the baseline.
- No discussion of false-negative leakage.
- No knowledge of curriculum or distillation.

## Common follow-up

"How would you mine hard negatives without polluting your training set with false negatives?"

The L6 answer:

> "Three patterns. (1) Mine candidates with the model, then filter out any candidate that has a known label (positive or negative) from the labeled set. (2) Use a stronger model (cross-encoder, larger LM) to score the mined candidates and exclude those scoring above a threshold (likely false negatives). (3) Use a margin loss (margin > 0 between positive and negative scores) that's somewhat tolerant of weak negatives. In practice, (1) + (3) is the common recipe. (2) is heavier but worth it for high-stakes domains."

---

*Related: [Two-tower vs cross-encoder: when to use which?](/questions/two-tower-vs-cross-encoder/), [Designing a RAG system that actually works](/guides/designing-rag-that-works/), [Cross-entropy and softmax](/concepts/cross-entropy-softmax/).*
