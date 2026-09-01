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

<!-- visual:negative-hardness-label-gate -->
<figure class="learning-figure plot-panel" aria-labelledby="negative-gate-visual-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="negative-gate-visual-title">Separate “hard for the model” from “safe to label negative.”</p>
	<svg viewBox="0 0 360 460" role="img" aria-labelledby="negative-gate-svg-title negative-gate-svg-desc">
		<title id="negative-gate-svg-title">Three negative candidates with different hardness and label validity</title>
		<desc id="negative-gate-svg-desc">For the query capital of Australia, a random passage about photosynthesis has low model similarity and is a verified nonmatch, so pushing its score down is correct but gives a weak gradient. A mined passage saying Sydney is Australia's largest city has high similarity but does not answer the query, so pushing it down gives a useful strong gradient. A mined passage saying Canberra is Australia's capital has the highest similarity and is actually relevant despite lacking a label. Treating it as negative would push a valid answer down, so it must be excluded or reviewed. The diagram concludes that mining selects by model confusion while a separate relevance gate authorizes the negative label.</desc>
		<text class="viz-axis-label" x="10" y="20">QUERY</text>
		<rect class="viz-node viz-node--input" x="10" y="30" width="340" height="42" rx="4"></rect><text class="viz-callout" x="180" y="49" text-anchor="middle">“What is the capital of Australia?”</text><text class="viz-label" x="180" y="64" text-anchor="middle">known positive: Canberra</text>
		<text class="viz-axis-label" x="10" y="98">CANDIDATE SOURCE + PASSAGE</text><text class="viz-axis-label" x="350" y="98" text-anchor="end">TRAINING DECISION</text>
		<rect class="viz-node" x="10" y="110" width="340" height="82" rx="4"></rect>
		<text class="viz-callout" x="22" y="130">RANDOM · LOW SIMILARITY</text><text class="viz-label" x="22" y="148">“Photosynthesis converts sunlight…”</text><text class="viz-node-value" x="104" y="170">VERIFIED NONMATCH ✓</text><text class="viz-callout" x="338" y="159" text-anchor="end">PUSH DOWN</text><text class="viz-label" x="338" y="177" text-anchor="end">correct · weak signal</text>
		<rect class="viz-node viz-node--focus" x="10" y="204" width="340" height="92" rx="4"></rect>
		<text class="viz-callout" x="22" y="224">BM25 / MODEL-MINED · HIGH SIMILARITY</text><text class="viz-label" x="22" y="242">“Australia's largest city is Sydney…”</text><text class="viz-node-value" x="104" y="266">VERIFIED NONMATCH ✓</text><text class="viz-callout" x="338" y="253" text-anchor="end">PUSH DOWN</text><text class="viz-label" x="338" y="271" text-anchor="end">correct · strong signal</text><text class="viz-label" x="180" y="288" text-anchor="middle">useful hard negative: plausible, but does not answer</text>
		<rect x="10" y="308" width="340" height="100" rx="4" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:2;stroke-dasharray:6 4"></rect>
		<text class="viz-callout" x="22" y="328">MINED · HIGHEST SIMILARITY</text><text class="viz-label" x="22" y="346">“Canberra is Australia's capital…”</text><text class="viz-node-value" x="118" y="370">UNLABELED, ACTUALLY RELEVANT ✕</text><text class="viz-callout" x="338" y="357" text-anchor="end">DO NOT PUSH</text><text class="viz-label" x="338" y="375" text-anchor="end">exclude or review</text><text class="viz-label" x="180" y="398" text-anchor="middle">false negative: hardness came from being a valid answer</text>
		<path d="M180 410V424" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2"></path><path class="viz-arrow-forward" d="M180 432 l-5 -9 h10 Z"></path>
		<rect class="viz-node" x="10" y="434" width="340" height="22" rx="4"></rect><text class="viz-node-value" x="180" y="449">MINE BY CONFUSION · LABEL BY RELEVANCE</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> compare the last two rows. Both rank near the query, so both are hard for the current model; only the Sydney passage is known not to answer it. Pushing that verified nonmatch down teaches a fine distinction. Pushing the unlabeled Canberra passage down teaches the opposite of the task. Mine by similarity, then use labels, answer checks, or a stronger teacher to filter before assigning the negative gradient. Original schematic checked against <a href="https://aclanthology.org/2020.emnlp-main.550/">Dense Passage Retrieval</a>, <a href="https://aclanthology.org/2021.naacl-main.466/">RocketQA</a>, and the <a href="https://www.sbert.net/examples/sentence_transformer/training/ms_marco/README.html">Sentence Transformers MS MARCO training guide</a>.</figcaption>
</figure>

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
