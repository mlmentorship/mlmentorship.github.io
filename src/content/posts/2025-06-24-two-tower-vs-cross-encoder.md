---
title: "Two-tower vs cross-encoder: when to use which?"
description: "The recsys / search architecture decision that comes up in every retrieval interview. The right answer is 'both, in sequence.'"
date: "2025-06-24"
draft: false
tags: ["questions"]
category: "questions"
---


> *Asked in: recsys, search, and retrieval interviews.*

Choose how to score query-item relevance when a large candidate collection must be searched under serving constraints.

## Problem

**Context.** A retrieval service must match queries against many items, then produce a short relevance-ranked result list.
Two-tower models encode queries and items separately; cross-encoders jointly process each query-item pair.

**Task.** Compare their representation, training, retrieval, and serving trade-offs, then justify a design for the workload.
Explain how a two-stage design would be evaluated if you choose one.

**Constraints.** Account for candidate recall, latency, index freshness, memory, and training-serving consistency.
Do not assume either architecture is universally more accurate; distinguish measured relevance from representational capacity.

<!-- visual:two-tower-cross-encoder-fanout -->
<figure class="learning-figure editorial-illustration" aria-labelledby="two-tower-cross-encoder-fanout-edition-title"><span class="visual-fragment-anchor" id="two-stage-fanout-title" aria-hidden="true"></span><p class="visual-title" id="two-tower-cross-encoder-fanout-edition-title">Retrieve reusable vectors, then jointly encode only the surviving pairs</p><svg viewBox="0 0 360 628" role="img" aria-labelledby="two-tower-cross-encoder-fanout-edition-svg-title two-tower-cross-encoder-fanout-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="two-tower-cross-encoder-fanout-edition-svg-title">Retrieve reusable vectors, then jointly encode only the surviving pairs</title><desc id="two-tower-cross-encoder-fanout-edition-svg-desc">A toy six-item corpus has independently stored item vectors. One query vector retrieves IDs two, four, six, so only three query/item pairs receive fresh joint scoring. A token-level example pairs query red with red running shoes in I4. Missing item I3 cannot be restored by the reranker.</desc><text x="180" y="22" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Toy corpus N=6; retained K=3</tspan></text><text x="180" y="63" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">OFFLINE: six reusable item vectors</tspan></text><g data-editorial-node><rect x="9" y="84" width="51" height="43" rx="3" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.3"></rect><text x="34.5" y="110.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="34.5" dy="0">I1</tspan></text></g><g data-editorial-node><rect x="67" y="84" width="51" height="43" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="92.5" y="110.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="92.5" dy="0">I2</tspan></text></g><g data-editorial-node><rect x="125" y="84" width="51" height="43" rx="3" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.3"></rect><text x="150.5" y="110.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="150.5" dy="0">I3</tspan></text></g><g data-editorial-node><rect x="183" y="84" width="51" height="43" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="208.5" y="110.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="208.5" dy="0">I4</tspan></text></g><g data-editorial-node><rect x="241" y="84" width="51" height="43" rx="3" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.3"></rect><text x="266.5" y="110.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="266.5" dy="0">I5</tspan></text></g><g data-editorial-node><rect x="299" y="84" width="51" height="43" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="324.5" y="110.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="324.5" dy="0">I6</tspan></text></g><g data-editorial-node><rect x="25" y="166" width="310" height="62" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="180" y="192.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Query vector: &quot;red shoes&quot;</tspan><tspan x="180" dy="19">vector search -&gt; IDs {2,4,6}</tspan></text></g><text x="180" y="265" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Only these three pairs pay for joint encoding</tspan></text><path d="M180 228L60 305" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M63.357831749229184,298.8579347167367L60,305L66.98261395948775,304.5069459535033" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><g data-editorial-node><rect x="5" y="309" width="110" height="82" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="60" y="345.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="60" dy="0">query + I2</tspan><tspan x="60" dy="19">JOINT score</tspan></text></g><path d="M180 228L180 305" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M176.64402122977057,298.85692206676737L180,305L183.35597877022943,298.85692206676737" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><g data-editorial-node><rect x="125" y="309" width="110" height="82" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="180" y="345.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">query + I4</tspan><tspan x="180" dy="19">JOINT score</tspan></text></g><path d="M180 228L300 305" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M293.01738604051224,304.5069459535033L300,305L296.6421682507708,298.8579347167367" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><g data-editorial-node><rect x="245" y="309" width="110" height="82" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="300" y="345.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="300" dy="0">query + I6</tspan><tspan x="300" dy="19">JOINT score</tspan></text></g><g data-editorial-node><rect x="20" y="429" width="320" height="80" rx="3" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.3"></rect><text x="180" y="445.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Example interaction in pair I4:</tspan><tspan x="180" dy="19">query &quot;red&quot; &lt;-&gt; item &quot;red running</tspan><tspan x="180" dy="19">shoes&quot;</tspan><tspan x="180" dy="19">score can inspect tokens together</tspan></text></g><text x="180" y="552" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Missing I3 cannot be restored by reranking</tspan><tspan x="180" dy="19">{2,4,6}.</tspan></text><text x="180" y="599" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Candidate recall bounds the final result.</tspan></text></svg><figcaption><strong>Read it this way:</strong> All six items can reuse offline vectors, but only the three retrieved IDs get joint token interaction here. The reranker cannot recover an item outside that candidate set. <span class="visual-reader-check" data-reader-check="two-tower-cross-encoder-fanout"><strong>Check:</strong> Why jointly encode only K pairs? Can reranking recover an item outside those candidates?</span></figcaption></figure>

## What an L4 answer sounds like

> "Two-tower is faster but less accurate. Cross-encoder is more accurate but slower. So you use two-tower when speed matters and cross-encoder when accuracy matters."

Right idea, missing the architectural point. You've heard the comparison but haven't built either.

## What an L5 answer sounds like

> "These solve different problems and the right answer is usually 'both, in sequence.'
>
> **Two-tower** independently encodes queries and items, allowing item-vector reuse. Search cost and recall depend on catalog size, dimension, hardware, index parameters, filters, and freshness. A single-vector interaction compresses pair detail; it does not imply a universal latency or relevance score. Keep query and item/index versions compatible, as in the [two-tower mechanism](/concepts/two-tower-retrieval/).
>
> **Cross-encoder** takes (query, item) as a *joint* input, runs them through the same model, and outputs a relevance score. Strictly more expressive because it can model cross-attention between query and item tokens. The disadvantage: you have to score every (query, item) pair at serving time, which is O(N) per query, infeasible for large catalogs.
>
> **The standard production architecture is two-stage**: use a two-tower for *retrieval* (reduce billions to ~thousands of candidates), then a cross-encoder (or other reranker) for *ranking* (reorder the top thousands down to the final ~10). This gets you the speed of two-tower for the bulk of the work and the accuracy of cross-encoder where it matters.
>
> A cross-encoder can improve bounded candidate scoring, but it cannot recover an omitted relevant item. Measure candidate recall, final relevance, cost, and latency separately; the extra stage is worthwhile only when the observed gain justifies its budget."

This is L5. You've named the architectures, the trade-offs, the standard composition, and the operational reality.

## What an L6 answer sounds like

> "...and a few more things that come up in practice:
>
> **Representation does not uniquely determine the loss.** Contrastive, pointwise, pairwise, or listwise objectives require appropriate labels and candidate sets. A two-tower objective need not be symmetric, and separately scored pairs need not be statistically independent.
>
> **The 'right' two-tower architecture has subtleties.** Modern variants include ColBERT (multi-vector representations, one embedding per token, late interaction at scoring time), which gets some of the accuracy of cross-encoder while keeping retrieval tractable; and SPLADE (sparse representations) for hybrid lexical-dense retrieval. These bridge the gap.
>
> **Negative semantics need auditing.** Hard-negative mining is a candidate intervention, not an automatic gain. A plausible retrieved item can be another valid positive; exclude or model such cases instead of training them as known negatives. Compare mining policies with matched evaluation and budgets.
>
> **Cross-encoder distillation back into two-tower** is a real lever. You can train your two-tower to mimic the cross-encoder's scores on a labeled set, gaining some of the cross-encoder's quality without its serving cost.
>
> **Late-interaction models** (ColBERT and similar) are increasingly common as a third option. Multi-vector retrieval, late interaction at scoring, a middle ground between two-tower and cross-encoder. More memory-intensive but better quality.
>
> **For RAG specifically**: the standard stack is bi-encoder retrieval (top 50-100) → cross-encoder rerank (top 5-10) → LLM generation. Each stage is its own engineering and quality optimization."

This is L6. You have connected the textbook comparison to newer methods such as ColBERT, SPLADE, and distillation, plus the practical training discipline of hard negative mining.

## The tells that get you a strong-hire vote

- You frame it as **two stages, not two choices**.
- You bring up **hard negative mining** as the actual quality lever for two-tower.
- You mention **ColBERT or other late-interaction models** as a third architecture.
- You discuss **distillation** from cross-encoder to two-tower.
- You connect to **RAG**: this is the same architectural decision in disguise.

## The tells that get you down-leveled

- Treating it as a binary choice (one or the other).
- "Cross-encoder is slow" without quantifying ("would need to score every item per query, which is O(N)").
- No mention of reranking as a standard architectural step.
- No knowledge of training-data subtleties (hard negatives).
- Confusing two-tower with siamese networks (they're related but two-tower has different towers for query and item, often with different inputs).

## A common follow-up

"How would you train a good two-tower from scratch?"

The L6 answer:

> "Three phases:
>
> 1. **Warm-up with in-batch negatives**: train with simple contrastive loss where every other item in the batch is a negative. Gets you a reasonable starting point quickly.
> 2. **Hard negative mining**: use the warm-up model (or a separate strong model) to mine hard negatives, items that have high similarity score to the query but are not the gold positive. Train with these as negatives. This is where quality really comes from.
> 3. **Optional: distillation from a cross-encoder**: train a strong cross-encoder on a labeled set, score (query, candidate) pairs, then train the two-tower to match those scores. Distillation often outperforms direct supervised training.
>
> Throughout: be careful about the eval set. Standard MTEB benchmarks are public and over-fit; you want a domain-specific eval set built from production samples (or prospective production samples)."

If you can have this conversation fluently, you're at strong-senior depth.

---

*Related: [Designing a RAG system that actually works](/guides/designing-rag-that-works/), [Design YouTube's recommender](/questions/design-youtube-recommender/).*
