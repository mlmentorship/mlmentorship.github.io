---
title: "Design a complementary purchase shelf"
description: "Distinguish complements from similarity, compare co-purchase rates with lift, and separate attributed orders from useful incremental purchases."
date: "2026-02-13"
draft: false
tags: ["questions"]
category: "questions"
---


A shopper chose a printer; compatible ink and another printer solve different tasks, so choose the shelf's purpose first.

## Problem

Build an illustrative complementary shelf, not a claim about current Amazon internals.
In 1,000 fictional baskets, 100 contain the printer, 80 contain ink and 50 contain both.
Another add-on appears in 300 baskets and 60 printer baskets.
Which evidence should retrieve candidates, and how would you measure useful purchase effects?

<span id="what-people-also-bought-actually-means"></span>

## Picture: Hold the Printer Fixed

<!-- visual:people-also-bought-three-relations -->
<figure class="learning-figure" aria-labelledby="people-also-bought-relations-title">
	<p class="visual-kicker">One anchor, three kinds of evidence</p>
	<p class="visual-title" id="people-also-bought-relations-title">Compatible ink can complete a printer purchase; another printer can help compare alternatives.</p>
	<div class="visual-grid--two" role="group" aria-label="Printer anchor with co-purchase, co-view and attribute-similarity evidence">
		<section class="visual-panel">
			<h4>Anchor: printer</h4>
			<p>The declared task is completing this purchase, not restarting printer comparison.</p>
		</section>
		<section class="visual-panel">
			<h4>Co-purchase: ink</h4>
			<p>50 printer baskets contain ink. Association suggests a candidate; verify the cartridge fits this printer.</p>
		</section>
		<section class="visual-panel">
			<h4>Co-view: another printer</h4>
			<p>Shared inspection can indicate alternatives, but does not prove substitution.</p>
		</section>
		<section class="visual-panel">
			<h4>Attribute similarity: similar printer</h4>
			<p>Nearby features support matching, including cold items; they do not establish complementarity.</p>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> keep the printer fixed and change the relation sought. Observed behavior supplies evidence, not compatibility or causality. Verify product fit before display.</figcaption>
</figure>

## Calculate Two Different Associations

Define one basket and one observation window; count each item once per basket here.
The conditional rate divides joint baskets by printer baskets.
Lift divides that rate by the candidate's overall basket rate.

| Candidate | Joint baskets | Conditional rate | Overall rate | Lift |
| --- | ---: | ---: | ---: | ---: |
| Ink | 50 | 50/100 = 0.50 | 80/1,000 = 0.08 | 6.25 |
| Popular add-on | 60 | 60/100 = 0.60 | 300/1,000 = 0.30 | 2.00 |

Counts and lift disagree on order.
Neither proves compatibility, reliable confidence or incremental customer value.

## Decision: Retrieve, Check Fit, Then Rank

Union decayed item-pair candidates with content-based cold-item candidates.
Apply current stock, region, device compatibility and customer-state rules before display.
Predict declared click/cart/purchase/return outcomes for the anchor, candidate, customer and context.
Loss weights are not automatically calibrated product utility.

## Change 1: Help Compare Printers

**New task.** The shopper has not selected a printer.

**Picture and change.** Retrieve substitutes and compare useful attributes, rather than optimize complementary attach rate.

**Check.** Ink need not help this decision.

**Use when.** The surface's purpose changes.

<span id="common-follow-up"></span>

## Change 2: The Customer Already Bought Ink

**New task.** Support replenishment, not permanent exclusion.

**Picture and change.** Use category, recency, purchase/cart state and exact consumable compatibility.

**Check.** A prior ink purchase is not a universal reason to suppress another.

**Use when.** Repeat purchases are legitimate.

<span id="tells-that-get-you-down-leveled"></span>

## Failure: Similarity Is Called Useful Incremental Revenue

Cosine-near printers can crowd out compatible ink.
Last-touch purchases are observable attribution, not purchases caused by the shelf.

## Transfer Check

Does high co-purchase lift prove displaying the pair causes more useful purchases?

### Reasoned Answer

No. Identify the display effect through a valid design, with return maturity and customer outcomes.

<!-- learning:core-end -->
<div class="learning-extension" data-learning-extension>

<span id="what-an-l5-answer-sounds-like"></span>
<span id="what-an-l6-answer-adds"></span>

## Observation, Decay and Evaluation

Recommendations change tomorrow's co-purchase data.
Log source, candidates, versions, propensity where known, position, actual exposure and mature outcomes.
Randomized eligible exposure/positions can identify specific effects under their assumptions; position features alone do not remove selection.
Other causal designs require explicit identification assumptions and support.

Decay an event aged $d$ days with $2^{-d/h}$ for half-life $h$.
At $d=h=30$, its weight is 0.5, not zero.
Choose decay by category and validate sparse-pair uncertainty; do not impose arbitrary popularity penalties or broad cold-item boosts.

Offline held-out pairs test association prediction, not causal basket lift.
Online assignment must match the estimand and cover returns, satisfaction, long-term value and displacement.
Keep attribution separate from randomized or otherwise identified incremental effects.

Customer-state caches need identity, freshness and invalidation contracts; recheck current exclusions.
Durable purchases may warrant exclusion, whereas consumables need replenishment rules.
Views and cart state are not interchangeable with purchases.

<span id="tells-that-get-you-a-strong-hire-vote"></span>

## Sources and Related Mechanisms

- [Linden et al.: item-to-item recommendations](https://doi.org/10.1109/MIC.2003.1167344).
- [McAuley et al.: complements and substitutes](https://doi.org/10.1145/2783258.2783381).

---

*Related: [Design YouTube's recommender](/questions/design-youtube-recommender/), [Two-tower vs cross-encoder: when to use which?](/questions/two-tower-vs-cross-encoder/), [System design case study: personalized search ranking](/guides/personalized-search-ranking/).*

</div>
