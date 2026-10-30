---
title: "Design a complementary purchase shelf"
description: "Distinguish complements from similarity, compare co-purchase rates with lift, and separate attributed orders from useful incremental purchases."
date: "2026-02-13"
draft: false
tags: ["questions"]
category: "questions"
---


Design a complementary-product shelf and distinguish useful add-ons from substitutes or merely popular items.

## Problem

**Context.** A shopper has selected a printer, and the hypothetical store has basket co-occurrence counts for possible add-ons.

**Task.** Compare the evidence below, propose candidate retrieval and ranking, and define an experiment for useful purchase effects.
Explain how compatibility, eligibility, and the shelf's purpose affect selection.

**Constraints.** Observed co-purchases are associations, not proof that showing an item causes incremental value.
Account for item popularity, exposure, and substitutes; this is not a claim about Amazon's internal implementation.

Build an illustrative complementary shelf, not a claim about current Amazon internals.
In 1,000 fictional baskets, 100 contain the printer, 80 contain ink and 50 contain both.
Another add-on appears in 300 baskets and 60 printer baskets.
Which evidence should retrieve candidates, and how would you measure useful purchase effects?

<span id="what-people-also-bought-actually-means"></span>

## Picture: Hold the Printer Fixed

<!-- visual:people-also-bought-three-relations -->
<figure class="learning-figure editorial-illustration" aria-labelledby="people-also-bought-three-relations-edition-title"><span class="visual-fragment-anchor" id="people-also-bought-relations-title" aria-hidden="true"></span><p class="visual-title" id="people-also-bought-three-relations-edition-title">Co-purchase suggests a candidate; product fit establishes a complement</p><div class="visual-argument" data-visual-argument="people-also-bought-three-relations" data-argument-kind="matrix"><p class="visual-argument-premise">Co-purchase evidence does not establish product compatibility</p><svg viewBox="0 0 360 148" role="img" aria-label="Co-purchase evidence does not establish product compatibility. A frequently co-purchased case can still fail the phone fit requirement. Complementarity needs a separate compatibility relation."><title>Co-purchase evidence does not establish product compatibility</title><desc>A frequently co-purchased case can still fail the phone fit requirement. Complementarity needs a separate compatibility relation.</desc><text x="105.13333333333333" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="105.13333333333333" dy="0">co-buy</tspan></text><text x="199.08" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="199.08" dy="0">fits</tspan></text><text x="293.02666666666664" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="293.02666666666664" dy="0">candidate</tspan></text><text x="46.16" y="54" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="46.16" dy="0">case</tspan><tspan x="46.16" dy="18">A</tspan></text><rect data-proof-cell="0,0" x="58.16" y="36" width="90.94666666666667" height="47" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke)"></rect><text x="103.63333333333333" y="65" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="103.63333333333333" dy="0">high</tspan></text><rect data-proof-cell="0,1" x="152.10666666666668" y="36" width="90.94666666666667" height="47" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke)"></rect><text x="197.58" y="65" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="197.58" dy="0">NO</tspan></text><path d="M156.10666666666668 79L239.05333333333334 40" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><rect data-proof-cell="0,2" x="246.05333333333334" y="36" width="90.94666666666667" height="47" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke)"></rect><text x="291.5266666666667" y="65" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="291.5266666666667" dy="0">exclude</tspan></text><path d="M250.05333333333334 79L333 40" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><text x="46.16" y="104" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="46.16" dy="0">case</tspan><tspan x="46.16" dy="18">B</tspan></text><rect data-proof-cell="1,0" x="58.16" y="86" width="90.94666666666667" height="47" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke)"></rect><text x="103.63333333333333" y="115" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="103.63333333333333" dy="0">lower</tspan></text><rect data-proof-cell="1,1" x="152.10666666666668" y="86" width="90.94666666666667" height="47" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke)"></rect><text x="197.58" y="115" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="197.58" dy="0">yes</tspan></text><rect data-proof-cell="1,2" x="246.05333333333334" y="86" width="90.94666666666667" height="47" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke)"></rect><text x="291.5266666666667" y="115" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="291.5266666666667" dy="0">eligible</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content"><svg viewBox="0 0 360 496" role="img" aria-labelledby="people-also-bought-three-relations-edition-svg-title people-also-bought-three-relations-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="people-also-bought-three-relations-edition-svg-title">An ink association becomes a complement only after product-fit verification</title><desc id="people-also-bought-three-relations-edition-svg-desc">A toy printer P1 is the fixed purchase anchor. Co-purchase links ink C7 with a count of fifty baskets and no known denominator. Co-view links another printer P2; attributes link similar printer P3. Only verified C7-to-P1 compatibility can admit ink to the complementary shelf.</desc><g data-editorial-node><rect x="92" y="24" width="176" height="62" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="180" y="41" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Anchor printer P1</tspan><tspan x="180" dy="19">task: complete</tspan><tspan x="180" dy="19">purchase</tspan></text></g><path d="M180 86L60 144" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path><path d="M64.0704969981701,138.3051730326648L60,144L66.99133057065477,144.3482769757365" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path><text x="60" y="132" style="font-size:12px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="60" dy="0">Co-purchase</tspan></text><g data-editorial-node><rect x="5" y="154" width="110" height="108" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="60" y="184.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="60" dy="0">ink C7</tspan><tspan x="60" dy="19">50 baskets</tspan><tspan x="60" dy="19">denominator</tspan><tspan x="60" dy="19">unknown</tspan></text></g><path d="M180 86L180 144" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M176.64402122977057,137.8569220667674L180,144L183.35597877022943,137.8569220667674" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="180" y="132" style="font-size:12px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Co-view</tspan></text><g data-editorial-node><rect x="125" y="154" width="110" height="108" rx="3" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.3"></rect><text x="180" y="194" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">printer P2</tspan><tspan x="180" dy="19">possible</tspan><tspan x="180" dy="19">alternative</tspan></text></g><path d="M180 86L300 144" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M293.0086694293452,144.34827697573652L300,144L295.9295030018299,138.3051730326648" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="300" y="132" style="font-size:12px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="300" dy="0">Attributes</tspan></text><g data-editorial-node><rect x="245" y="154" width="110" height="108" rx="3" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.3"></rect><text x="300" y="194" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="300" dy="0">printer P3</tspan><tspan x="300" dy="19">similar</tspan><tspan x="300" dy="19">features</tspan></text></g><path d="M60 262L60 300" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path><path d="M56.64402122977058,293.85692206676737L60,300L63.35597877022942,293.85692206676737" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path><g data-editorial-node><rect x="8" y="305" width="344" height="78" rx="3" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.3"></rect><text x="180" y="330" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Does current catalog verify C7 fits P1?</tspan><tspan x="180" dy="19">YES: complementary shelf</tspan><tspan x="180" dy="19">NO / UNKNOWN: withhold C7</tspan></text></g><text x="180" y="425" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Observed association is not product</tspan><tspan x="180" dy="19">compatibility.</tspan></text><text x="180" y="466" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">P2 and P3 answer a different comparison task.</tspan></text></svg><div class="visual-derivation-caption">Each edge names a different relation to P1. Fifty baskets is a count, not a rate. Verify C7 fits P1 before treating it as a complementary purchase.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="people-also-bought-three-relations"><strong>Check:</strong> Which candidate could complete this purchase? What must be verified beyond observing co-purchase?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="people-also-bought-three-relations">A frequently co-purchased case can still fail the phone fit requirement. Complementarity needs a separate compatibility relation.</span></figcaption></figure>

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
