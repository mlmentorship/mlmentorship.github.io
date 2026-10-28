---
title: "Content-based filtering"
description: "Content-based filtering scores item features against a user profile. It handles item cold-start and often complements collaborative filtering."
date: "2026-05-31"
draft: false
tags: ["concepts"]
category: "concepts"
---

Content-based filtering can score a new item's features against a known profile before that item receives any interactions.

## Summary

Can a telescope guide with no interactions outrank a popular drama using this user's history?
Use features ordered as science, drama and short.
The liked vectors are $x_1=[1,0,1]$ and $x_2=[1,1,0]$.

## The mechanism

Their mean is $p=[1,0.5,0.5]$, with squared length $1.5$.
For nonzero vectors, cosine is the dot product divided by both lengths.
The guide $[1,0,1]$ scores $1.5/\sqrt{3}\approx0.866025$.
The drama $[0,1,0]$ scores $0.5/\sqrt{1.5}\approx0.408248$.
Neither candidate's interaction count enters this runtime calculation.

More generally, use $p=(\sum_i a_i x_i)/(\sum_i a_i)$ for nonnegative weights with positive total.
Define feature version, metric, recency weights and eligibility exclusions separately.
Zero profiles or zero candidate vectors need an explicit fallback; their cosine is undefined.
Learned encoders may still have used cross-user evidence during training.

<!-- visual:content-profile-scores-new-item -->
<figure class="learning-figure plot-panel" aria-labelledby="content-profile-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="content-profile-title">Why can a content model rank a brand-new item?</p>
	<svg viewBox="0 0 360 390" role="img" aria-labelledby="content-profile-svg-title content-profile-svg-desc">
		<title id="content-profile-svg-title">Building a user profile and scoring a new item in the same feature space</title>
		<desc id="content-profile-svg-desc">Two liked items have binary vectors over science, drama, and short features. Averaging them gives the user profile one, one half, one half. A brand-new telescope guide has zero interactions but a cosine similarity of 0.87 to that profile, so it ranks above a popular drama with 12,000 interactions and a similarity of 0.41. The content score never uses other users' interaction counts.</desc>
		<defs><marker id="content-profile-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path class="viz-arrow-forward" d="M0,0 L7,3.5 L0,7 Z"></path></marker></defs>
		<text class="viz-axis-label" x="18" y="25">1 · REPRESENT LIKED ITEMS</text>
		<text class="viz-label" x="180" y="44" text-anchor="middle">features = [science, drama, short]</text>
		<rect class="viz-node viz-node--input" x="18" y="58" width="148" height="52" rx="4"></rect>
		<text class="viz-callout" x="30" y="78">Liked · Space 101</text>
		<text class="viz-label" x="30" y="98">x₁ = [1, 0, 1]</text>
		<rect class="viz-node viz-node--input" x="194" y="58" width="148" height="52" rx="4"></rect>
		<text class="viz-callout" x="206" y="78">Liked · Lab Story</text>
		<text class="viz-label" x="206" y="98">x₂ = [1, 1, 0]</text>
		<path d="M92 116V138H180V154" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#content-profile-arrow)"></path>
		<path d="M268 116V138H180" style="fill:none;stroke:var(--viz-edge);stroke-width:2"></path>
		<text class="viz-axis-label" x="18" y="152">2 · AVERAGE VECTORS</text>
		<rect class="viz-node viz-node--focus" x="98" y="164" width="164" height="56" rx="4"></rect>
		<text class="viz-callout" x="180" y="185" text-anchor="middle">user profile</text>
		<text class="viz-callout" x="180" y="205" text-anchor="middle">p = [1, 0.5, 0.5]</text>
		<path d="M180 226V251" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#content-profile-arrow)"></path>
		<text class="viz-axis-label" x="18" y="248">3 · SCORE ITEMS</text>
		<rect class="viz-node viz-node--output" x="18" y="263" width="158" height="78" rx="4"></rect>
		<text class="viz-callout" x="30" y="284">#1 · Telescope guide</text>
		<text class="viz-label" x="30" y="302">new · 0 interactions</text>
		<text class="viz-label" x="30" y="319">x = [1, 0, 1]</text>
		<text class="viz-callout" x="30" y="335">cos(p, x) = 0.87</text>
		<rect class="viz-node" x="194" y="263" width="148" height="78" rx="4"></rect>
		<text class="viz-callout" x="206" y="284">#2 · Popular drama</text>
		<text class="viz-label" x="206" y="302">12k interactions</text>
		<text class="viz-label" x="206" y="319">x = [0, 1, 0]</text>
		<text class="viz-callout" x="206" y="335">cos(p, x) = 0.41</text>
		<path d="M28 359H332" style="fill:none;stroke:var(--c-rule);stroke-width:1"></path>
		<text class="viz-axis-label" x="180" y="378" text-anchor="middle">OTHER USERS' COUNTS ARE NOT INPUTS TO THIS SCORE</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> average the liked vectors, then score each candidate in the same feature space. The interaction counts are annotations, not score inputs. Original vectors and layout; mechanism checked against <a href="https://developers.google.com/machine-learning/recommendation/content-based/basics">Google's recommendation documentation</a>. <span class="visual-reader-check" data-reader-check="content-profile-scores-new-item"><strong>Check:</strong> Why can the zero-interaction telescope item outrank the popular drama?</span></figcaption>
</figure>

## Change the evidence or list policy

### A user has no history

**New task:** recommend before any liked-item profile exists.

**Picture and change:** an empty mean is not the profile above.
Request an optional genre preference or use a separately evaluated contextual prior.

**Check:** explicit drama preference gives $p=[0,1,0]$.
Drama now scores one; the telescope guide scores zero.
Absent that evidence, do not pretend a zero profile has meaningful cosine scores.

**Use when:** the missing evidence belongs to the user, not merely a new item.

### Avoid a duplicate list

**New task:** return two items from different categories.

**Picture and change:** two identical guides both score $0.866025$; drama scores $0.408248$.
Unconstrained top-two returns both guides, totaling $1.73205$.

**Check:** a one-per-category constraint selects a guide and drama, totaling $1.274274$.
The reduced similarity sum buys the stated diversity constraint, not guaranteed user benefit.

**Use when:** repeated near-duplicates violate the list requirement; evaluate discovery and useful outcomes separately.

## Reconstruct the information path

Can a content item tower prove that the entire recommender uses no collaborative information?

<details>
<summary>Reasoned answer</summary>

No. Inspect user inputs, training objectives and encoder provenance.
Two items with identical features remain indistinguishable to this score even when their quality differs.
A similarity contribution is not a causal explanation of satisfaction.

</details>

<!-- learning:core-end -->

<div data-learning-extension>

## Content-based vs collaborative filtering

| Mechanism | Required evidence | Test separately |
| --- | --- | --- |
| Content score | compatible item features and user/context profile | new-item feature coverage |
| Collaborative score | shared interaction structure or learned behavioral factors | interaction sparsity and exposure |
| Hybrid | explicitly combined feature and behavioral paths | cold/warm quality and resource costs |

## Strengths and weaknesses

Features enable scoring before item interactions, but missing attributes and domain shifts can defeat that benefit.
Concentrated recommendations establish overspecialization, not a causal filter-bubble conclusion.
Neither approach has universal accuracy, niche-user or discovery superiority.

## Hybrid systems (what's actually deployed)

Mixtures, feature-rich two-tower models and graph side-information can combine evidence.
Specify fusion weights and compare baselines on cold/warm, diversity and eligibility slices.
Do not assume a mandatory age-based handoff to collaborative filtering.

## What an interviewer expects you to say

Trace profile construction, candidate information, score, exclusions and evidence gaps before choosing a hybrid.

## Common confusions

Item cold-start is not user cold-start.
A named architecture does not specify its actual information sources.

---

*Related: [Matrix factorization for recsys](/concepts/matrix-factorization-recsys/), [Two-tower retrieval](/concepts/two-tower-retrieval/), [How would you do cold-start for a new user?](/questions/cold-start-new-user/), [Knowledge-graph embeddings](/concepts/knowledge-graph-embeddings/), [TF-IDF and BM25](/concepts/tf-idf-and-bm25/).*

</div>
