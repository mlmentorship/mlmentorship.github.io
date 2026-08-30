---
title: "Content-based filtering"
description: "Content-based filtering scores item features against a user profile. It handles item cold-start and often complements collaborative filtering."
date: "2026-05-31"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Content-based filtering recommends items whose **features** (genre, text, tags, embeddings) match a **profile built from the items a user already engaged with**: it scores item–profile similarity, using **no other users' behavior**.

Content-based filtering is the standard answer to the **item cold-start** problem: a brand-new item with zero interactions has no collaborative signal, but it *does* have features, so a content model can recommend it on day one. It also drives **explainability** ("because you watched X") and works for niche users with unique tastes. Every recsys interview expects you to contrast it with collaborative filtering and explain why production systems combine them.

## The mechanism

1. **Item representation.** Turn each item into a feature vector: structured attributes (genre, brand, price), TF-IDF / embeddings of text, image/audio embeddings, or learned content encoders.
2. **User profile.** Aggregate the representations of items the user engaged with, e.g. the (weighted) average of liked-item vectors, or a learned user encoder.
3. **Score and rank.** Recommend items with the highest similarity (cosine / dot product) to the profile, excluding already-seen items.

This is structurally a **two-tower** idea (a user/profile tower and an item/content tower) when both sides are learned, which is why content features feed naturally into modern retrieval models.

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
		<text class="viz-axis-label" x="18" y="152">2 · AVERAGE INTO ONE PROFILE</text>
		<rect class="viz-node viz-node--focus" x="98" y="164" width="164" height="56" rx="4"></rect>
		<text class="viz-callout" x="180" y="185" text-anchor="middle">user profile</text>
		<text class="viz-callout" x="180" y="205" text-anchor="middle">p = [1, 0.5, 0.5]</text>
		<path d="M180 226V251" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#content-profile-arrow)"></path>
		<text class="viz-axis-label" x="18" y="248">3 · SCORE CANDIDATE FEATURES</text>
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
	<figcaption><strong>Read it this way:</strong> average the two liked-item vectors to build the profile, then compare that profile only with each candidate's features. The telescope guide can rank first on day one because its feature vector already exists; the 0 and 12k interaction counts are shown only to emphasize that this content score does not use them. The vectors and layout are original; mechanism checked against <a href="https://developers.google.com/machine-learning/recommendation/content-based/basics">Google's recommendation-systems documentation</a>.</figcaption>
</figure>

## Content-based vs collaborative filtering

| | **Content-based** | **Collaborative filtering** |
| --- | --- | --- |
| Signal | item **features** + this user's history | the **user–item interaction matrix** |
| New item (item cold-start) | **works** (has features) | fails (no interactions) |
| New user | needs a little history | fails (no interactions) |
| Serendipity / discovery | weak (stays near known tastes → **filter bubble**) | strong (finds non-obvious patterns) |
| Niche users | strong | weak |
| Needs other users? | no | yes |
| Quality ceiling | limited by feature quality | learns latent taste it can't name |

The crisp summary: **content-based asks "what is this item like?"; collaborative asks "who else behaved like you?"** They fail in opposite situations, which is exactly why they're combined.

## Strengths and weaknesses

**Strengths**: handles item cold-start, needs no other users, recommendations are explainable, works for unique tastes.

**Weaknesses**:

- **Limited serendipity**: recommendations cluster around what the user already likes (the **filter-bubble / over-specialization** problem).
- **Feature-bound**: quality is capped by how good your item features are; it can't discover preferences your features don't encode.
- **Still has user cold-start**: a brand-new user with no history has no profile.

## Hybrid systems (what's actually deployed)

Production recommenders blend both:

- **Cold-start handoff**: content-based for new items/users, sliding to collaborative as interactions accumulate.
- **Feature-rich two-tower / wide-and-deep models** that take both content features *and* collaborative IDs as input, learning a single ranker.
- **Knowledge-graph and embedding side-information** layered onto collaborative factors.

So "content-based vs collaborative" is rarely a real either/or in 2026: the design question is *how* to fuse them.

## What an interviewer expects you to say

1. Define it as **profile (from the user's items) × item features**, with **no reliance on other users**.
2. Lead with its killer use case: **item cold-start** and **explainability**.
3. Contrast cleanly with collaborative filtering on the cold-start and serendipity axes ("what is this item like" vs "who behaves like you").
4. Name its weaknesses: **over-specialization / filter bubble**, **feature-quality ceiling**, and remaining **user cold-start**.
5. Conclude with **hybrid** systems and feature-rich two-tower models as the production reality.

## Common confusions

- **"Content-based solves all cold-start."** It solves **item** cold-start; a brand-new **user** still has no profile.
- **"It's just collaborative filtering with features."** It uses *no* cross-user signal; that's the defining difference and the source of both its cold-start strength and its serendipity weakness.
- **"It's more accurate than collaborative filtering."** Usually the opposite once interaction data exists: collaborative filtering learns latent preferences content features can't capture. Content shines specifically when behavioral data is sparse.
- **"Two-tower retrieval is collaborative filtering."** Two-tower can be either or both: with content features in the item tower it's content-based; with pure ID embeddings it's collaborative.

---

*Related: [Matrix factorization for recsys](/concepts/matrix-factorization-recsys/), [Two-tower retrieval](/concepts/two-tower-retrieval/), [How would you do cold-start for a new user?](/questions/cold-start-new-user/), [Knowledge-graph embeddings](/concepts/knowledge-graph-embeddings/), [TF-IDF and BM25](/concepts/tf-idf-and-bm25/).*
