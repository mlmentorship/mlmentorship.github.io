---
title: "Design Spotify's homepage"
description: "Compose coherent music shelves under page-level utility, repetition, editorial, freshness and privacy constraints rather than sorting unrelated local scores."
date: "2025-12-12"
draft: false
tags: ["questions"]
category: "questions"
---


Design a music homepage whose selected shelves work together rather than merely ranking each shelf independently.

## Problem

**Context.** A Spotify-like homepage combines shelves for different listening intents; the invented candidates below can overlap.

**Task.** Choose two shelves under the stated page-utility model, then explain selection, ordering, feedback, and evaluation in a full design.
Distinguish choosing a pair from choosing its display order.

**Constraints.** Use the stipulated common utility units and pair penalties only for this toy calculation.
Respect eligibility and policy boundaries; do not present the example as Spotify's internal implementation or a causal user-benefit estimate.

Design a Spotify-like homepage with coherent shelves for familiar favorites, discovery, new releases and editorial selections.
These names illustrate the surface, not Spotify's current internal implementation.

Choose two shelves from three eligible candidates.
For this toy calculation only, local scores share utility units; showing A with B incurs an overlap penalty of four.
Other pair penalties are zero, and every pair meets the hard policy constraints.

| Shelf | Intent and illustrative contents | Local score |
| --- | --- | ---: |
| A | Familiar: songs 1, 2 | 6 |
| B | Repeat: songs 1, 3 | 5 |
| C | Discovery: songs 4, 5 | 4 |

Which pair wins page utility, and does that calculation establish its display order?

## Picture: A Shelf Is One Page Candidate

Keep each shelf's coherent item list intact, then compare whole-page choices.

$$
U(\{i,j\})=s_i+s_j-\operatorname{penalty}(i,j).
$$

| Pair | Raw local score | Pair penalty | Page utility |
| --- | ---: | ---: | ---: |
| A+B | 11 | 4 | 7 |
| A+C | 10 | 0 | 10 |
| B+C | 9 | 0 | 9 |

Sorting local scores selects A+B; the explicit page objective selects A+C.
This symmetric utility determines membership, not relative position.
Ordering needs a position-sensitive policy and exposure evidence.

<p class="visual-kicker">Local scores versus whole-page utility</p>
<!-- visual:spotify-two-level-shelf-composition -->
<figure class="learning-figure editorial-illustration" aria-labelledby="spotify-two-level-shelf-composition-edition-title"><p class="visual-title" id="spotify-two-level-shelf-composition-edition-title">The lower-scoring shelf wins when it avoids an expensive duplicate</p><div class="visual-argument" data-visual-argument="spotify-two-level-shelf-composition" data-argument-kind="bars" data-argument-composition="endpoint"><svg viewBox="0 0 360 394" role="img" aria-label="A lower-scoring shelf can improve the whole page. Shelf scores are A=6, B=5, C=4. A+B repeats a song and pays penalty 4; A+C wins at 10."><title>A lower-scoring shelf can improve the whole page</title><desc>Shelf scores are A=6, B=5, C=4. A+B repeats a song and pays penalty 4; A+C wins at 10.</desc><text x="18" y="23" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="18" dy="0">A + B - duplicate penalty</tspan></text><path d="M28 64L332 64" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><rect data-proof-value="7" data-proof-min="0" data-proof-max="10" data-proof-zero="28" x="28" y="46" width="212.79999999999998" height="35" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.75"></rect><circle cx="240.79999999999998" cy="63.5" r="4" style="fill:var(--viz-warning-stroke);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><path d="M240.79999999999998 69L240.79999999999998 86" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><text x="240.79999999999998" y="106" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="240.79999999999998" dy="0">7 toy utility</tspan></text><text x="18" y="140" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="18" dy="0">A + C</tspan></text><path d="M28 181L332 181" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><rect data-proof-value="10" data-proof-min="0" data-proof-max="10" data-proof-zero="28" x="28" y="163" width="304" height="35" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></rect><circle cx="332" cy="180.5" r="4" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><path d="M332 186L332 203" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75"></path><text x="269" y="223" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-output-stroke)"><tspan x="269" dy="0">10 toy utility</tspan></text><text x="18" y="257" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="18" dy="0">B + C</tspan></text><path d="M28 298L332 298" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><rect data-proof-value="9" data-proof-min="0" data-proof-max="10" data-proof-zero="28" x="28" y="280" width="273.6" height="35" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></rect><circle cx="301.6" cy="297.5" r="4" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><path d="M301.6 303L301.6 320" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><text x="273.5" y="340" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-input-stroke)"><tspan x="273.5" dy="0">9 toy utility</tspan></text><text x="28" y="374" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="28" dy="0">0</tspan></text><text x="332" y="374" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="332" dy="0">10</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content"><svg viewBox="0 0 360 448" role="img" aria-labelledby="spotify-two-level-shelf-composition-edition-svg-title spotify-two-level-shelf-composition-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="spotify-two-level-shelf-composition-edition-svg-title">Two strong shelves can make a weaker page when they repeat the same song</title><desc id="spotify-two-level-shelf-composition-edition-svg-desc">Toy shelves A, B, and C contain song IDs one-two, one-three, and four-five, with local scores six, five, and four. Repeated song one creates the declared four-unit overlap penalty for A plus B, giving seven. A plus C gives ten and B plus C nine. Membership does not determine display order.</desc><text x="180" y="22" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Toy whole-page utility, not measured</tspan><tspan x="180" dy="19">listener value</tspan></text><rect x="22" y="61" width="316" height="61" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="47" y="97" style="font-size:18px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="47" dy="0">A</tspan></text><g data-editorial-node><rect x="80" y="73" width="72" height="36" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="116" y="96" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="116" dy="0">song 1</tspan></text></g><g data-editorial-node><rect x="165" y="73" width="72" height="36" rx="3" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.3"></rect><text x="201" y="96" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="201" dy="0">song 2</tspan></text></g><text x="286" y="97" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="286" dy="0">score 6</tspan></text><rect x="22" y="139" width="316" height="61" rx="3" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.3"></rect><text x="47" y="175" style="font-size:18px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="47" dy="0">B</tspan></text><g data-editorial-node><rect x="80" y="151" width="72" height="36" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="116" y="174" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="116" dy="0">song 1</tspan></text></g><g data-editorial-node><rect x="165" y="151" width="72" height="36" rx="3" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.3"></rect><text x="201" y="174" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="201" dy="0">song 3</tspan></text></g><text x="286" y="175" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="286" dy="0">score 5</tspan></text><rect x="22" y="217" width="316" height="61" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="47" y="253" style="font-size:18px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="47" dy="0">C</tspan></text><g data-editorial-node><rect x="80" y="229" width="72" height="36" rx="3" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.3"></rect><text x="116" y="252" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="116" dy="0">song 4</tspan></text></g><g data-editorial-node><rect x="165" y="229" width="72" height="36" rx="3" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.3"></rect><text x="201" y="252" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="201" dy="0">song 5</tspan></text></g><text x="286" y="253" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="286" dy="0">score 4</tspan></text><text x="180" y="326" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">A + B: 6 + 5 - 4 repeated-song penalty = 7</tspan></text><text x="180" y="370" style="font-size:14px;font-weight:700;fill:var(--viz-output-stroke);text-anchor:middle"><tspan x="180" dy="0">A + C = 10 &gt; B + C = 9</tspan></text><text x="180" y="412" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Choose intact A and C; display order is a</tspan><tspan x="180" dy="19">separate choice.</tspan></text></svg><div class="visual-derivation-caption">Song 1 appears in both A and B. The declared overlap penalty makes A + C the best pair. Shelf order remains unresolved.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="spotify-two-level-shelf-composition"><strong>Check:</strong> Why is shelf C selected over higher-scoring B when composing the whole page?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="spotify-two-level-shelf-composition">Shelf scores are A=6, B=5, C=4. A+B repeats a song and pays penalty 4; A+C wins at 10.</span></figcaption></figure>

<span id="whats-distinctive-about-a-homepage"></span>
<span id="what-an-l5-answer-sounds-like"></span>

## Decision Contract

| Stage | Required state |
| --- | --- |
| Shelf | ID, intent, content membership, eligibility, score semantics and generator version |
| Page | Candidate shelves, session context, position/exposure, utility and hard constraints |
| Editorial | Authorized curation, market/rights constraints, commitments and overrides |
| Freshness | Cache/version dependencies, privacy scope, expiry and invalidation |
| Evaluation | Separate local, page and mature long-term targets, assignment, uncertainty and guardrails |

Local click, completion and listen-time scores do not automatically share units.
Use a defensible composer model or explicit common utility, and validate whole-page interactions.
Specify diversity within shelves, across shelf intents and over time.
Keep safety/rights constraints separate from soft novelty or repetition penalties.

<span id="common-follow-up"></span>

## Change 1: No Personal History or Preferences

**New task.** A new user supplies no optional preferences.

**Picture and change.** Use permitted context, eligible editorial/content priors and bounded exploration.
Test new-user outcomes and abandonment; no universal familiar-first ordering is established.

**Check.** Weak evidence does not authorize demographic inference or a fixed interaction-count switch.

**Use when.** Personalized shelf scores lack support.

## Change 2: One Item Just Finished

**New task.** Recent activity changes one shelf while most cached candidates remain valid.

**Picture and change.** Invalidate affected compatible state when useful; recheck page-wide repetition and policy constraints.
Keep version, session, privacy and eligibility dependencies in cache keys/invalidation rules.

**Check.** A high hit rate can hide stale or duplicated content.

**Use when.** Recomputing every shelf is unnecessary.

<span id="tells-that-get-you-a-strong-hire-vote"></span>
<span id="tells-that-get-you-down-leveled"></span>

## Failure: Every Shelf Wins the Same Click

Independent local optimization can repeat one artist and intent across the page.
Local engagement then overstates the value of the composed session.

## Transfer Check

Must every shelf have a learned multi-stage ranker?

### Reasoned Answer

No. Recently played can use recency; editorial shelves can be curated.
Use retrieval and learned ranking only when justified by the shelf's requirements.

<!-- learning:core-end -->
<div class="learning-extension" data-learning-extension>

<span id="what-an-l6-answer-adds"></span>

## Shelf and Page Evidence

Familiar, discovery and new-release shelves need different definitions of useful listening, novelty and freshness.
Do not define discovery solely as distance from history or assume novelty improves retention.
Content features can support eligible new items before interactions; any exposure boost remains bounded by policy and quality.

Multi-task heads can estimate listen, completion, save, share and return outcomes on their proper populations.
Combining calibrated heads does not automatically yield a calibrated utility or correct exposure bias.
Log position, slate, eligibility and selection policy; counterfactual estimates require support and valid assumptions.

Choose an end-to-end latency/cost target from product needs, then measure it under load.
Cache duration depends on content, policy and session change, not a universal refresh interval.
Compare composer changes with an appropriate randomized baseline and predefined local/page/long-term targets.
Set experiment duration from outcome maturity, cycles and power; there is no mandatory four-to-eight-week window.
Check attrition, creator/content coverage, satisfaction and operational guardrails before staged release and rollback readiness.

## Sources

- [Spotify shelf-generation research](https://arxiv.org/abs/2607.25823).
- [Spotify homepage calibration](https://research.atspotify.com/2025/9/calibrated-recommendations-with-contextual-bandits-on-spotify-homepage).
- [Multi-carousel evaluation study](https://doi.org/10.3389/fdata.2022.910030).

</div>

---

*Related: [Design YouTube's recommender](/questions/design-youtube-recommender/), [Two-tower vs cross-encoder: when to use which?](/questions/two-tower-vs-cross-encoder/), [System design case study: building personalized search ranking](/guides/personalized-search-ranking/).*
