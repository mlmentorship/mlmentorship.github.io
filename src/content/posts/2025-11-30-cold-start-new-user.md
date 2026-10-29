---
title: "How would you do cold-start for a new user?"
description: "Use permitted context, item content, optional preferences and bounded exploration while measuring feedback uncertainty and whole-cohort outcomes."
date: "2025-11-30"
draft: false
tags: ["questions"]
category: "questions"
---


Design recommendations for a user with no interaction history and explain how limited early feedback should update decisions.

## Problem

**Context.** A new user can receive recommendations from eligible strategies A and B before any personal behavior is available.
The teaching fixture below models saves after confirmed exposure.

**Task.** Propose initial recommendations, update the stated probability model after the observations, and identify what remains uncertain.
Explain how feedback could change the mix without assuming a fixed interaction count establishes reliable personalization.

**Constraints.** Use only permitted context and optional preferences; preserve eligibility and safety boundaries.
Respect the reward's observation window, and distinguish a matured no-save from dislike or unobserved exposure.

A new user can receive eligible recommendations from strategies A and B.
Use optional preferences, permitted request context and item content before behavior exists.
Decide how early feedback should change the mix without claiming certainty or bypassing safety.

For a teaching model, strategy A has independent Bernoulli rewards with stable save probability $p$.
A success means a save within 24 hours after confirmed exposure; a matured no-save is failure for this reward.
That failure does not establish dislike.
Start with $p\sim\operatorname{Beta}(1,1)$ and observe success, success, failure.
What changes, and what remains unknown?

## Picture: Update Belief, Not a Session Counter

The Beta prior contributes one unit to each parameter, not two observed interactions.
Add the two successes to the first parameter and the failure to the second.

$$
p\mid S,S,F\sim\operatorname{Beta}(3,2),\qquad
\mathbb{E}[p\mid S,S,F]=\frac{3}{3+2}=0.60.
$$

The mean rises from 0.50 to 0.60, but probability mass remains spread across possible values.
Untried B keeps its own uncertainty; A's three observations do not establish B's value.

<!-- visual:cold-start-signal-handoff -->
<figure class="learning-figure" aria-labelledby="cold-start-handoff-title">
	<p class="visual-kicker">Three mature observations, still uncertain</p>
	<p class="visual-title" id="cold-start-handoff-title">Three observations reshape uncertainty; they do not prove a fixed success rate</p>
	<div class="visual-scroll">
		<svg viewBox="0 0 360 270" role="img" aria-labelledby="cold-start-beta-title cold-start-beta-desc" style="width:400px;min-width:400px">
			<title id="cold-start-beta-title">Uniform Beta prior and the posterior after two successes and one failure</title>
			<desc id="cold-start-beta-desc">The dashed Beta(1,1) prior has constant density one. The solid Beta(3,2) posterior has density twelve times p squared times one minus p. Both integrate to one. The posterior mean is 0.60 and its peak is at two thirds, with substantial uncertainty remaining.</desc>
			<g fill="none" stroke="currentColor">
				<path d="M40 30 H70" stroke-dasharray="5 4" stroke-width="1.5" />
				<path d="M40 55 H70" stroke-width="3" />
				<path d="M60 75 V210 H310" />
				<path d="M60 150 H300" stroke-dasharray="5 4" stroke-width="1.5" />
				<path data-beta-posterior d="M60 210 C140 210 220 -30 300 210" stroke-width="3" />
				<path d="M204 80 V210" stroke-dasharray="2 4" />
			</g>
			<g fill="currentColor" font-size="14">
				<text x="80" y="35">Prior: Beta(1,1)</text>
				<text x="80" y="60">After S, S, F: Beta(3,2)</text>
				<text x="10" y="72">Density</text>
				<text x="48" y="214" text-anchor="end">0</text>
				<text x="48" y="154" text-anchor="end">1</text>
				<text x="48" y="94" text-anchor="end">2</text>
				<text x="60" y="232" text-anchor="middle">0</text>
				<text x="180" y="232" text-anchor="middle">0.5</text>
				<text x="300" y="232" text-anchor="middle">1</text>
				<text x="215" y="90">Mean 0.60</text>
				<text x="180" y="260" text-anchor="middle">Strategy A's save probability p</text>
			</g>
		</svg>
	</div>
	<figcaption><strong>Read it this way:</strong> the solid curve shifts belief toward larger save probabilities without collapsing to one value. These are illustrative distributions under the stated reward assumptions, not a production user classifier. <span class="visual-reader-check" data-reader-check="cold-start-signal-handoff"><strong>Check:</strong> Why do two successes and one failure yield Beta(3,2), not certainty about the save rate?</span></figcaption>
</figure>

<span id="what-an-l4-answer-sounds-like"></span>
<span id="what-an-l5-answer-sounds-like"></span>

## Decision Contract

First filter eligible items/actions using safety, availability, consent and privacy rules.
Then blend available preference, context, content and behavior evidence using calibrated confidence, not a universal interaction count.
Bound exploration against an acceptable fallback.
Keep exposure, strategy/policy, selection probability, context, feedback window and missingness with each update.
Pending or unexamined impressions are not observed dislikes.

## Change 1: Onboarding Causes Abandonment

**New task.** Preference questions produce richer profiles among completers but increase abandonment.

**Picture and change.** Compare optional/minimal questions against context/content fallback, assigning users before onboarding.
Retain abandoners in whole-cohort retention and successful-recommendation outcomes.

**Check.** Better recommendations among completers can coexist with worse results for all assigned users.

**Use when.** Collecting signal has product cost.

## Change 2: The Item Is New

**New task.** A new item has content but no engagement history.

**Picture and change.** Retrieve it through eligible content features and consider bounded exploration with quality and item/creator coverage checks.

**Check.** A boost cannot bypass eligibility or guarantee value.

**Use when.** The cold side is the catalog, not the user.

<span id="tells-that-get-you-a-strong-hire-vote"></span>
<span id="tells-that-get-you-down-leveled"></span>

## Failure: Unshown Means Disliked

If popular items receive every impression, unseen categories have no feedback.
Treating their missing clicks as negative preferences reinforces the same selection without learning about alternatives.

## Transfer Check

Does higher engagement in session ten prove that learning improved cold-start?

### Reasoned Answer

No. Returners differ from users who left, and intent may change.
Compare identified cohorts with explicit follow-up maturity and censoring.

<!-- learning:core-end -->
<div class="learning-extension" data-learning-extension>

<span id="what-an-l6-answer-adds"></span>

## Signal Sources and Exploration Limits

Content features and two-tower models can connect eligible items with stated or observed interests before item-specific interaction exists.
Similar-user aggregates can supply priors only when their use is permitted and relevant.
Available demographics do not imply permission or useful prediction.

An epsilon-greedy or Thompson-sampling strategy needs eligible actions, defined rewards, sufficient exposure support and a nonstationarity policy.
The Beta example assumes stable, independent rewards; actual slates, repeated exposures and changing intent can violate that assumption.
No algorithm guarantees faster learning or retention.
Continue only justified exploration after early sessions; reset or adapt when evidence becomes stale.

<span id="common-follow-up"></span>

## Evaluate New-User Outcomes

Define success and time to useful recommendations before testing; report users who never reach the threshold.
Compare new-user strategies with cohort assignment, uncertainty, sample/power planning and privacy/safety/quality guardrails.
Measure first-session outcomes, mature day-N retention and time-to-event outcomes with explicit censoring.
Use the full assigned cohort, not only onboarding completers or returning users.

## Sources

- [Schein et al.: cold-start recommendation](https://doi.org/10.1145/564376.564421).
- [Li et al.: contextual-bandit feedback](https://arxiv.org/abs/1003.0146).

</div>

---

*Related: [Design YouTube's recommender](/questions/design-youtube-recommender/), [Design Spotify's homepage](/questions/design-spotify-homepage/), [System design case study: building personalized search ranking](/guides/personalized-search-ranking/).*
