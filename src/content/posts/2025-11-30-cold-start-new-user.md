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
	<p class="visual-title" id="cold-start-handoff-title">Three observations reshape uncertainty; they do not prove a fixed success rate</p><div class="visual-argument" data-visual-argument="cold-start-signal-handoff" data-argument-kind="curve" data-argument-composition="density-update"><svg viewBox="0 0 360 543" role="img" aria-label="Three outcomes update a distribution, not a certainty. Begin with an even spread of possible success rates. Success favors higher rates; failure favors lower ones. After two successes and one failure, the average becomes 60%, but many rates remain possible. Only shown outcomes count: missing exposure is not failure. Bounded exploration gathers evidence without treating the estimate as certainty."><title>Three outcomes update a distribution, not a certainty</title><desc>Begin with an even spread of possible success rates. Success favors higher rates; failure favors lower ones. After two successes and one failure, the average becomes 60%, but many rates remain possible. Only shown outcomes count: missing exposure is not failure. Bounded exploration gathers evidence without treating the estimate as certainty.</desc><text x="20" y="27" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Evidence reshapes a distribution</tspan></text><text x="20" y="70" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">Prior: every probability has equal</tspan><tspan x="20" dy="18">density</tspan></text><path d="M35 181L35 119L39.53125 119L44.0625 119L48.59375 119L53.125 119L57.65625 119L62.1875 119L66.71875 119L71.25 119L75.78125 119L80.3125 119L84.84375 119L89.375 119L93.90625 119L98.4375 119L102.96875 119L107.5 119L112.03125 119L116.5625 119L121.09375 119L125.625 119L130.15625 119L134.6875 119L139.21875 119L143.75 119L148.28125 119L152.8125 119L157.34375 119L161.875 119L166.40625 119L170.9375 119L175.46875 119L180 119L184.53125 119L189.0625 119L193.59375 119L198.125 119L202.65625 119L207.1875 119L209 119L211.71875 119L216.25 119L220.78125 119L225.3125 119L229.84375 119L234.375 119L238.90625 119L243.4375 119L247.96875 119L252.5 119L257.03125 119L261.5625 119L266.09375 119L270.625 119L275.15625 119L279.6875 119L284.21875 119L288.75 119L293.28125 119L297.8125 119L302.34375 119L306.875 119L311.40625 119L315.9375 119L320.46875 119L325 119L325 181Z" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:2"></path><text x="20" y="219" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Two successes; one failure</tspan></text><circle cx="113" cy="254" r="14" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="180" cy="254" r="14" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="247" cy="254" r="14" style="fill:var(--viz-warning-stroke);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><path d="M241 254L253 254" style="fill:none;stroke:var(--c-text);stroke-width:1.75"></path><text x="20" y="310" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Combine evidence; reshape uncertainty</tspan></text><path d="M35 465L35 465L39.53125 464.8211975097656L44.0625 464.296142578125L48.59375 463.4418640136719L53.125 462.275390625L57.65625 460.8137512207031L62.1875 459.073974609375L66.71875 457.0730895996094L71.25 454.828125L75.78125 452.3561096191406L80.3125 449.674072265625L84.84375 446.7990417480469L89.375 443.748046875L93.90625 440.5381164550781L98.4375 437.186279296875L102.96875 433.7095642089844L107.5 430.125L112.03125 426.4496154785156L116.5625 422.700439453125L121.09375 418.8945007324219L125.625 415.048828125L130.15625 411.1804504394531L134.6875 407.306396484375L139.21875 403.4436950683594L143.75 399.609375L148.28125 395.8204650878906L152.8125 392.093994140625L157.34375 388.4469909667969L161.875 384.896484375L166.40625 381.4595031738281L170.9375 378.153076171875L175.46875 374.9942321777344L180 372L184.53125 369.1874084472656L189.0625 366.573486328125L193.59375 364.1752624511719L198.125 362.009765625L202.65625 360.0940246582031L207.1875 358.445068359375L209 357.864L211.71875 357.0799255371094L216.25 356.015625L220.78125 355.2691955566406L225.3125 354.857666015625L229.84375 354.7980651855469L234.375 355.107421875L238.90625 355.8027648925781L243.4375 356.901123046875L247.96875 358.4195251464844L252.5 360.375L257.03125 362.7845764160156L261.5625 365.665283203125L266.09375 369.0341491699219L270.625 372.908203125L275.15625 377.3044738769531L279.6875 382.239990234375L284.21875 387.7317810058594L288.75 393.796875L293.28125 400.4523010253906L297.8125 407.715087890625L302.34375 415.6022644042969L306.875 424.130859375L311.40625 433.3179016113281L315.9375 443.180419921875L320.46875 453.7354431152344L325 465L325 465Z" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:2"></path><path d="M209 342L209 472" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><text x="20" y="507" style="font-size:20px;font-weight:700;text-anchor:start;fill:var(--viz-output-stroke)"><tspan x="20" dy="0">Mean 0.6; uncertainty</tspan><tspan x="20" dy="25">remains</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
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
	<div class="visual-derivation-caption"><strong>Read it this way:</strong> the solid curve shifts belief toward larger save probabilities without collapsing to one value. These are illustrative distributions under the stated reward assumptions, not a production user classifier.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="cold-start-signal-handoff"><strong>Check:</strong> Why do two successes and one failure yield Beta(3,2), not certainty about the save rate?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="cold-start-signal-handoff">Begin with an even spread of possible success rates. Success favors higher rates; failure favors lower ones. After two successes and one failure, the average becomes 60%, but many rates remain possible. Only shown outcomes count: missing exposure is not failure. Bounded exploration gathers evidence without treating the estimate as certainty. <strong>Do three observations make every probability except 0.6 impossible?</strong></span></figcaption>
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
