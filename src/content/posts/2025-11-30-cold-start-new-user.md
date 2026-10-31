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
	<p class="visual-title" id="cold-start-handoff-title">Three observations reshape uncertainty; they do not prove a fixed success rate</p><div class="visual-argument" data-visual-argument="cold-start-signal-handoff" data-argument-kind="curve" data-argument-composition="landmark"><svg viewBox="0 0 360 439" role="img" aria-label="Three outcomes update a distribution, not a certainty. Starting from Beta(1,1), two successes and one failure give Beta(3,2). Its mean is 0.6, but substantial uncertainty remains."><title>Three outcomes update a distribution, not a certainty</title><desc>Starting from Beta(1,1), two successes and one failure give Beta(3,2). Its mean is 0.6, but substantial uncertainty remains.</desc><g opacity=".35"><path d="M48 177.5L330 177.5" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path></g><g opacity=".35"><path d="M48 135L330 135" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path></g><g opacity=".35"><path d="M48 92.5L330 92.5" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path></g><path d="M48 40L48 220" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M48 220L334 220" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="48" y="245" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="48" dy="0">0</tspan></text><text x="330" y="245" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="330" dy="0">1</tspan></text><text x="40" y="54" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="40" dy="0">2</tspan></text><text x="40" y="220" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="40" dy="0">0</tspan></text><path data-proof-landmark="0.6" d="M217.2 40V220" style="stroke:var(--viz-focus-stroke);stroke-width:1.5;stroke-dasharray:2 4"></path><text x="217.2" y="245" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-focus-stroke)"><tspan x="217.2" dy="0">0.6</tspan></text><path data-proof-series="Prior: Beta(1,1)" d="M48 135L52.40625 135L56.8125 135L61.21875 135L65.625 135L70.03125 135L74.4375 135L78.84375 135L83.25 135L87.65625 135L92.0625 135L96.46875 135L100.875 135L105.28125 135L109.6875 135L114.09375 135L118.5 135L122.90625 135L127.3125 135L131.71875 135L136.125 135L140.53125 135L144.9375 135L149.34375 135L153.75 135L158.15625 135L162.5625 135L166.96875 135L171.375 135L175.78125 135L180.1875 135L184.59375 135L189 135L193.40625 135L197.8125 135L202.21875 135L206.625 135L211.03125 135L215.4375 135L217.2 135L219.84375 135L224.25 135L228.65625 135L233.0625 135L237.46875 135L241.875 135L246.28125 135L250.6875 135L255.09375 135L259.5 135L263.90625 135L268.3125 135L272.71875 135L277.125 135L281.53125 135L285.9375 135L290.34375 135L294.75 135L299.15625 135L303.5625 135L307.96875 135L312.375 135L316.78125 135L321.1875 135L325.59375 135L330 135" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:3"></path><g data-proof-observation="0" data-proof-x="0.6" data-proof-y="1"><circle cx="217.2" cy="135" r="5" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><path data-proof-series="Posterior: Beta(3,2)" d="M48 220L52.40625 219.75486755371094L56.8125 219.0350341796875L61.21875 217.8638458251953L65.625 216.2646484375L70.03125 214.2607879638672L74.4375 211.8756103515625L78.84375 209.13246154785156L83.25 206.0546875L87.65625 202.66563415527344L92.0625 198.9886474609375L96.46875 195.0470733642578L100.875 190.8642578125L105.28125 186.4635467529297L109.6875 181.8682861328125L114.09375 177.10182189941406L118.5 172.1875L122.90625 167.14866638183594L127.3125 162.0086669921875L131.71875 156.7908477783203L136.125 151.5185546875L140.53125 146.2151336669922L144.9375 140.9039306640625L149.34375 135.60829162597656L153.75 130.3515625L158.15625 125.15708923339844L162.5625 120.0482177734375L166.96875 115.04829406738281L171.375 110.1806640625L175.78125 105.46867370605469L180.1875 100.9356689453125L184.59375 96.60499572753906L189 92.5L193.40625 88.64402770996094L197.8125 85.0604248046875L202.21875 81.77253723144531L206.625 78.8037109375L211.03125 76.17729187011719L215.4375 73.9166259765625L217.2 73.11999999999998L219.84375 72.04505920410156L224.25 70.5859375L228.65625 69.56260681152344L233.0625 68.9984130859375L237.46875 68.91670227050781L241.875 69.3408203125L246.28125 70.29411315917969L250.6875 71.7999267578125L255.09375 73.88160705566406L259.5 76.5625L263.90625 79.86595153808594L268.3125 83.8153076171875L272.71875 88.43391418457031L277.125 93.7451171875L281.53125 99.77226257324219L285.9375 106.5386962890625L290.34375 114.06776428222656L294.75 122.3828125L299.15625 131.50718688964844L303.5625 141.4642333984375L307.96875 152.2772979736328L312.375 163.9697265625L316.78125 176.5648651123047L321.1875 190.0860595703125L325.59375 204.55665588378906L330 220" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:3;stroke-dasharray:7 4"></path><g data-proof-observation="1" data-proof-x="0.6" data-proof-y="1.7280000000000002"><circle cx="217.2" cy="73.11999999999998" r="5" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><text x="190" y="273" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="190" dy="0">unknown success probability</tspan></text><text x="20" y="20" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">probability density</tspan></text><path d="M20 306L48 306" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><text x="60" y="310" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="60" dy="0">Prior: Beta(1,1)</tspan></text><text x="60" y="333" style="font-size:18px;font-weight:700;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="60" dy="0">1 at 0.6</tspan></text><path d="M20 368L48 368" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><text x="60" y="372" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="60" dy="0">Posterior: Beta(3,2)</tspan></text><text x="60" y="395" style="font-size:18px;font-weight:700;text-anchor:start;fill:var(--viz-output-stroke)"><tspan x="60" dy="0">1.728 at 0.6</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
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
	<div class="visual-derivation-caption"><strong>Read it this way:</strong> the solid curve shifts belief toward larger save probabilities without collapsing to one value. These are illustrative distributions under the stated reward assumptions, not a production user classifier.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="cold-start-signal-handoff"><strong>Check:</strong> Why do two successes and one failure yield Beta(3,2), not certainty about the save rate?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="cold-start-signal-handoff">Starting from Beta(1,1), two successes and one failure give Beta(3,2). Its mean is 0.6, but substantial uncertainty remains.</span></figcaption>
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
