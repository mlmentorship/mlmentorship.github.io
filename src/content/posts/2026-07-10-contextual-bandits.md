---
title: "Contextual bandits"
description: "Choose actions from context while balancing reward and uncertainty. The bridge between supervised prediction, experimentation, and reinforcement learning."
date: "2026-07-10"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Bandits show up everywhere a system chooses an action and only learns about the action it took: recommendation, notifications, ranking, treatment selection, adaptive experiments. They formalize the cost that logged production data hides: you know what happened under the policy you ran, not what would have happened under the alternatives you never tried.

Formally, a contextual bandit sees context $x_t$, chooses action $a_t$, and receives a reward only for that action, maximizing cumulative reward while learning which actions work for which contexts. It sits between two neighbors: unlike full reinforcement learning, the action does not drive a persistent state transition; unlike supervised learning, the labels for the actions you did not choose are missing by design.

<!-- visual:contextual-bandit-partial-feedback-support -->
<figure class="learning-figure" aria-labelledby="bandit-support-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="bandit-support-title">Why can a logged bandit event evaluate some target policies but not others?</p>
	<div class="visual-grid--two" role="group" aria-label="A logged contextual-bandit event followed by an off-policy support check">
		<section class="visual-panel plot-panel">
			<svg viewBox="0 0 300 230" role="img" aria-labelledby="bandit-log-title bandit-log-desc">
				<title id="bandit-log-title">The logging policy reveals one reward and two unknown counterfactuals</title>
				<desc id="bandit-log-desc">For context x, logging policy mu assigns probabilities 0.70 to action A, 0.30 to B, and zero to C. It chooses B and observes reward one. Rewards for unchosen A and C are unknown, shown as question marks rather than zeros.</desc>
				<rect class="viz-plot-bg" x="8" y="25" width="284" height="196" rx="5"></rect>
				<text class="viz-axis-label" x="12" y="16">1 · ONE LOGGED EVENT</text>
				<text class="viz-label" x="20" y="46">context x · logging policy μ</text>
				<text class="viz-axis-label" x="25" y="67">ACTION</text>
				<text class="viz-axis-label" x="117" y="67">μ(a|x)</text>
				<text class="viz-axis-label" x="210" y="67">REWARD</text>
				<rect class="viz-node" x="18" y="77" width="58" height="34" rx="3"></rect>
				<rect class="viz-node viz-node--focus" x="18" y="121" width="58" height="34" rx="3"></rect>
				<rect class="viz-node" x="18" y="165" width="58" height="34" rx="3"></rect>
				<text class="viz-node-label" x="47" y="99">A</text>
				<text class="viz-node-label" x="47" y="143">B</text>
				<text class="viz-node-label" x="47" y="187">C</text>
				<text class="viz-callout" x="135" y="99" text-anchor="middle">0.70</text>
				<text class="viz-callout" x="135" y="143" text-anchor="middle">0.30 · chosen</text>
				<text class="viz-callout" x="135" y="187" text-anchor="middle">0</text>
				<text class="viz-callout" x="232" y="99" text-anchor="middle">? unknown</text>
				<text class="viz-callout" x="232" y="143" text-anchor="middle">1 observed</text>
				<text class="viz-callout" x="232" y="187" text-anchor="middle">? unknown</text>
				<path d="M78 138H91" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2"></path>
				<text class="viz-axis-label" x="20" y="208">UNCHOSEN REWARDS ARE MISSING,</text>
				<text class="viz-axis-label" x="20" y="220">NOT ZERO</text>
			</svg>
		</section>
		<section class="visual-panel plot-panel">
			<svg viewBox="0 0 300 230" role="img" aria-labelledby="bandit-target-title bandit-target-desc">
				<title id="bandit-target-title">The target policy fails the support check on action C</title>
				<desc id="bandit-target-desc">Target policy pi assigns probabilities 0.20 to A, 0.30 to B, and 0.50 to C. Actions A and B have positive logging probability and are supported. C is unsupported because the target probability is positive while the logging probability is zero, so this target policy cannot be identified from these logs.</desc>
				<rect class="viz-plot-bg" x="8" y="25" width="284" height="196" rx="5"></rect>
				<text class="viz-axis-label" x="12" y="16">2 · TARGET POLICY SUPPORT CHECK</text>
				<text class="viz-label" x="20" y="46">same context x · target policy π</text>
				<text class="viz-axis-label" x="25" y="67">ACTION</text>
				<text class="viz-axis-label" x="101" y="67">π(a|x)</text>
				<text class="viz-axis-label" x="204" y="67">LOG SUPPORT?</text>
				<rect class="viz-node" x="18" y="77" width="58" height="34" rx="3"></rect>
				<rect class="viz-node" x="18" y="121" width="58" height="34" rx="3"></rect>
				<rect x="18" y="165" width="58" height="34" rx="3" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:2"></rect>
				<text class="viz-node-label" x="47" y="99">A</text>
				<text class="viz-node-label" x="47" y="143">B</text>
				<text class="viz-node-label" x="47" y="187">C</text>
				<text class="viz-callout" x="117" y="99" text-anchor="middle">0.20</text>
				<text class="viz-callout" x="117" y="143" text-anchor="middle">0.30</text>
				<text class="viz-callout" x="117" y="187" text-anchor="middle">0.50</text>
				<text class="viz-callout" x="224" y="99" text-anchor="middle">yes · μ(A|x) = 0.70</text>
				<text class="viz-callout" x="224" y="143" text-anchor="middle">yes · μ(B|x) = 0.30</text>
				<text class="viz-callout" x="224" y="187" text-anchor="middle">NO · μ(C|x) = 0</text>
				<text class="viz-axis-label" x="20" y="208">π(C|x) &gt; 0 BUT μ(C|x) = 0</text>
				<text class="viz-axis-label" x="20" y="220">→ NOT IDENTIFIABLE</text>
			</svg>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> start with the left panel: one decision logs one chosen action and one reward; the other rewards remain unknown counterfactuals. Then align the same action rows on the right. The target policy puts probability on C, but the logging policy never tried C in this context, so no propensity correction can recover its reward from these logs. Original synthetic example informed by <a href="https://arxiv.org/abs/1003.5956">Li et al. on unbiased offline evaluation</a> and <a href="https://arxiv.org/abs/1103.4601">Dudík et al. on doubly robust evaluation</a>.</figcaption>
</figure>

## Core approaches

- **Epsilon-greedy:** exploit most of the time; pick randomly with probability $\epsilon$.
- **UCB:** choose the highest estimated reward plus an uncertainty bonus.
- **Thompson sampling:** sample parameters from the posterior and act greedily under that sample.
- **LinUCB / linear Thompson sampling:** assume expected reward is linear in the context features.

## Off-policy evaluation

Logged data needs propensities. Inverse propensity scoring estimates a target policy by weighting reward by the probability of the logged action; doubly robust estimators combine a reward model with that correction. Without exploration support, a policy that chooses actions absent from the log is not identifiable offline: you have no evidence about what they would have returned.

## In an interview

1. Separate contextual bandits from supervised learning and MDPs.
2. Define regret and the exploration-exploitation trade-off.
3. Describe UCB or Thompson sampling.
4. Explain logging propensities and off-policy evaluation.
5. Cover delayed rewards, non-stationarity, safety constraints, and feedback loops.

## Common confusions

- **"An A/B test is a bandit."** A fixed A/B test explores with a static policy; a bandit adapts assignment over time.
- **"Bandits always beat experiments."** Adaptive policies complicate inference and can chase short-term proxies.
- **"Use historical clicks as labels."** Only the actions the logging policy chose have outcomes; selection bias is the whole problem.

*Related: [exploration versus exploitation](/concepts/exploration-vs-exploitation/), [A/B testing for ML](/concepts/ab-testing-for-ml/), and [policy gradient](/concepts/policy-gradient/).*
