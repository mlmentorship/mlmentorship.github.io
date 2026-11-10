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
	<p class="visual-title" id="bandit-support-title">Context and uncertainty guide choices; only the chosen action supplies feedback</p><div class="visual-argument" data-visual-argument="contextual-bandit-partial-feedback-support" data-argument-kind="ledger" data-argument-composition="context-choice-and-feedback"><svg viewBox="0 0 360 1841" role="img" aria-label="Context and uncertainty guide choices; only the chosen action supplies feedback. Context changes expected rewards. Greedy choice follows the larger estimate; exploration can try an uncertain action. Here it chooses B, observes only B's reward, and updates B's estimate. The ranges and choice are illustrative, not fitted confidence bounds. Offline evaluation also needs logging support. An action never chosen has unknown reward, not zero reward."><title>Context and uncertainty guide choices; only the chosen action supplies feedback</title><desc>Context changes expected rewards. Greedy choice follows the larger estimate; exploration can try an uncertain action. Here it chooses B, observes only B's reward, and updates B's estimate. The ranges and choice are illustrative, not fitted confidence bounds. Offline evaluation also needs logging support. An action never chosen has unknown reward, not zero reward.</desc><text x="20" y="27" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Context changes reward; uncertainty</tspan><tspan x="20" dy="18">can change choice</tspan></text><text x="20" y="104" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Context 1</tspan></text><text x="75" y="148" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="75" dy="0">0</tspan></text><text x="285" y="148" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="285" dy="0">1</tspan></text><text x="20" y="200" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="20" dy="0">A</tspan></text><path d="M75 194L285 194" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M190.5 194L211.5 194" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><g data-picture-bandit-context="0" data-picture-bandit-action="0" data-picture-bandit-mean="0.6"><circle cx="201" cy="194" r="8" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><text x="201" y="233" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="201" dy="0">0.6</tspan></text><text x="20" y="282" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-output-stroke)"><tspan x="20" dy="0">B</tspan></text><path d="M75 276L285 276" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M96 276L264 276" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75"></path><g data-picture-bandit-context="0" data-picture-bandit-action="1" data-picture-bandit-mean="0.3"><circle cx="138" cy="276" r="8" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><text x="138" y="315" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="138" dy="0">0.3</tspan></text><text x="20" y="346" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Greedy favors A; trying the high</tspan><tspan x="20" dy="18">upper range chooses B.</tspan></text><text x="20" y="474" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Context 2</tspan></text><text x="75" y="518" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="75" dy="0">0</tspan></text><text x="285" y="518" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="285" dy="0">1</tspan></text><text x="20" y="570" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="20" dy="0">A</tspan></text><path d="M75 564L285 564" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M106.5 564L127.5 564" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><g data-picture-bandit-context="1" data-picture-bandit-action="0" data-picture-bandit-mean="0.2"><circle cx="117" cy="564" r="8" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><text x="117" y="603" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="117" dy="0">0.2</tspan></text><text x="20" y="652" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-output-stroke)"><tspan x="20" dy="0">B</tspan></text><path d="M75 646L285 646" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M211.5 646L232.5 646" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75"></path><g data-picture-bandit-context="1" data-picture-bandit-action="1" data-picture-bandit-mean="0.7"><circle cx="222" cy="646" r="8" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><text x="222" y="685" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="222" dy="0">0.7</tspan></text><text x="20" y="716" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">This context favors B even without</tspan><tspan x="20" dy="18">exploration.</tspan></text><text x="20" y="794" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Dot: estimate. Colored span:</tspan><tspan x="20" dy="18">illustrative uncertainty.</tspan></text><text x="20" y="887" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Choose B; observe only B's reward</tspan></text><circle cx="65" cy="958" r="24" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.75"></circle><text x="65" y="964" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="65" dy="0">A</tspan></text><path d="M65 991L65 1039M61 1031L65 1039L69 1031" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:2;stroke-linejoin:round"></path><text x="65" y="1070" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="65" dy="0">?</tspan></text><circle cx="180" cy="958" r="24" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="180" y="964" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="180" dy="0">B</tspan></text><path d="M180 991L180 1039M176 1031L180 1039L184 1031" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:2;stroke-linejoin:round"></path><text x="180" y="1070" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-output-stroke)"><tspan x="180" dy="0">1</tspan></text><circle cx="295" cy="958" r="24" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.75"></circle><text x="295" y="964" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="295" dy="0">C</tspan></text><path d="M295 991L295 1039M291 1031L295 1039L299 1031" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:2;stroke-linejoin:round"></path><text x="295" y="1070" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="295" dy="0">?</tspan></text><text x="20" y="1146" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Update B: 0.3 becomes 0.364 after</tspan><tspan x="20" dy="18">this success</tspan></text><text x="20" y="1224" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">No observed A or C reward accompanies</tspan><tspan x="20" dy="18">this choice.</tspan></text><text x="20" y="1302" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Separate offline logger: C is never</tspan><tspan x="20" dy="18">selected</tspan></text><g transform="translate(0 1370)"><text x="20" y="27" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">The logger decides which paths exist</tspan></text><circle cx="65" cy="104" r="24" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="65" y="110" style="font-size:20px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="65" dy="0">A</tspan></text><text x="65" y="164" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-input-stroke)"><tspan x="65" dy="0">0.6</tspan></text><path data-picture-support="0" d="M65 180V278" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:18"></path><rect x="20" y="290" width="90" height="54" rx="4" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke)"></rect><text x="65" y="322" style="font-size:20px;font-weight:700;text-anchor:middle;fill:var(--viz-input-stroke)"><tspan x="65" dy="0">seen</tspan></text><circle cx="180" cy="104" r="24" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="180" y="110" style="font-size:20px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="180" dy="0">B</tspan></text><text x="180" y="164" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-input-stroke)"><tspan x="180" dy="0">0.4</tspan></text><path data-picture-support="1" d="M180 180V278" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:12"></path><rect x="135" y="290" width="90" height="54" rx="4" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke)"></rect><text x="180" y="322" style="font-size:20px;font-weight:700;text-anchor:middle;fill:var(--viz-input-stroke)"><tspan x="180" dy="0">seen</tspan></text><circle cx="295" cy="104" r="24" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><text x="295" y="110" style="font-size:20px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="295" dy="0">C</tspan></text><text x="295" y="164" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="295" dy="0">0</tspan></text><path d="M295 180L295 211" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><path d="M286 218L304 236" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><path d="M304 218L286 236" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><path d="M295 243L295 278" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><rect x="250" y="290" width="90" height="54" rx="4" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke)"></rect><text x="295" y="322" style="font-size:20px;font-weight:700;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="295" dy="0">?</tspan></text><text x="20" y="398" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">No path means no observed reward</tspan></text><text x="20" y="443" style="font-size:22px;font-weight:700;text-anchor:start;fill:var(--viz-warning-stroke)"><tspan x="20" dy="0">Unknown is not zero</tspan></text></g></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
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
	<div class="visual-derivation-caption"><strong>Read it this way:</strong> start with the left panel: one decision logs one chosen action and one reward; the other rewards remain unknown counterfactuals. Then align the same action rows on the right. The target policy puts probability on C, but the logging policy never tried C in this context, so no propensity correction can recover its reward from these logs. Original synthetic example informed by <a href="https://arxiv.org/abs/1003.5956">Li et al. on unbiased offline evaluation</a> and <a href="https://arxiv.org/abs/1103.4601">Dudík et al. on doubly robust evaluation</a>.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="contextual-bandit-partial-feedback-support"><strong>Check:</strong> Why is the target policy not identifiable when it assigns mass to C?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="contextual-bandit-partial-feedback-support">Context changes expected rewards. Greedy choice follows the larger estimate; exploration can try an uncertain action. Here it chooses B, observes only B's reward, and updates B's estimate. The ranges and choice are illustrative, not fitted confidence bounds. Offline evaluation also needs logging support. An action never chosen has unknown reward, not zero reward. <strong>Can multiplying observed A and B records reveal C rewards?</strong></span></figcaption>
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
