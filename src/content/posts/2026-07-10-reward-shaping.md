---
title: "Reward shaping"
description: "Modify learning signals without accidentally changing the task, creating reward hacking, or hiding specification failure."
date: "2026-07-10"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Sparse or delayed rewards (robotics, games, long-horizon agents, recommender objectives) make credit assignment hard: the agent rarely sees a signal, so it rarely learns. Reward shaping adds auxiliary feedback to densify that signal. The danger is that changing the reward can change the optimal policy, and a badly shaped reward produces confident, well-optimized behavior that does the wrong thing: circling near a waypoint, farming easy interactions, or maximizing proxy engagement while destroying long-term value.

A shaped reward has the form

$$r'(s,a,s') = r(s,a,s') + F(s,a,s').$$

## Potential-based shaping

The safe construction makes $F$ a difference of a potential function $\Phi$:

$$F(s,a,s') = \gamma \Phi(s') - \Phi(s).$$

This shifts value estimates while preserving the set of optimal policies under standard assumptions. It rewards progress toward useful states without redefining the final objective, which is why it is the default when you must shape at all.

<!-- visual:potential-shaping-cancels-backtracking -->
<figure class="learning-figure plot-panel" aria-labelledby="potential-shaping-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="potential-shaping-title">Why can one progress signal preserve the task while another creates a reward loop?</p>
	<svg viewBox="0 0 360 530" role="img" aria-labelledby="potential-shaping-svg-title potential-shaping-svg-desc">
		<title id="potential-shaping-svg-title">Potential differences cancel on backtracking, but event bonuses accumulate</title>
		<desc id="potential-shaping-svg-desc">Two panels use the same finite episodic path with discount gamma equal to 1. In the safe panel, start, waypoint, and terminal goal have potentials negative 2, negative 1, and 0. Potential shaping gives plus 1 from start to waypoint, plus 1 from waypoint to goal, and negative 1 when backtracking from waypoint to start. A backtrack followed by another advance therefore adds 0. The original goal-path return is 10 and the shaped return is 12, a fixed plus 2 boundary shift from this start. In the unsafe panel, entering the waypoint pays an arbitrary plus 1 event bonus, but leaving pays 0. Every loop through the waypoint earns another plus 1 without completing the goal, so the proxy can change the preferred behavior.</desc>
		<defs>
			<marker id="potential-shaping-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" style="fill:var(--viz-edge)"></path></marker>
			<marker id="potential-shaping-warning-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" style="fill:var(--viz-warning-stroke)"></path></marker>
		</defs>
		<rect class="viz-plot-bg" x="8" y="8" width="344" height="244" rx="5"></rect>
		<text class="viz-axis-label" x="18" y="29">SAFE · POTENTIAL DIFFERENCE</text>
		<text class="viz-label" x="342" y="29" text-anchor="end">finite episode · γ = 1</text>
		<rect class="viz-node viz-node--input" x="18" y="70" width="82" height="58" rx="4"></rect>
		<text class="viz-callout" x="59" y="93" text-anchor="middle">START</text>
		<text class="viz-label" x="59" y="113" text-anchor="middle">Φ = −2</text>
		<rect class="viz-node viz-node--focus" x="139" y="70" width="82" height="58" rx="4"></rect>
		<text class="viz-callout" x="180" y="93" text-anchor="middle">WAYPOINT</text>
		<text class="viz-label" x="180" y="113" text-anchor="middle">Φ = −1</text>
		<rect class="viz-node viz-node--output" x="260" y="70" width="82" height="58" rx="4"></rect>
		<text class="viz-callout" x="301" y="93" text-anchor="middle">GOAL</text>
		<text class="viz-label" x="301" y="113" text-anchor="middle">terminal · Φ = 0</text>
		<path d="M100 99H136 M221 99H257" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#potential-shaping-arrow)"></path>
		<text class="viz-label" x="119" y="62" text-anchor="middle">r = 0</text>
		<text class="viz-callout" x="119" y="48" text-anchor="middle">F = +1</text>
		<text class="viz-label" x="240" y="62" text-anchor="middle">r = +10</text>
		<text class="viz-callout" x="240" y="48" text-anchor="middle">F = +1</text>
		<path d="M154 129C131 163 89 163 65 131" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:2;stroke-dasharray:6 4;marker-end:url(#potential-shaping-warning-arrow)"></path>
		<text class="viz-gradient-label" x="110" y="178">backtrack F = −1 · advance F = +1 · lap = 0</text>
		<rect class="viz-node" x="18" y="192" width="324" height="45" rx="4"></rect>
		<text class="viz-callout" x="180" y="211" text-anchor="middle">goal path: original = 0 + 10 = 10</text>
		<text class="viz-label" x="180" y="228" text-anchor="middle">shaped = (0 + 1) + (10 + 1) = 12 · fixed boundary shift +2</text>
		<rect x="8" y="270" width="344" height="252" rx="5" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.5"></rect>
		<text class="viz-axis-label" x="18" y="291">UNSAFE · ARBITRARY EVENT BONUS</text>
		<text class="viz-label" x="342" y="291" text-anchor="end">same states · no potential</text>
		<rect class="viz-node viz-node--input" x="18" y="332" width="82" height="58" rx="4"></rect>
		<text class="viz-callout" x="59" y="355" text-anchor="middle">START</text>
		<text class="viz-label" x="59" y="375" text-anchor="middle">task incomplete</text>
		<rect class="viz-node viz-node--focus" x="139" y="332" width="82" height="58" rx="4"></rect>
		<text class="viz-callout" x="180" y="355" text-anchor="middle">WAYPOINT</text>
		<text class="viz-label" x="180" y="375" text-anchor="middle">entry bonus +1</text>
		<rect class="viz-node viz-node--output" x="260" y="332" width="82" height="58" rx="4"></rect>
		<text class="viz-callout" x="301" y="355" text-anchor="middle">GOAL</text>
		<text class="viz-label" x="301" y="375" text-anchor="middle">task reward +10</text>
		<path d="M100 361H136 M221 361H257" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#potential-shaping-arrow)"></path>
		<text class="viz-callout" x="119" y="321" text-anchor="middle">BONUS +1</text>
		<text class="viz-label" x="240" y="321" text-anchor="middle">reward +10</text>
		<path d="M154 391C131 425 89 425 65 393" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:2;stroke-dasharray:6 4;marker-end:url(#potential-shaping-warning-arrow)"></path>
		<text class="viz-gradient-label" x="109" y="440">leave +0 · re-enter +1 · every lap earns +1</text>
		<rect x="18" y="456" width="324" height="49" rx="4" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:2"></rect>
		<text class="viz-callout" x="180" y="476" text-anchor="middle">LOOP RETURN GROWS WITH VISITS</text>
		<text class="viz-label" x="180" y="494" text-anchor="middle">high shaped reward no longer proves task progress</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> compare the dashed return edges. With potential differences, the <code>+1</code> earned approaching the waypoint is repaid as <code>−1</code> when the agent backtracks, so a lap adds zero. A standalone waypoint bonus has no departure penalty, so repeated visits keep paying without reaching the goal. The example uses <code>γ = 1</code> and zero terminal potential; for general discounted returns, the intermediate terms still telescope to boundary potentials. Original schematic checked against <a href="https://www.cs.utexas.edu/~shivaram/readings/b2hd-NgHR1999.html">Ng, Harada, and Russell (1999)</a> and the episodic analysis by <a href="https://www.ifaamas.org/Proceedings/aamas2017/pdfs/p565.pdf">Grześ (2017)</a>.</figcaption>
</figure>

## Design procedure

1. Write down the true objective and the behavior you will not accept.
2. Identify why credit assignment is hard.
3. Prefer state potentials or demonstrations over arbitrary event bonuses.
4. Check whether a policy can maximize the shaped reward without doing the task.
5. Evaluate on the original reward and independent guardrails.
6. Anneal or remove the shaping once it is no longer needed.

## In an interview

Explain sparse credit assignment, potential-based shaping, reward hacking, and how you would red-team the proxy. The senior move is to separate optimization failure from specification failure: a perfectly optimized bad reward is not an algorithm bug, it is a spec bug.

## Common confusions

- **"More detailed reward is always better."** More terms mean more loopholes and unstable scales.
- **"Human preference solves specification."** Preference data still has annotator, coverage, and manipulation limits.
- **"The training reward is the evaluation."** Evaluate against independent task outcomes and safety constraints.

*Related: [policy gradient](/concepts/policy-gradient/), [PPO](/concepts/ppo/), and [RLHF and DPO](/concepts/rlhf-and-dpo/).*
