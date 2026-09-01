---
title: "Multi-agent reinforcement learning"
description: "Learning when other agents change the environment: non-stationarity, credit assignment, coordination, competition, and evaluation."
date: "2026-07-10"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Single-agent RL assumes a stationary environment. The moment other learning agents share it, that assumption breaks: markets, game-playing, traffic, negotiation, ad auctions, and self-play training are all multi-agent. Multi-agent RL studies environments where several agents act at once, and each agent's reward and transitions depend on the other agents' policies, which are themselves changing.

## Why it is harder than single-agent RL

- **Non-stationarity:** from one agent's view, the environment shifts as the others learn.
- **Credit assignment:** a shared team reward does not say whose action helped.
- **Partial observability:** agents usually see only local information.
- **Coordination equilibria:** several stable conventions can coexist.
- **Opponent modeling:** competitive agents adapt strategically to you.
- **Evaluation:** doing well against one set of opponents need not generalize.

## Centralized training, decentralized execution

The common pattern trains critics or value functions with global state and all agents' actions, while each deployed policy acts on its local observation only. MADDPG and value-decomposition methods apply this idea differently.

<!-- visual:multi-agent-ctde-information-boundary -->
<figure class="learning-figure visual-wide plot-panel" aria-labelledby="ctde-visual-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="ctde-visual-title">Separate information used to learn coordinated policies from information each policy can use when acting.</p>
	<div class="visual-scroll">
		<svg viewBox="0 0 760 360" role="img" aria-labelledby="ctde-svg-title ctde-svg-desc">
			<title id="ctde-svg-title">Centralized training and decentralized execution information paths</title>
			<desc id="ctde-svg-desc">During training, agent one and agent two each map a local observation to an action. A centralized critic additionally receives global state and both actions, then sends dashed learning signals back to both policies. During execution, the critic and global state are absent. Each trained policy independently maps only its own local observation to its own action.</desc>
			<defs>
				<marker id="arrow-forward" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="viz-arrow-forward" d="M0,0 L8,4 L0,8 Z"></path></marker>
				<marker id="arrow-backward" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="viz-arrow-backward" d="M0,0 L8,4 L0,8 Z"></path></marker>
			</defs>
			<text class="viz-axis-label" x="20" y="25">TRAINING · extra information may shape the update</text>
			<rect class="viz-node viz-node--input" x="20" y="45" width="110" height="52" rx="9"></rect>
			<text class="viz-node-label" x="75" y="66">Agent 1 sees</text><text class="viz-node-value" x="75" y="84">local o₁</text>
			<rect class="viz-node viz-node--focus" x="190" y="45" width="110" height="52" rx="9"></rect>
			<text class="viz-node-label" x="245" y="66">Policy π₁</text><text class="viz-node-value" x="245" y="84">parameters θ₁</text>
			<rect class="viz-node viz-node--output" x="360" y="45" width="90" height="52" rx="9"></rect>
			<text class="viz-node-label" x="405" y="66">Action</text><text class="viz-node-value" x="405" y="84">a₁</text>
			<path class="viz-forward" d="M130 71 H190"></path><path class="viz-forward" d="M300 71 H360"></path>
			<rect class="viz-node viz-node--input" x="20" y="120" width="110" height="52" rx="9"></rect>
			<text class="viz-node-label" x="75" y="141">Agent 2 sees</text><text class="viz-node-value" x="75" y="159">local o₂</text>
			<rect class="viz-node viz-node--focus" x="190" y="120" width="110" height="52" rx="9"></rect>
			<text class="viz-node-label" x="245" y="141">Policy π₂</text><text class="viz-node-value" x="245" y="159">parameters θ₂</text>
			<rect class="viz-node viz-node--output" x="360" y="120" width="90" height="52" rx="9"></rect>
			<text class="viz-node-label" x="405" y="141">Action</text><text class="viz-node-value" x="405" y="159">a₂</text>
			<path class="viz-forward" d="M130 146 H190"></path><path class="viz-forward" d="M300 146 H360"></path>
			<rect class="viz-node" x="515" y="45" width="115" height="52" rx="9"></rect>
			<text class="viz-node-label" x="572" y="66">Global state</text><text class="viz-node-value" x="572" y="84">s</text>
			<rect class="viz-node viz-node--focus" x="515" y="120" width="200" height="52" rx="9"></rect>
			<text class="viz-node-label" x="615" y="141">Centralized critic</text><text class="viz-node-value" x="615" y="159">Qᵢ(s, a₁, a₂)</text>
			<path class="viz-forward" d="M450 71 C485 71 480 132 515 142"></path><path class="viz-forward" d="M450 146 H515"></path><path class="viz-forward" d="M572 97 V120"></path>
			<path class="viz-backward" d="M555 172 C510 205 310 205 268 172"></path><path class="viz-backward" d="M675 172 C655 226 325 235 275 97"></path>
			<text class="viz-edge-label" x="475" y="219">dashed: learning signal updates policies during training</text>
			<line class="viz-gridline" x1="20" y1="242" x2="740" y2="242"></line>
			<text class="viz-axis-label" x="20" y="270">EXECUTION · the critic and global state are not inputs</text>
			<rect class="viz-node viz-node--input" x="20" y="288" width="90" height="48" rx="9"></rect><text class="viz-node-label" x="65" y="317">local o₁</text>
			<rect class="viz-node viz-node--focus" x="145" y="288" width="105" height="48" rx="9"></rect><text class="viz-node-label" x="197" y="309">trained π₁</text><text class="viz-node-value" x="197" y="326">uses o₁ only</text>
			<rect class="viz-node viz-node--output" x="285" y="288" width="60" height="48" rx="9"></rect><text class="viz-node-label" x="315" y="317">a₁</text>
			<path class="viz-forward" d="M110 312 H145"></path><path class="viz-forward" d="M250 312 H285"></path>
			<rect class="viz-node viz-node--input" x="405" y="288" width="90" height="48" rx="9"></rect><text class="viz-node-label" x="450" y="317">local o₂</text>
			<rect class="viz-node viz-node--focus" x="530" y="288" width="105" height="48" rx="9"></rect><text class="viz-node-label" x="582" y="309">trained π₂</text><text class="viz-node-value" x="582" y="326">uses o₂ only</text>
			<rect class="viz-node viz-node--output" x="670" y="288" width="60" height="48" rx="9"></rect><text class="viz-node-label" x="700" y="317">a₂</text>
			<path class="viz-forward" d="M495 312 H530"></path><path class="viz-forward" d="M635 312 H670"></path>
		</svg>
	</div>
	<figcaption><strong>Read it this way:</strong> read the solid paths first: each actor always chooses from its own observation. During training only, the critic may inspect global state and the joint action, then send dashed learning signals into both policies. Below the divider, that privileged path is gone; execution remains decentralized. Original schematic checked against <a href="https://arxiv.org/abs/1706.02275">MADDPG</a> and <a href="https://proceedings.mlr.press/v80/rashid18a.html">QMIX</a>.</figcaption>
</figure>

## Cooperative and competitive methods

**Cooperative.** Value decomposition builds a team value from per-agent values: VDN sums them; QMIX uses a monotonic mixing network so decentralized greedy actions stay consistent with the joint value.

**Competitive and mixed.** Self-play, population-based training, opponent sampling, and league systems reduce overfitting to a single opponent. Nash equilibrium is a useful reference point but hard to compute in large stochastic games.

## In an interview

1. Name the source of non-stationarity.
2. Clarify cooperative, competitive, or mixed incentives.
3. Explain centralized training with decentralized execution.
4. Address credit assignment and communication.
5. Evaluate against diverse policies, unseen partners, and exploiters.

## Common confusions

- **"Just treat other agents as part of the environment."** Their policies change and respond strategically.
- **"Self-play guarantees robustness."** It can cycle or overfit to its own population.
- **"A team reward creates teamwork."** It can also create free-riding and muddy credit assignment.

*Related: [actor-critic methods](/concepts/actor-critic-methods/), [exploration versus exploitation](/concepts/exploration-vs-exploitation/), and [PPO](/concepts/ppo/).*
