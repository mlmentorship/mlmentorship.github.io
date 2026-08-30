---
title: "How do you A/B test a chatbot?"
description: "Chatbot A/B testing has all the hard parts of regular A/B testing plus delayed feedback, conversational state, and metrics that are hard to define."
date: "2026-05-01"
draft: false
tags: ["questions"]
category: "questions"
---


> *Asked in: LLM-team and product-ML interviews.*

The L4 candidate proposes thumbs up/down. The L6 candidate identifies the right unit of randomization, the right metric, and the operational constraints of testing on conversations rather than single requests.

## What an L4 answer sounds like

> "Send some traffic to the new model, some to the old, compare thumbs up/down or session length."

This is right at the slogan level. Each piece has problems. You've heard A/B testing but haven't tested a stateful product.

## What an L5 answer sounds like

> "Three things to get right:
>
> **1. Unit of randomization.** Per-request randomization gives the most statistical power but breaks conversation continuity (the same user might get model A and B in alternating turns, which corrupts both). Per-conversation or per-user randomization keeps state coherent but reduces N.
>
> **2. Metric definition.** Thumbs up/down are sparse and biased toward dissatisfaction. Better signals:
> - Implicit: regeneration rate, edit-distance from response to user follow-up, copy-rate, abandonment within N turns
> - Outcome: did the user complete their task (depends on what the bot does), did they return next session
> - Explicit (sparse but high-signal): post-conversation satisfaction surveys, NPS
>
> **3. Statistical setup**: pre-commit to primary metric and sample size. Account for novelty effects (run for at least 2-4 weeks). Cut by user segment and by query type to catch slice regressions."

<!-- visual:chatbot-randomization-unit -->
<figure class="learning-figure" aria-labelledby="chatbot-randomization-title">
	<p class="visual-kicker">Unit of randomization</p>
	<p class="visual-title" id="chatbot-randomization-title">Assign once per conversation, not again at every turn.</p>
	<div class="visual-panel plot-panel">
		<svg viewBox="0 0 360 520" role="img" aria-labelledby="chatbot-randomization-svg-title chatbot-randomization-svg-desc">
			<title id="chatbot-randomization-svg-title">Per-request versus per-conversation chatbot assignment</title>
			<desc id="chatbot-randomization-svg-desc">Two panels show the same three-turn conversation. In the first, each request is randomized separately, producing Model A, then Model B, then Model A; a dashed warning states that the conversation outcome mixes both treatments. In the second, the conversation is assigned once to Model B, so all three turns use Model B and the outcome is attributable to one treatment.</desc>
			<defs>
				<marker id="arrow-forward" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="viz-arrow-forward" d="M0,0 L8,4 L0,8 Z"></path></marker>
			</defs>
			<rect class="viz-plot-bg" x="4" y="4" width="352" height="242" rx="4"></rect>
			<text class="viz-callout" x="16" y="27">PER REQUEST: assign again at each turn</text>
			<text class="viz-axis-label" x="16" y="55">USER TURN</text>
			<text class="viz-axis-label" x="238" y="55">MODEL</text>
			<text class="viz-label" x="16" y="88">1 · “Plan a trip”</text>
			<text class="viz-label" x="16" y="128">2 · “Make it cheaper”</text>
			<text class="viz-label" x="16" y="168">3 · “Book option two”</text>
			<path class="viz-forward" d="M152 83 H226"></path>
			<path class="viz-forward" d="M152 123 H226"></path>
			<path class="viz-forward" d="M152 163 H226"></path>
			<rect class="viz-node viz-node--input" x="236" y="66" width="104" height="28" rx="4"></rect>
			<rect class="viz-node viz-node--focus" x="236" y="106" width="104" height="28" rx="4"></rect>
			<rect class="viz-node viz-node--input" x="236" y="146" width="104" height="28" rx="4"></rect>
			<text class="viz-node-value" x="288" y="84">Model A</text>
			<text class="viz-node-value" x="288" y="124">Model B</text>
			<text class="viz-node-value" x="288" y="164">Model A</text>
			<path class="viz-operating-guide" d="M10 188 H350 V234 H10 Z"></path>
			<text class="viz-callout" x="20" y="208">MIXED EXPOSURE</text>
			<text class="viz-label" x="20" y="226">One conversation outcome cannot isolate A or B.</text>
			<rect class="viz-plot-bg" x="4" y="260" width="352" height="256" rx="4"></rect>
			<text class="viz-callout" x="16" y="283">PER CONVERSATION: assign once</text>
			<rect class="viz-node viz-node--focus" x="16" y="298" width="324" height="32" rx="4"></rect>
			<text class="viz-node-value" x="178" y="318">Conversation 42 → Model B</text>
			<text class="viz-axis-label" x="16" y="359">USER TURN</text>
			<text class="viz-axis-label" x="238" y="359">MODEL</text>
			<text class="viz-label" x="16" y="390">1 · “Plan a trip”</text>
			<text class="viz-label" x="16" y="426">2 · “Make it cheaper”</text>
			<text class="viz-label" x="16" y="462">3 · “Book option two”</text>
			<path class="viz-forward" d="M152 385 H226"></path>
			<path class="viz-forward" d="M152 421 H226"></path>
			<path class="viz-forward" d="M152 457 H226"></path>
			<rect class="viz-node viz-node--focus" x="236" y="368" width="104" height="28" rx="4"></rect>
			<rect class="viz-node viz-node--focus" x="236" y="404" width="104" height="28" rx="4"></rect>
			<rect class="viz-node viz-node--focus" x="236" y="440" width="104" height="28" rx="4"></rect>
			<text class="viz-node-value" x="288" y="386">Model B</text>
			<text class="viz-node-value" x="288" y="422">Model B</text>
			<text class="viz-node-value" x="288" y="458">Model B</text>
			<text class="viz-callout" x="16" y="494">COHERENT EXPOSURE · outcome belongs to B</text>
		</svg>
	</div>
	<figcaption><strong>Read it this way:</strong> follow each row from the user turn to its assigned model. Re-randomizing requests produces an A–B–A conversation, so a conversation-level outcome mixes treatments. Assigning conversation 42 once keeps every turn on B and makes its outcome attributable to B.</figcaption>
</figure>

This is L5. You've named the three concrete problems and the typical fixes.

## What an L6 answer adds

> "...practical things that bite:
>
> **Implicit metrics drift over time.** Users change behavior in response to model behavior. A model that 'wins' at week 1 because it surfaces options the old one didn't may 'lose' at week 8 because users adapted to the old patterns. Track both short-term and long-term metrics; gate on long-term where feasible.
>
> **Cost is a first-class metric.** A 1% lift on satisfaction at 3x the inference cost is usually a regression. Report cost per conversation alongside quality metrics; gate launches on Pareto improvement.
>
> **Heterogeneous treatment effects matter.** Power users react very differently than casual users. Slice your A/B by user-tier, query-complexity, language, region. The aggregate average can hide a regression on the most valuable cohort.
>
> **Sample ratio mismatch (SRM)** check is mandatory. If your randomization is broken, the assignment ratio won't match expected; you'll see ghost effects from cohort imbalance, not from the model.
>
> **Multiple comparisons**: looking at 30 metrics, some will be 'significant' by chance. Pre-specify a primary metric. Treat the rest as guardrails (one-sided tests with stricter thresholds) or apply Bonferroni / Benjamini-Hochberg.
>
> **Holdback experiments** for long-term metrics: keep a small fraction of users on the old model permanently to detect slow drift in the new model that A/B tests miss."

## Tells that get you a strong-hire vote

- You name the **unit-of-randomization trade-off** explicitly.
- You discuss **implicit vs explicit signals** and their biases.
- You bring up **novelty effects** with a 2-4 week observation horizon.
- You mention **cost** as a first-class metric.
- You insist on **slice analysis** and **SRM checks**.

## Tells that get you down-leveled

- "Just measure thumbs up/down."
- Per-request randomization without flagging the state-breaking issue.
- No discussion of statistical power.
- No mention of cost.

## Common follow-up

"What if your primary metric is statistically insignificant but you have positive directional signal across all guardrails?"

The L6 answer:

> "First, check power: did you have enough N to detect the size of the effect that matters to you? If yes, the result is genuinely null and you should not ship; positive directional signal across guardrails is what you'd see by random chance roughly half the time. If no, run longer. Don't rationalize a non-significant primary as a win because the secondaries 'look right'; this is one of the most common decision errors in A/B-driven product teams."

---

*Related: [A/B testing for ML systems](/concepts/ab-testing-for-ml/), [How would you evaluate an LLM application?](/questions/how-would-you-evaluate-an-llm-application/).*
