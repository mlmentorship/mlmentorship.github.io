---
title: "Present a technical ML project and defend it under questions"
description: "A senior project presentation should expose three decisions, one failure, honest attribution, and enough depth to survive interruption."
date: "2026-07-11"
draft: false
tags: ["questions"]
category: "questions"
---

> Prepare a 30-minute presentation about a consequential technical project. Expect 15 minutes of questions on decisions, failures, ownership, and impact.

Build the presentation around decisions, not chronology. The audience is not hiring the project. It is deciding whether your judgment, scope, and technical depth transfer to its problems.

## Choose the project with the best evidence

A strong project has five properties:

- You owned at least one consequential technical decision.
- There were credible alternatives.
- Something failed or contradicted the first plan.
- The outcome was measured.
- You can descend from architecture to implementation detail.

Prestige is not on the list. A famous system where you changed one component produces a weaker presentation than a bounded system whose difficult calls were yours.

## Open with the answer

The first two minutes should establish:

1. the user or research problem;
2. why it mattered;
3. what you personally owned;
4. the measured result;
5. the three decisions the presentation will defend.

Do not begin with your resume, company history, or a 12-box architecture. Those delay the claim and invite the audience to guess why the work mattered.

## The decision spine

For each major decision, use the same compact structure:

- **Constraint:** what made the obvious approach insufficient?
- **Alternatives:** what credible options existed?
- **Evidence then:** what did you know before the outcome?
- **Choice:** what did you decide and own?
- **Result:** what happened, including a failure or surprise?
- **Update:** what would you do differently now?

One decision should support a technical descent. Show a bottleneck calculation, loss or metric choice, failure trace, data-contract issue, ablation, or serving invariant. The detail proves that the architecture slide reflects work you understand.

<!-- visual:technical-project-decision-spine -->
<figure class="learning-figure" aria-labelledby="technical-project-decision-spine-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="technical-project-decision-spine-title">How does a decision spine keep a project presentation coherent under questions?</p>
	<div class="visual-grid--two" role="group" aria-label="Chronological project tour compared with three self-contained decision records">
		<section class="visual-panel">
			<h4>FRAGILE: ONE CHRONOLOGICAL ROUTE</h4>
			<p><strong>Setup &rarr; data &rarr; model &rarr; launch</strong><br />Ownership and alternatives are scattered across stages. A question about why the model changed forces the story to backtrack.</p>
		</section>
		<section class="visual-panel">
			<h4>DEFENSIBLE: THREE DECISION RECORDS</h4>
			<p><strong>Decision 1 &middot; Decision 2 &middot; Decision 3</strong><br />Each carries its own constraint, alternatives, evidence then, choice, result, and update. Questions can enter any record without losing the claim.</p>
		</section>
	</div>
	<section class="visual-panel" role="note" aria-label="Question paths that test each decision record">
		<h4>CHANGE DEPTH, NOT STRUCTURE</h4>
		<p><strong>Why this choice?</strong> Compare alternatives and prior evidence. <strong>What failed?</strong> Trace the result and update. <strong>Was it yours?</strong> Name your choice and collaborators. <strong>Can you prove it?</strong> Descend into one calculation, trace, ablation, or invariant.</p>
	</section>
	<figcaption><strong>Read it this way:</strong> the left-hand deck works only in delivery order. The right-hand deck has three independent evidence units, so an interruption changes which decision you open or how deeply you inspect it, not the logic of the presentation. This original comparison was checked against <a href="https://pure.psu.edu/en/publications/how-the-design-of-presentation-slides-affects-audience-comprehens/">Garner and Alley's assertion-evidence presentation study</a> and the <a href="https://www.opm.gov/policy-data-oversight/assessment-and-selection/structured-interviews/guide.pdf">U.S. Office of Personnel Management's structured-interview guide</a>.</figcaption>
</figure>

## What an L4 answer sounds like

> "We built a recommendation model. I will explain the data pipeline, architecture, training, and launch. Engagement increased 8%."

The project may be good, but the presentation hides the candidate. "We" owns every choice, the architecture becomes a tour, and the outcome has no attribution or guardrail. Questions quickly expose that the candidate memorized the final system rather than drove it.

## What an L5 answer adds

An L5 presentation has a clear personal thread. It explains why a baseline failed, compares alternatives, shows one implementation detail, and connects offline evidence to an online decision. The candidate can answer who owned each component, where they disagreed, and what metric blocked launch.

The failure is specific:

> "Our first sequence model improved NDCG but increased p99 latency by 70 ms. I owned the decision to keep the candidate model and replace the online feature join. A shadow test showed that the join, not inference, caused most of the tail. That preserved the gain without spending the latency budget."

This is a decision with evidence, not a list of tasks.

## What an L6 answer adds

An L6 candidate shows how one project changed a broader system or decision process while remaining technically credible. They explain cross-team incentives, the adoption path, and the second-order effect of the launch. They do not inflate attribution.

Under questions, they answer directly before adding context. They can say:

- "No. That result does not prove causality; the holdback does."
- "I did not own the training platform. I owned the model and the launch decision."
- "That assumption was wrong. The slice analysis changed my position."
- "With half the compute, I would keep the data intervention and drop the architecture search."

They use counterfactuals to demonstrate a model of the system, not attachment to the historical solution.

## Tells that get you a strong-hire vote

- The opening gives problem, ownership, outcome, and roadmap in two minutes.
- Three decisions organize the deck.
- Alternatives are credible, not straw men.
- One failed approach changes the narrative.
- Architecture descends into a concrete mechanism or bottleneck.
- Impact is separated into observed outcome and personal contribution.
- You answer the question asked before returning to the deck.
- You update when a challenge exposes a weak assumption.

## Tells that get you down-leveled

- A chronological diary of meetings and implementation steps.
- Dense architecture slides with no decision.
- "We" throughout, followed by vague ownership under questioning.
- Only successful experiments.
- Offline metrics presented as product impact.
- A polished monologue that breaks when interrupted.
- Strategic claims with no hands-on technical descent.

## Common follow-up

"What would your strongest critic say you got wrong?"

A strong answer names a real criticism, not a disguised compliment. State the critic's evidence, where you agree, and the decision you would revisit. If no reasonable critic exists in the story, the presentation probably removed the hard part.

Use the [presentation protocol](/prep/presentation/) and attempt it before reading this page again.

*Related: [your most ambitious project](/questions/most-ambitious-project/), [scope an ambiguous problem](/questions/scope-ambiguous-problem/), and [technical presentation practice](/prep/presentation/).*
