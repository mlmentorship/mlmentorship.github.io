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
