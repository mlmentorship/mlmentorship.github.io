---
title: "Synthetic data generation and verification"
description: "Generate training examples for a clear capability, verify them with independent evidence, and protect diversity and held-out evaluation."
date: "2026-08-28"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Synthetic data is created by a model, simulator, program, or rule instead of being recorded directly from the target population.

## Why AI labs care

Synthetic data can create examples that are rare, expensive, private, or easy to verify. It is used for:

- instruction tuning;
- math and code reasoning;
- tool-use trajectories;
- safety and red-team cases;
- multilingual data;
- simulations for robotics and agents;
- distillation from a stronger model.

Generation is cheap. Reliable verification and useful diversity are the hard parts.

## Start with the capability

Define what the new data should teach.

Weak goal: "generate more reasoning data."

Better goal: "generate algebra problems that require two substitutions, include a unique numeric answer, and remain correct after changing variable names."

The goal determines the generator, verifier, filters, and held-out test.

## A simple pipeline

1. **Specify:** define task families, difficulty, format, allowed tools, and failure rules.
2. **Generate:** use programs, simulators, models, or mixtures of sources.
3. **Verify:** check correctness with evidence that does not rely only on the generator.
4. **Filter:** remove invalid, trivial, unsafe, duplicate, and low-value examples.
5. **Balance:** control source, topic, language, difficulty, and style mixtures.
6. **Train:** mix synthetic and non-synthetic data with clear provenance.
7. **Evaluate:** use held-out real and synthetic tests that were not part of generation or filtering.
8. **Audit:** inspect model failures and update the pipeline.

<!-- visual:synthetic-data-two-independent-evidence-gates -->
<figure class="learning-figure plot-panel visual-wide" aria-labelledby="synthetic-evidence-visual-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="synthetic-evidence-visual-title">Separate evidence that selects training rows from evidence that confirms the learned capability.</p>
	<div class="visual-scroll">
		<svg viewBox="0 0 760 390" role="img" aria-labelledby="synthetic-evidence-svg-title synthetic-evidence-svg-desc">
			<title id="synthetic-evidence-svg-title">Two independent evidence gates for synthetic training data</title>
			<desc id="synthetic-evidence-svg-desc">In the development lane, a capability specification guides a generator that creates candidates. Independent evidence such as tests, a solver, or simulator state checks correctness. A separate coverage and decontamination gate rejects repetitive or leaked examples. Accepted examples enter a provenance-tracked training mixture and produce a frozen student. Below a heavy boundary, fresh held-out task families that were never used for generation, filtering, or training meet the frozen student only at final evaluation. That evaluation supports a scoped capability claim and has no feedback arrow into development.</desc>
			<rect class="viz-plot-bg" x="12" y="12" width="736" height="366" rx="5"></rect>
			<text class="viz-axis-label" x="28" y="38">DEVELOPMENT LANE · EVIDENCE MAY SELECT TRAINING ROWS</text>
			<rect class="viz-node viz-node--input" x="28" y="55" width="132" height="60" rx="5"></rect>
			<text class="viz-node-label" x="94" y="79">1 · CAPABILITY</text>
			<text class="viz-node-value" x="94" y="98">families · difficulty</text>
			<path d="M160 85H192" style="fill:none;stroke:var(--viz-edge);stroke-width:2"></path>
			<path d="M192 80L205 85L192 90Z" style="fill:var(--viz-edge)"></path>
			<rect class="viz-node viz-node--input" x="205" y="55" width="132" height="60" rx="5"></rect>
			<text class="viz-node-label" x="271" y="79">2 · GENERATE</text>
			<text class="viz-node-value" x="271" y="98">model · program · sim</text>
			<path d="M337 85H369" style="fill:none;stroke:var(--viz-edge);stroke-width:2"></path>
			<path d="M369 80L382 85L369 90Z" style="fill:var(--viz-edge)"></path>
			<rect class="viz-node" x="382" y="55" width="132" height="60" rx="5"></rect>
			<text class="viz-node-label" x="448" y="79">CANDIDATES</text>
			<text class="viz-node-value" x="448" y="98">large count ≠ quality</text>
			<path d="M514 85H546" style="fill:none;stroke:var(--viz-edge);stroke-width:2"></path>
			<path d="M546 80L559 85L546 90Z" style="fill:var(--viz-edge)"></path>
			<rect class="viz-node viz-node--focus" x="559" y="48" width="165" height="74" rx="5"></rect>
			<text class="viz-node-label" x="641" y="72">3 · VERIFY</text>
			<text class="viz-node-value" x="641" y="91">tests · solver · simulator</text>
			<text class="viz-node-value" x="641" y="106">not generator confidence</text>
			<path d="M641 122V145" style="fill:none;stroke:var(--viz-edge);stroke-width:2"></path>
			<path d="M636 145L641 158L646 145Z" style="fill:var(--viz-edge)"></path>
			<rect class="viz-node viz-node--focus" x="559" y="158" width="165" height="64" rx="5"></rect>
			<text class="viz-node-label" x="641" y="182">4 · COVERAGE GATE</text>
			<text class="viz-node-value" x="641" y="201">balance · dedup · decontam</text>
			<path d="M559 190H519" style="fill:none;stroke:var(--viz-edge);stroke-width:2"></path>
			<path d="M519 185L506 190L519 195Z" style="fill:var(--viz-edge)"></path>
			<rect class="viz-node viz-node--output" x="341" y="158" width="165" height="64" rx="5"></rect>
			<text class="viz-node-label" x="423" y="182">TRAINING MIX</text>
			<text class="viz-node-value" x="423" y="201">accepted + real · provenance</text>
			<path d="M341 190H301" style="fill:none;stroke:var(--viz-edge);stroke-width:2"></path>
			<path d="M301 185L288 190L301 195Z" style="fill:var(--viz-edge)"></path>
			<rect class="viz-node" x="123" y="158" width="165" height="64" rx="5"></rect>
			<text class="viz-node-label" x="205" y="182">FROZEN STUDENT</text>
			<text class="viz-node-value" x="205" y="201">chosen before final test</text>
			<path d="M28 249H724" style="fill:none;stroke:var(--c-text-soft);stroke-width:3"></path>
			<text class="viz-axis-label" x="28" y="270">CONFIRMATION LANE · FRESH TASKS NEVER FEED GENERATION, FILTERING, OR TRAINING</text>
			<rect class="viz-node viz-node--input" x="493" y="292" width="190" height="60" rx="5" style="stroke-dasharray:5 3"></rect>
			<text class="viz-node-label" x="588" y="316">FRESH HELD-OUT FAMILIES</text>
			<text class="viz-node-value" x="588" y="335">new sources · tasks · difficulty</text>
			<path d="M493 322H438" style="fill:none;stroke:var(--viz-edge);stroke-width:2;stroke-dasharray:5 3"></path>
			<path d="M438 317L425 322L438 327Z" style="fill:var(--viz-edge)"></path>
			<path d="M205 222V272L330 292" style="fill:none;stroke:var(--viz-edge);stroke-width:2"></path>
			<path d="M326 286L339 294L324 296Z" style="fill:var(--viz-edge)"></path>
			<rect class="viz-node viz-node--focus" x="339" y="292" width="86" height="60" rx="5"></rect>
			<text class="viz-node-label" x="382" y="316">FINAL</text>
			<text class="viz-node-label" x="382" y="335">EVALUATION</text>
			<path d="M339 322H284" style="fill:none;stroke:var(--viz-edge);stroke-width:2"></path>
			<path d="M284 317L271 322L284 327Z" style="fill:var(--viz-edge)"></path>
			<rect class="viz-node viz-node--output" x="81" y="292" width="190" height="60" rx="5"></rect>
			<text class="viz-node-label" x="176" y="316">SCOPED CLAIM</text>
			<text class="viz-node-value" x="176" y="335">learned capability, not verifier</text>
			<text class="viz-gradient-label" x="380" y="370">NO RETURN ARROW · final evidence does not select the system.</text>
		</svg>
	</div>
	<figcaption><strong>Read it this way:</strong> follow the top lane into training: correctness evidence and coverage checks decide which generated rows may teach the student. Then cross the heavy line only after freezing the student; fresh held-out families meet it at final evaluation and never flow back into development.</figcaption>
</figure>

## Verification methods

Use the strongest available evidence:

- execute code and tests;
- use a symbolic or numeric solver;
- check simulator state;
- compare with a trusted database;
- require agreement among independent methods;
- ask trained humans to review a sample;
- use a model judge only after human calibration.

A model saying "this answer is correct" is weak evidence when that model generated the answer.

## Diversity and coverage

A generator tends to repeat its common patterns. Large row counts can hide low task diversity.

Measure:

- unique task structures;
- source and template frequency;
- difficulty distribution;
- language and domain coverage;
- exact and semantic duplication;
- answer and style distribution;
- verifier failure by slice;
- overlap with evaluation sets.

Sample from explicit task families instead of asking for unrestricted variety.

## Failure modes

### Error amplification

The student learns generator mistakes that pass weak filters.

### Low diversity

The dataset contains many surface variations of a small number of tasks.

### Generator imitation

The student learns the teacher's style and blind spots instead of the target capability.

### Verifier overfitting

Generated examples are chosen because one verifier accepts them. Training then improves that score without improving the real task.

### Model collapse

Repeated training on model-generated data can narrow the distribution and lose rare behavior. Keep high-quality real data and track coverage across generations.

### Contamination

The generator may reproduce public benchmark items or close variants. Decontaminate against evaluation data before training.

## Small example: code repair data

A team wants synthetic tasks for a coding agent.

For each task:

1. Start from a working repository snapshot.
2. Apply one recorded fault.
3. Confirm that a focused test fails.
4. Keep hidden regression tests unchanged.
5. Ask the agent to repair the repository.
6. Grade focused tests, regressions, patch scope, and forbidden edits separately.
7. Hold out repositories and fault families from training.

This pipeline has clear ground truth. It is stronger than asking a model to invent a bug and judge its own repair.

## Mixing synthetic and real data

Synthetic data should solve a known coverage problem. Keep enough real data to preserve the target distribution and natural variation.

Run mixture experiments. Compare:

- real data only;
- synthetic data only;
- several mixed ratios;
- results by task family and difficulty;
- memorization and contamination checks.

The largest mixture is not automatically the best.

## In an interview

Use this order:

1. Define the missing capability or slice.
2. Choose a generator suited to that target.
3. Define independent verification.
4. Control diversity, difficulty, and duplication.
5. Track provenance and evaluation overlap.
6. Mix with real data and run ablations.
7. Evaluate on held-out families and real tasks.
8. Inspect failures before scaling volume.

## Common mistakes

- Generating data before defining the target gap.
- Using the same model to generate and judge without another check.
- Reporting example count instead of task coverage.
- Keeping only outputs that match one model's style.
- Training on benchmark variants.
- Replacing all real data with synthetic data.
- Evaluating only with the generation verifier.

*Related: [foundation-model data curation](/concepts/foundation-model-data-curation/), [preference data and reward models](/concepts/preference-data-and-reward-models/), and [RL environments and graders](/concepts/rl-environments-and-graders/).*