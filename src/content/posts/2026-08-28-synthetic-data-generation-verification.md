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