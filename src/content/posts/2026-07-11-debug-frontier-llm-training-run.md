---
title: "Debug a frontier LLM training run that suddenly diverges"
description: "A loss spike is an incident: preserve evidence, classify the failure, isolate the first bad transition, and recover without hiding the cause."
date: "2026-07-11"
draft: false
tags: ["questions"]
category: "questions"
---

> A large language-model run is stable for 18,000 steps, then loss spikes, gradient norms jump, and two ranks report NaNs. What do you do in the first hour?

Investigate a distributed language-model training incident while preserving evidence and preventing an unsafe or uninformative restart.

## Problem

**Context.** A hypothetical run is stable for 18,000 steps, then loss and gradient norms spike and two ranks report NaNs.
You have the first hour to coordinate containment and diagnosis.

**Task.** Preserve the relevant run state, prioritize hypotheses, choose discriminating replay or numerical checks, and define a safe recovery decision.
Specify communication, ownership, and evidence needed before resuming expensive training.

**Constraints.** Do not overwrite the first bad batch, checkpoints, rank logs, optimizer state, or environment differences.
Separate data, arithmetic, optimizer, collective, and hardware explanations; a successful restart alone does not establish the cause.

## The first-hour sequence

1. **Stop or quarantine the run.** Prevent more bad checkpoints and wasted compute.
2. **Preserve evidence.** Save the last good and first bad checkpoints, batch identifiers, rank-local logs, gradient norms, scaler state, learning rate, data shard, code revision, topology, and hardware events.
3. **Classify the symptom.** Did loss jump globally, on one rank, in one tensor, after a restart, or only in evaluation?
4. **Find the first divergence.** Compare the last good step and first bad step, then narrow by rank, layer, tensor, and sample.
5. **Reproduce cheaply.** Replay the offending batch and checkpoint on fewer devices where possible.
6. **Test one causal family at a time.** Data, numerics, optimizer state, distributed state, hardware, or code/config drift.
7. **Recover from a known-good state.** Apply the smallest justified mitigation and retain the original failure for root-cause work.

<!-- visual:frontier-training-divergence-zoom -->
<figure class="learning-figure editorial-illustration" aria-labelledby="frontier-training-divergence-zoom-edition-title"><p class="visual-title" id="frontier-training-divergence-zoom-edition-title">Find the first failing rank and step before searching for its cause</p><div class="visual-argument" data-visual-argument="frontier-training-divergence-zoom" data-argument-kind="matrix" data-argument-composition="cells"><svg viewBox="0 0 360 240" role="img" aria-label="Locate the first failing rank and step before blaming later symptoms. The first nonfinite value appears at rank 1, step 2. The later failures across all ranks do not identify their original cause."><title>Locate the first failing rank and step before blaming later symptoms</title><desc>The first nonfinite value appears at rank 1, step 2. The later failures across all ranks do not identify their original cause.</desc><text x="105.13333333333333" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="105.13333333333333" dy="0">step 1</tspan></text><text x="199.08" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="199.08" dy="0">step 2</tspan></text><text x="293.02666666666664" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="293.02666666666664" dy="0">step 3</tspan></text><text x="46.16" y="54" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="46.16" dy="0">rank</tspan><tspan x="46.16" dy="18">0</tspan></text><g data-proof-status="value"><rect data-proof-cell="0,0" x="58.16" y="36" width="88.94666666666667" height="59" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1"></rect><text x="102.63333333333333" y="71" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="102.63333333333333" dy="0">ok</tspan></text></g><g data-proof-status="value"><rect data-proof-cell="0,1" x="152.10666666666668" y="36" width="88.94666666666667" height="59" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1"></rect><text x="196.58" y="71" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="196.58" dy="0">ok</tspan></text></g><g data-proof-status="value"><rect data-proof-cell="0,2" x="246.05333333333334" y="36" width="88.94666666666667" height="59" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1"></rect><text x="290.5266666666667" y="71" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="290.5266666666667" dy="0">bad</tspan></text><path data-proof-warning="0,2" d="M250.05333333333334 44v43" style="stroke:var(--viz-warning-stroke);stroke-width:3;stroke-dasharray:3 3"></path></g><text x="46.16" y="118" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="46.16" dy="0">rank</tspan><tspan x="46.16" dy="18">1</tspan></text><g data-proof-status="value"><rect data-proof-cell="1,0" x="58.16" y="100" width="88.94666666666667" height="59" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1"></rect><text x="102.63333333333333" y="135" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="102.63333333333333" dy="0">ok</tspan></text></g><g data-proof-status="value"><rect data-proof-cell="1,1" x="152.10666666666668" y="100" width="88.94666666666667" height="59" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></rect><text x="196.58" y="135" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="196.58" dy="0">FIRST</tspan></text><path d="M157.10666666666668 153L236.05333333333334 153" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path></g><g data-proof-status="value"><rect data-proof-cell="1,2" x="246.05333333333334" y="100" width="88.94666666666667" height="59" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1"></rect><text x="290.5266666666667" y="135" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="290.5266666666667" dy="0">bad</tspan></text><path data-proof-warning="1,2" d="M250.05333333333334 108v43" style="stroke:var(--viz-warning-stroke);stroke-width:3;stroke-dasharray:3 3"></path></g><text x="46.16" y="182" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="46.16" dy="0">rank</tspan><tspan x="46.16" dy="18">2</tspan></text><g data-proof-status="value"><rect data-proof-cell="2,0" x="58.16" y="164" width="88.94666666666667" height="59" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1"></rect><text x="102.63333333333333" y="199" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="102.63333333333333" dy="0">ok</tspan></text></g><g data-proof-status="value"><rect data-proof-cell="2,1" x="152.10666666666668" y="164" width="88.94666666666667" height="59" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1"></rect><text x="196.58" y="199" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="196.58" dy="0">ok</tspan></text></g><g data-proof-status="value"><rect data-proof-cell="2,2" x="246.05333333333334" y="164" width="88.94666666666667" height="59" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1"></rect><text x="290.5266666666667" y="199" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="290.5266666666667" dy="0">bad</tspan></text><path data-proof-warning="2,2" d="M250.05333333333334 172v43" style="stroke:var(--viz-warning-stroke);stroke-width:3;stroke-dasharray:3 3"></path></g></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content"><svg viewBox="0 0 360 423" role="img" aria-labelledby="frontier-training-divergence-zoom-edition-svg-title frontier-training-divergence-zoom-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="frontier-training-divergence-zoom-edition-svg-title">Find the first bad cell, then zoom into its cause</title><desc id="frontier-training-divergence-zoom-edition-svg-desc">A fictional four-step, four-rank grid first becomes non-finite at rank 2, step 3. That cell is traced to layer 6 and a gradient tensor. A one-condition replay tests a candidate explanation; recovery from a known-good checkpoint remains a separate task.</desc><text x="180" y="22" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Fictional alarm grid: find the first bad</tspan><tspan x="180" dy="19">cell</tspan></text><text x="110" y="55" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="110" dy="0">s1</tspan></text><text x="172" y="55" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="172" dy="0">s2</tspan></text><text x="234" y="55" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="234" dy="0">s3</tspan></text><text x="296" y="55" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="296" dy="0">s4</tspan></text><text x="50" y="88" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="50" dy="0">rank 0</tspan></text><rect x="88" y="68" width="44" height="29" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="110" y="87" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="110" dy="0">ok</tspan></text><rect x="150" y="68" width="44" height="29" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="172" y="87" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="172" dy="0">ok</tspan></text><rect x="212" y="68" width="44" height="29" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="234" y="87" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="234" dy="0">ok</tspan></text><rect x="274" y="68" width="44" height="29" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="296" y="87" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="296" dy="0">ok</tspan></text><text x="50" y="125" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="50" dy="0">rank 1</tspan></text><rect x="88" y="105" width="44" height="29" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="110" y="124" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="110" dy="0">ok</tspan></text><rect x="150" y="105" width="44" height="29" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="172" y="124" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="172" dy="0">ok</tspan></text><rect x="212" y="105" width="44" height="29" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="234" y="124" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="234" dy="0">ok</tspan></text><rect x="274" y="105" width="44" height="29" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="296" y="124" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="296" dy="0">ok</tspan></text><text x="50" y="162" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="50" dy="0">rank 2</tspan></text><rect x="88" y="142" width="44" height="29" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="110" y="161" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="110" dy="0">ok</tspan></text><rect x="150" y="142" width="44" height="29" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="172" y="161" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="172" dy="0">ok</tspan></text><rect x="212" y="142" width="44" height="29" rx="3" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.3"></rect><text x="234" y="161" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="234" dy="0">NaN</tspan></text><rect x="274" y="142" width="44" height="29" rx="3" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.3"></rect><text x="296" y="161" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="296" dy="0">NaN</tspan></text><text x="50" y="199" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="50" dy="0">rank 3</tspan></text><rect x="88" y="179" width="44" height="29" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="110" y="198" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="110" dy="0">ok</tspan></text><rect x="150" y="179" width="44" height="29" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="172" y="198" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="172" dy="0">ok</tspan></text><rect x="212" y="179" width="44" height="29" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="234" y="198" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="234" dy="0">ok</tspan></text><rect x="274" y="179" width="44" height="29" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="296" y="198" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="296" dy="0">ok</tspan></text><path d="M234 173L180 244" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><path d="M181.0476412741913,237.07884057685342L180,244L186.38999783197582,241.14204133911204" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><g data-editorial-node><rect x="25" y="248" width="310" height="63" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="180" y="275" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">rank 2 / step 3 -&gt; layer 6 -&gt;</tspan><tspan x="180" dy="19">gradient tensor</tspan></text></g><text x="180" y="345" style="font-size:12px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Replay: change ONE condition; compare first</tspan><tspan x="180" dy="19">failure</tspan></text><text x="180" y="387" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Recovery from a good checkpoint is a separate</tspan><tspan x="180" dy="19">task.</tspan></text></svg><div class="visual-derivation-caption">This fictional grid localizes the first failure; it does not prove its cause. Change one condition in replay. Recovery and diagnosis remain separate.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="frontier-training-divergence-zoom"><strong>Check:</strong> Which observation localizes the first failure? Why does a successful restart not prove its cause?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="frontier-training-divergence-zoom">The first nonfinite value appears at rank 1, step 2. The later failures across all ranks do not identify their original cause.</span></figcaption></figure>

## The failure families

### Data

Corrupt tokens, extreme sequence lengths, changed packing, bad loss masks, malformed labels, duplicated batches, or a sudden mixture shift can create a real gradient outlier. Check token counts, per-domain loss, sequence-length distribution, invalid IDs, mask density, and sample fingerprints.

### Numerical behavior

FP16 overflow, an incorrect loss scale, unstable normalization, softmax or log operations outside FP32, exploding activation norms, or a kernel regression can create Inf or NaN values. Find the first non-finite tensor. "NaNs in the loss" is the end of the chain, not the diagnosis.

### Optimizer and schedule

A bad resume can restore weights but not optimizer moments, scheduler step, gradient scaler, or RNG state. A learning-rate discontinuity, missing clipping, changed accumulation factor, or corrupted moment tensor can destabilize an otherwise valid batch.

### Distributed state

Rank desynchronization, a skipped collective, inconsistent data, stale parameters, partial checkpoint restore, or silent communication corruption can make one rank diverge before the aggregate exposes it. Compare checksums and key tensor statistics across ranks.

### Hardware and software

ECC events, a failing device, network errors, driver or compiler changes, and a new fused kernel belong in the incident timeline. Correlation is not proof, but a failure starting immediately after a topology or binary change deserves isolation.

## What an L4 answer sounds like

> "I would lower the learning rate, enable gradient clipping, and restart from the last checkpoint."

Those mitigations may resume training, but they can also hide corrupt data, a bad restore, or a rank-local failure. The candidate is treating a symptom as an optimization setting.

## What an L5 answer adds

An L5 candidate preserves evidence, identifies the first bad step, compares rank-local metrics, replays the batch, and checks data, numerics, optimizer state, and recent changes in a disciplined order. They distinguish recovery from root cause and validate the repaired run against the pre-incident trajectory.

They propose concrete probes:

- activation and gradient norm by layer;
- first non-finite tensor hooks;
- batch hash and sequence-length summary;
- optimizer moment checksums;
- cross-rank parameter checksums;
- replay with fused kernels disabled;
- replay in FP32 on a reduced model or batch.

## What an L6 answer adds

An L6 candidate runs the incident as a system problem. They assign an evidence-preservation owner, define a safe recovery path, estimate the cost of waiting versus resuming, and prevent the fix from contaminating scientific comparability.

They ask whether the run can continue without invalidating downstream conclusions. A changed data filter, clipping threshold, kernel, or schedule creates a new training regime. The checkpoint lineage and experiment record must say so.

They also turn recurrence into infrastructure:

- automatic finite checks at high-value tensor boundaries;
- rank-local anomaly capture before collective reduction;
- batch lineage sufficient to replay one step;
- checkpoint completeness validation;
- canary jobs for compiler or kernel changes;
- alerts on layerwise norm distributions, not only scalar loss;
- a recovery playbook with an explicit evidence-retention gate.

## Tells that get you a strong-hire vote

- You preserve the first bad transition before restarting.
- You distinguish global from rank-local divergence.
- You identify the first non-finite tensor rather than staring at final loss.
- Data, optimizer, distributed state, and software changes all enter the timeline.
- You reproduce on the smallest faithful setup.
- Recovery and root cause are separate decisions.
- You state whether the mitigation changes experimental validity.

## Tells that get you down-leveled

- "Lower the learning rate" as the opening move.
- Restarting before saving the offending batch and state.
- Looking only at aggregate metrics after reduction.
- Assuming every spike is a bad sample.
- Changing five stabilizers at once.
- Declaring success when the run no longer crashes.

## Common follow-up

"The offending batch trains cleanly when replayed on one GPU. What next?"

That result lowers the probability of a deterministic data or model bug and raises distributed, precision, topology, and timing hypotheses. Reproduce with the original world size, compare rank inputs and RNG, disable overlapping or fused paths, and bisect topology or software changes. Do not conclude that the batch was innocent merely because the reduced setup changed the failure conditions.

Try the [broken training lab](/prep/labs/broken-training/) before using this page as a checklist.

*Related: [train a 100B parameter model](/questions/train-100b-model/), [debug a model that is not learning](/questions/debug-model-not-learning/), and [loss spikes at scale](/concepts/loss-spikes-at-scale/).*
