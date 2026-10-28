---
title: "Foundation-model data curation"
description: "Source, filter, deduplicate, decontaminate, mix, and audit data as a versioned model input rather than a pile of tokens."
date: "2026-07-11"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Data curation selects a versioned training distribution, with evidence for both retention and rejection.
Mixture weights control exposure, not just how many unique records are stored.

Suppose post-gate corpora A and B contain 600 and 200 eligible token positions.
Unique here means positions in the deduplicated corpus, not distinct vocabulary IDs.
Plan a 1000-token training budget with intended shares $0.6$ and $0.4$.
How often is each source consumed, and can a restricted record be traced to affected checkpoints?

<p class="visual-kicker">One record, explicit decisions</p>
<!-- visual:foundation-data-curation-lineage -->
<figure class="learning-figure editorial-illustration" aria-labelledby="foundation-data-curation-lineage-edition-title"><p class="visual-title" id="foundation-data-curation-lineage-edition-title">Being listed in a manifest is not proof that a sample was consumed</p><svg viewBox="0 0 360 460" role="img" aria-labelledby="foundation-data-curation-lineage-edition-svg-title foundation-data-curation-lineage-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="foundation-data-curation-lineage-edition-svg-title">Being listed in a manifest is not proof that a sample was consumed</title><desc id="foundation-data-curation-lineage-edition-svg-desc">Illustrative stable IDs separate accepted R17 from rejected R18. R18 keeps its policy-rejection reason and has no shard membership. R17 appears in S1 and M1, then has separate consumption evidence before checkpoint C200. Membership and consumption are different claims.</desc><text x="180" y="22" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Illustrative stable IDs; rejection evidence</tspan><tspan x="180" dy="19">is retained</tspan></text><text x="90" y="70" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">R17: ACCEPTED</tspan></text><text x="270" y="70" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">R18: REJECTED</tspan></text><path d="M15 91L345 91" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="180" y="111" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Versioned gate</tspan></text><g data-editorial-node><rect x="15" y="125" width="150" height="67" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="90" y="163.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">policy permits</tspan></text></g><g data-editorial-node><rect x="195" y="125" width="150" height="67" rx="3" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.3"></rect><text x="270" y="163.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">policy denies</tspan></text></g><path d="M15 207L345 207" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="180" y="227" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Data manifest</tspan></text><g data-editorial-node><rect x="15" y="241" width="150" height="73" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="90" y="282.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">S1 -&gt; M1</tspan></text></g><g data-editorial-node><rect x="195" y="241" width="150" height="73" rx="3" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.3"></rect><text x="270" y="263.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">reason retained</tspan><tspan x="270" dy="19">no shard</tspan><tspan x="270" dy="19">membership</tspan></text></g><path d="M15 329L345 329" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="180" y="349" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Before checkpoint C200</tspan></text><g data-editorial-node><rect x="15" y="363" width="150" height="67" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="90" y="392" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">consumption log:</tspan><tspan x="90" dy="19">R17 read</tspan></text></g><g data-editorial-node><rect x="195" y="363" width="150" height="67" rx="3" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.3"></rect><text x="270" y="401.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">not consumed</tspan></text></g></svg><figcaption><strong>Read it this way:</strong> Rejected R18 retains its reason without entering a shard. Accepted R17 needs a consumption record, not just manifest membership, before linking it to C200. <span class="visual-reader-check" data-reader-check="foundation-data-curation-lineage"><strong>Check:</strong> Which receipt proves R17 was consumed, rather than merely present in a manifest?</span></figcaption></figure>
<p class="diagram-source">References: <a href="https://arxiv.org/abs/2406.11794">DataComp-LM</a>, <a href="https://aclanthology.org/2022.acl-long.577/">Lee et al. on deduplication</a>, <a href="https://arxiv.org/abs/2303.03915">ROOTS</a>, and <a href="https://arxiv.org/abs/2203.15556">Hoffmann et al.</a></p>

## Calculate realized exposure

Suppose consumption logs record exactly 600 draws from A and 400 from B.
Intended weights alone would not establish those realized counts.

$$
\operatorname{effective\ epochs}(s)=\frac{\text{realized token draws from }s}{\text{eligible unique token positions in }s}.
$$

| Source | Eligible positions | Intended share | Realized draws | Effective epochs |
| --- | --- | --- | --- | --- |
| A | $600$ | $0.6$ | $600$ | $1$ |
| B | $200$ | $0.4$ | $400$ | $2$ |

B contains one quarter of eligible positions but receives two effective epochs.
Reducing its realized draws to 200 gives one effective epoch, without establishing a capability improvement.

## Keep the decision artifact

R17's record binds its source, parsing version, decision and shard.
M1 also fixes source snapshots, transforms, gates, tokenizer, sampling policy, weights and removal-list version.
T1 binds that manifest; C200 binds actual consumption up to its saved step.
Store a protected reason reference for R18, not secret material in ordinary logs.

A later restriction on R17 can trace through S1, M1 and T1 to affected consumption and checkpoints.
Removing its raw file does not remove derived records or learned effects.

## Change the selection rule

### 1. Deduplicate spans rather than whole documents

**New task:** two documents share a license block but contain different proofs.

**Picture and change:** retain both record identities and unique proofs.
Handle repeated training spans without discarding required legal notices or provenance.
Version the deduplication representation and action.

**Check:** both unique proof sections remain available after handling repetition.

**Use when:** document similarity hides valuable nonrepeated material.

### 2. Tighten a quality threshold

**New task:** a stricter prose filter removes many low-resource-language records.

**Picture and change:** audit accepted and rejected slices, including a controlled below-threshold sample.
Compare threshold variants under equal training budgets.

**Check:** a higher accepted mean score with lower language coverage does not establish a better mixture.
Use fresh-task and coverage evidence.

**Use when:** selection changes the population beyond its intended target.

## A filter cannot validate itself

A polished-prose judge can reject correct informal technical answers.
Its accepted mean score then rises while useful coverage falls.
Independent audit must test the capability and risk that motivated the filter.

## Synthetic data

Synthetic examples can target rare skills or provide verifier checks.
Record generator provenance, verification, diversity and evaluation exclusions.
Generator, filter and student can share the same unchecked error; agreement is not independent validation.

## Measuring curation

Measure source coverage, duplication, policy errors, known evaluation overlap, repetition, task outcomes and memorization.
Audit whether removal requests reach derived shards, manifests and consumed checkpoints.
Decontamination reduces known overlap but cannot establish complete evaluation independence.

## Try changing a source share

A source has 50 eligible token positions and receives 250 draws.
What happens if its realized draws halve at fixed total budget?

<details>
<summary>Check your answer</summary>

Exposure falls from five effective epochs to $2.5$.
Then test whether changed repetition affects memorization or capability; exposure arithmetic alone does not decide.

</details>

<!-- learning:core-end -->

<div data-learning-extension>

## Pipeline stages

### Acquisition and provenance

Record source, usage basis, collection time, language, content type and access constraints before transformations.

### Parsing and normalization

Preserve meaningful code, tables, formulas and dialects while extracting text, normalizing encoding and removing boilerplate.

### Policy and privacy filtering

Check secrets, personal information, malware and source restrictions.
Audit thresholds and false positives by slice; filtering changes the population.

### Quality filtering

Heuristics and learned filters can detect spam or incoherence while also rejecting valuable unusual material.

### Deduplication

Hashes detect exact matches; MinHash, locality-sensitive hashing, suffix or n-gram methods identify particular near-duplicate patterns.
Embedding similarity can collapse distinct examples.
State the unit, threshold and action.

### Decontamination

Search prompts, answers, source overlap and transformed variants.
Exact matching misses formatting and paraphrase changes.

### Mixture design

Choose weights from capability, coverage and risk requirements, not raw size alone.
Track actual repetitions and sampling behavior.

### Versioning and audit

Version raw snapshots, parsing code, filter models, thresholds, dedup indexes, tokenizer, weights and removal lists.
Retain aggregate and sampled audits.

## Common confusions

- Quality depends on the intended task and population, not one score.
- Deduplication changes exposure as well as compute.
- Known-overlap checks do not certify independence.
- Synthetic generation still needs verification and diversity controls.
- Mixture selection is a model-design choice.
- Raw-file deletion is not model unlearning.

## In an interview

Connect the target capability to source decisions, measured exposure and independent evidence.

*Related: [design a foundation-model data platform](/questions/design-foundation-model-data-platform/), [ML data lineage and versioning](/concepts/ml-data-lineage-versioning/), [lessons from Marin 8B](/guides/lessons-from-marin-8b/), and [preference data and reward models](/concepts/preference-data-and-reward-models/).*

</div>
