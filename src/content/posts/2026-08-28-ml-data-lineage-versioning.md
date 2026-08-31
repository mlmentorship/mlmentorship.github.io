---
title: "ML data lineage, versioning, and reproducibility"
description: "Trace a model from raw data through transforms, features, labels, training, evaluation, and deployment. Version each contract for replay, rollback, deletion, and incident response."
date: "2026-08-28"
draft: false
tags: ["concepts"]
category: "concepts"
aliases: ["ML data lineage", "dataset versioning", "model lineage", "training data provenance", "ML reproducibility", "data deletion lineage"]
roles: ["Machine Learning Engineer", "Research Engineer", "ML Platform", "Safety and Evals"]
rounds: ["ML infrastructure", "ML system design", "Project deep-dive"]
difficulty: "Intermediate"
priority: "Core"
prerequisites: ["reproducibility-fair-model-comparison", "data-leakage-point-in-time-correctness"]
---

## Summary

ML lineage records how raw data, transformation code, feature and label definitions, training configuration, evaluation, and deployment produced a model. Versioning gives each input and contract a stable identity.

Together they answer three operational questions: Can the team reproduce this result? Which models are affected by a bad data source? Which compatible artifacts must be restored during rollback?

## The lineage graph

A useful lineage graph includes:

```text
raw sources
  -> snapshots and manifests
  -> parsing and validation
  -> feature and label definitions
  -> training dataset
  -> code, config, environment, and seeds
  -> model artifact
  -> evaluation artifact
  -> deployment and policy version
```

Each node needs an identifier. Each edge records the transformation and its inputs.

A dashboard link is not enough. The metadata must remain queryable after jobs, tables, or deployments change.

## What to version

Version the contracts that can change model behavior:

- raw snapshot or source offsets;
- schema and semantic field definitions;
- transformation code and parameters;
- event-time and availability-time rules;
- feature definitions;
- label definition and maturity window;
- sampling and filtering rules;
- train, validation, and test split manifests;
- model code and configuration;
- dependencies, container, hardware, and precision;
- random seeds and checkpoint source;
- evaluation data, prompts, rubrics, and scoring code;
- serving feature and threshold versions.

A dataset version should identify content, not only a mutable table name. Use immutable snapshots, manifests, transaction-log versions, or stable source offsets.

## Dataset manifests

A dataset manifest can store:

- snapshot ID and creation time;
- source IDs or partitions;
- schema version;
- row or sample counts;
- time range;
- hashes or checksums;
- transformation commit;
- feature and label spec versions;
- exclusion and deletion lists;
- quality statistics;
- access policy.

Large datasets do not require a full physical copy for every version. A manifest can reference immutable files or a versioned table. The references must remain valid for the retention period.

## Backward and forward tracing

**Backward tracing** starts from a model or prediction and asks which data, code, and configuration produced it.

**Forward tracing** starts from a bad source, transform, or label and asks which datasets, models, evaluations, and deployments depend on it.

Both directions matter during incidents. A team that can reproduce one model but cannot find all affected models still lacks operational lineage.

<!-- visual:ml-lineage-two-way-incident-trace -->
<figure class="learning-figure plot-panel" aria-labelledby="lineage-trace-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="lineage-trace-title">Use one immutable graph for backward reconstruction and forward blast-radius tracing</p>
	<svg viewBox="0 0 360 430" role="img" aria-labelledby="lineage-trace-svg-title lineage-trace-svg-desc">
		<title id="lineage-trace-svg-title">Two directions through one versioned ML lineage graph</title>
		<desc id="lineage-trace-svg-desc">An original lineage graph begins with raw source payments at snapshot 42, then currency transform version 2, then training manifest 88. The manifest branches to training runs 501 and 502, which produce model A version 17 and model B version 9. Model A version 17 reaches production deployment 31, while model B version 9 reaches shadow deployment 12. A dashed backward trace starts at production deployment 31 and follows model A, training run 501, manifest 88, transform version 2, and snapshot 42 to reconstruct its inputs. Solid forward traces start at the bad currency transform version 2 and reach both models and both deployments, exposing the complete blast radius.</desc>
		<rect class="viz-plot-bg" x="4" y="4" width="352" height="422" rx="6"></rect>
		<text class="viz-callout" x="16" y="24">DEPENDENCIES · each node has an immutable ID</text>
		<rect class="viz-node viz-node--input" x="70" y="38" width="220" height="42" rx="5"></rect>
		<text class="viz-node-label" x="180" y="55">raw source · payments</text><text class="viz-node-value" x="180" y="70">snapshot @42</text>
		<path d="M180 80V94" style="fill:none;stroke:var(--viz-edge);stroke-width:2"></path><path d="M174 90L180 98L186 90Z" style="fill:var(--viz-edge)"></path>
		<rect class="viz-node viz-node--focus" x="70" y="98" width="220" height="48" rx="5" style="stroke-dasharray:6 3"></rect>
		<text class="viz-node-label" x="180" y="116">currency transform @v2</text><text class="viz-node-value" x="180" y="134">incident: dollars interpreted as cents</text>
		<path d="M180 146V160" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2.5"></path><path d="M174 156L180 164L186 156Z" style="fill:var(--viz-focus-stroke)"></path>
		<rect class="viz-node viz-node--output" x="70" y="164" width="220" height="42" rx="5"></rect>
		<text class="viz-node-label" x="180" y="181">training manifest @88</text><text class="viz-node-value" x="180" y="196">content-addressed inputs + contracts</text>
		<path d="M180 206V220H91V234M180 220H269V234" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2.5"></path>
		<path d="M85 230L91 238L97 230ZM263 230L269 238L275 230Z" style="fill:var(--viz-focus-stroke)"></path>
		<rect class="viz-node" x="12" y="238" width="158" height="54" rx="5"></rect>
		<text class="viz-node-label" x="91" y="256">training run @501</text><text class="viz-node-value" x="91" y="273">model A @17</text><text class="viz-label" x="91" y="286">evaluation @A17</text>
		<rect class="viz-node" x="190" y="238" width="158" height="54" rx="5"></rect>
		<text class="viz-node-label" x="269" y="256">training run @502</text><text class="viz-node-value" x="269" y="273">model B @9</text><text class="viz-label" x="269" y="286">evaluation @B9</text>
		<path d="M91 292V306M269 292V306" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2.5"></path>
		<path d="M85 302L91 310L97 302ZM263 302L269 310L275 302Z" style="fill:var(--viz-focus-stroke)"></path>
		<rect class="viz-node viz-node--output" x="12" y="310" width="158" height="48" rx="5"></rect>
		<text class="viz-node-label" x="91" y="329">production @31</text><text class="viz-node-value" x="91" y="346">model A + feature contract @7</text>
		<rect class="viz-node viz-node--output" x="190" y="310" width="158" height="48" rx="5"></rect>
		<text class="viz-node-label" x="269" y="329">shadow @12</text><text class="viz-node-value" x="269" y="346">model B + feature contract @7</text>
		<path d="M44 310H8V122H64" style="fill:none;stroke:var(--viz-edge);stroke-width:2;stroke-dasharray:6 4"></path><path d="M58 116L68 122L58 128Z" style="fill:var(--viz-edge)"></path>
		<text class="viz-callout" x="16" y="380">BACKWARD ← start at production @31: what made it?</text>
		<path class="viz-operating-guide" d="M16 389H344" style="stroke-dasharray:6 4"></path>
		<text class="viz-callout" x="16" y="410">FORWARD → start at transform @v2: what is affected?</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> the arrows record creation dependencies, not merely pipeline order. To reproduce <code>production @31</code>, trace backward through its exact model, run, manifest, transform, and source snapshot. When <code>transform @v2</code> is found faulty, traverse forward through every branch: both models and both deployments are in scope. Stable IDs make both answers possible after jobs and mutable table names have changed.</figcaption>
</figure>

## Point-in-time lineage

Store event time and availability time for temporal features. Also store the query or feature-view version that enforced the cutoff.

Without this information, an old training dataset may be physically reproducible but logically wrong. Replaying a current query against historical tables can introduce future data or changed backfills.

Lineage should identify the exact historical feature values or the immutable inputs needed to rebuild them.

## Reproducibility levels

Several guarantees are useful:

1. **Artifact reproducibility:** retrieve the exact model and dataset artifacts.
2. **Pipeline reproducibility:** rerun the recorded pipeline from fixed inputs.
3. **Statistical reproducibility:** obtain compatible results despite nondeterministic hardware or training.
4. **Decision reproducibility:** reconstruct why a model passed its launch gate.

Bit-for-bit equality may be infeasible for distributed training. Record tolerances and expected variation instead of claiming exact determinism.

## Schema and feature compatibility

A model depends on feature names, types, shapes, units, missing-value rules, and category vocabularies. A rollback can fail when the old model receives a new incompatible feature contract.

Version models and feature views together. Validate compatibility before deployment. Preserve the serving path required by the rollback model or provide an explicit adapter with tests.

## Deletion and retention

A deletion request or source restriction should support forward tracing:

1. Find raw records and derived samples.
2. Update exclusion or suppression lists.
3. Identify affected datasets and models.
4. Apply the required removal, retraining, or approved unlearning process.
5. Verify serving and future training no longer use the data.
6. Record the action without retaining prohibited content.

Deleting a raw file does not remove its influence from a trained model. The correct response depends on law, policy, risk, and available technical controls.

Retention rules must balance reproducibility, storage cost, privacy, licenses, and incident needs.

## Rollback

A safe rollback restores a compatible set:

- model artifact;
- preprocessing code;
- feature contract;
- calibration and thresholds;
- policy configuration;
- evaluation and approval record.

Rolling back only model weights can produce training-serving skew or invalid decisions.

## Worked incident

A team changes a currency feature from dollars to cents without changing the field name. Model quality drops after the next training run.

Backward lineage identifies the new feature transform and dataset version. Forward lineage finds two trained models and one shadow deployment that consumed it. The team restores the prior feature view, retrains from the last valid snapshot, and adds unit and range checks to the schema contract.

Without versioned feature definitions, the same rows and code commit would not explain the change.

## System design

A practical platform separates:

- immutable artifacts in object storage or versioned tables;
- a metadata service for nodes, edges, schemas, and ownership;
- orchestration events that record run inputs and outputs;
- a model registry for approval and deployment state;
- access controls and audit logs;
- retention and deletion workflows.

Prefer automatic capture from pipelines. Manual metadata fields decay quickly. Allow human annotations for decisions that code cannot infer.

## In an interview

Use this order:

1. State the incident, replay, deletion, and rollback goals.
2. Draw the lineage graph from source to deployment.
3. Give every artifact and contract an immutable version.
4. Support backward and forward tracing.
5. include event-time and availability-time rules.
6. Define reproducibility and retention guarantees.
7. Roll back model, features, calibration, and policy together.

## Common mistakes

- Versioning a table name without versioning its contents.
- Recording model code but not data and label definitions.
- Recomputing historical features with current logic.
- Claiming exact reproducibility on nondeterministic hardware without a tolerance.
- Treating a model registry as complete data lineage.
- Deleting raw data while ignoring derived artifacts and models.
- Rolling back weights without the matching feature contract.

## Practice next

Apply this design in [multi-team ML platform design](/questions/design-multi-team-ml-platform/), [feature-store design](/questions/design-feature-store/), [point-in-time correctness](/concepts/data-leakage-point-in-time-correctness/), and [foundation-model data curation](/concepts/foundation-model-data-curation/).
