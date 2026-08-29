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
