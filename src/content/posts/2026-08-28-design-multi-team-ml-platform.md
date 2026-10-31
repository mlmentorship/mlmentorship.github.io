---
title: "Design a multi-team ML platform"
description: "A staff-level case covering platform scope, data contracts, training, deployment, governance, migration, ownership, cost, and adoption across product teams."
date: "2026-08-28"
draft: false
tags: ["questions", "system-design", "staff", "principal", "ml-platform"]
category: "questions"
aliases: ["ML platform system design", "multi-team machine learning platform", "model platform", "ML developer platform"]
roles: ["Applied Scientist", "Machine Learning Engineer", "Research Scientist", "Research Engineer"]
rounds: ["ML system design", "Systems and infrastructure", "Technical strategy"]
difficulty: "Advanced"
priority: "Role-specific"
prerequisites: ["ml-data-lineage-versioning", "design-feature-store", "design-ml-monitoring"]
---

Design a shared ML platform that improves reproducibility and release control without forcing eight teams into one execution runtime.

<span id="the-prompt"></span>

## Problem

**Context.** Eight ML teams with about 50 practitioners and 30 models operate fragmented pipelines.
The detailed scenario below includes retries and regional failure.

**Task.** Define a two-year platform direction, a useful first-quarter release, incremental adoption, and authoritative publication, promotion, and recovery.
Resolve how attempts A and B publish results for one logical run while existing executors remain.

**Constraints.** No wholesale rewrite or forced migration is allowed.
Separate execution attempts from committed release identity, preserve deletion and access obligations, and state ownership and degraded-operation rules.

An organization has eight ML teams, about 50 scientists/engineers and roughly 30 models.
Separate pipelines produce irreproducible data, training-serving skew, incomparable experiments, ambiguous releases, idle compute and poorly traced deletions.
Some teams avoid the shared platform because it slows delivery.

Design a two-year direction with a useful production release this quarter, without a rewrite or forced migration.
Two validated attempts, A and B, belong to logical training run R.
A finishes first; B finishes later.
The deployment client may retry, and a metadata region can fail during canary.
Define authoritative publication, promotion and recovery while existing executors remain.

## Picture: Many Attempts, One Published Result

Allocate R before execution; A and B write private artifact paths.
Both may execute, but publication changes one shared record using expected-state compare-and-swap.

| Event | Expected state | Authoritative result |
| --- | --- | --- |
| R created | No run record | `pending`, version 0 |
| A publishes validated artifact A | `pending`, version 0 | Artifact A, version 1 |
| B finishes and tries publication | `pending`, version 0 | Rejected; A remains authoritative |
| Deployment request K for A | Evidence valid, caller authorized | One operation D1 |
| Retry K with the same arguments | K already binds this request | Return D1 |
| Reuse K for different arguments | Binding differs | Reject, not a second rollout |

B's artifact and costs remain traceable under its attempt identity.
Do not advertise exactly-once execution; the property is one unambiguous publication and deployment operation.
Deduplication cannot repair missing evidence, incompatible schemas or wrong model semantics.

<p class="visual-kicker">Execution identity versus publication authority</p>
<!-- visual:multi-team-platform-contract-boundary -->
<figure class="learning-figure editorial-illustration" aria-labelledby="multi-team-platform-contract-boundary-edition-title"><p class="visual-title" id="multi-team-platform-contract-boundary-edition-title">A stale attempt cannot overwrite the first valid committed result</p><div class="visual-argument" data-visual-argument="multi-team-platform-contract-boundary" data-argument-kind="matrix" data-argument-composition="cells"><svg viewBox="0 0 360 176" role="img" aria-label="Only the first valid compare-and-swap owns the result. A commits version 0 to 1. B still expects version 0, so its later write is rejected rather than overwriting A."><title>Only the first valid compare-and-swap owns the result</title><desc>A commits version 0 to 1. B still expects version 0, so its later write is rejected rather than overwriting A.</desc><text x="126.48333333333335" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="126.48333333333335" dy="0">expected</tspan></text><text x="211.89000000000001" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="211.89000000000001" dy="0">actual</tspan></text><text x="297.29666666666674" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="297.29666666666674" dy="0">result</tspan></text><text x="71.78" y="54" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="71.78" dy="0">attempt</tspan><tspan x="71.78" dy="18">A</tspan></text><g data-proof-status="value"><rect data-proof-cell="0,0" x="83.78" y="36" width="80.40666666666668" height="59" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1"></rect><text x="123.98333333333335" y="71" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="123.98333333333335" dy="0">0</tspan></text></g><g data-proof-status="value"><rect data-proof-cell="0,1" x="169.18666666666667" y="36" width="80.40666666666668" height="59" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1"></rect><text x="209.39000000000001" y="71" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="209.39000000000001" dy="0">0</tspan></text></g><g data-proof-status="value"><rect data-proof-cell="0,2" x="254.59333333333336" y="36" width="80.40666666666668" height="59" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></rect><text x="294.7966666666667" y="71" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="294.7966666666667" dy="0">v1</tspan></text><path d="M259.59333333333336 89L330.00000000000006 89" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path></g><text x="71.78" y="118" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="71.78" dy="0">attempt</tspan><tspan x="71.78" dy="18">B</tspan></text><g data-proof-status="value"><rect data-proof-cell="1,0" x="83.78" y="100" width="80.40666666666668" height="59" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1"></rect><text x="123.98333333333335" y="135" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="123.98333333333335" dy="0">0</tspan></text></g><g data-proof-status="value"><rect data-proof-cell="1,1" x="169.18666666666667" y="100" width="80.40666666666668" height="59" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1"></rect><text x="209.39000000000001" y="135" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="209.39000000000001" dy="0">1</tspan></text></g><g data-proof-status="stopped"><rect data-proof-cell="1,2" x="254.59333333333336" y="100" width="80.40666666666668" height="59" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1"></rect><path d="M286.7966666666667 110l16 16m0-16l-16 16" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:2.5"></path><text x="294.7966666666667" y="146" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="294.7966666666667" dy="0">REJECT</tspan></text><path data-proof-warning="1,2" d="M258.59333333333336 108v43" style="stroke:var(--viz-warning-stroke);stroke-width:3;stroke-dasharray:3 3"></path></g></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content"><svg viewBox="0 0 360 405" role="img" aria-labelledby="multi-team-platform-contract-boundary-edition-svg-title multi-team-platform-contract-boundary-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="multi-team-platform-contract-boundary-edition-svg-title">The first valid commit wins; a late worker cannot overwrite it</title><desc id="multi-team-platform-contract-boundary-edition-svg-desc">Attempts A and B share run R at expected version zero. A commits artifact A and advances the run to version one. B then fails its stale version-zero compare-and-swap and retains only attempt evidence. Deployment key K retries the same D1, while writer epochs remain separately checked.</desc><text x="180" y="22" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">One run R, two attempts, expected version 0</tspan></text><g data-editorial-node><rect x="15" y="53" width="150" height="61" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="90" y="79" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">attempt A</tspan><tspan x="90" dy="19">artifact A</tspan></text></g><g data-editorial-node><rect x="195" y="53" width="150" height="61" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="270" y="79" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">attempt B</tspan><tspan x="270" dy="19">artifact B</tspan></text></g><path d="M90 114L90 148" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75"></path><path d="M86.64402122977057,141.8569220667674L90,148L93.35597877022943,141.8569220667674" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75"></path><path d="M270 114L270 210" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><path d="M266.6440212297706,203.8569220667674L270,210L273.3559787702294,203.8569220667674" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><g data-editorial-node><rect x="15" y="152" width="150" height="61" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.3"></rect><text x="90" y="168.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">A commits v0 -&gt;</tspan><tspan x="90" dy="19">v1</tspan><tspan x="90" dy="19">artifact A wins</tspan></text></g><g data-editorial-node><rect x="195" y="214" width="150" height="61" rx="3" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.3"></rect><text x="270" y="230.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">B expects v0</tspan><tspan x="270" dy="19">actual v1:</tspan><tspan x="270" dy="19">reject</tspan></text></g><text x="180" y="312" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">B cannot overwrite A</tspan><tspan x="180" dy="19">its attempt evidence remains</tspan></text><text x="180" y="367" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Retry deployment key K -&gt; same D1; recheck</tspan><tspan x="180" dy="19">writer epoch.</tspan></text></svg><div class="visual-derivation-caption">A advances the version before B arrives. B cannot replace the winning artifact. Deployment idempotency and writer authority remain separate contracts.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="multi-team-platform-contract-boundary"><strong>Check:</strong> Why can late attempt B not replace A? Why must retrying deployment key K return D1?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="multi-team-platform-contract-boundary">A commits version 0 to 1. B still expects version 0, so its later write is rejected rather than overwriting A.</span></figcaption></figure>

<span id="state-the-decision-before-drawing-boxes"></span>

## Standardize the Boundary, Not Every Runtime

Share immutable identities, lineage, evidence, promotion policy and deployment state.
Adapters expose existing training, feature and serving capabilities without pretending their semantics are identical.
Keep high-volume payloads outside the transactional metadata store.
Pick an online-serving pilot and a scheduled/accelerator-training pilot with willing owners.
Compare release time, reliability, support burden and useful cost before expanding shared execution.

## Establish Invariants and Service Levels

| Invariant | Required contract |
| --- | --- |
| Immutable identity | Resolve code, configuration, environment, inputs, evidence and artifact digests |
| Authorized promotion | Record actor, policy, evidence and expected-state transition |
| Temporal inputs | Source, transformation, entity, event/availability times and freshness |
| Unambiguous retries | Logical versus attempt IDs, private writes and atomic validated publication |
| Compatible rollback | Retain prior artifacts, serving dependencies and rollback window; no retraining required |
| Bounded shared failures | Quotas, isolated rollout, degradation and tested recovery |
| Temporary exceptions | Named owner, reason, bounded scope and review date |

Service levels belong to individual capabilities and product risk, not one platform-wide percentage.
The reference below gives scenario targets, not guarantees.

## Change 1: Deployment Fits, Shared Training Does Not

**New task.** A research team accepts release controls but rejects the shared training executor.

**Picture and change.** Keep identity, evidence and promotion authority shared; permit a justified adapter or local execution path.
Preserve resource, output and failure semantics explicitly.

**Check.** Partial adoption can be a useful platform boundary, not noncompliance.
Compare delivered outcomes and support before imposing consolidation.

**Use when.** Workload semantics differ materially.

## Change 2: Metadata Fails During Canary

**New task.** The metadata region fails while approved versions serve a bounded canary.

**Picture and change.** Retain the last approved compatible serving/fallback state; freeze unverifiable promotions.
Recover a fenced writer with declared recovery point/time.
Enforce controller epoch and desired-state version at the mutation boundary, not only inside the controller.

**Check.** An old controller cannot overwrite a newer approved rollout after failover.
Return incomplete lineage as incomplete, not an authoritative empty dependency set.

**Use when.** Serving availability and control-plane availability diverge.

## Change 3: The Pilots Do Not Deliver Value

**New task.** After two quarters, neither pilot releases faster and support load rises.

**Picture and change.** Inspect delays, workarounds, reliability gains and ownership.
Repair a narrow contract, narrow the platform or stop the capability; fund exit and retirement work.

**Check.** Forced usage and higher utilization do not establish useful delivery.

**Use when.** Centralization has not earned its cost.

## Failure: Two Controllers Own Production

Unfenced writers or split migration authority can each issue apparently valid but conflicting desired state.
Require one authoritative transition path, version checks and fencing during handover.

## Transfer Check

Utilization rises from 35% to 70%, but experiment cycle time doubles.
Did scheduling improve?

### Reasoned Answer

Not necessarily. Separate queue/execution time, useful experiment decisions, reliability, priorities and total cost.
Occupancy can rise while the platform's user outcome worsens.

<!-- learning:core-end -->
<div class="learning-extension" data-learning-extension>

## Clarify the organization and workload

<span id="product-and-risk"></span>
<span id="workload"></span>
<span id="organization"></span>

| Boundary | Questions that change the design |
| --- | --- |
| Product/risk | Consequences, review/audit, rollback deadline, reliable existing paths and shared-feature blast radius? |
| Workload | Arrival rate, accelerators/memory/distribution, batch/stream/online, service targets and data/artifact/read sizes? |
| Organization | Operators, source/semantic owners, extension contributors, supported systems and failed prior standardization? |

Assume warehouse, object storage, stream bus, container compute and two serving patterns.
Most models train daily/weekly; five need sub-20-ms online features; four require approval/audit.

## Define success for the platform

<span id="delivery"></span>
<span id="reliability"></span>
<span id="quality-and-safety"></span>
<span id="efficiency"></span>

| Goal | Outcome evidence |
| --- | --- |
| Delivery | Change-to-shadow/setup time, ticket-free releases, reproduction and diagnosis time |
| Reliability | Rollback time/success, clean job recovery, feature SLOs, skew incidents and missing release evidence |
| Quality/safety | Critical slices, staged regressions, expiring exceptions, deletion proof and valid evaluations |
| Efficiency | Workload utilization, retained storage, team/product cost, retired duplicates and support versus reusable work |

Compare both pilots against their baselines, including migration/support costs; forced component usage is not useful delivery.

## Capability-Specific Service Levels

| Capability | Example target | Failure behavior |
| --- | --- | --- |
| Metadata reads | 99.9% monthly availability | Show last known state; block unsafe promotion |
| Training submission | 99.5% acceptance within one minute | Queue locally; do not lose the request |
| Online feature read | Product-specific p99 and availability | Fall back to cached, default, or reduced model path |
| Model deployment | Start staged rollout within ten minutes | Leave current version serving |
| Rollback | Restore a compatible version within five minutes | Invoke product fallback if restoration fails |
| Lineage query | Complete common dependency query within seconds | Return partial results with missing edge state |

Derive real targets from product risk and current baselines.

## Separate the control plane from data planes

The control plane owns desired state, identity, policy and evidence; data planes scan, train, materialize and infer.
This permits independent execution changes and failures without immediate migration.
Adapters must expose preemption, region, hardware, consistency and rollback differences, not hide them.

## Use immutable identities and explicit lineage

A model version should reference immutable objects rather than mutable names.

```text
ModelVersion
  model_id
  version_id
  source_revision
  package_digest
  environment_digest
  training_run_id
  dataset_snapshot_ids[]
  feature_view_versions[]
  evaluation_bundle_id
  artifact_uri
  artifact_digest
  input_schema_version
  output_schema_version
  owner
  created_at
```

Aliases such as `fraud-primary` may move; immutable versions do not.
Typed lineage connects events, snapshots, features, runs, artifacts, evaluation, deployment, predictions and outcomes.
Keep identities/pointers/edges in metadata and high-cardinality payloads in appropriate stores.

### Reproduction levels

Declare traceable inputs/transformations, re-runnable specifications, numerical tolerance or bitwise identity separately.
Hardware/library changes can defeat bitwise identity; record the workload's actual promise and test it.

## Build the data and feature contract

Do not require online materialization for every workload.

```text
FeatureDefinition
  name, owner, entity_keys, dtype, null_policy
  source_id, transformation_revision
  event_time, availability_time, freshness_slo
  retention, privacy_class, fallback, consumers, deprecation
```

### Point-in-time training data

Distinguish event time, platform ingestion and production availability.
Historical assembly selects the latest valid feature available no later than the prediction time.
An earlier event timestamp does not make a later correction available to that prediction.

### Batch and streaming consistency

Choose a shared transformation, a shared versioned event log, or independently implemented paths with continuous sampled comparison.
Record discrepancies; similar-looking SQL and streaming code need not agree.

### Late and corrected data

Declare watermark and late-event policy: ignore, update future state or create revised history.
Corrections create new snapshots rather than mutate old evidence.

## Make training runs reproducible and schedulable

A training submission should be a declarative run specification.

```text
TrainingRunSpec
  code_package
  environment
  command
  input_snapshots
  feature_versions
  parameters
  random_seed_policy
  resource_request
  retry_policy
  output_contract
  owner
  budget_tag
```

The orchestrator validates the specification, resolves immutable inputs, and assigns a run identifier. An execution adapter translates the request to the current batch or accelerator system.

### Retry semantics

The core's R/A/B ledger defines private attempts and atomic publication.
Keep every attempt's outcome and resource use, including failed or losing attempts.

### Resource scheduling

Separate debugging, production retraining, bounded experiments, sweeps and emergency work.
Enforce team budgets, job limits, queue age and preemption; declare gang scheduling and checkpoint support.
Justify reservations and report queue time separately from execution and useful decisions.

### Experiment comparison

Record baseline, changed variables, shared snapshots, budgets, repeated seeds, metrics/guardrails and decision status.
Mechanical checks support, not replace, scientific review by the owning team.

## Treat evaluation as a versioned product

```text
EvaluationBundle
  model_version, dataset, slices, metric_code_versions
  uncertainty_or_repeated_runs, baseline_comparison
  calibration, thresholds, risk_tier_checks
  exclusions, unresolved_failures, reviewers, expires_at
```

Freeze and link the bundle to its exact model.

### Promotion policy

Mechanically check required metrics/slices, schema compatibility, complete lineage, evidence validity and reviewer authority.
Domain trade-offs remain explicit decisions; bounded deviations record approver, rationale, scope and expiration.

### Evaluation expiration

Source schemas, label definitions, policy and serving interfaces can invalidate dependent evidence before its calendar expiry.
Mark affected models for review rather than reuse a stale green status.

## Design deployment as a state machine

A deployment controller should reconcile declared state with serving systems.

```text
registered -> evaluated -> approved -> shadow -> canary -> production -> retired
```

Transitions require evidence and authorization. The controller records who requested the transition, what changed, and which policy allowed it.

### Compatibility checks

Verify features/input schema, output consumers, preprocessing/tokenization, runtime/hardware/memory, fallback and regional/privacy rules.
Declare compatibility ranges; breaking interfaces need versioned endpoints or justified dual-read/write migration.

### Release stages

| Stage | Evidence/limit |
| --- | --- |
| Offline | Contract, quality and policy checks |
| Shadow | Integration without action exposure; not causal user lift |
| Canary | Bounded exposure/stops; small samples can miss regressions |
| Experiment | Justified randomization for product effects |
| Progressive/full | System/product guardrails and retained compatible rollback version |

### Rollback

Switch authorized desired state to a compatible prior version without rebuilding or retraining.
Restore required feature/schema dependencies too; an old artifact against incompatible inputs is not a valid rollback.

## Join model and system observability

Monitor system latency, errors, queues, resources and cost.
Track data validity and freshness, model behavior, supported quality and mature outcomes, including harm.
Each alert names owner, severity, deadline and runbook.

### Correlation identifiers

A prediction record should link request, model version, feature versions, and deployment stage. Sensitive raw features should stay out of general logs.

Use sampled traces for detailed debugging and aggregated metrics for broad monitoring. Retention should follow privacy and incident requirements.

### Delayed outcomes

Join outcomes by stable decision identifiers. Report metric maturity so teams can distinguish early partial labels from complete windows.

For selective labels, preserve exploration or audit samples where policy permits. Otherwise the monitoring loop can confirm only outcomes for actions the current model chose.

## Design for failure and degraded operation

<span id="metadata-service-unavailable"></span>
<span id="feature-freshness-breach"></span>
<span id="training-worker-loss"></span>
<span id="corrupt-dataset-snapshot"></span>
<span id="bad-shared-transformation"></span>
<span id="region-failure"></span>
<span id="accidental-duplicate-deployment-request"></span>

| Failure | Safe response |
| --- | --- |
| Metadata | Continue approved serving, freeze unverifiable promotion; queue only recoverable submissions |
| Freshness | Product-approved cached/default/reduced/simple/refusal fallback; no invented semantics |
| Worker | Supported checkpoint recovery, private attempt output, no partial publication, waste recorded |
| Corrupt snapshot | Quarantine dependencies found by lineage and preserve incident evidence |
| Shared transform | Stop materialization, invalidate dependent features, find deployments; stronger shared-blast-radius rollout |
| Region | Declared recovery point/time, fenced failover, regional serving state; no unnecessary active-active writers |
| Duplicate request | Same logical key/arguments returns the same operation |

## Add security, privacy, and governance at the contracts

Separate user/workload authentication and authorize by team, environment, class and action.
Use short-lived credentials, package signatures/digests, isolated secrets, export controls, dependency/image checks and classified retention.
Audit promotion, exception and access events.

### Privacy deletion

Trace subject/source identity to derived snapshots, features, runs and models.
Approved policy selects deletion, rebuilding, expiration or proof of non-impact; record residual copies.
Graph traversal does not remove learned influence from trained parameters.

### Risk tiers

Tier by consequence, reversibility, sensitivity, regulatory exposure and autonomy.
Match evidence, reviewers, retention and rollout controls to risk; preserve a justified low-risk fast path.

## Control multi-tenancy and cost

Isolate namespaces; enforce quotas, budgets, priorities, API limits, retention and expiring reservations.
Attribute costs by run/deployment/team/product and test noisy neighbors.

### Chargeback versus showback

Showback reports usage without budget transfer.
Chargeback needs reliable attribution and team control over charged decisions; uncontrollable overhead encourages avoidance.

### Unit economics

Track cost per completed run, accepted decision, thousand predictions, artifact-month and onboarded-team support.
Cheap accelerator-hours can coexist with wasted experiments.

## Decide what to build, buy, or keep local

Compare differentiation, existing scale/policy/integration fit, migration, incident operation and accumulated exit cost per capability.
Adopt suitable tracking/storage, integrate identity/lineage/policy/deployment, keep specialized runtimes local and defer unproven universal feature services.

### Exit costs

Keep metadata exportable. Store artifacts in open formats and stable object paths. Put policy and run specifications in versioned code.

Vendor abstraction should protect identities and evidence. Do not build a lowest-common-denominator wrapper over every vendor feature. That wrapper becomes another platform with worse capabilities.

## Migrate through representative slices

<span id="phase-0-baseline-and-select-pilots"></span>
<span id="phase-1-identity-and-evidence"></span>
<span id="phase-2-paved-workflows"></span>
<span id="phase-3-shared-data-and-scheduling"></span>
<span id="phase-4-deprecate-with-evidence"></span>

| Phase | Evidence before expansion |
| --- | --- |
| 0 | Baseline setup/release/incidents/reproduction/support; representative online and training pilots |
| 1 | Identity, run/artifact/evaluation/deployment records; reproduce, inspect and roll back through existing executors |
| 2 | Templates/self-service/policy/adapters; ticket-free completion and first-shadow time |
| 3 | Feature/time/queue/cost sharing justified by duplication or reliability, not adoption quotas |
| 4 | Tested consumers, reliability, owner/deadline, compatibility window and rollback before retiring an old path |

### Migration safety

Use dual registration or shadow metadata before the platform controls production. Compare the platform's reconstructed lineage and deployment state against the current source of truth.

When confidence is high, move one authority at a time. Split authority creates incidents when two systems both believe they control production.

## Define ownership as part of the architecture

<span id="platform-team-owns"></span>
<span id="product-ml-teams-own"></span>
<span id="data-owners-own"></span>
<span id="risk-or-review-functions-own"></span>

| Accountable owner | Duties |
| --- | --- |
| Platform | Control durability/availability, APIs/clients, common adapters, policy mechanics, telemetry/incidents, migration and deprecation |
| Product ML | Objectives/labels, feature semantics, evaluation/slices, guardrails/fallback, model incidents, approvals/exceptions |
| Data | Source quality/semantics, backfills/corrections, privacy and retention |
| Risk/review | Requirements, tiered approval authority, exceptions and audit interpretation |

## Treat adoption as an engineering problem

### Build a paved path

Provide working templates, simple submission, automatic lineage, default evaluation/release, actionable errors and bounded escape hatches.
Repeated exceptions reveal missing capabilities or appropriate local ownership.

### Measure user experience

Inspect waiting, unrecorded manual work, support classes, abandoned onboarding, local client wrappers and useful diagnosis time.

### Create contribution boundaries

Review domain adapters/evaluators/templates; retain stronger ownership of identity, authorization, state transitions and compatibility.

### Avoid mandatory adoption too early

Require shared controls where shared risk needs one authority, not to inflate usage.
Check workarounds and delivery rather than mandatory-adoption dashboards.

## Make the staff-level decisions explicit

Cross-team integration requires contracts, representative pilots, single-authority migration, tiered policy, useful adoption and reversible sourcing.
Defend one invariant under changed workload or failure conditions.

## Add the principal-level portfolio view

### Choose the shared boundary

Share identity, promotion, lineage, access, cost attribution and release evidence where inconsistency creates shared risk/cost.
Keep specialized runtimes, labels, evaluators, fallback and early research local when semantics differ.
State evidence that would move this boundary.

### Balance the portfolio

Fund migration/reliability, product enablement, specialization, retirement and boundary experiments alongside shared services.
Count dual-operation costs and displaced product work.

### Define decision checkpoints

Quarterly, expand/repair/narrow/stop using release reliability, support, exception clusters, shared incidents, retired duplicates and tested vendor exit.

### Plan for organizational change

Delegate authority with review policy, runbooks and succession; one expert must not become the only release path.

### Preserve option value

Exportable identities/evidence, bounded adapters and staged authority transfer preserve future direction changes.

## Walk through one model lifecycle

A fraud team updates a transaction model with a new device-risk feature.

1. The source owner publishes a versioned event schema.
2. The feature definition declares entity key, event time, availability time, freshness, fallback, and privacy class.
3. The pipeline materializes a new offline feature snapshot and shadows the online value.
4. A consistency job compares sampled batch and online values.
5. The training specification references immutable data, feature, code, and environment versions.
6. Record three approved seed runs and one evaluation bundle; each run has its own publication identity.
7. Evaluation compares the current production model on the same snapshot and reports cost-aware thresholds, calibration, critical slices, and delayed-label maturity.
8. Policy checks lineage, schema compatibility, required evidence, and reviewer approval.
9. The deployment controller shadows the model, then canaries it on a bounded traffic slice.
10. System telemetry checks latency and errors. Product telemetry checks intervention volume and early guardrails.
11. Mature chargeback labels later join to decision identifiers.
12. The experiment owner makes the ship decision with the risk team.
13. Progressive rollout increases traffic while the prior compatible model remains available.
14. A device-pipeline freshness breach triggers the declared fallback and pages the source owner.
15. If calibration regresses, rollback changes the production alias to the prior model and preserves incident evidence.

Each step exposes a boundary for a changed-condition check.

## Compare rejected alternatives

A strong design explains what it did not choose.

### One universal runtime

A single runtime simplifies operations but can block specialized workloads and create a large migration. Use common run contracts with several adapters first. Consolidate only after workload evidence supports it.

### Metadata captured after execution

Post-hoc scraping is easy to add but misses failed runs, manual changes, and exact inputs. Make identity allocation and run submission part of the normal path. Use scraping only during migration.

### Platform-owned model quality

Central ownership sounds consistent but removes domain accountability. The platform should enforce evidence contracts. Product teams retain decisions about labels, metrics, and acceptable trade-offs.

### Strict mandatory path

A strict path reduces variation but can freeze research and encourage hidden workarounds. Require shared production identities and promotion records. Allow explicit exceptions for execution and evaluation needs.

### Immediate active-active control plane

Active-active writes improve theoretical regional availability but add conflict and state-transition complexity. Begin with a single writer, durable replicas, and tested failover unless business requirements demand more.

## Structure the 45-minute interview

<span id="minutes-0-to-5-clarify-and-frame"></span>
<span id="minutes-5-to-12-define-success-and-invariants"></span>
<span id="minutes-12-to-22-draw-the-architecture"></span>
<span id="minutes-22-to-32-take-one-deep-dive"></span>
<span id="minutes-32-to-39-migration-and-ownership"></span>
<span id="minutes-39-to-45-principal-trade-offs-and-follow-up"></span>

| Minutes | Work |
| --- | --- |
| 0-5 | Users, risk, workload, failures, quarter constraint and boundary |
| 5-12 | Delivery/reliability/quality/cost, identity, promotion, rollback and blast radius |
| 12-22 | Data, training, evaluation, deployment and telemetry across control/data planes |
| 22-32 | One temporal/retry/evaluation/rollback/lineage/isolation mechanism |
| 32-39 | Pilots, authority transfer, owners and useful adoption |
| 39-45 | Shared/local, build/buy/exit, quarterly decisions and residual risk |

## Distinguish answer levels

<span id="mid-level-answer"></span>
<span id="senior-answer"></span>
<span id="staff-answer"></span>
<span id="principal-answer"></span>

| Scope archetype | Evidence |
| --- | --- |
| Components | Correct tools/subsystems without a complete multi-team operating contract |
| Product lifecycle | Connected data, training, evaluation, deployment, monitoring and rollback |
| Cross-team integration | Narrow shared boundary, safe migration, ownership and useful adoption; precise technical follow-ups |
| Portfolio | Standards versus specialization, multi-quarter investment/retirement, delegation, reversibility and stop evidence |

Company titles vary; these are scope distinctions, not hiring predictions.

## Observer scorecard

Score each dimension from 0 to 2 before discussing the answer.

| Dimension | 0 | 1 | 2 |
| --- | --- | --- | --- |
| Framing | Starts with tools | Names users and scale | Changes platform scope from workload and risk |
| Architecture | Lists disconnected components | Connects the lifecycle | Defines control, data, state, and failure boundaries |
| Technical depth | Uses platform slogans | Explains one subsystem | Defends invariants and changed conditions precisely |
| Migration | Assumes a rewrite | Suggests pilots | Transfers authority safely and retires old paths |
| Ownership | Platform owns everything | Names team roles | Aligns responsibility with semantics and incidents |
| Adoption | Measures component usage | Mentions developer experience | Uses workflow outcomes, escape hatches, and support evidence |
| Strategy | Proposes a roadmap | Prioritizes capabilities | Balances portfolio, reversibility, and stop decisions |
| Communication | Wanders through details | Uses a clear structure | Preserves the decision while changing depth on demand |

Use scores to locate practice gaps, not as a validated hiring rubric.

## Strong signals

Inspectable identities, temporal inputs, publication, promotion, rollback, owners and measured pilots support the design.
Preserve risk tiers, portable evidence and explicit narrowing/stopping decisions.

## Weak signals

Mutable artifacts, unsupported reproducibility, unfenced writers, blind mandatory migration and unowned semantics violate those contracts.
More teams or more boxes do not establish broader technical judgment.

## Changed-condition follow-ups

1. The company acquires a business in another cloud. Which contracts remain shared, and which execution layers stay separate?
2. A regulator requires deletion evidence within 30 days. How do lineage, retention, and retraining policy change?
3. One research team needs a runtime the platform does not support. When do you add an adapter, permit an exception, or reject the workload?
4. A shared feature causes three product incidents. Do you centralize more control or return ownership to teams?
5. Accelerator utilization rises from 35% to 70%, but experiment cycle time doubles. Is the scheduler successful?
6. A vendor doubles its price after two years. What can move without losing run history or production safety?
7. Product teams adopt deployment controls but reject the training path. Is partial adoption acceptable?
8. The metadata database loses a region during a rollout. What continues, what stops, and who has authority?
9. A principal engineer wants one universal serving runtime. What evidence would support or reject that direction?
10. After two quarters, neither pilot ships faster. What do you repair, narrow, or stop?

For each condition, name the changed invariant, responsible boundary, evidence and worsened trade-off.

---

## Sources

- [Sculley et al.: Hidden Technical Debt in Machine Learning Systems](https://research.google/pubs/hidden-technical-debt-in-machine-learning-systems/).
- [MLflow model lifecycle](https://mlflow.org/docs/latest/ml/model-registry/).
- [Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml).

*Related: [enterprise agent-platform design](/questions/design-enterprise-agent-platform/), [ML data lineage and versioning](/concepts/ml-data-lineage-versioning/), [design a feature store](/questions/design-feature-store/), [design ML monitoring](/questions/design-ml-monitoring/), and [senior through senior-principal ML scope](/guides/l5-vs-l6-faang-ml/).*

</div>
