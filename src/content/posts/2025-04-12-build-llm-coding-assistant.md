---
title: "Build an LLM coding assistant from scratch"
description: "Bind inline suggestions to editor state, retrieve authorized context, discard stale responses and verify a separate tool-enabled task workflow."
date: "2025-04-12"
draft: false
tags: ["questions"]
category: "questions"
---


An inline completion belongs to one editor state; typing can make a valid-looking response stale before it arrives.

## Problem

A hypothetical request R1 targets document version 41 at cursor 80.
The user types, producing version 42 and cursor 81 before R1 returns.
R2 targets the new state.
R1's suggested code parses correctly. Which result may the editor display or accept?

## Picture

Treat each response as a proposal for its captured document, cursor, context and policy, not as free-floating code.
Cancellation can save work, but a late response still needs a validity check.

<!-- visual:coding-assistant-two-latency-paths -->
```mermaid
flowchart TB
  accTitle: An inline response must still match the current editor state
  accDescr: Request R1 captures document version 41 and cursor 80. Authorized context and the buffer prefix and suffix form its bounded fill-in-the-middle prompt. Typing advances the current editor to version 42 and cursor 81 while R1 runs. R1 is discarded on return because its snapshot is stale. R2 may display only while its version, cursor, context, deadline and authorization remain valid; accepting it requires a version-aware editor transaction. A separate explicit task path grants scoped tools and evaluates patches with review and tests, never using an inline response as authorization.
  Scope{"Inline suggestion<br/>or explicit coding task?"}
  Trigger["R1 captures version 41<br/>cursor 80 · request identity"]
  Policy["Check read + egress policy<br/>before retrieving or sending code"]
  Index[("Authorized context<br/>buffer · symbols · imports · tests")]
  Prompt["Bounded FIM snapshot<br/>prefix + suffix + context revision"]
  Fast["Approved code model<br/>R1 in flight"]
  Current["Typing advances editor<br/>version 42 · cursor 81"]
  Valid{"Response still matches?<br/>state · deadline · authorization"}
  Drop["R1: discard stale response<br/>never overwrite new text"]
  Ghost["R2: show only while valid<br/>version-check acceptance · undo"]
  InlineEval["Exposure funnel + retained edits<br/>language · task · latency"]
  Deep["Explicit scoped task<br/>authorized tools · patch review"]
  TaskEval["Task outcomes<br/>tests · regressions · safety"]
  Scope -->|"inline"| Trigger
  Trigger --> Policy --> Index
  Index --> Prompt
  Prompt --> Fast
  Trigger -->|"user types"| Current
  Fast --> Valid
  Current --> Valid
  Valid -->|"R1 mismatches"| Drop
  Current -->|"new request R2"| Policy
  Valid -->|"R2 matches"| Ghost
  Ghost --> InlineEval
  Scope -->|"task authorization"| Deep
  Deep --> TaskEval
  class Scope,Prompt,Valid viz-focus
  class Trigger,Current viz-input
  class Index,Policy viz-state
  class Drop viz-warning
  class Ghost,Deep viz-output
```
<p class="diagram-caption"><strong>Read it this way:</strong> R1 can be syntactically correct but invalid for the current editor. R2 also needs a fresh check when displayed and accepted.</p>
<p class="diagram-source">References: <a href="https://arxiv.org/abs/2207.14255">FIM</a>, <a href="https://arxiv.org/abs/2303.12570">RepoCoder</a>, and <a href="https://docs.github.com/en/copilot/concepts/context/repository-indexing">repository indexing</a>.</p>

## Capture and check request identity

| Captured field | Validity dependency |
| --- | --- |
| Request/document ID, version, cursor/selection | Locates the suggestion's exact edit target |
| Prefix/suffix hashes and context revision | Detects changed prompt dependencies |
| Policy identity and live authorization | Prevents reuse after access or egress changes |
| Cancellation and deadline | Prevents abandoned or late presentation |
| Suggestion/edit IDs | Connects exposure, acceptance and ordinary undo |

Compare these fields on return; recheck state and authorization at acceptance.
Use the editor platform's version-aware APIs for an atomic, undoable edit.
A successful earlier check does not prevent a later race.

| Event | Current state | Decision |
| --- | --- | --- |
| R1 captured | Version 41, cursor 80 | Start bounded work |
| User types | Version 42, cursor 81 | Cancel R1; capture R2 |
| R1 returns | Still version 42 | Discard despite valid syntax |
| R2 returns | Its snapshot still matches | Show only if authorized and before deadline |

## Frame before architecture

Scope completion type, repository context, deployment policy and measured latency target first.
Inline suggestions, function generation, chat and agent tasks need different interaction and evaluation contracts.

For FIM-capable models, serialize prefix, suffix and insertion position using the supported template.
Retrieve bounded relevant context: imports, definitions, structural neighbors, recent edits/navigation and tests.
BM25, embeddings and symbol/import graphs are options; benchmark retrieval quality under access and prompt-budget constraints.

## Change the workflow or deployment policy

### 1. Request a multi-file refactor

**Picture and change:** move to explicit scope, tool authorization, patch review and task-level verification.
Use tests, regressions, safety and task outcomes rather than inline acceptance.

**Check:** an inline completion cannot authorize a shell command.

**Use when:** work requires actions across files, not a cursor-bound insertion.

### 2. Keep private code on device

**Picture and change:** use approved local inference and context processing; disable incompatible remote routes and content telemetry.

**Check:** remote embedding uploads also violate the policy, even if generation is local.
Credential filtering alone does not authorize egress.

**Use when:** residency covers source and its derived representations.

## Failure and transfer

A cached suggestion keyed only by prefix can survive a changed suffix or revoked permission.
Cache identity must cover its actual dependencies, model and prompt configuration; recheck access on reuse.

Acceptance rises after fewer suggestions are shown. Did useful assistance improve?

<details>
<summary>Check your answer</summary>

Not necessarily. Compare eligible opportunities, shown suggestions, acceptance, retained output and user outcomes, not acceptance's selected denominator alone.

</details>

<!-- learning:core-end -->

<div data-learning-extension>

<span id="what-an-l4-answer-sounds-like"></span>
<span id="what-an-l5-answer-sounds-like"></span>

## Choose measured model paths

A small fast model for inline work and another model for explicit tasks is one design, not a universal requirement.
Size and latency depend on hardware, context, batching and network conditions.
FIM capability requires compatible training and formatting; a template alone does not create it.

<span id="what-an-l6-answer-adds"></span>

## Speculation and caching

Debounce speculative requests, cancel obsolete work and still reject stale results.
Measure saved wait against wasted compute and privacy exposure.

Embedding caches need content/model revisions and access controls.
Prefix KV reuse requires the actual compatible serialized prefix, model state, positions and attention rules.
It is different from reusing a complete suggestion after the suffix changes.

## Common follow-up

For a fast model, compare an existing FIM-trained code model, authorized continued pretraining and distillation.
Train only on approved code/prompts and retain permissions through data collection, storage and evaluation.
Distillation can specialize to production prompts, but quality and out-of-distribution behavior require measurement.
No model-size range or training method guarantees the best result.

<span id="tells-that-get-you-a-strong-hire-vote"></span>
<span id="tells-that-get-you-down-leveled"></span>

## Evaluate the complete product

Slice retained edits, dismissals, latency and task outcomes by language and task type.
Compare the previous assistant and appropriate no-assistant baselines.
Filter sensitive content, honor retention controls and test tools under real task boundaries.
No architecture vocabulary or answer guarantees a hiring outcome.

*Related: [design an AI coding product](/questions/design-ai-coding-product/), [enterprise agent-platform design](/questions/design-enterprise-agent-platform/), [build evals for a coding assistant](/questions/evals-for-coding-assistant/), and [reduce LLM inference cost](/questions/reduce-llm-inference-cost-10x/).*

</div>
