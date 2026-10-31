---
title: "Build an LLM coding assistant from scratch"
description: "Bind inline suggestions to editor state, retrieve authorized context, discard stale responses and verify a separate tool-enabled task workflow."
date: "2025-04-12"
draft: false
tags: ["questions"]
category: "questions"
---


Design an inline coding assistant that displays suggestions for the correct editor state despite overlapping requests.

## Problem

**Context.** An editor requests model-generated code while the developer continues typing, so requests may complete out of order.

**Task.** Decide which result may be displayed or accepted in the race below.
Define request identity, stale-result handling, acceptance checks, and the surrounding latency-sensitive serving path.

**Constraints.** A syntactically valid suggestion can target an obsolete document or cursor.
Preserve developer acceptance authority; cancellation alone does not prove an old response cannot arrive.

A hypothetical request R1 targets document version 41 at cursor 80.
The user types, producing version 42 and cursor 81 before R1 returns.
R2 targets the new state.
R1's suggested code parses correctly. Which result may the editor display or accept?

## Picture

Treat each response as a proposal for its captured document, cursor, context and policy, not as free-floating code.
Cancellation can save work, but a late response still needs a validity check.

<!-- visual:coding-assistant-two-latency-paths -->
<figure class="learning-figure editorial-illustration" aria-labelledby="coding-assistant-two-latency-paths-edition-title"><p class="visual-title" id="coding-assistant-two-latency-paths-edition-title">A correct suggestion for an old document must still be discarded</p><div class="visual-argument" data-visual-argument="coding-assistant-two-latency-paths" data-argument-kind="timeline" data-argument-composition="shared-clock"><svg viewBox="0 0 360 459" role="img" aria-label="A fast answer to an old document is still stale. The request used document version 7. A version-8 edit before its reply makes that reply inapplicable without revalidation."><title>A fast answer to an old document is still stale</title><desc>The request used document version 7. A version-8 edit before its reply makes that reply inapplicable without revalidation.</desc><text x="20" y="25" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Request for version 7</tspan></text><path d="M30 63L330 63" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><circle cx="30" cy="63" r="7" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="53" y="94" style="font-size:18px;font-weight:700;text-anchor:middle;fill:var(--viz-input-stroke)"><tspan x="53" dy="0">0</tspan></text><text x="20" y="122" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">User edits to version 8</tspan></text><path d="M30 160L330 160" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><circle cx="130" cy="160" r="7" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><text x="130" y="191" style="font-size:18px;font-weight:700;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="130" dy="0">1</tspan></text><text x="20" y="219" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Version-7 reply arrives</tspan></text><path d="M30 257L330 257" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><circle cx="230" cy="257" r="7" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><text x="230" y="288" style="font-size:18px;font-weight:700;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="230" dy="0">2</tspan></text><text x="20" y="316" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Version-8 context revalidated</tspan></text><path d="M30 354L330 354" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><circle cx="330" cy="354" r="7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="307" y="385" style="font-size:18px;font-weight:700;text-anchor:middle;fill:var(--viz-output-stroke)"><tspan x="307" dy="0">3</tspan></text><path d="M30 404L330 404" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M323 399L330 404" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M323 409L330 404" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="180" y="430" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="180" dy="0">event order, not elapsed time</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content"><svg viewBox="0 0 360 391" role="img" aria-labelledby="coding-assistant-two-latency-paths-edition-svg-title coding-assistant-two-latency-paths-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="coding-assistant-two-latency-paths-edition-svg-title">A correct suggestion for yesterday's buffer is still stale</title><desc id="coding-assistant-two-latency-paths-edition-svg-desc">Editor and server lanes share one ordered time direction. R1 captures version 41, typing advances the document to version 42, and R1 returns stale and is discarded. R2 targets version 42 but still needs current version, deadline, and policy checks before acceptance. Tool tasks require separate authorization.</desc><text x="90" y="25" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">EDITOR</tspan></text><text x="270" y="25" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">SERVER</tspan></text><path d="M90 45L90 340" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><path d="M270 45L270 340" style="fill:none;stroke:var(--viz-state-stroke);stroke-width:1.75"></path><rect x="20" y="61" width="140" height="38" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="90" y="85" style="font-size:13px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">version 41</tspan></text><rect x="200" y="61" width="140" height="38" rx="3" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.3"></rect><text x="270" y="85" style="font-size:13px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">R1 captures v41</tspan></text><rect x="20" y="125" width="140" height="38" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="90" y="149" style="font-size:13px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">typing: v42</tspan></text><rect x="200" y="125" width="140" height="38" rx="3" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.3"></rect><text x="270" y="149" style="font-size:13px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">R1 still running</tspan></text><rect x="20" y="189" width="140" height="38" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="90" y="213" style="font-size:13px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">keep new text</tspan></text><rect x="200" y="189" width="140" height="38" rx="3" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.3"></rect><text x="270" y="213" style="font-size:13px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">R1 returns v41</tspan></text><rect x="20" y="270" width="140" height="38" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="90" y="294" style="font-size:13px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">current: v42</tspan></text><rect x="200" y="270" width="140" height="38" rx="3" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.3"></rect><text x="270" y="294" style="font-size:13px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">R2 returns v42</tspan></text><path d="M160 81L200 81" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M193.8569220667674,84.35597877022943L200,81L193.8569220667674,77.64402122977057" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M200 209L160 209" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><path d="M166.1430779332326,205.64402122977057L160,209L166.1430779332326,212.35597877022943" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><text x="180" y="251" style="font-size:14px;font-weight:700;fill:var(--viz-warning-stroke);text-anchor:middle"><tspan x="180" dy="0">41 != 42: DISCARD R1</tspan></text><path d="M200 290L160 290" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75"></path><path d="M166.1430779332326,286.6440212297706L160,290L166.1430779332326,293.3559787702294" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75"></path><text x="180" y="334" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">R2 still needs live version + policy checks</tspan></text><text x="180" y="374" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">A tool task needs separate authorization</tspan></text></svg><div class="visual-derivation-caption">R1 cannot overwrite version 42, even when its code is correct. R2 still needs live version, deadline, and policy checks. Vertical spacing is schematic.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="coding-assistant-two-latency-paths"><strong>Check:</strong> Why must R1 never be inserted even if its proposed code is correct?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="coding-assistant-two-latency-paths">The request used document version 7. A version-8 edit before its reply makes that reply inapplicable without revalidation.</span></figcaption></figure>
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
