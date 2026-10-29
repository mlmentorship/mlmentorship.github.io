---
title: "Design an LLM red-team and security evaluation program"
description: "Start from assets and attacker capability, test the whole agentic system adaptively, and turn findings into measurable release gates."
date: "2026-07-11"
draft: false
tags: ["questions"]
category: "questions"
---

> Design the security evaluation for an enterprise agent that reads documents, calls tools, writes code, and acts on user accounts.

Design an authorized security-evaluation program for an enterprise agent, covering its complete tool-using system and operational controls.

## Problem

**Context.** A hypothetical enterprise agent reads documents, calls tools, writes code, and acts on user accounts.
Untrusted content, identity boundaries, retained state, and approval workflows can affect its behavior.

**Task.** Define the threat model, bounded test environment, representative test categories, evidence, prioritization, remediation, and release gates.
Explain how findings become reproducible regressions and inform incident response.

**Constraints.** Use authorized synthetic accounts and data, with explicit test scope, containment, and stop conditions.
Evaluate actual permission and state boundaries rather than only response wording; no test should create uncontrolled real-world effects.

## Start with the threat model

Name:

- **assets:** secrets, user data, money, code execution, model weights, policy integrity;
- **actors:** malicious user, compromised document, untrusted tool, insider, external service, curious tenant;
- **capabilities:** prompt control, file upload, repeated queries, account access, tool-result injection, collusion;
- **boundaries:** tenant isolation, model-to-tool authorization, network, sandbox, memory, human approval;
- **success conditions:** exfiltration, unauthorized action, policy bypass, persistence, denial of service, model extraction;
- **acceptable residual risk:** severity and likelihood by product surface.

The threat model decides the test suite. "Try jailbreak prompts" does not.

## Build attack families

### Instruction and context attacks

Direct jailbreaks, indirect prompt injection through retrieved content, instruction hierarchy conflicts, obfuscation, multilingual variants, and long-context hiding.

### Tool and action attacks

Confused deputy behavior, argument injection, excessive permissions, forged tool results, unsafe retries, missing confirmation, cross-tool data flow, and sandbox escape.

### Data and privacy attacks

Training-data extraction, system-prompt leakage, cross-tenant memory leakage, retrieval exfiltration, membership inference, and sensitive-log exposure.

### Integrity and availability attacks

Data poisoning, memory poisoning, denial through long contexts or tool loops, grader manipulation, and attacks that increase cost or latency rather than changing text.

### Agentic attacks

Multi-step plans, delayed triggers, persistence across sessions, colluding agents, unsafe generated code, and attacks that exploit the gap between local tool success and global policy.

## Measure more than attack success rate

Track:

- success by attack family, severity, attacker capability, and target asset;
- attempts or query budget to success;
- transfer across paraphrase, language, model version, and tool configuration;
- false refusals and task-quality regression under defenses;
- time to detection and containment;
- exploit reproducibility;
- residual risk after system-level controls.

Use adaptive attackers. A static prompt list becomes a regression suite, not a red team.

**Learning objective:** Trace how an adaptive red-team finding becomes an owned mitigation, a residual-risk release decision, and a static regression test while production evidence updates the threat model.

<!-- visual:adaptive-red-team-release-loop -->
<figure class="learning-figure editorial-illustration" aria-labelledby="adaptive-red-team-release-loop-edition-title"><p class="visual-title" id="adaptive-red-team-release-loop-edition-title">Blocking one reproduced attack leaves other paths untested</p><svg viewBox="0 0 360 419" role="img" aria-labelledby="adaptive-red-team-release-loop-edition-svg-title adaptive-red-team-release-loop-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="adaptive-red-team-release-loop-edition-svg-title">One blocked path leaves untested paths open questions</title><desc id="adaptive-red-team-release-loop-edition-svg-desc">A fictional private read-only canary scenario compares three access paths before and after a mitigation. The reproduced direct-request failure becomes blocked. Quoted-text and tool-output paths remain question marks, not assumed successes. Adaptive coverage and residual-risk evidence still govern release.</desc><text x="180" y="22" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Fictional boundary: private read-only</tspan><tspan x="180" dy="19">canary</tspan></text><text x="79" y="66" style="font-size:12px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="79" dy="0">PATH</tspan></text><text x="202" y="66" style="font-size:12px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="202" dy="0">BEFORE</tspan></text><text x="300" y="66" style="font-size:12px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="300" dy="0">AFTER</tspan></text><text x="79" y="110" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="79" dy="0">direct request</tspan></text><g data-editorial-node><rect x="160" y="88" width="82" height="45" rx="3" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.3"></rect><text x="201" y="115.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="201" dy="0">leaks</tspan></text></g><g data-editorial-node><rect x="258" y="88" width="82" height="45" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.3"></rect><text x="299" y="115.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="299" dy="0">blocked</tspan></text></g><text x="79" y="168" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="79" dy="0">quoted text</tspan></text><g data-editorial-node><rect x="160" y="146" width="82" height="45" rx="3" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.3"></rect><text x="201" y="173.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="201" dy="0">?</tspan></text></g><g data-editorial-node><rect x="258" y="146" width="82" height="45" rx="3" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.3"></rect><text x="299" y="173.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="299" dy="0">?</tspan></text></g><text x="79" y="226" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="79" dy="0">tool output</tspan></text><g data-editorial-node><rect x="160" y="204" width="82" height="45" rx="3" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.3"></rect><text x="201" y="231.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="201" dy="0">?</tspan></text></g><g data-editorial-node><rect x="258" y="204" width="82" height="45" rx="3" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.3"></rect><text x="299" y="231.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="299" dy="0">?</tspan></text></g><text x="180" y="293" style="font-size:14px;font-weight:700;fill:var(--viz-warning-stroke);text-anchor:middle"><tspan x="180" dy="0">One repaired path != all paths</tspan><tspan x="180" dy="19">covered</tspan></text><text x="180" y="337" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Adaptive testing targets the remaining question</tspan><tspan x="180" dy="19">marks.</tspan></text><text x="180" y="383" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Release still needs residual-risk + utility</tspan><tspan x="180" dy="19">evidence.</tspan></text></svg><figcaption><strong>Read it this way:</strong> This fictional grid distinguishes a reproduced repair from missing coverage. Question marks are unknown outcomes. One blocked path cannot certify the whole system. <span class="visual-reader-check" data-reader-check="adaptive-red-team-release-loop"><strong>Check:</strong> Why does passing every stored regression not establish adequate coverage of the current threat model?</span></figcaption></figure>

## What an L4 answer sounds like

> "Collect jailbreak prompts, run them before launch, add filters, and block anything unsafe. Monitor abuse in production."

This tests one input channel, assumes a binary definition of unsafe, and does not connect findings to assets, permissions, or release decisions.

## What an L5 answer adds

An L5 candidate defines attacker capability and assets, builds an attack taxonomy, tests direct and indirect injection, evaluates tools and authorization, and reports severity-weighted results with utility regression. They separate model mitigations from system controls:

- least-privilege tool scopes;
- typed arguments and validation;
- provenance labels for untrusted text;
- confirmation for consequential actions;
- sandbox and network restrictions;
- secret isolation;
- rate limits and anomaly detection;
- output and action monitoring;
- incident response and revocation.

## What an L6 answer adds

An L6 candidate builds an operating program. Findings have owners, reproduction artifacts, severity, fix deadlines, retest status, and release gates. Novel attacks become tests without teaching the model only one string pattern.

They anticipate evaluator failure. A model judge can be attacked by the same output it grades. Human reviewers can miss encoded exfiltration. Attack generation can optimize against a proxy while ignoring realistic reachability. Independent tools, deterministic policy checks, and manual review cover different failure modes.

They also define production feedback. Near misses, blocked tool calls, anomalous retries, permission denials, and user reports update the threat model. Logs preserve enough evidence without becoming another secret store.

## Tells that get you a strong-hire vote

- Assets, actors, capabilities, and trust boundaries come first.
- Indirect prompt injection and tool output are first-class inputs.
- Consequential actions have authorization and confirmation semantics.
- Results are sliced by severity and attacker budget.
- Defenses are evaluated for utility and false refusals.
- Adaptive attacks complement static regression suites.
- Findings map to owners and release gates.
- Production telemetry updates the threat model.

## Tells that get you down-leveled

- Treating red teaming as a prompt list.
- Model filters as the only defense.
- No tool, memory, identity, or tenant boundary.
- Attack success rate with no severity.
- Ignoring false refusals and task degradation.
- Trusting one LLM judge.
- No incident or retest workflow.

## Common follow-up

"Your prompt-injection success rate falls from 8% to 1%, but false refusals triple. Do you ship?"

Not from those aggregates. Slice by action severity, attacker capability, user workflow, and refusal cost. A 1% path to unauthorized payment can block launch; an 8% failure to summarize untrusted text may be containable with provenance and tool restrictions. The decision follows residual risk and system controls, not one average.

*Related: [enterprise agent-platform design](/questions/design-enterprise-agent-platform/), [LLM security threat models](/concepts/llm-security-threat-models/), [content moderation](/questions/content-moderation/), and [handle hallucinations in production](/questions/handle-hallucinations-in-production/).*
