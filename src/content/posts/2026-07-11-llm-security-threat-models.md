---
title: "LLM security threat models"
description: "Map assets, actors, trust boundaries, and attack paths across prompts, retrieval, tools, memory, generated code, and tenants."
date: "2026-07-11"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A threat model traces an attacker's reachable path to an asset; model influence does not itself grant permission.

## Problem

Use synthetic records and controlled endpoints, not real tenant data.
A research assistant may summarize tenant A's approved document A1 for an internal destination.
An attacker controls one uploaded document, not the verified user identity or policy.
Suppose the model proposes exporting tenant B's record B7 to an external mock destination.
The arguments satisfy the tool's syntax.

Specify the boundary that must deny this request and the evidence needed to detect compromise through other channels.

## Picture: Trace Influence, Then Trace Authority

The upload can influence the model's proposed action.
It cannot create tenant B permission.
The gateway checks the exact action, tenant, resource, destination and purpose before issuing credentials or dispatching.
It denies the proposal and records a redacted reason class.

Also inspect the answer, caches and logs.
A refusal after an earlier disclosure is not a secure outcome.
Keep a synthetic compromise canary in the observer, outside model context.
Check observed channels and actual effects against that oracle.

## Assets

The protected asset here is tenant B's content, including copies in outputs or derived stores.
Credentials, action authority, model artifacts, code sandboxes, policy, grader integrity, availability and audit evidence can require separate paths.
Rank consequences, not just unusual model text.

## Actors and Capabilities

For this example, allow document edits and at most 100 adaptive proposals in the simulator.
Do not assume access to identity, policy, credentials, other tenants or the observer's canary.
State authentication, modalities, defense knowledge, tool scope and data-poisoning access when those assumptions change.

## Trust Boundaries

| Field | Contract for this case |
| --- | --- |
| Asset | B7 content and its derived copies |
| Actor | Attacker controlling one upload |
| Control surface | Untrusted uploaded text, preserved through context assembly |
| Budget | Up to 100 adaptive simulated proposals |
| Path | Upload → model proposal → tool gateway → executor → protected record/destination |
| Trust boundary | Verified authority, separate from model-produced arguments |
| Compromise oracle | Unauthorized effect or canary disclosure in answer, tools, caches or logs |
| Control owner | Identity/policy owner and tool operator; storage/channel owners cover other paths |
| Residual assumption | All relevant paths reach correct enforcement; no bypass or hidden credential |

<p class="visual-kicker">Influence is not authority</p>
<!-- visual:llm-security-influence-authority-path -->
<figure class="learning-figure editorial-illustration" aria-labelledby="llm-security-influence-authority-path-edition-title"><p class="visual-title" id="llm-security-influence-authority-path-edition-title">Untrusted text can propose an export without authorizing it</p><div class="visual-argument" data-visual-argument="llm-security-influence-authority-path" data-argument-kind="connections"><p class="visual-argument-premise">Untrusted content can suggest an action without authorizing it</p><svg viewBox="0 0 360 354" role="img" aria-label="Untrusted content can suggest an action without authorizing it. Retrieved text may contain an export instruction. That instruction cannot cross the authority boundary into an executable export."><title>Untrusted content can suggest an action without authorizing it</title><desc>Retrieved text may contain an export instruction. That instruction cannot cross the authority boundary into an executable export.</desc><path data-proof-edge="0,1" d="M197 38H356V126H180V136" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path><path d="M176 129L180 136" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path><path d="M184 129L180 136" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path><path data-proof-edge="1,2" d="M197 154H353V242H180V252" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><path d="M176 245L180 252" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><path d="M184 245L180 252" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><path d="M347 192L359 204" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><path d="M359 192L347 204" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><circle cx="180" cy="38" r="16" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><text x="180" y="42" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="180" dy="0">1</tspan></text><text x="180" y="77" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="180" dy="0">Untrusted document</tspan><tspan x="180" dy="18">text</tspan></text><circle cx="180" cy="154" r="16" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><text x="180" y="158" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="180" dy="0">2</tspan></text><text x="180" y="193" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="180" dy="0">Suggested export</tspan><tspan x="180" dy="18">action</tspan></text><circle cx="180" cy="270" r="16" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="180" y="274" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="180" dy="0">3</tspan></text><text x="180" y="309" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="180" dy="0">Authorized tool</tspan><tspan x="180" dy="18">dispatch</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content"><svg viewBox="0 0 360 390" role="img" aria-labelledby="llm-security-influence-authority-path-edition-svg-title llm-security-influence-authority-path-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="llm-security-influence-authority-path-edition-svg-title">Untrusted context can propose an action without acquiring authority to execute it</title><desc id="llm-security-influence-authority-path-edition-svg-desc">Untrusted context proposes export of B7 from tenant B to an external mock destination. Verified authority A1 permits only a tenant-A internal summary. Exact action, tenant, resource, destination, and purpose do not match, so the proposal ends before credentials or dispatch. Answer, cache, and log disclosure still need separate inspection.</desc><text x="180" y="22" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Untrusted proposal != verified authority A1</tspan></text><g transform="translate(0 0)"><g data-editorial-node><rect x="30" y="43" width="300" height="58" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="180" y="67.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Proposal: export B7 / tenant B</tspan><tspan x="180" dy="19">external mock destination</tspan></text></g><path d="M180 101L180 128" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M176.64402122977057,121.8569220667674L180,128L183.35597877022943,121.8569220667674" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><g data-editorial-node><rect x="30" y="132" width="300" height="76" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="180" y="156" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">A1 permits tenant A</tspan><tspan x="180" dy="19">internal summary only</tspan><tspan x="180" dy="19">exact fields do not match</tspan></text></g><path d="M180 208L180 229" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M90 229L270 229" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M90 229L90 266" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75"></path><path d="M86.64402122977057,259.85692206676737L90,266L93.35597877022943,259.85692206676737" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75"></path><path d="M270 229L270 266" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><path d="M266.6440212297706,259.85692206676737L270,266L273.3559787702294,259.85692206676737" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><text x="90" y="253" style="font-size:12px;font-weight:700;fill:var(--viz-output-stroke);text-anchor:middle"><tspan x="90" dy="0">MATCH REQUIRED</tspan></text><text x="270" y="253" style="font-size:12px;font-weight:700;fill:var(--viz-warning-stroke);text-anchor:middle"><tspan x="270" dy="0">THIS PROPOSAL</tspan></text><g data-editorial-node><rect x="14" y="270" width="152" height="100" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.3"></rect><text x="90" y="296.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">scope still</tspan><tspan x="90" dy="19">bounded</tspan><tspan x="90" dy="19">inspect answer,</tspan><tspan x="90" dy="19">cache, and logs</tspan></text></g><g data-editorial-node><rect x="194" y="270" width="152" height="100" rx="3" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.3"></rect><text x="270" y="306" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">DENY</tspan><tspan x="270" dy="19">no credential</tspan><tspan x="270" dy="19">no dispatch</tspan></text></g></g></svg><div class="visual-derivation-caption">The proposed action cannot borrow authority from retrieved text. Its fields fail the verified scope. Blocking tools does not settle answer, cache, or log disclosure.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="llm-security-influence-authority-path"><strong>Check:</strong> Why can text influence the model's proposal without granting permission to export B7?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="llm-security-influence-authority-path">Retrieved text may contain an export instruction. That instruction cannot cross the authority boundary into an executable export.</span></figcaption></figure>

  ## Check the Proposed Scope

  This fixture checks one boundary, not a production authorization system.
  Assume schema-validated strings and trusted authority supplied outside the model.

  ```python
  def permitted(proposal, authority):
    return (
      proposal.get("action") in authority["actions"]
      and proposal.get("tenant") == authority["tenant"]
      and proposal.get("resource") in authority["resources"]
      and proposal.get("destination") in authority["destinations"]
      and proposal.get("purpose") == authority["purpose"]
    )


  trusted = {
    "actions": {"summarize_document"},
    "tenant": "A",
    "resources": {"A1"},
    "destinations": {"internal"},
    "purpose": "research-summary",
  }
  proposal = {
    "action": "export_document",
    "tenant": "B",
    "resource": "B7",
    "destination": "external-mock",
    "purpose": "research-summary",
  }
  assert not permitted(proposal, trusted)
  ```

  Schema validity establishes neither resource ownership nor permission.
  Confirmation cannot grant authority absent from the confirming person's scope.

  ## Change 1: The Upload Can Persist in Memory

  **New task.** A document can propose memory used next session.

  **Picture and change.** Govern writes, provenance, expiry, reads and deletion separately.
  Keep current authorization outside inferred memory.
  Test a later session after the entry is stored and after delegation changes.

  **Check.** A remembered instruction cannot widen current tenant, resource or destination scope.

  **Use when.** Influence survives the initial context.

  ## Change 2: The Target Is Availability

  **New task.** Repeated calls consume capacity rather than disclose records.

  **Picture and change.** Enforce token, call, time, concurrency and tenant budgets at actual resource boundaries.
  Reserve capacity before expensive work; reconcile actual usage and stop at hard limits.

  **Check.** With a four-call limit, the fifth dispatch must be blocked.
  A harmless final answer after unbounded work still fails the availability contract.

  **Use when.** Resource consumption is the compromise outcome.

  ## Failure: A Delimiter Replaces Enforcement

  Marking uploaded text as “data” cannot contain an executor holding broad reusable credentials.
  Remove unnecessary authority, enforce all dispatch paths and test bypasses.

  ## Transfer Check

  Does the assistant's final refusal establish that nothing was compromised?

  ### Reasoned Answer

  No. Inspect earlier calls, arguments, outputs, caches, logs and effects under the declared attacker capabilities.
  Report the tested boundary and residual assumptions, not a universal security claim.

  <!-- learning:core-end -->
  <div class="learning-extension" data-learning-extension>

## Major attack classes

<span id="prompt-injection"></span>
<span id="data-exfiltration"></span>
<span id="tool-misuse-and-confused-deputy"></span>
<span id="model-and-data-attacks"></span>
<span id="availability-and-cost-attacks"></span>

| Class | Compromise to test |
| --- | --- |
| Direct/indirect injection | User, retrieval, file, tool, plugin or agent content redirects a reachable action |
| Disclosure | Protected context, retrieval, memory, tool data or logs reach an unauthorized reader |
| Confused deputy | A legitimate service applies its authority to an attacker's request |
| Model/data attacks | Extraction, membership inference, poisoning, backdoors or adversarial inputs violate declared protections |
| Availability/cost | Context, loops, branches, malformed input or scheduling exceeds resource/fairness limits |

## Defense in depth

Minimize data and authority, preserve provenance, validate arguments, isolate tenants and bound generated code/network access.
Monitor effects, govern confirmation, redact evidence, and name revocation/recovery owners.
Sandbox limits and memory/cache isolation require independent tests.
Monitor alarms are not enforcement unless a trusted component acts on them.

## Common confusions

Internal text can remain untrusted; natural-language delimiters do not enforce access.
Security addresses adversarial asset compromise; safety also includes non-adversarial harm.
Report severity, reachability, attacker budget, transfer, channel coverage and residual uncertainty alongside attack success.

## In an interview

Start with one reachable path, an observable compromise criterion and a named control owner.
Then vary capabilities and test residual paths with controlled artifacts.

## Sources

- [Greshake et al.: Indirect prompt injection](https://arxiv.org/abs/2302.12173).
- [OWASP: Prompt injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/).
- [NIST SP 800-207: Zero Trust Architecture](https://doi.org/10.6028/NIST.SP.800-207).

*Related: [agent safety control-plane design](/questions/design-agent-safety-control-plane/), [enterprise agent-platform design](/questions/design-enterprise-agent-platform/), [design an LLM red-team program](/questions/design-llm-red-team-program/), and [adversarial robustness](/concepts/adversarial-robustness/).*

</div>
