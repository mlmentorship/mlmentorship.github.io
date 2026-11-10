---
title: "When would you fine-tune vs prompt vs RAG?"
description: "Locate a failed requirement, test the suspected cause and compare prompts, retrieval, training or a simpler implementation."
date: "2025-05-21"
draft: false
tags: ["questions"]
category: "questions"
---


Choose how to repair a model service that uses an outdated policy and violates its required output format.

## Problem

**Context.** A fictional support service must follow an authorized policy revision and return a defined JSON response.
Its baseline fails both policy correctness and output-format checks.

**Task.** Design tests that isolate those failures, then choose among prompting, retrieval, fine-tuning, or a justified combination.
Specify success measurements and rollback conditions.

**Constraints.** Use only authorized policy evidence. Separate changing facts from learning behavior or enforcing an exact output contract.
Compare operational cost and freshness needs without assuming one intervention fixes both failures.

A fictional support service must apply yesterday's authorized policy revision B and return two JSON fields.
Policy A allowed refunds within seven days; B allows thirty days.
The request concerns day fourteen.
The baseline applies A and returns malformed JSON.

Choose tests that separate missing policy evidence from output-format failure.
Then decide what to change, what to measure and how to revert.

## Picture: Two Failed Checks, Not Three Ranked Techniques

The current-policy check expects eligibility under B.
The format check expects a boolean `eligible` and a `policy_version` string, with no other fields.
Passing either check does not imply passing the other.

Use this stipulated diagnostic ledger; these are example outputs, not measured model runs.

| Intervention | Example output | Policy check | JSON/schema check |
| --- | --- | --- | --- |
| Baseline | `eligible = false` | Fail | Fail |
| Supply authorized B directly | `Eligible under policy B.` | Pass | Fail |
| Supply B and an output contract | `{"eligible":true,"policy_version":"B"}` | Pass | Pass |

Supplying known-correct context bypasses retrieval to test whether the model can use the needed evidence.
It does not prove a retrieval system will find B.
If the model still misreads B, better retrieval alone cannot resolve that failure.

<span id="what-an-l5-answer-sounds-like"></span>

<!-- visual:llm-adaptation-layers -->
<figure class="learning-figure editorial-illustration" aria-labelledby="llm-adaptation-layers-edition-title"><span class="visual-fragment-anchor" id="llm-adaptation-title" aria-hidden="true"></span><p class="visual-title" id="llm-adaptation-layers-edition-title">Isolate missing policy evidence from output-format failure before choosing an intervention</p><div class="visual-argument" data-visual-argument="llm-adaptation-layers" data-argument-kind="matrix" data-argument-composition="adaptation-diagnostic"><svg viewBox="0 0 360 2452" role="img" aria-label="Isolate missing policy evidence from output-format failure before choosing an intervention. Policy correctness and output format are independent checks. Supplied policy B fixes eligibility in the day-fourteen case, but not JSON. Adding an output contract passes both stipulated checks; valid JSON using stale A still fails policy. Direct context tests evidence use, not retrieval coverage. Prompting, authorized retrieval and training can compose. Evaluate persistent behavior errors on held-out tasks; compare quality, cost, freshness and rollback."><title>Isolate missing policy evidence from output-format failure before choosing an intervention</title><desc>Policy correctness and output format are independent checks. Supplied policy B fixes eligibility in the day-fourteen case, but not JSON. Adding an output contract passes both stipulated checks; valid JSON using stale A still fails policy. Direct context tests evidence use, not retrieval coverage. Prompting, authorized retrieval and training can compose. Evaluate persistent behavior errors on held-out tasks; compare quality, cost, freshness and rollback.</desc><text x="20" y="27" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Change one suspected cause; keep both</tspan><tspan x="20" dy="18">checks</tspan></text><text x="20" y="109" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Fictional refund case: day fourteen.</tspan><tspan x="20" dy="18">Old A allows seven days; authorized B</tspan><tspan x="20" dy="18">allows thirty. Expected: eligible</tspan><tspan x="20" dy="18">under B.</tspan></text><text x="20" y="223" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-focus-stroke)"><tspan x="20" dy="0">Stipulated example responses, not</tspan><tspan x="20" dy="18">measured model runs</tspan></text><text x="20" y="305" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Baseline</tspan></text><text x="20" y="362" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Stale A; no working output contract</tspan></text><text x="20" y="428" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--viz-focus-stroke)"><tspan x="20" dy="0">Bare text: eligible false</tspan></text><circle cx="90" cy="513" r="23" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><text x="90" y="519" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="90" dy="0">no</tspan></text><text x="90" y="565" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="90" dy="0">Policy B</tspan></text><circle cx="265" cy="513" r="23" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><text x="265" y="519" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="265" dy="0">no</tspan></text><text x="265" y="565" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="265" dy="0">Schema</tspan></text><path d="M20 615L340 615" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="20" y="685" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Supply authorized B directly</tspan></text><text x="20" y="742" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Correct context only; retrieval</tspan><tspan x="20" dy="18">bypassed</tspan></text><text x="20" y="808" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--viz-focus-stroke)"><tspan x="20" dy="0">Eligible under policy B.</tspan></text><circle cx="90" cy="893" r="23" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="90" y="899" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="90" dy="0">yes</tspan></text><text x="90" y="945" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="90" dy="0">Policy B</tspan></text><circle cx="265" cy="893" r="23" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><text x="265" y="899" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="265" dy="0">no</tspan></text><text x="265" y="945" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="265" dy="0">Schema</tspan></text><path d="M20 995L340 995" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="20" y="1065" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Supply B and output contract</tspan></text><text x="20" y="1122" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Correct context plus the required</tspan><tspan x="20" dy="18">two-field schema</tspan></text><text x="20" y="1188" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--viz-focus-stroke)"><tspan x="20" dy="0">{&quot;eligible&quot;:true,&quot;policy_version&quot;:&quot;B&quot;</tspan><tspan x="20" dy="18">}</tspan></text><circle cx="90" cy="1273" r="23" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="90" y="1279" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="90" dy="0">yes</tspan></text><text x="90" y="1325" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="90" dy="0">Policy B</tspan></text><circle cx="265" cy="1273" r="23" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="265" y="1279" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="265" dy="0">yes</tspan></text><text x="265" y="1325" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="265" dy="0">Schema</tspan></text><path d="M20 1375L340 1375" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="20" y="1445" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Control: valid JSON, stale A</tspan></text><text x="20" y="1502" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Repair formatting without changing</tspan><tspan x="20" dy="18">the applied policy</tspan></text><text x="20" y="1568" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--viz-focus-stroke)"><tspan x="20" dy="0">{&quot;eligible&quot;:false,&quot;policy_version&quot;:&quot;A</tspan><tspan x="20" dy="18">&quot;}</tspan></text><circle cx="90" cy="1653" r="23" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><text x="90" y="1659" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="90" dy="0">no</tspan></text><text x="90" y="1705" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="90" dy="0">Policy B</tspan></text><circle cx="265" cy="1653" r="23" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="265" y="1659" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="265" dy="0">yes</tspan></text><text x="265" y="1705" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="265" dy="0">Schema</tspan></text><path d="M20 1755L340 1755" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="20" y="1878" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Direct B tests whether the model can</tspan><tspan x="20" dy="18">use evidence. It does not show that</tspan><tspan x="20" dy="18">an index retrieves B or that access</tspan><tspan x="20" dy="18">filtering works.</tspan></text><text x="20" y="2002" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Persistent behavior gaps: evaluate</tspan><tspan x="20" dy="18">representative training. Changing</tspan><tspan x="20" dy="18">facts: maintain authorized sources</tspan><tspan x="20" dy="18">and versions. These are composable</tspan><tspan x="20" dy="18">changes, not a mandatory ladder.</tspan></text><text x="20" y="2144" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Equal-quality cost toy: prompt adds</tspan><tspan x="20" dy="18">0.002 per request; tuning costs 2,000</tspan><tspan x="20" dy="18">plus 0.001 per request. Break-even:</tspan><tspan x="20" dy="18">two million requests. A 500 refresh</tspan><tspan x="20" dy="18">moves it to 2.5 million.</tspan></text><text x="20" y="2310" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--viz-warning-stroke)"><tspan x="20" dy="0">Include quality, freshness, latency,</tspan><tspan x="20" dy="18">failures, maintenance and rollback.</tspan><tspan x="20" dy="18">An exact whole-identifier grammar may</tspan><tspan x="20" dy="18">need a parser, not generation.</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content"><svg viewBox="0 0 360 560" role="img" aria-labelledby="llm-adaptation-layers-edition-svg-title llm-adaptation-layers-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="llm-adaptation-layers-edition-svg-title">Valid JSON can still apply the wrong policy</title><desc id="llm-adaptation-layers-edition-svg-desc">Policy A permits seven days and applicable B thirty days on a shared day axis. Day fourteen exceeds A but fits B. A response falsely declaring ineligibility under A is structurally valid but fails the independent current-evidence check. Release is held.</desc><text x="180" y="22" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Toy policy limits; both shown on the same day</tspan><tspan x="180" dy="19">axis</tspan></text><text x="18" y="68" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="18" dy="0">A: 7 days</tspan></text><rect x="40" y="83" width="63" height="22" rx="3" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.3"></rect><text x="18" y="143" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="18" dy="0">B: 30 days; applicable current version</tspan></text><rect x="40" y="158" width="270" height="22" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.3"></rect><path d="M166 51L166 195" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75;stroke-dasharray:5 4"></path><text x="166" y="218" style="font-size:12px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="166" dy="0">day 14</tspan></text><text x="40" y="218" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="40" dy="0">0</tspan></text><text x="310" y="218" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="310" dy="0">30</tspan></text><g data-editorial-node><rect x="25" y="251" width="310" height="74" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="180" y="283.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Counterexample: valid JSON</tspan><tspan x="180" dy="19">{&quot;eligible&quot;:false,&quot;policy_version&quot;:&quot;A&quot;}</tspan></text></g><g data-editorial-node><rect x="15" y="362" width="150" height="77" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.3"></rect><text x="90" y="396" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">Schema check</tspan><tspan x="90" dy="19">PASS</tspan></text></g><g data-editorial-node><rect x="195" y="362" width="150" height="77" rx="3" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.3"></rect><text x="270" y="396" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">Current evidence</tspan><tspan x="270" dy="19">FAIL: stale A</tspan></text></g><path d="M110 325L90 362" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M89.96886181709273,355.0000692565165L90,362L95.87341458428078,358.1917194009425" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M250 325L270 362" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M264.1265854157192,358.1917194009425L270,362L270.0311381829073,355.0000692565165" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="180" y="486" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">HOLD: valid structure cannot repair wrong</tspan><tspan x="180" dy="19">policy.</tspan></text><text x="180" y="524" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">B permits day 14; authorization still</tspan><tspan x="180" dy="19">applies.</tspan></text></svg><div class="visual-derivation-caption">The day-14 marker lies beyond A and inside B. Syntax passes while current-policy evidence fails. Neither a schema nor fine-tuning supplies missing current authority.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="llm-adaptation-layers"><strong>Check:</strong> Can structurally valid JSON establish that policy B was the applicable authorized evidence?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="llm-adaptation-layers">Policy correctness and output format are independent checks. Supplied policy B fixes eligibility in the day-fourteen case, but not JSON. Adding an output contract passes both stipulated checks; valid JSON using stale A still fails policy. Direct context tests evidence use, not retrieval coverage. Prompting, authorized retrieval and training can compose. Evaluate persistent behavior errors on held-out tasks; compare quality, cost, freshness and rollback. <strong>If valid JSON still applies policy A, which check fails?</strong></span></figcaption></figure>

## Make the Decision Inspectable

| Failed requirement | Candidate cause | Intervention | Matched check | Operating cost | Rollback |
| --- | --- | --- | --- | --- | --- |
| Current policy | Missing or stale B | Authorized retrieval | Same policy/version cases | Index freshness, retrieval latency | Restore prior index/configuration |
| JSON contract | Unclear output rules | Prompt/examples, supported constrained generation, validator | Syntax/types plus separate semantic checks | Context tokens, validation/retry | Restore prompt/schema version |
| Persistent task error | Learned behavior misses target | Representative fine-tuning experiment | Held-out tasks, same release limits | Training, serving, refresh, validation | Restore base/checkpoint |

These interventions compose; they are not mandatory steps in a ladder.
Training can learn facts, and retrieved text can change behavior.
The categories identify where the system changes, not exclusive effects.
Keep the complete release contract fixed when comparing candidates.

## Change 1: Request Volume Changes the Economics

**New task.** A prompt adds 0.002 currency units per request.
Tuning costs 2,000 initially and reduces that increment to 0.001.
Assume equal quality and exclude costs common to both systems.

**Picture and change.** Accumulate each variable cost, then add fixed training and separate refresh costs.

```python
from decimal import Decimal


def adaptation_costs(requests, refresh_cost=Decimal("0")):
    if requests < 0 or refresh_cost < 0:
        raise ValueError("cost inputs must be nonnegative")
    prompt_cost = Decimal("0.002") * requests
    tuned_cost = Decimal("2000") + refresh_cost + Decimal("0.001") * requests
    return prompt_cost, tuned_cost


assert adaptation_costs(2_000_000) == (Decimal("4000"), Decimal("4000"))
```

**Check.** The simple break-even is two million requests, not two million tokens.
A 500-unit refresh moves it to 2.5 million requests.

**Use when.** Measured recurring savings can repay adaptation before the model or task changes.

<span id="a-common-follow-up-when-does-it-make-sense-to-not-use-an-llm-at-all"></span>

## Change 2: The Task Is an Exact Grammar

**New task.** Accept a whole identifier `ACCT-` followed by six ASCII digits.
Preserve leading zeros and reject surrounding text.

**Picture and change.** Match the declared grammar instead of sampling a generated answer.

```python
import re


def parse_account_id(text):
    matched = re.fullmatch(r"ACCT-([0-9]{6})", text)
    return matched.group(1) if matched else None


assert parse_account_id("ACCT-000123") == "000123"
assert parse_account_id("ACCT-000123\n") is None
```

**Check.** Syntax validity does not establish that an account exists or that the caller may access it.

**Use when.** A parser, classifier or task-specific model meets the actual requirement more predictably.

## Failure: Training Becomes a Freshness Strategy

A weekly-changing policy needs source/version maintenance even if the model once learned it.
Likewise, finding B does not ensure correct interpretation or safe handling of untrusted content.

## Transfer Check

Could one system use fine-tuning, retrieval and prompting together?

### Reasoned Answer

Yes. A tuned model can receive current authorized evidence and request-specific instructions.
Evaluate the combined system; neither parameter updates nor citations remove schema, access or semantic checks.

<!-- learning:core-end -->
<div class="learning-extension" data-learning-extension>

<span id="what-an-l4-answer-sounds-like"></span>
<span id="the-tells-that-get-you-down-leveled"></span>

## Localize Failures Before Expanding the System

Audit retrieval coverage, document access, applicable versions, chunk selection and reranking separately from answer faithfulness.
Use direct correct context as a diagnostic, not as a production retrieval result.
Inspect real prompt contents before concluding that a fact is absent from the model.
Training a generator cannot authorize an inaccessible source or repair an index automatically.

<span id="what-an-l6-answer-sounds-like"></span>
<span id="the-tells-that-get-you-a-strong-hire-vote"></span>

## Compare Lifecycle Costs

Prompt costs depend on token counts, caching, provider pricing, latency and retries, not length alone.
Fine-tuning adds data preparation, privacy review, training, validation, checkpoint serving and refresh work.
A provider deprecation or model-family change can require new adaptation and qualification.
Retrieval adds ingestion, access filtering, versioning, evaluation, storage and serving costs.

Report cost per successful task, tail latency, invalid answers and severe failures under the same task distribution.
Shorter prompts or smaller models are potential savings, not guaranteed savings.
Include a named owner, versioned artifacts and a tested rollback for each changed component.

## Test a Simpler Baseline

A label classifier, parser or conventional ranking model may satisfy a narrow task without generative output.
Choose using representative quality, latency, cost and maintenance evidence, not the label “LLM application.”

*Related: [Evaluate an LLM application](/questions/how-would-you-evaluate-an-llm-application/).*

</div>
