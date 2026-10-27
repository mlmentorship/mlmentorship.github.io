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
<figure class="learning-figure" aria-labelledby="llm-adaptation-title">
	<p class="visual-kicker">Independent release requirements</p>
	<p class="visual-title" id="llm-adaptation-title">Current evidence and valid output need separate checks.</p>
	<div class="visual-grid--two" role="group" aria-label="Policy evidence and output validation for the same refund request">
		<section class="visual-panel">
			<h4>1. Supply the applicable policy</h4>
			<p>Day fourteen exceeds A's seven days but fits B's thirty days.</p>
			<p>Supply authorized B directly, then test whether retrieval returns that same applicable revision.</p>
			<p>Version, freshness, access and correct interpretation all matter.</p>
		</section>
		<section class="visual-panel">
			<h4>2. Validate the output</h4>
			<p><code>{"eligible":true,"policy_version":"B"}</code></p>
			<p>Check field names, types and allowed values separately from policy correctness.</p>
			<p>A tuned model still needs current evidence, a request contract and release checks.</p>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> the left panel tests applicable evidence; the right tests output structure. Neither check establishes the other.</figcaption>
</figure>

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
