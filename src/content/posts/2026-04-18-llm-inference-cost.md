---
title: "How to think about LLM inference cost"
description: "Count each billed model call once, including cache state, outputs, tools and retries; compare complete cost under quality and authority limits."
date: "2026-04-18"
draft: false
tags: ["guides"]
category: "guides"
---


One user request can trigger many billed calls; count each once before diagnosing repeated context, retries or expensive tails.

## Problem

Three model calls consume 1,200, 1,400 and 1,600 input tokens, plus 200 output tokens each.
Each input includes the same 1,000-token prefix.
The first use is uncached; the next two receive a prefix discount.
Tools cost 0.001 dollars in total.

Use fictional prices per million tokens: 1 dollar uncached input, 0.2 cached input and 4 output.
There are no additional cache-write charges in this example.
Compute the request total and compare it with the same calls without caching.
These are teaching prices, not current provider quotes.

## Picture: Price Three Rows, Not Five Overlapping Buckets

| Call | Total input | Cached input | Uncached input | Output | Model cost, dollars |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1,200 | 0 | 1,200 | 200 | 0.0020 |
| 2 | 1,400 | 1,000 | 400 | 200 | 0.0014 |
| 3 | 1,600 | 1,000 | 600 | 200 | 0.0016 |
| Total | 4,200 | 2,000 | 2,200 | 600 | 0.0050 |

The prefix and history are already inside each input count.
Do not add them again as separate token charges.
Add tools once: the full request costs 0.006 dollars.
Without caching, it costs 0.0076 dollars; the saving is about 21.05%.
The 80% cached-token discount does not apply to the entire request.

<!-- visual:llm-cost-compounds-across-agent-calls -->
<figure class="learning-figure" aria-labelledby="llm-cost-compounding-title">
	<p class="visual-kicker">Input tokens, not currency</p>
	<p class="visual-title" id="llm-cost-compounding-title">The same prefix recurs, while other input grows.</p>
	<div class="visual-scroll">
		<svg viewBox="0 0 360 470" style="width: 400px; min-width: 400px" role="img" aria-labelledby="llm-cost-svg-title llm-cost-svg-desc">
			<title id="llm-cost-svg-title">Three input bars show 1,200, 1,400 and 1,600 tokens</title>
			<desc id="llm-cost-svg-desc">Each input bar contains a 1,000-token stable prefix and respectively 200, 400 or 600 other input tokens. Prefix width is 180 units; the other widths are 36, 72 and 108, so total lengths scale exactly with token counts. Call one prices the prefix as uncached; calls two and three price it as cached. Each call also has 200 output tokens, priced separately and not included in the input bars. An extra call adds a ledger row with its own actual cache and output usage.</desc>
			<defs>
				<pattern id="llm-cost-fixed-hatch" width="8" height="8" patternUnits="userSpaceOnUse"><rect width="8" height="8" fill="var(--viz-state-bg)"></rect><path d="M-2 2 L2 -2 M0 8 L8 0 M6 10 L10 6" stroke="var(--viz-state-stroke)" stroke-width="1"></path></pattern>
				<pattern id="llm-cost-history-dots" width="8" height="8" patternUnits="userSpaceOnUse"><rect width="8" height="8" fill="var(--viz-focus-bg)"></rect><circle cx="2" cy="2" r="1.2" fill="var(--viz-focus-stroke)"></circle><circle cx="6" cy="6" r="1.2" fill="var(--viz-focus-stroke)"></circle></pattern>
				<marker id="llm-cost-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" fill="var(--viz-edge)"></path></marker>
			</defs>
			<text class="viz-axis-label" x="18" y="24">ONE USER REQUEST</text>
			<path d="M180 34 V58" fill="none" stroke="var(--viz-edge)" stroke-width="2" marker-end="url(#llm-cost-arrow)"></path>
			<text class="viz-axis-label" x="18" y="82">CALL 1: 1,200 IN + 200 OUT</text>
			<rect x="18" y="94" width="180" height="54" rx="3" fill="url(#llm-cost-fixed-hatch)" stroke="var(--viz-state-stroke)" stroke-width="1.5"></rect>
			<rect class="viz-node viz-node--input" x="198" y="94" width="36" height="54" rx="3"></rect>
			<text class="viz-edge-label" x="108" y="117"><tspan x="108">1,000 prefix</tspan><tspan x="108" dy="13">uncached</tspan></text>
			<text class="viz-edge-label" x="216" y="117"><tspan x="216">other</tspan><tspan x="216" dy="13">200</tspan></text>
			<path d="M180 157 V181" fill="none" stroke="var(--viz-edge)" stroke-width="2" marker-end="url(#llm-cost-arrow)"></path>
			<text class="viz-axis-label" x="18" y="205">CALL 2: 1,400 IN + 200 OUT</text>
			<rect x="18" y="217" width="180" height="54" rx="3" fill="url(#llm-cost-fixed-hatch)" stroke="var(--viz-state-stroke)" stroke-width="1.5"></rect>
			<rect x="198" y="217" width="72" height="54" rx="3" fill="url(#llm-cost-history-dots)" stroke="var(--viz-focus-stroke)" stroke-width="1.5"></rect>
			<text class="viz-edge-label" x="108" y="240"><tspan x="108">1,000 prefix</tspan><tspan x="108" dy="13">cached</tspan></text>
			<text class="viz-edge-label" x="234" y="240"><tspan x="234">other</tspan><tspan x="234" dy="13">400</tspan></text>
			<path d="M180 280 V304" fill="none" stroke="var(--viz-edge)" stroke-width="2" marker-end="url(#llm-cost-arrow)"></path>
			<text class="viz-axis-label" x="18" y="328">CALL 3: 1,600 IN + 200 OUT</text>
			<rect x="18" y="340" width="180" height="54" rx="3" fill="url(#llm-cost-fixed-hatch)" stroke="var(--viz-state-stroke)" stroke-width="1.5"></rect>
			<rect x="198" y="340" width="108" height="54" rx="3" fill="url(#llm-cost-history-dots)" stroke="var(--viz-focus-stroke)" stroke-width="1.5"></rect>
			<text class="viz-edge-label" x="108" y="363"><tspan x="108">1,000 prefix</tspan><tspan x="108" dy="13">cached</tspan></text>
			<text class="viz-edge-label" x="252" y="363"><tspan x="252">other</tspan><tspan x="252" dy="13">600</tspan></text>
			<path d="M306 367 H344 V420 H232" fill="none" stroke="var(--viz-warning-stroke)" stroke-width="2" stroke-dasharray="6 4"></path>
			<rect x="18" y="411" width="214" height="40" rx="4" fill="var(--viz-warning-bg)" stroke="var(--viz-warning-stroke)" stroke-width="1.5" stroke-dasharray="5 3"></rect>
			<text class="viz-gradient-label" x="125" y="428"><tspan x="125">Extra call: another row</tspan><tspan x="125" dy="13">price its actual usage</tspan></text>
		</svg>
	</div>
	<figcaption><strong>Read it this way:</strong> each input bar repeats 1,000 prefix tokens. Other input grows from 200 to 400 to 600 tokens. Widths represent token counts, not cost. The 200 output tokens per call are separate. Another call adds its actual usage, not an assumed full-price retry.</figcaption>
</figure>

## A Practical Cost Model

For each call, input is partitioned into uncached and cached tokens.
Rates below have currency-per-token units; add explicit write/tool charges and allocated costs only once.

$$
C_{\mathrm{request}}=
\sum_{\mathrm{calls}}
\left(I_u r_u+I_c r_c+O r_o+C_{\mathrm{write}}+C_{\mathrm{tool}}\right)
+C_{\mathrm{allocated}}.
$$

```python
from decimal import Decimal


def model_cost(calls):
	total = Decimal("0")
	million = Decimal("1000000")
	for call in calls:
		input_tokens = call["input_tokens"]
		cached = call["cached_input_tokens"]
		output = call["output_tokens"]
		if not 0 <= cached <= input_tokens or output < 0:
			raise ValueError("invalid token partition")
		total += (
			Decimal(input_tokens - cached) * Decimal("1")
			+ Decimal(cached) * Decimal("0.2")
			+ Decimal(output) * Decimal("4")
		) / million
		total += call.get("cache_write_cost", Decimal("0"))
	return total


calls = [
	{"input_tokens": 1200, "cached_input_tokens": 0, "output_tokens": 200},
	{"input_tokens": 1400, "cached_input_tokens": 1000, "output_tokens": 200},
	{"input_tokens": 1600, "cached_input_tokens": 1000, "output_tokens": 200},
]
assert model_cost(calls) == Decimal("0.005")
assert model_cost(calls) + Decimal("0.001") == Decimal("0.006")
```

Reconcile request/call/attempt IDs, usage categories, cache state, rate version and tool receipts against billed totals.
Include every retry, cancellation and failed task that incurred charges.

## Change 1: Stable Prefixes Dominate

**New task.** Most input repeats an authorized prefix.

**Picture and change.** Measure eligible cache hits, not just repeated text.
Check provider/model support, exact-prefix rules, expiry, write/miss charges and tenant isolation.
Compare safe prompt trimming under the same evaluation.

**Check.** Cached input does not discount outputs, tools or newly added history.
Authorization and source freshness still apply independently of cache reuse.

**Use when.** Measured eligible repeated work makes caching worthwhile.

## Change 2: Long Agent Runs Dominate

**New task.** Retries and growing tool history make some tasks expensive.

**Picture and change.** Bound steps, retries, tokens, time and money at enforced limits.
Summarize only while preserving source evidence, authority, unresolved effects and stop conditions.
Charge the summary calls too.

**Check.** With first input $I_0$ and $h$ additional history tokens per step, cumulative input is

$$
I_{\mathrm{total}}=mI_0+h\frac{m(m-1)}2.
$$

Our three-call ledger fits $I_0=1200$, $h=200$, giving 4,200 input tokens.
Bounded fixed-size context instead gives approximately linear cumulative input, excluding additional compaction work.

**Use when.** Measured multi-call tails dominate spend.
An exhausted run is incomplete, not a cheap success.

<span id="what-an-interviewer-expects-you-to-discuss"></span>

## Preserve Release Requirements

Compare quality, severe failures, authority, latency and cost per successful task on the same population.
Keep a rollback for prompt, routing, retrieval, model and budget changes.
Lower cost can come from removing waste or from losing useful work; distinguish them.

## Failure: Charge the Same Work Twice

Prompt and retrieval tokens cannot be added again after counting full input.
Actual tail calls already appear in the ledger; do not add another arbitrary tail charge.
A dimensionless multiplier cannot be added directly to currency.

## Transfer Check

Must cumulative input grow quadratically with the number of calls?

### Reasoned Answer

No. Quadratic growth requires each call to resend roughly linearly growing history.
Fixed-size context can grow approximately linearly, and cache pricing changes cost separately from token count.

<!-- learning:core-end -->
<div class="learning-extension" data-learning-extension>

## The Five Layers of Cost

Treat these as diagnostic views of one ledger, not five additive charges.

## Layer 1: Per-Request Token Cost

Separate input/output rates and usage classes.
Benchmark smaller models, distillation, quantization and batching under the same quality/latency requirements.
Price ratios and savings depend on the actual service.

## Layer 2: Prompt Amortization

Measure repeated instructions, examples and schemas.
Compare trimming, eligible caching and adaptation, including training, refresh and validation costs.
For example, 5,000 system tokens plus 100 query tokens make the system portion about 98% of uncached input.
That is not 98% of the full bill.

## Layer 3: Retrieval / Context Overhead

Measure retrieved passages, conversation history and tool results.
Reranking, field selection and compression can reduce tokens but add compute or remove necessary evidence.
Validate both effects; fewer chunks are not automatically better.

## Layer 4: Multi-Step / Agentic Cost

Measure planning, tools, verification, synthesis and retries separately.
Evaluate smaller-model routing and safe parallel work without speculating unauthorized or irreversible side effects.
Concurrent calls still consume resources and money.

## Layer 5: Tail Cost

Inspect p50/p95/p99, cohort sizes, completion and expensive traces.
Apply input/output/step caps and tenant quotas; prompt wording alone is not a hard limit.
Forecast tails with explicit assumptions rather than claiming universal percentile multipliers.

<span id="the-thing-nobody-mentions"></span>

## Changes Can Interact

Shorter context can reduce input work yet worsen answers and increase retries.
Evaluate complete request economics, not independent percentage reductions multiplied without evidence.

Self-hosted cost uses allocated accelerator capacity/time, utilization, memory, networking and operations.
Do not charge the same work both a hypothetical API token bill and its hosting cost.
Distinct external calls can still add real API charges.

## Provider References

Verify current dated pricing and cache semantics before applying any real rate.

- [OpenAI prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching).
- [Anthropic prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching).
- [Amazon Bedrock prompt caching](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html).
- [PagedAttention](https://arxiv.org/abs/2309.06180), a serving-memory design, not a provider pricing schedule.

*Related: [Designing a RAG system that actually works](/guides/designing-rag-that-works/), [When would you fine-tune vs prompt vs RAG?](/questions/fine-tune-vs-prompt-vs-rag/).*

</div>
