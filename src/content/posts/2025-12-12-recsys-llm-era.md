---
title: "Where an LLM belongs in a recommender"
description: "Choose a changed task, budget optional inference, compare simpler baselines and validate generated recommendations against the current catalog."
date: "2025-12-12"
draft: false
tags: ["questions"]
category: "questions"
---


Choose one changed task, then budget the LLM stage and validate every catalog item before display.

## Problem

A fictional recommender handles 1,000 requests/second.
An optional reranker receives 10% of requests.
Its mean service time is 0.1 seconds; each call bills 2,000 input tokens.
Assume stable flow, no retries and this service boundary only.
Estimate its load, and assess a proposed 50-millisecond hard request deadline.

## Picture: Optional Work Still Consumes Capacity

<p class="visual-kicker">Choose placement, not a universal replacement</p>
<p class="visual-title">Intent, stored content and bounded reranking solve different tasks; every displayed item must remain valid.</p>

<!-- visual:recsys-llm-bounded-augmentation -->
```mermaid
flowchart TB
  accTitle: Optional LLM paths require budgets and live catalog validation
  accDescr: Requests can use fast retrieval directly or an optional intent helper. Item content can be encoded offline by a chosen simpler or language model. Retrieval and ranking use stored representations and permitted behavioral evidence. A bounded optional reranker has a deadline and fast fallback. An alternative semantic-ID generator must resolve IDs into catalog candidates before ranking. All outputs pass current catalog, permission and composition checks. This is an illustrative design, not a prevalence claim.
  Request["REQUEST<br/>query/session"] ==> Retrieve["FAST RETRIEVAL<br/>candidate set"]
  Request -.->|"optional"| Intent["INTENT HELPER<br/>clarify/rewrite"]
  Intent --> Retrieve
  Content["ITEM CONTENT"] -.->|"offline"| Encode["CONTENT ENCODER<br/>compare simpler/LLM"]
  Encode --> Catalog[("STORED FEATURES")]
  Catalog ==> Retrieve
  Retrieve ==> Rank["RANKER<br/>bounded top N"]
  Catalog --> Rank
  Behavior["PERMITTED HISTORY"] ==> Retrieve
  Behavior ==> Rank
  Rank -.->|"approved route"| Rerank["OPTIONAL RERANK<br/>deadline + token cap"]
  Rerank --> Validate["CURRENT CATALOG<br/>permission + composition"]
  Rank ==>|"fast fallback"| Validate
  Validate ==> Slate["VALID SLATE<br/>evidence-backed reasons"]
  Intent -.->|"alternative retrieval"| Generate["SEMANTIC ID<br/>sequence generation"]
  Generate --> Resolve["RESOLVE ID<br/>versioned catalog"]
  Resolve --> Rank
  class Request,Behavior,Content viz-input
  class Intent,Encode,Rerank viz-focus
  class Catalog,Resolve viz-state
  class Validate viz-warning
  class Slate viz-output
```

<p class="diagram-caption"><strong>Read it this way:</strong> follow the fast path, then inspect each optional branch's cost and validation boundary. Semantic-ID generation changes retrieval, not permission to display an unresolved item.</p>

## Calculate the Routed Load

| Quantity | Calculation | Meaning |
| --- | --- | --- |
| Reranker arrivals | 1,000 x 0.10 = 100/second | This route only |
| Mean calls in service | 100 x 0.1 = 10 | Little's law at the service boundary |
| Billed input rate | 100 x 2,000 = 200,000 tokens/second | Output, retries and other stages additional |

Ten is mean occupancy, not a safe capacity limit or p99 bound.
As a counterexample, let every routed call take 100 milliseconds and every bypass take five.
The service assumptions hold, but 10% of whole requests already exceed 50 milliseconds before other work.
Use a deadline-aware optional route with valid fallback; count timeout/cancellation costs and quality loss.

<span id="what-hasnt-changed"></span>

## Contract: Budget the Stage, Validate the Output

Specify route predicate, candidate/token caps, deadlines, model/prompt/catalog identity and fallback compatibility.
Keep current existence, permission, region, availability and composition checks across every path.
Support explanations with actual decision evidence, not fluent guesses.
Predictive quality does not settle causal product value or participant policy.

<span id="common-follow-up"></span>

## Change 1: Move Content Work Offline

**New task.** New items have text, but request latency is strict.

**Picture and change.** Compare stored LLM and simpler content features under the same publication-freshness and cold-item evaluation.

**Check.** Classical content models also represent items without engagement.

**Use when.** Representation, not dialogue, is the bottleneck.

## Change 2: Clarify Constraints Through Conversation

**New task.** Users need iterative preference clarification.

**Picture and change.** Retrieve/rank through tools, confirm constraints and revalidate catalog choices.

**Check.** Compare task completion and interaction friction, not just pleasant chat.

**Use when.** Dialogue solves a real expression problem; preserve the effective low-friction surface otherwise.

<span id="tells-that-get-you-down-leveled"></span>

## Failure: A Plausible ID Is Displayed Without Resolution

A generated sequence can be invalid, ambiguous, stale or unauthorized.
An explanation cannot repair a missing item or invent its selection reason.

## Transfer Check

Does an unseen semantic-ID sequence prove discovery of a valid new item?

### Reasoned Answer

No. Resolve it against the versioned catalog and check identity, eligibility and representation quality.

<!-- learning:core-end -->
<div class="learning-extension" data-learning-extension>

<span id="what-an-l5-answer-sounds-like"></span>
<span id="what-an-l6-answer-adds"></span>

## Compare at the Changed Boundary

| Placement | Compare and control |
| --- | --- |
| Intent | Rewrite/disambiguation versus simpler parsing; preserve user constraints and timeout behavior |
| Stored content | Matched simpler encoder, freshness, versioned vectors, publication cost and cold/warm slices |
| Top-N reranking | Relevance/satisfaction, bounded work, tail latency, billing, fallback and evidence-backed reasons |
| Conversation | Completion, useful clarification, privacy and extra effort versus existing surfaces |
| Generated retrieval | Semantic-token mapping, collisions/invalid IDs, updates, current catalog checks and supported relevance |

Semantic IDs encode catalog items as token sequences, not unrestricted permission to invent products.
New-item support depends on mapping/publication, representation and retrieval evidence.

Compare automated judges with human-reviewed slices, failure types and bias.
Judge agreement is not ground truth or causal preference.
Use product experiments or other justified identification for incremental outcomes.
Keep calibration, support, feedback, cold-start and diversity contracts appropriate to the selected architecture.
Neither behavioral dominance nor a universal two-stage design follows from using an LLM.

<span id="tells-that-get-you-a-strong-hire-vote"></span>

## Sources

These papers document particular designs, not current adoption or universal performance.

- [YouTube retrieve-then-rank architecture, 2016](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/).
- [Language-based item representations, 2023](https://arxiv.org/abs/2305.13731).
- [LLM reranking, 2023](https://arxiv.org/abs/2305.08845).
- [Generative retrieval with semantic IDs, 2023](https://arxiv.org/abs/2305.05065).

---

*Related: [Design YouTube's recommender](/questions/design-youtube-recommender/), [Two-tower vs cross-encoder: when to use which?](/questions/two-tower-vs-cross-encoder/), [How would you do cold-start for a new user?](/questions/cold-start-new-user/), [Designing a RAG system that actually works](/guides/designing-rag-that-works/).*

</div>
