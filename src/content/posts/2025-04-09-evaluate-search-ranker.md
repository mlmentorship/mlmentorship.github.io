---
title: "How would you evaluate a search ranker?"
description: "Evaluate a search ranker with judged queries, explicit averaging weights, exposure controls, slice guardrails and a recorded release decision."
date: "2025-04-09"
draft: false
tags: ["questions"]
category: "questions"
---


Decide whether a proposed search ranker improves the relevant metric and satisfies a required query-slice quality gate.

## Problem

**Context.** Two rankers have relevance judgments for frequent head queries and less frequent tail queries.
NDCG@10 summarizes ranking quality over the first ten results.

**Task.** Compute the traffic-weighted comparison below, apply the tail gate, and identify evidence still needed for a launch decision.

**Constraints.** Use the stated traffic weights rather than equal slice weights.
These invented offline scores do not by themselves measure user outcomes or causal product impact.

Compare A and B on two equally sized sets of judged queries: head and tail.
Suppose traffic is 90% head and 10% tail.
These are fictional slice means and traffic weights, not measured production results.

| Ranker | Head NDCG@10 | Tail NDCG@10 |
| --- | ---: | ---: |
| A | 0.80 | 0.80 |
| B | 0.90 | 0.70 |

The proposed offline gate requires tail NDCG@10 of at least 0.78.
Does B improve the headline metric, pass the gate or establish better user outcomes?

## Picture: The Weights Change the Headline

Imagine one equal-sized stack of judged queries for each slice.
Equal-slice averaging gives both rankers 0.80.
Traffic averaging gives the head stack nine times the tail stack's weight.

$$
\begin{aligned}
A_{\text{traffic}}&=0.9(0.80)+0.1(0.80)=0.80\\
B_{\text{traffic}}&=0.9(0.90)+0.1(0.70)=0.88.
\end{aligned}
$$

B gains 0.08 in traffic-weighted NDCG but loses 0.10 on tail queries.
It fails the declared 0.78 tail gate.
Neither weighted average establishes causal user benefit.

<p class="visual-title">A weighted gain cannot erase a failed tail guardrail</p>

<!-- visual:search-ranker-evidence-flywheel -->
```mermaid
flowchart TB
  accTitle: A weighted NDCG improvement does not override the tail guardrail
  accDescr: Human judgments define a versioned relevance dataset. Fixed head and tail slices give A scores 0.80 and 0.80, and B scores 0.90 and 0.70. With 90 percent head traffic, traffic-weighted NDCG is 0.80 for A and 0.88 for B. B nevertheless fails the required tail score of at least 0.78 and returns for diagnosis. Candidates that pass declared screening can enter an appropriately randomized product experiment. Its estimand, uncertainty and guardrails determine a staged release decision. Production failures inform a separately versioned judging refresh rather than silently changing a running comparison.
  Human["Human rubric + sampled candidates<br/>versioned relevance judgments"]
  Slices["Fixed head / tail slices<br/>A: 0.80 / 0.80<br/>B: 0.90 / 0.70"]
  Weights["Traffic weights: 0.90 / 0.10<br/>weighted A: 0.80<br/>weighted B: 0.88"]
  Gate{"Tail at least 0.78?<br/>B: 0.70 fails"}
  Hold["Hold B; diagnose tail failure<br/>average gain is not permission"]
  Online["Passing candidate only:<br/>randomized product experiment<br/>estimand + uncertainty + guardrails"]
  Release{"Release evidence sufficient?"}
  Ship["Staged release<br/>monitor + compatible rollback"]
  Refresh["Failure sampling + new judgments<br/>new dataset version"]
  Human --> Slices --> Weights --> Gate
  Gate -->|"B fails"| Hold
  Gate -->|"candidate passes"| Online --> Release
  Release -->|"yes"| Ship --> Refresh
  Release -->|"no or uncertain"| Hold
  Refresh -.-> Human
  class Human,Slices,Refresh viz-state
  class Weights,Gate,Online,Release viz-focus
  class Hold viz-warning
  class Ship viz-output
```

<p class="diagram-caption"><strong>Read it this way:</strong> calculate the weighted average, then apply the tail constraint. Passing offline screening permits further evaluation; it does not prove product lift.</p>

<span id="what-an-l4-answer-sounds-like"></span>
<span id="what-an-l5-answer-sounds-like"></span>

## Decision Contract: Three Different Questions

| Layer | Measures | Important limit |
| --- | --- | --- |
| Human judgments | Relevance under a documented rubric | Rater agreement is not truth or user satisfaction |
| Offline metrics | Fixed-label ranking comparison, including slices | NDCG: graded ordering; MRR: first relevant result; Recall@K: relevant-result coverage |
| Online experiment | Defined behavioral/product effect under randomized exposure | Clicks depend on position and prior exposure; no-click can also mean success |

Freeze an evaluation manifest: query/candidate/judgment/exposure versions, cutoff, gains, weights, slices and uncertainty.
Also declare randomization unit, target effect, outcome maturity, guardrails, decision owner and rollback conditions.
Define zero-ideal-gain queries and relevance thresholds before computing metrics.

## Change 1: New Documents Have No Judgments

**New task.** A new retriever finds documents absent from the old judging pool.

**Picture and change.** Mark them unknown, pool candidates from multiple systems and obtain rubric-based judgments.
Version the resulting comparison and report judged coverage by slice.

**Check.** The old ranker must not win merely because its documents already have labels.

**Use when.** Candidate support changes.

## Change 2: The Goal Is Long-Term Satisfaction

**New task.** Relative click preference is insufficient for the release decision.

**Picture and change.** Choose a product experiment with suitable randomization, duration, mature outcomes and safety/cost/latency guardrails.
Interleaving can still compare rankers through attributed clicks under its protocol.

**Check.** Interleaving preference is not automatically long-term causal value or universally more powerful than A/B testing.

**Use when.** The target effect changes.

<span id="tells-that-get-you-a-strong-hire-vote"></span>
<span id="tells-that-get-you-down-leveled"></span>

## Failure: Production Clicks Become Relevance Truth

Sampling only the current ranker's top results excludes unseen candidates.
Treating their raw clicks as relevance labels also confounds exposure, position and usefulness.
More data from that same selection process does not resolve missing support.

## Transfer Check

Does refreshing labels every quarter keep the evaluation valid?

### Reasoned Answer

No fixed calendar guarantees validity.
Monitor intents, candidates, policy, data and raters; refresh versioned rubrics while protecting ongoing comparisons.

<!-- learning:core-end -->
<div class="learning-extension" data-learning-extension>

<span id="what-an-l6-answer-adds"></span>

## Position and Counterfactual Evidence

Position-aware prediction alone does not identify relevance independently of exposure.
Randomized swaps and explicit click models need their own design and assumptions; they are not synonyms for interleaving.
Cascade and dynamic Bayesian click models impose behavioral assumptions, not guaranteed bias removal.

Logged-policy evaluation requires valid exposure propensities, support for evaluated actions, an identified outcome and controlled confounding/interference assumptions.
Inverse propensity weighting can have high variance.
It cannot recover unsupported actions by assigning them invented probabilities.
Record clipping and sensitivity analyses when used; correction changes uncertainty and sometimes the target.

<span id="common-follow-up"></span>

## Build and Maintain the Judged Set

Sample production queries across head, torso, tail, verticals and relevant user segments.
Record sampling weights and privacy constraints; pool candidate documents beyond the current ranker's results.
Document graded relevance criteria, rater training, adjudication and disagreement.
Measure agreement by slice and investigate shared rubric errors; agreement alone cannot establish validity.
Rejudge ambiguous/new candidates and retain a protected comparison set with versioned updates.
There is no universal agreement target or refresh interval.

</div>

---

*Related: [learning-to-rank losses](/concepts/learning-to-rank-losses/), [counterfactual ranking](/concepts/position-bias-counterfactual-learning-to-rank/), [A/B testing for ML systems](/concepts/ab-testing-for-ml/), and [personalized search ranking](/guides/personalized-search-ranking/).*
