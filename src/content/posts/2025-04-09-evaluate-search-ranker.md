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

<!-- visual:search-ranker-evidence-flywheel -->
<figure class="learning-figure editorial-illustration" aria-labelledby="search-ranker-evidence-flywheel-edition-title"><p class="visual-title" id="search-ranker-evidence-flywheel-edition-title">The average rises to 0.88 while the tail still fails its floor</p><div class="visual-argument" data-visual-argument="search-ranker-evidence-flywheel" data-argument-kind="matrix"><p class="visual-argument-premise">A better weighted average can conceal a failing tail</p><svg viewBox="0 0 360 142" role="img" aria-label="A better weighted average can conceal a failing tail. At head/tail weights 90%/10%, B averages 0.88 versus A at 0.80. Its tail score 0.70 misses the 0.78 floor."><title>A better weighted average can conceal a failing tail</title><desc>At head/tail weights 90%/10%, B averages 0.88 versus A at 0.80. Its tail score 0.70 misses the 0.78 floor.</desc><text x="105" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="105" dy="0">head</tspan></text><text x="199" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="199" dy="0">tail</tspan></text><text x="293" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="293" dy="0">weighted</tspan></text><text x="46" y="54" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="46" dy="0">A</tspan></text><rect data-proof-cell="0,0" x="58" y="36" width="91" height="44" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke)"></rect><text x="103.5" y="63.5" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="103.5" dy="0">0.8</tspan></text><rect data-proof-cell="0,1" x="152" y="36" width="91" height="44" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke)"></rect><text x="197.5" y="63.5" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="197.5" dy="0">0.8</tspan></text><rect data-proof-cell="0,2" x="246" y="36" width="91" height="44" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke)"></rect><text x="291.5" y="63.5" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="291.5" dy="0">0.8</tspan></text><text x="46" y="101" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="46" dy="0">B</tspan></text><rect data-proof-cell="1,0" x="58" y="83" width="91" height="44" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke)"></rect><text x="103.5" y="110.5" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="103.5" dy="0">0.9</tspan></text><rect data-proof-cell="1,1" x="152" y="83" width="91" height="44" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke)"></rect><text x="197.5" y="110.5" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="197.5" dy="0">0.7</tspan></text><path d="M156 123L239 87" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><rect data-proof-cell="1,2" x="246" y="83" width="91" height="44" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke)"></rect><text x="291.5" y="110.5" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="291.5" dy="0">0.88</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content"><svg viewBox="0 0 360 445" role="img" aria-labelledby="search-ranker-evidence-flywheel-edition-svg-title search-ranker-evidence-flywheel-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="search-ranker-evidence-flywheel-edition-svg-title">The better average still fails the tail floor</title><desc id="search-ranker-evidence-flywheel-edition-svg-desc">Fixed illustrative head and tail scores are A 0.80 and 0.80, and B 0.90 and 0.70. Shared-scale bars show each matched slice. The 90-to-10 traffic mixture gives averages 0.80 and 0.88, but B fails the declared tail floor 0.78 and must hold.</desc><text x="180" y="22" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Illustrative fixed slices; head/tail</tspan><tspan x="180" dy="19">weights 90% / 10%</tspan></text><text x="18" y="72" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="18" dy="0">HEAD</tspan></text><text x="35" y="108" style="font-size:13px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="35" dy="0">A</tspan></text><rect x="60" y="89" width="208" height="26" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="330" y="108" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="330" dy="0">0.80</tspan></text><text x="35" y="145" style="font-size:13px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="35" dy="0">B</tspan></text><rect x="60" y="126" width="234" height="26" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="330" y="145" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="330" dy="0">0.90</tspan></text><text x="18" y="210" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="18" dy="0">TAIL</tspan></text><text x="35" y="246" style="font-size:13px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="35" dy="0">A</tspan></text><rect x="60" y="227" width="208" height="26" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="330" y="246" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="330" dy="0">0.80</tspan></text><text x="35" y="283" style="font-size:13px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="35" dy="0">B</tspan></text><rect x="60" y="264" width="182" height="26" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="330" y="283" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="330" dy="0">0.70</tspan></text><path d="M262.8 220L262.8 294" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:5 4"></path><text x="262.8" y="320" style="font-size:12px;font-weight:400;fill:var(--viz-warning-stroke);text-anchor:middle"><tspan x="262.8" dy="0">floor 0.78</tspan></text><text x="180" y="377" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Weighted: A = 0.80; B = 0.88</tspan></text><text x="180" y="418" style="font-size:14px;font-weight:700;fill:var(--viz-warning-stroke);text-anchor:middle"><tspan x="180" dy="0">B tail 0.70 &lt; 0.78 -&gt; HOLD</tspan></text></svg><div class="visual-derivation-caption">Traffic weighting improves B's average while its tail breaches the floor. Hold B and diagnose the slice. An aggregate improvement is not release permission.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="search-ranker-evidence-flywheel"><strong>Check:</strong> Why must B be held despite improving the weighted average from 0.80 to 0.88?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="search-ranker-evidence-flywheel">At head/tail weights 90%/10%, B averages 0.88 versus A at 0.80. Its tail score 0.70 misses the 0.78 floor.</span></figcaption></figure>

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
