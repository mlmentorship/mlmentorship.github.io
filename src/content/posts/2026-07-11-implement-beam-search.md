---
title: "Implement beam search with EOS and length normalization"
description: "The implementation tests bounded hypothesis state, log-probability accounting, finished-sequence handling, and whether the decoder fits the task."
date: "2026-07-11"
draft: false
tags: ["questions"]
category: "questions"
---

> Implement beam search around a next-token function. Keep at most $k$ live hypotheses, stop expanding finished sequences, and return the best completed result.

Implement a bounded sequence search that handles end-of-sequence tokens and a consistently declared final scoring rule.

## Problem

**Context.** A next-token function maps a token prefix to log probabilities over a fixed vocabulary.
Inputs include beginning/end token IDs, positive beam width `k`, and a maximum number of newly generated tokens.

**Task.** Return the highest-final-score completed sequence found; if none completes within the budget, return the best remaining live sequence.
Specify the length-normalization rule, including whether beginning and end tokens count, before comparing results.

**Constraints.** Keep at most `k` live hypotheses and never expand a completed sequence.
Accumulate token log probabilities consistently, define tie behavior, and respect the generation budget.
Beam search is approximate search over sequences, not a guarantee of finding the global highest-probability completion.

## The baseline algorithm

Represent each hypothesis as `(tokens, cumulative_log_probability)`.

At each step:

1. call the model for every live prefix;
2. add each candidate token's log probability to the prefix score;
3. move EOS candidates to the finished pool;
4. keep only the top $k$ unfinished candidates;
5. stop when no live candidates remain, the token budget is exhausted, or a valid early-stop bound proves no live beam can beat the best finished one;
6. rank finished hypotheses with one consistent final scoring rule.

<!-- visual:beam-search-live-finished-transition -->
<figure class="learning-figure editorial-illustration" data-illustration-print-page aria-labelledby="beam-search-live-finished-transition-edition-title"><span class="visual-fragment-anchor" id="beam-pools-title" aria-hidden="true"></span><span class="visual-fragment-anchor" id="beam-expand-title" aria-hidden="true"></span><span class="visual-fragment-anchor" id="beam-expand-desc" aria-hidden="true"></span><span class="visual-fragment-anchor" id="beam-prune-title" aria-hidden="true"></span><span class="visual-fragment-anchor" id="beam-prune-desc" aria-hidden="true"></span><p class="visual-title" id="beam-search-live-finished-transition-edition-title">Finished beams stop expanding even while they remain final candidates</p><svg viewBox="0 0 360 1041" role="img" aria-labelledby="beam-search-live-finished-transition-edition-svg-title beam-search-live-finished-transition-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="beam-search-live-finished-transition-edition-svg-title">Route EOS out of live search, then compare completions with the declared length rule</title><desc id="beam-search-live-finished-transition-edition-svg-desc">A normalized toy starts A with probability 0.6 and B with 0.4. A has next-token probabilities 0.5, 0.3, 0.2 for x, EOS, z. B has 0.6, 0.25, 0.15 for y, x, EOS. Log totals are computed from those probabilities. EOS candidates enter a separate finished enclosure and never expand. Live pruning retains A x and B y. At a later step A x emits EOS with probability 0.5. Final comparison includes EOS and excludes BOS, using alpha one: A EOS has probability 0.18, length two and final score about -0.857; A x EOS has probability 0.15, length three and final score about -0.632. The longer completion wins only under this declared normalization.</desc><text x="180" y="22" style="font-size:12px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">NORMALIZED TOY PROBABILITIES; BEAM WIDTH 2</tspan></text><text x="180" y="54" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Child score = parent log score + log token</tspan><tspan x="180" dy="19">probability</tspan></text><text x="18" y="88" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="18" dy="0">A: p = 0.6; log p = -0.511</tspan></text><text x="210" y="114" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="210" dy="0">token p</tspan></text><text x="302" y="114" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="302" dy="0">log total</tspan></text><g data-beam-parent="A" data-beam-token="x" data-token-probability="0.5" data-parent-probability="0.6" data-log-score="-1.203972804325936"><text x="24" y="142" style="font-size:14px;font-weight:700;fill:var(--viz-input-stroke);text-anchor:start"><tspan x="24" dy="0">A x</tspan></text><text x="210" y="142" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="210" dy="0">0.5</tspan></text><text x="302" y="142" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="302" dy="0">-1.204</tspan></text></g><g data-beam-parent="A" data-beam-token="EOS" data-token-probability="0.3" data-parent-probability="0.6" data-log-score="-1.7147984280919268"><text x="24" y="176" style="font-size:14px;font-weight:700;fill:var(--viz-state-stroke);text-anchor:start"><tspan x="24" dy="0">A EOS</tspan></text><text x="210" y="176" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="210" dy="0">0.3</tspan></text><text x="302" y="176" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="302" dy="0">-1.715</tspan></text></g><g data-beam-parent="A" data-beam-token="z" data-token-probability="0.2" data-parent-probability="0.6" data-log-score="-2.120263536200091"><text x="24" y="210" style="font-size:14px;font-weight:700;fill:var(--viz-input-stroke);text-anchor:start"><tspan x="24" dy="0">A z</tspan></text><text x="210" y="210" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="210" dy="0">0.2</tspan></text><text x="302" y="210" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="302" dy="0">-2.120</tspan></text></g><text x="18" y="247" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="18" dy="0">B: p = 0.4; log p = -0.916</tspan></text><text x="210" y="273" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="210" dy="0">token p</tspan></text><text x="302" y="273" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="302" dy="0">log total</tspan></text><g data-beam-parent="B" data-beam-token="y" data-token-probability="0.6" data-parent-probability="0.4" data-log-score="-1.4271163556401456"><text x="24" y="301" style="font-size:14px;font-weight:700;fill:var(--viz-input-stroke);text-anchor:start"><tspan x="24" dy="0">B y</tspan></text><text x="210" y="301" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="210" dy="0">0.6</tspan></text><text x="302" y="301" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="302" dy="0">-1.427</tspan></text></g><g data-beam-parent="B" data-beam-token="x" data-token-probability="0.25" data-parent-probability="0.4" data-log-score="-2.3025850929940455"><text x="24" y="335" style="font-size:14px;font-weight:700;fill:var(--viz-input-stroke);text-anchor:start"><tspan x="24" dy="0">B x</tspan></text><text x="210" y="335" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="210" dy="0">0.25</tspan></text><text x="302" y="335" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="302" dy="0">-2.303</tspan></text></g><g data-beam-parent="B" data-beam-token="EOS" data-token-probability="0.15" data-parent-probability="0.4" data-log-score="-2.8134107167600364"><text x="24" y="369" style="font-size:14px;font-weight:700;fill:var(--viz-state-stroke);text-anchor:start"><tspan x="24" dy="0">B EOS</tspan></text><text x="210" y="369" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="210" dy="0">0.15</tspan></text><text x="302" y="369" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="302" dy="0">-2.813</tspan></text></g><path d="M20 395L340 395" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="180" y="425" style="font-size:12px;font-weight:700;fill:var(--viz-state-stroke);text-anchor:middle"><tspan x="180" dy="0">EOS LEAVES THE FRONTIER: NEVER EXPAND IT</tspan></text><rect x="20" y="444" width="320" height="86" rx="3" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.3;stroke-dasharray:5 3"></rect><text x="180" y="473" style="font-size:14px;font-weight:400;fill:var(--viz-state-stroke);text-anchor:middle"><tspan x="180" dy="0">A EOS: -1.715; finished</tspan></text><text x="180" y="507" style="font-size:14px;font-weight:400;fill:var(--viz-state-stroke);text-anchor:middle"><tspan x="180" dy="0">B EOS: -2.813; finished</tspan></text><text x="180" y="568" style="font-size:12px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">PRUNE ONLY LIVE CANDIDATES: KEEP TWO</tspan></text><g data-beam-live="A x" data-kept="true"><text x="180" y="604" style="font-size:14px;font-weight:700;fill:var(--viz-output-stroke);text-anchor:middle"><tspan x="180" dy="0">KEEP A x: -1.204</tspan></text></g><g data-beam-live="B y" data-kept="true"><text x="180" y="634" style="font-size:14px;font-weight:700;fill:var(--viz-output-stroke);text-anchor:middle"><tspan x="180" dy="0">KEEP B y: -1.427</tspan></text></g><g data-beam-live="A z" data-kept="false"><text x="180" y="664" style="font-size:14px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">drop A z: -2.120</tspan></text></g><g data-beam-live="B x" data-kept="false"><text x="180" y="694" style="font-size:14px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">drop B x: -2.303</tspan></text></g><path d="M20 716L340 716" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="180" y="750" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">LATER: p(EOS | A x) = 0.5</tspan></text><text x="180" y="783" style="font-size:13px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Final score = log p / length; alpha = 1</tspan></text><text x="180" y="812" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Length includes EOS, excludes BOS.</tspan></text><text x="210" y="845" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="210" dy="0">raw</tspan></text><text x="302" y="845" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="302" dy="0">final</tspan></text><g data-beam-completion="A EOS" data-probability="0.18" data-length="2" data-log-score="-1.7147984280919268" data-final-score="-0.8573992140459634"><text x="20" y="879" style="font-size:13px;font-weight:400;fill:var(--c-text);text-anchor:start"><tspan x="20" dy="0">A EOS; n=2</tspan></text><text x="210" y="879" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="210" dy="0">-1.715</tspan></text><text x="302" y="879" style="font-size:14px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="302" dy="0">-0.857</tspan></text></g><g data-beam-completion="A x EOS" data-probability="0.15" data-length="3" data-log-score="-1.8971199848858813" data-final-score="-0.6323733282952938"><text x="20" y="917" style="font-size:13px;font-weight:400;fill:var(--c-text);text-anchor:start"><tspan x="20" dy="0">A x EOS; n=3</tspan></text><text x="210" y="917" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="210" dy="0">-1.897</tspan></text><text x="302" y="917" style="font-size:14px;font-weight:700;fill:var(--viz-output-stroke);text-anchor:middle"><tspan x="302" dy="0">-0.632</tspan></text></g><text x="180" y="958" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Raw prefers A EOS; this final rule prefers A</tspan><tspan x="180" dy="19">x EOS.</tspan></text><text x="180" y="1005" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Only if no completion exists, rank remaining</tspan><tspan x="180" dy="19">live beams.</tspan></text></svg><figcaption><strong>Read it this way:</strong> Each parent distributes total token probability one. EOS candidates leave live search before pruning. The later final comparison divides log probability by the declared length. It can prefer a longer completion despite its lower raw probability. <span class="visual-reader-check" data-reader-check="beam-search-live-finished-transition"><strong>Check:</strong> Which score ranks completed sequences at the end? Why does an EOS sequence never return to live expansion?</span></figcaption></figure>

The raw sequence score is:

$$
S(y) = \sum_{t=1}^{|y|} \log p(y_t \mid y_{<t}).
$$

Because log probabilities are non-positive, raw score prefers shorter sequences. A simple length penalty is:

$$
S_{lp}(y) = \frac{S(y)}{|y|^\alpha}.
$$

State whether length includes BOS or EOS. Consistency matters more than one universal convention.

## Reference outline

```python
live = [((bos,), 0.0)]
finished = []

for _ in range(max_new_tokens):
    candidates = []
    for prefix, score in live:
        for token, token_logp in enumerate(step(prefix)):
            next_prefix = prefix + (token,)
            next_score = score + token_logp
            if token == eos:
                finished.append((next_prefix, next_score))
            else:
                candidates.append((next_prefix, next_score))
    live = top_k(candidates, beam_size)
    if not live:
        break

pool = finished or live
return max(pool, key=normalized_score)[0]
```

A production implementation batches beams and keeps tensor state. The scalar outline makes semantics easier to verify first.

## What an L4 answer sounds like

The candidate keeps top tokens independently at each step, which is not beam search over sequences. Or they expand finished beams, compare normalized and unnormalized scores mid-loop, and return the best live beam even when a better completed sequence exists.

## What an L5 answer adds

An L5 candidate uses cumulative log probability, maintains separate live and finished pools, defines length normalization, handles no-EOS fallback, and tests a case where the locally best first token does not produce the globally best sequence.

They test:

- beam size one equals greedy decoding;
- immediate EOS;
- no EOS before the budget;
- ties with deterministic ordering;
- very small probabilities without underflow;
- a finished beam is never expanded;
- length penalty changes the selected sequence in a controlled example.

## What an L6 answer adds

An L6 candidate connects sequence state to model state. Each beam carries or indexes a KV cache. Expanding and pruning hypotheses requires cache reordering without copying full prefixes unnecessarily. Finished beams release or freeze their state.

They discuss when beam search is the wrong decoder. Maximizing model likelihood often produces bland text for open-ended chat. Beam search is better suited to tasks with a constrained or reference-like answer, such as some translation, speech, or structured generation settings. For code or reasoning, sampling multiple candidates plus verification may outperform a narrow likelihood beam.

They also know early stopping is subtle under length normalization. A live beam with a worse current normalized score can improve relative ranking as length changes, so the stopping bound must match the final scoring rule.

## Tells that get you a strong-hire vote

- Scores accumulate in log space.
- Live and finished hypotheses are separate.
- EOS stops expansion.
- Length conventions are explicit and consistent.
- Tests include a globally non-greedy optimum.
- Beam state is connected to KV-cache reorder semantics.
- You state where beam search fits and where it does not.

## Tells that get you down-leveled

- Top-$k$ token sampling described as beam search.
- Multiplying raw probabilities until they underflow.
- Continuing after EOS.
- Comparing different score conventions in one pool.
- Returning only live hypotheses when completed ones exist.
- Claiming larger beam always improves user-visible quality.

## Common follow-up

"How would you add constrained decoding?"

Track the constraint state with each hypothesis and mask tokens that would make completion invalid. For a finite-state grammar, each beam carries a grammar state; token transitions update it. The pruning score remains probabilistic, but validity is a hard gate.

Use the [beam-search starter](/prep/labs/implementation/) before relying on the outline.

*Related: [decoding strategies](/concepts/decoding-strategies/), [KV cache](/concepts/kv-cache/), and [speculative decoding](/concepts/speculative-decoding/).*
