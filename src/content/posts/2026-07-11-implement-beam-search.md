---
title: "Implement beam search with EOS and length normalization"
description: "The implementation tests bounded hypothesis state, log-probability accounting, finished-sequence handling, and whether the decoder fits the task."
date: "2026-07-11"
draft: false
tags: ["questions"]
category: "questions"
---

> Implement beam search around a next-token function. Keep at most $k$ live hypotheses, stop expanding finished sequences, and return the best completed result.

Use log probabilities and separate live from finished hypotheses. Most bugs come from mixing those two states, normalizing at inconsistent times, or continuing to expand EOS.

## The baseline algorithm

Represent each hypothesis as `(tokens, cumulative_log_probability)`.

At each step:

1. call the model for every live prefix;
2. add each candidate token's log probability to the prefix score;
3. move EOS candidates to the finished pool;
4. keep only the top $k$ unfinished candidates;
5. stop when no live candidates remain, the token budget is exhausted, or a valid early-stop bound proves no live beam can beat the best finished one;
6. rank finished hypotheses with one consistent final scoring rule.

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
