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

<!-- visual:beam-search-live-finished-transition -->
<figure class="learning-figure" aria-labelledby="beam-pools-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="beam-pools-title">When does an EOS hypothesis leave the expandable beam?</p>
	<div class="visual-grid--two" role="group" aria-label="One beam-search step routes EOS candidates to a finished pool before pruning only the live candidates">
		<section class="visual-panel plot-panel">
			<svg viewBox="0 0 300 250" role="img" aria-labelledby="beam-expand-title beam-expand-desc">
				<title id="beam-expand-title">Every live prefix expands and accumulates log probability</title>
				<desc id="beam-expand-desc">With beam size two, live prefix A has cumulative log score negative 0.20 and live prefix B has negative 0.35. A expands to A x at negative 0.30, A EOS at negative 0.45, and A z at negative 0.90. B expands to B y at negative 0.40, B x at negative 0.75, and B EOS at negative 1.15. Each child score is its parent score plus the new token log probability.</desc>
				<rect class="viz-plot-bg" x="8" y="25" width="284" height="217" rx="5"></rect>
				<text class="viz-axis-label" x="12" y="16">1 · EXPAND EVERY LIVE PREFIX</text>
				<path d="M98 73L174 52M98 73L174 86M98 73L174 120M98 177L174 154M98 177L174 188M98 177L174 222" style="fill:none;stroke:var(--viz-edge);stroke-width:1.4"></path>
				<path d="M168 49L174 52L169 56M168 82L174 86L168 89M168 116L174 120L168 123M168 150L174 154L168 157M168 184L174 188L168 191M168 218L174 222L168 225" style="fill:none;stroke:var(--viz-edge);stroke-width:1.4"></path>
				<rect class="viz-node viz-node--input" x="18" y="52" width="80" height="42" rx="4"></rect>
				<rect class="viz-node viz-node--input" x="18" y="156" width="80" height="42" rx="4"></rect>
				<text class="viz-node-label" x="58" y="70" text-anchor="middle">LIVE A</text>
				<text class="viz-label" x="58" y="86" text-anchor="middle">-0.20</text>
				<text class="viz-node-label" x="58" y="174" text-anchor="middle">LIVE B</text>
				<text class="viz-label" x="58" y="190" text-anchor="middle">-0.35</text>
				<rect class="viz-node" x="176" y="39" width="108" height="26" rx="4"></rect>
				<rect x="176" y="73" width="108" height="26" rx="4" style="fill:var(--viz-surface);stroke:var(--viz-edge);stroke-width:1.5;stroke-dasharray:4 3"></rect>
				<rect class="viz-node" x="176" y="107" width="108" height="26" rx="4"></rect>
				<rect class="viz-node" x="176" y="141" width="108" height="26" rx="4"></rect>
				<rect class="viz-node" x="176" y="175" width="108" height="26" rx="4"></rect>
				<rect x="176" y="209" width="108" height="26" rx="4" style="fill:var(--viz-surface);stroke:var(--viz-edge);stroke-width:1.5;stroke-dasharray:4 3"></rect>
				<g class="viz-node-label" text-anchor="middle"><text x="230" y="56">A x · -0.30</text><text x="230" y="90">A &lt;EOS&gt; · -0.45</text><text x="230" y="124">A z · -0.90</text><text x="230" y="158">B y · -0.40</text><text x="230" y="192">B x · -0.75</text><text x="230" y="226">B &lt;EOS&gt; · -1.15</text></g>
			</svg>
		</section>
		<section class="visual-panel plot-panel">
			<svg viewBox="0 0 300 250" role="img" aria-labelledby="beam-prune-title beam-prune-desc">
				<title id="beam-prune-title">EOS routes to finished before top-k prunes the live candidates</title>
				<desc id="beam-prune-desc">A dashed finished pool contains A EOS at negative 0.45 and B EOS at negative 1.15; these records have no outgoing arrows and are not part of top-k pruning. The live pool ranks four non-EOS candidates. With beam size two it keeps A x at negative 0.30 and B y at negative 0.40, and prunes B x at negative 0.75 and A z at negative 0.90. Final ranking uses completed hypotheses, falling back to live hypotheses only if none completed.</desc>
				<rect class="viz-plot-bg" x="8" y="25" width="284" height="217" rx="5"></rect>
				<text class="viz-axis-label" x="12" y="16">2 · ROUTE EOS, THEN PRUNE LIVE ONLY</text>
				<rect x="16" y="47" width="128" height="83" rx="5" style="fill:var(--viz-surface);stroke:var(--viz-edge);stroke-width:1.5;stroke-dasharray:5 3"></rect>
				<text class="viz-axis-label" x="24" y="63">FINISHED · STOP</text>
				<text class="viz-node-label" x="80" y="88" text-anchor="middle">A &lt;EOS&gt; · -0.45</text>
				<text class="viz-node-label" x="80" y="111" text-anchor="middle">B &lt;EOS&gt; · -1.15</text>
				<rect x="154" y="47" width="130" height="149" rx="5" style="fill:var(--viz-surface);stroke:var(--viz-edge);stroke-width:1.5"></rect>
				<text class="viz-axis-label" x="162" y="63">LIVE · TOP 2 OF 4</text>
				<rect class="viz-node viz-node--focus" x="163" y="72" width="112" height="25" rx="4"></rect>
				<rect class="viz-node viz-node--focus" x="163" y="103" width="112" height="25" rx="4"></rect>
				<rect class="viz-node" x="163" y="134" width="112" height="25" rx="4"></rect>
				<rect class="viz-node" x="163" y="165" width="112" height="25" rx="4"></rect>
				<g class="viz-node-label" text-anchor="middle"><text x="219" y="89">KEEP A x · -0.30</text><text x="219" y="120">KEEP B y · -0.40</text><text x="219" y="151">PRUNE B x · -0.75</text><text x="219" y="182">PRUNE A z · -0.90</text></g>
				<path d="M166 135L272 158M272 135L166 158M166 166L272 189M272 166L166 189" style="fill:none;stroke:var(--viz-edge);stroke-width:1.2"></path>
				<text class="viz-callout" x="150" y="216" text-anchor="middle">FINAL: rank FINISHED; use LIVE only if none finished</text>
				<text class="viz-label" x="150" y="233" text-anchor="middle">No outgoing edge means no expansion after EOS.</text>
			</svg>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> add each token's log probability, then route by state before pruning. Dashed <code>&lt;EOS&gt;</code> records leave the live frontier and are never expanded again; top-<var>k</var> compares only the four non-EOS candidates. Here it keeps <code>A x</code> and <code>B y</code>, while both completed records remain available for the final scoring rule. Original worked example checked against <a href="https://arxiv.org/abs/1609.08144">Wu et al. (2016)</a>, <a href="https://arxiv.org/abs/1808.10006">Murray and Chiang (2018)</a>, and the <a href="https://huggingface.co/docs/transformers/main/en/generation_strategies">Transformers generation documentation</a>.</figcaption>
</figure>

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
