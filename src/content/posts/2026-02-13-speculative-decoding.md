---
title: "Speculative decoding"
description: "Draft and verify tokens while preserving the target sampling law. Determine when acceptance outweighs drafting and verification cost."
date: "2026-02-13"
draft: false
tags: ["concepts"]
category: "concepts"
---


## Summary

A small "draft" model proposes K candidate tokens cheaply; a single parallel forward pass of the large target model verifies them using a rejection-sampling rule that **provably preserves the target model's output distribution**.

Low-batch autoregressive decoding can be **memory-bound**: reading weights for the next token leaves arithmetic capacity underused.
That is a workload condition, not a rule for every serving configuration.
Large batches, long contexts, and KV-cache traffic can change the bottleneck.

Speculation batches target verification of several proposed positions into one pass.
That pass is not necessarily as cheap as ordinary one-token decoding, and rejected drafts still cost work.
The exact acceptance/correction scheme preserves the target distribution; latency improves only when accepted output amortizes the extra work.

The original [Leviathan et al. study](https://proceedings.mlr.press/v202/leviathan23a.html) reported 2-3x acceleration for T5-XXL against its T5X baseline.
That is a result for the paper's configuration, not a speed guarantee for another model, accelerator, or batch size.

<!-- visual:speculative-decoding-first-rejection-boundary -->
<figure class="learning-figure" aria-labelledby="speculative-rejection-visual-title">
	<p class="visual-kicker">Learning objective · one cycle with K = 5</p>
	<p class="visual-title" id="speculative-rejection-visual-title">Keep the accepted prefix; correct the first rejection</p>
	<svg viewBox="0 0 600 680" role="img" aria-labelledby="speculative-rejection-svg-title speculative-rejection-svg-desc" style="width: 100%; min-width: 0; height: auto;">
		<title id="speculative-rejection-svg-title">First rejection and its exact residual sampling correction</title>
		<desc id="speculative-rejection-svg-desc">One target pass scores five serial draft positions. The first two drafts are accepted; draft token A at position three is rejected, invalidating drafts four and five. At that position, conditional on the accepted prefix, the illustrative target distribution P over A,B,C is 0.2,0.3,0.5 and draft distribution D is 0.5,0.3,0.2. Accept proposed x with probability min(1,P(x)/D(x)). The positive part of P minus D is 0,0,0.3. Its normalized residual q is 0,0,1, so replacement is C, not an ordinary new draw from P. Commit the first two drafts and C, then restart from that changed prefix. Exactness assumes the stated sampling rule and compatible conditional distributions; no speedup is inferred.</desc>
		<g fill="var(--c-text)" font-family="var(--font-ui)" font-size="24">
			<text x="40" y="30" font-weight="650">One target pass; first rejection at 3</text>
			<g fill="var(--viz-input-bg)" stroke="var(--viz-input-stroke)"><rect x="40" y="68" width="88" height="44"/><rect x="144" y="68" width="88" height="44"/><rect x="248" y="68" width="88" height="44"/></g>
			<g fill="var(--viz-neutral-bg)" stroke="var(--viz-edge)" stroke-dasharray="4 3"><rect x="352" y="68" width="88" height="44"/><rect x="456" y="68" width="88" height="44"/></g>
			<g text-anchor="middle"><text x="84" y="98">draft 1</text><text x="188" y="98">draft 2</text><text x="292" y="98">A</text><text x="396" y="98">draft 4</text><text x="500" y="98">draft 5</text></g>
			<g font-size="20" text-anchor="middle"><text x="84" y="145">ACCEPT</text><text x="188" y="145">ACCEPT</text><text x="292" y="145">REJECT</text><text x="396" y="145">DISCARD</text><text x="500" y="145">DISCARD</text></g>
			<path d="M360 76L432 104 M464 76L536 104" stroke="var(--viz-warning-stroke)" stroke-width="2"/>
			<path d="M84 153V184 M188 153V184 M292 153V184" stroke="var(--viz-edge)" stroke-width="2"/>
			<g fill="var(--viz-output-bg)" stroke="var(--viz-output-stroke)"><rect x="40" y="190" width="88" height="44"/><rect x="144" y="190" width="88" height="44"/></g>
			<rect x="248" y="190" width="88" height="44" fill="var(--viz-focus-bg)" stroke="var(--viz-focus-stroke)"/>
			<g text-anchor="middle"><text x="84" y="220">draft 1</text><text x="188" y="220">draft 2</text><text x="292" y="220">C</text></g>
			<text x="354" y="209" font-size="22">Sample q;</text><text x="354" y="237" font-size="22">restart after C</text>
			<path d="M40 248V258H336V248" fill="none" stroke="var(--viz-output-stroke)" stroke-width="2"/><text x="188" y="286" text-anchor="middle" font-size="22">Committed prefix</text>
			<path d="M40 310H560" stroke="var(--c-rule)"/>
			<text x="40" y="346" font-size="22">At position 3, after the accepted prefix:</text>
			<text x="40" y="380" font-size="21">Accept x with min(1, P(x) / D(x)).</text>
			<g font-size="22"><text x="300" y="418">A</text><text x="400" y="418">B</text><text x="500" y="418">C</text></g>
			<g data-speculative-row="draft"><text x="40" y="454">Draft D</text><text x="300" y="454">0.5</text><text x="400" y="454">0.3</text><text x="500" y="454">0.2</text></g>
			<g data-speculative-row="target"><text x="40" y="490">Target P</text><text x="300" y="490">0.2</text><text x="400" y="490">0.3</text><text x="500" y="490">0.5</text></g>
			<g data-speculative-row="positive-part"><text x="40" y="526">Positive P - D</text><text x="300" y="526">0</text><text x="400" y="526">0</text><text x="500" y="526">0.3</text></g>
			<g data-speculative-row="correction"><text x="40" y="562">Normalize: q</text><text x="300" y="562">0</text><text x="400" y="562">0</text><text x="500" y="562">1</text></g>
			<text x="40" y="610" font-weight="650">q returns C with probability 1.</text>
			<text x="40" y="650" font-size="22">A fresh P draw is not this correction.</text>
		</g>
	</svg>
	<figcaption><strong>Read it this way:</strong> the rejected A is replaced by C under the illustrated residual distribution q. Later drafts followed A, so they cannot survive that change. Accepted mass plus the rejection correction recovers the target law; the picture does not predict speedup.</figcaption>
</figure>

## The mechanism

Setup: large target model **M** (the one whose distribution we want to sample from), small draft model **m** (cheap; e.g., a 1B distilled version of a 70B M).

Per cycle:

1. **Draft.** Run m for $K$ autoregressive steps, retaining each proposal and its conditional draft distribution $D$.
2. **Verify.** Evaluate M on the proposed continuation in parallel, obtaining the target conditionals $P$ for the proposals and the possible bonus position.
3. **Accept/reject.** At each proposed position, accept token $x$ with probability $\min(1,P(x)/D(x))$. On the first rejection, sample from the normalized positive part of $P-D$, discard the remaining proposals, and restart from the corrected prefix.
4. **Bonus token.** If every proposal is accepted, sample one additional target token. The cycle emits accepted proposals plus exactly one replacement-or-bonus target token, at the cost of drafting and verification.

## Why it's lossless

The accept/reject rule is a special case of rejection sampling chosen specifically so that the marginal distribution of the output token at each position is *exactly* p&#7704;. The proof is one page of careful algebra; the upshot is **the output stream is statistically indistinguishable from sampling from M directly**.

This is a distributional guarantee in exact arithmetic under the stated conditional probabilities and correction rule, not a speed guarantee.
Apply the intended sampling transformations consistently, maintain the correct accepted-prefix KV state, and distinguish mathematical exactness from bit-for-bit reproducibility.
An implementation error is not excused as a harmless property of speculation.

## Speedup analysis

For an illustrative model, assume independent acceptance with constant probability $\alpha$ at every position.
Expected emitted tokens, including the replacement or bonus, are

$$
\mathbb{E}[N]=\sum_{j=0}^{K}\alpha^j=\frac{1-\alpha^{K+1}}{1-\alpha},\qquad 0\leq\alpha<1.
$$

For $\alpha=1$, the limit is $K+1$. Do not add a second bonus to this count.
Real acceptance can depend on position and prefix, so measure accepted-prefix lengths for the serving workload.

Normalize ordinary target-token latency to one. Let $d$ be a draft-step cost and $v_K$ the measured verification-pass cost in those units.
Under this serial cost model, the speed ratio is $\mathbb{E}[N]/(Kd+v_K)$.
Improvement requires $\mathbb{E}[N]>Kd+v_K$; batch scheduling and overlap can require a richer model.

**Predict before revealing:** use $K=4$, $d=0.1$, $v_K=1$, and disjoint draft/target support, so the first proposal always rejects.
Does exact correction still make the cycle faster?

<details><summary>Separate output correctness from cost</summary>

No. Each cycle emits one corrected token for cost 1.4, giving speed ratio $1/1.4\approx0.714$.
The target sampling law remains correct, but this toy configuration is slower.
The verifier cost of one is an optimistic illustrative assumption, not a benchmark measurement.

</details>

Tune $K$ against measured draft, verification, acceptance, memory, and batching costs rather than relying on a universal block length.

## Variants

- **Self-speculation / Medusa.** Internal prediction heads can propose continuations without a separate draft model. Training and acceptance schemes vary; preserving measured generation quality is not the same claim as preserving the original sampling distribution.
- **EAGLE.** Predicts target-model features to produce proposals. Its benefit depends on proposal quality and implementation costs, not a universal ordering over other methods.
- **Tree speculation.** Verifies branching candidates together. A valid acceptance scheme must choose the continuation; simply selecting the longest plausible branch does not establish sampling exactness.
- **Lookahead decoding.** Uses parallel lookahead/n-gram proposals. State the decoding rule and its guarantee rather than assuming every variant implements the rejection sampler above.

Compare a named implementation on a dated, matched model/hardware/workload configuration before ranking variants by speed or engineering cost.

## What an interviewer expects you to say

If asked about speculative decoding:

1. State the workload and why low-batch decoding may be memory-bound.
2. Explain proposal, parallel verification, accepted-prefix retention, and first-rejection correction.
3. State the exact sampler's distributional guarantee and its assumptions.
4. Derive a break-even condition from acceptance and measured costs instead of promising a speedup multiple.
5. Compare variants by their proposal mechanism, acceptance guarantee, memory use, and scheduling overhead.

## Common confusions

- **"Every speculative method is exact."** The rejection/correction sampler above preserves the target law in exact arithmetic; a different acceptance heuristic needs its own analysis.
- **"It only helps for greedy decoding."** No, it works for sampling too, the rejection rule is *defined* in terms of probabilities precisely because sampling is the general case.
- **"The draft must be fine-tuned for correctness."** Matching the target helps acceptance, but the stated correction rule, not fine-tuning, establishes the target law.
- **"Any small model is plug-compatible."** The algorithm above assumes a shared token vocabulary. Cross-tokenizer methods require explicit alignment and their own correctness conditions.
- **"It must save FLOPs."** Drafting and rejected proposals can add computation. Lower latency, when achieved, does not imply less arithmetic.

## Why interviewers care

This question tests whether you understand:
1. When weight bandwidth, arithmetic, or KV-cache traffic limits decoding.
2. The difference between batched and sequential workloads on GPU.
3. Lossless vs. lossy optimizations (a common confusion).
4. That you've kept up with serving developments since 2023.

Explain how rejected-prefix KV state is discarded and how batching changes the cost comparison.
Those concrete mechanisms are stronger evidence of understanding than quoting a speedup or a method name.

## Reading list

- *Fast Inference from Transformers via Speculative Decoding* [(Leviathan et al., 2023)](https://arxiv.org/abs/2211.17192)
- *Accelerating Large Language Model Decoding with Speculative Sampling* [(Chen et al., 2023)](https://arxiv.org/abs/2302.01318)
- *Medusa: Simple LLM Inference Acceleration with Multiple Decoding Heads* [(Cai et al., 2024)](https://arxiv.org/abs/2401.10774)
- *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty* [(Li et al., 2024)](https://arxiv.org/abs/2401.15077)

---

*Related: [FlashAttention](/concepts/flashattention/) and [production LLM inference design](/questions/design-production-llm-inference-service/).*
