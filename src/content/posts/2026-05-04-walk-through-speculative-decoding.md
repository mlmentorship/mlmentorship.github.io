---
title: "Walk me through speculative decoding"
description: "The interview signal is whether you understand why decoding is memory-bound and why the verify pass is essentially free."
date: "2026-05-04"
draft: false
tags: ["questions"]
category: "questions"
---


> *Asked in: LLM-team and inference-platform interviews.*

Walk through speculative decoding and explain when it can accelerate sampling without changing the target model's output distribution.

## Problem

**Context.** An autoregressive target model defines next-token probabilities, while a cheaper draft model can propose several tokens.
Both models use compatible token identities and the same accepted prefix.

**Task.** Explain proposal, target verification, acceptance, rejection correction, and continuation through one speculative block.
Justify distribution preservation and identify the workload conditions needed for a speedup.

**Constraints.** Distinguish exact sampling from greedy agreement or approximate decoding.
Account for proposal probabilities, normalized correction, draft cost, acceptance rate, and hardware utilization; verification is not universally free.

<!-- visual:speculative-decoding-weight-read-ledger -->
<figure class="learning-figure" aria-labelledby="speculative-cost-title">
<p class="visual-kicker">Learning objective</p>
<h3 class="visual-title" id="speculative-cost-title">Accepted prefix length determines whether draft and verification work pays off</h3><div class="visual-argument" data-visual-argument="speculative-decoding-weight-read-ledger" data-argument-kind="bars" data-argument-composition="endpoint"><svg viewBox="0 0 360 394" role="img" aria-label="Accepted tokens must repay draft and verification work. With target verification cost 1 and three draft costs of 0.1, two accepted tokens cost 0.65 each. One accepted token would cost 1.3, exceeding baseline 1."><title>Accepted tokens must repay draft and verification work</title><desc>With target verification cost 1 and three draft costs of 0.1, two accepted tokens cost 0.65 each. One accepted token would cost 1.3, exceeding baseline 1.</desc><text x="18" y="23" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="18" dy="0">Baseline target cost per token</tspan></text><path d="M28 64L332 64" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><rect data-proof-value="1" data-proof-min="0" data-proof-max="1.3" data-proof-zero="28" x="28" y="46" width="233.8461538461538" height="35" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></rect><circle cx="261.8461538461538" cy="63.5" r="4" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><path d="M261.8461538461538 69L261.8461538461538 86" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><text x="201.5" y="106" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-input-stroke)"><tspan x="201.5" dy="0">1 toy cost per accepted token</tspan></text><text x="18" y="140" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="18" dy="0">Two accepted: 1.3 / 2</tspan></text><path d="M28 181L332 181" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><rect data-proof-value="0.65" data-proof-min="0" data-proof-max="1.3" data-proof-zero="28" x="28" y="163" width="152" height="35" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></rect><circle cx="180" cy="180.5" r="4" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><path d="M180 186L180 203" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75"></path><text x="180" y="223" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-output-stroke)"><tspan x="180" dy="0">0.65 toy cost per accepted token</tspan></text><text x="18" y="257" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="18" dy="0">Only one accepted: 1.3 / 1</tspan></text><path d="M28 298L332 298" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><rect data-proof-value="1.3" data-proof-min="0" data-proof-max="1.3" data-proof-zero="28" x="28" y="280" width="304" height="35" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.75"></rect><circle cx="332" cy="297.5" r="4" style="fill:var(--viz-warning-stroke);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><path d="M332 303L332 320" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><text x="192.5" y="340" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="192.5" dy="0">1.3 toy cost per accepted token</tspan></text><text x="28" y="374" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="28" dy="0">0</tspan></text><text x="332" y="374" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="332" dy="0">1.3</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
<div class="visual-grid--two" role="group" aria-label="Cost comparison: ordinary low-batch decode reads the target model weights once to score one next position, while speculative verification reads the same target weights once to score K drafted positions in parallel; latency stays similar only while both operations are limited by weight bandwidth rather than compute">
<section class="visual-panel">
<h4>ORDINARY DECODE &#183; 1 POSITION</h4>
<p><strong>Target-weight traffic</strong><br />Read the large model's weights from HBM once.</p>
<p><strong>Useful target work</strong><br />Score one next-token position from the cached prefix.</p>
<p><strong>Typical low-batch bottleneck</strong><br />Weight movement dominates; tensor-core capacity is underused.</p>
</section>
<section class="visual-panel">
<h4>VERIFY &#183; <var>K</var> DRAFTED POSITIONS</h4>
<p><strong>Target-weight traffic</strong><br />Read the same large-model weights from HBM once.</p>
<p><strong>Useful target work</strong><br />Score the short drafted block in parallel, yielding distributions for several positions.</p>
<p><strong>Conditional latency result</strong><br />More arithmetic, but near one decode step only while the pass remains memory-bound.</p>
</section>
</div>
<div class="visual-derivation-caption">speculative verification is not free computation. It amortizes one expensive target-weight read across several useful positions. The wall time can stay close to one target decode step while bandwidth remains the bottleneck; large <var>K</var>, batching, draft cost, and cache work can erase that advantage. Original cost ledger checked against <a href="https://arxiv.org/abs/2211.17192">Leviathan et al. (2023)</a> and <a href="https://arxiv.org/abs/2302.01318">Chen et al. (2023)</a>.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="speculative-decoding-weight-read-ledger"><strong>Check:</strong> Why can expensive drafting or a short accepted prefix erase verification savings?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="speculative-decoding-weight-read-ledger">With target verification cost 1 and three draft costs of 0.1, two accepted tokens cost 0.65 each. One accepted token would cost 1.3, exceeding baseline 1.</span></figcaption>
</figure>

## What an L4 answer sounds like

> "A small model proposes K tokens; the big model checks them in parallel. If they match, we accept; if not, we resample. This makes decoding faster."

True at a slogan level, missing the why. You've heard the technique but not the reason it works.

## What an L5 answer sounds like

> "Setup: target model M (large, expensive), draft model m (small, cheap, same tokenizer).
>
> Per cycle:
> 1. m generates K candidate tokens autoregressively (cheap).
> 2. M runs ONE forward pass on the prefix + K candidates in parallel.
> 3. From M's output we get probabilities p_M at every position.
> 4. Accept token i with probability `min(1, p_M(x_i) / p_m(x_i))`. If rejected at position i*, resample from `normalize(max(0, p_M - p_m))` and discard the rest.
> 5. If all K accepted, sample one bonus token from M.
>
> Why it's free: decoding is *memory-bound*. The cost of a decode step is dominated by reading M's weights from HBM, not by the matmul. Running M on K+1 tokens vs 1 token is approximately the same wall time because the matmul gets bigger but the weight-reading is identical.
>
> The accept/reject rule is a special case of rejection sampling chosen so the marginal output distribution is exactly p_M. The output is provably indistinguishable from sampling M directly. It is *not* an approximation."

This is L5. You've described the algorithm, given the memory-bound argument, and noted the lossless property.

## What an L6 answer adds

> "...practical points:
>
> **Speedup depends on draft acceptance rate.** Average accepted tokens per cycle is roughly `(1 - alpha^(K+1)) / (1 - alpha)` where alpha is the per-token acceptance probability. With a well-distilled draft alpha is around 0.6-0.8, giving 2-3x typical speedup. With a mismatched draft you can have negative speedup.
>
> **K = 4-8 is the typical sweet spot.** Too small and you don't amortize M's pass; too large and late drafts almost always get rejected.
>
> **Variants worth knowing**:
> - Self-speculation (Medusa): M itself produces K drafts via extra heads. No separate draft model.
> - EAGLE: a small feature regressor predicts M's hidden states cheaply.
> - Tree speculation: m proposes a tree of branches, M verifies all in one pass, longest accepted prefix wins.
>
> **Constraints**:
> - Tokenizer must match between m and M.
> - Sampling parameters (temperature, top-p) must be applied consistently.
> - Some serving systems (vLLM, TensorRT-LLM) implement this; rolling your own is non-trivial because of the KV-cache management.
>
> **Why people get this wrong in interviews**: they call it 'approximate.' It's not. The output distribution is exact (modulo floating point)."

## Tells that get you a strong-hire vote

- You explain **why decoding is memory-bound** before describing the algorithm.
- You give the **lossless proof intuition**.
- You name the **speedup formula**.
- You bring up **Medusa, EAGLE, or tree speculation** as variants.

## Tells that get you down-leveled

- Calling it approximate.
- No mention of memory-bound decoding.
- Confusing it with batching or with quantization.
- No knowledge of variants beyond vanilla speculative.

## Common follow-up

"What if your draft model is much worse than the target?"

The L6 answer:

> "Speedup degrades and can go below 1. With acceptance rate alpha = 0.4, the formula gives roughly 1.5 tokens per cycle, but you've paid K draft costs plus M's verify cost; net wall-clock can be slower than just running M directly. The fix is either to distill a better draft (matched to M's outputs on the production distribution) or to switch to self-speculation (Medusa) where M generates its own drafts cheaply."

---

*Related: [speculative decoding](/concepts/speculative-decoding/), [KV cache](/concepts/kv-cache/), and [reduce LLM inference cost](/questions/reduce-llm-inference-cost-10x/).*
