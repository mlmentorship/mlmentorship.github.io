---
title: "Walk me through speculative decoding"
description: "The interview signal is whether you understand why decoding is memory-bound and why the verify pass is essentially free."
date: "2026-05-04"
draft: false
tags: ["interviews"]
category: "interviews"
---


> *Asked in: LLM-team and inference-platform interviews.*

The L4 candidate describes a draft model and verification. The L6 candidate explains *why* the verify pass costs roughly the same as a single decode step (memory bandwidth, not compute), and proves the output distribution is preserved.

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

*Related reference: [Speculative decoding](/reference/speculative-decoding/), [KV cache](/reference/kv-cache/), [How would you reduce LLM inference cost by 10x?](/interviews/reduce-llm-inference-cost-10x/).*
