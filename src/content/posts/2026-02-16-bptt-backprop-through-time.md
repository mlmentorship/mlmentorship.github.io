---
title: "Explain backprop through time"
description: "BPTT is just backprop on the unrolled computation graph of a recurrent network. The interview signal is whether you understand truncation and what it costs."
date: "2026-02-16"
draft: false
tags: ["questions"]
category: "questions"
---


> *Asked in: ML breadth, especially in NLP and time-series interviews.*

The question is mechanical for L4 (define BPTT) and conceptual for L6 (truncated BPTT, vanishing/exploding gradients, and why transformers don't need it).

## What an L4 answer sounds like

> "BPTT applies backpropagation to RNNs by unrolling the network through time and computing gradients across all time steps."

Correct, no depth. You've memorized the term.

## What an L5 answer sounds like

> "An RNN at training time is computationally a deep feedforward network where the same weights appear at every time step. BPTT is just standard backpropagation applied to that unrolled graph.
>
> Two practical issues:
>
> 1. **Memory grows linearly with sequence length.** The forward activations at every time step must be cached for the backward pass. For long sequences, this is prohibitive.
>
> 2. **Gradients vanish or explode**. The gradient of the loss with respect to early-step weights involves a product of Jacobians, one per time step. If the Jacobian eigenvalues are < 1, the gradient vanishes; > 1, it explodes.
>
> Mitigations:
> - **Truncated BPTT (TBPTT)**: backprop only through K steps at a time, then detach. Trades exact gradients for tractable memory.
> - **Gradient clipping** for explosion.
> - **Architectures that mitigate vanishing**: LSTM, GRU (gated cells), residual connections, careful initialization."

This is L5. You've named the unrolling, the memory and gradient problems, and the standard mitigations.

## What an L6 answer sounds like

> "...two more things:
>
> **Truncated BPTT changes what the model can learn.** With truncation length K, the model can only learn dependencies up to ~K steps. This is why long-range dependencies are hard for vanilla RNNs even with TBPTT, and why architectures like LSTM (gated state that can persist information across many steps) and Transformers (parallel attention over all positions, no recurrence) became dominant.
>
> **Transformers replaced RNNs partly because they avoid BPTT entirely.** Self-attention computes all-to-all dependencies in one operation; the backward pass is parallel across positions. Memory still scales with sequence length squared (the attention matrix), which is why FlashAttention matters, but there's no sequential gradient chain to vanish or explode.
>
> **State-space models (Mamba, S4) are a recent middle ground**: they have recurrent structure for memory efficiency at long context, but use techniques (parallel scan, selective state) to avoid the worst BPTT problems."

## Tells that get you a strong-hire vote

- You frame BPTT as **standard backprop on the unrolled graph**, not a separate algorithm.
- You name **vanishing/exploding gradients** with the eigenvalue intuition.
- You mention **truncated BPTT** and what it sacrifices.
- You connect to why **Transformers replaced RNNs** for most sequence modeling.

## Tells that get you down-leveled

- Treating BPTT as fundamentally different from backprop.
- No mention of memory scaling.
- No knowledge of truncation.
- Recommending vanilla RNNs in 2026 for new sequence-modeling problems.

## Common follow-up

"Why doesn't the transformer have a vanishing gradient problem?"

The L6 answer:

> "Two reasons. First, the gradient path from the loss back to any token's representation goes through residual connections at every layer, with no multiplicative chain along a sequence axis. Second, attention provides a *direct* connection from any output position to any input position in a single layer, so dependencies don't have to be propagated through many sequential steps. The residual + attention combination breaks the multiplicative-Jacobian chain that causes vanishing in RNNs."

---

*Related: [Transformer architecture](/concepts/transformer-architecture/), [FlashAttention](/concepts/flashattention/), [Explain backprop](/questions/explain-backprop/).*
