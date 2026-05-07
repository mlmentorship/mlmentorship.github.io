---
title: "Sequence packing with block-diagonal masks"
description: "Concatenate multiple short examples into one fixed-length sequence to eliminate padding waste. The single largest throughput win for training on skewed-length corpora."
date: "2026-01-25"
draft: false
tags: ["reference"]
category: "reference"
---

## One-line definition

Sequence packing concatenates multiple training examples back-to-back into a single fixed-length sequence and uses a block-diagonal attention mask so each example only attends within itself, eliminating the FLOPs and memory wasted on padding tokens.

## Why it matters

Most NLP corpora have heavily skewed length distributions: many short examples, few long ones. With naive padding to the longest example in the batch, the wasted-token ratio is

$$
1 - \frac{\bar{\ell}}{\ell_{\max}}
$$

For C4-like web text this is often 50–80%. Padded positions cost full FLOPs and memory but contribute nothing to the loss. Sequence packing recovers nearly all of that throughput.

## The mechanism

1. Pick a fixed packed length $L$ (e.g., 8192).
2. Concatenate examples $e_1, e_2, \dots$ until adding the next would exceed $L$. Record the boundaries (cumulative sequence lengths, often called `cu_seqlens`).
3. Build a **block-diagonal attention mask**: token $i$ in example $e_a$ cannot attend to any token in $e_b \neq e_a$.
4. Compute attention with a kernel that respects `cu_seqlens` (FlashAttention-2 supports this natively via the `varlen` API).
5. Apply the loss only on response tokens within each example (mask the boundaries and any prompt tokens for SFT).

Position IDs reset at each example boundary so position 0 is the start of each packed example.

## Numbers

For pretraining on web text packed at $L = 8192$:

- Wasted tokens drop from ~50% (naive batching) to <2% (just the slack at the end of the packed sequence).
- Throughput per GPU roughly doubles.
- Quality is unchanged when the mask is correct.

Most modern training stacks pack by default. Llama, Mistral, Qwen, and major SFT toolkits (axolotl, TRL) all support it.

## Common implementation pitfalls

- **Forgetting to reset positions.** If position IDs continue across boundaries, attention learns to treat packed boundaries as long-range dependencies and quality drops.
- **Wrong loss masking.** Loss must not flow from one example to another; mask boundaries explicitly.
- **Mixing prompt and response in SFT without masking.** For SFT, mask out prompt tokens from the loss within each packed example.
- **Using a kernel that doesn't support varlen.** Without FlashAttention-2 varlen (or equivalent), the block-diagonal mask materializes the full $L \times L$ matrix and you lose the speedup.

## When not to pack

- Very long single examples that fill or exceed $L$ on their own (no concatenation possible; padding is already minimal).
- When examples have inter-document context that should attend across boundaries (rare).
- During inference, where examples come one at a time.

## Related

- [FlashAttention](/reference/flashattention/). The underlying kernel that makes varlen attention efficient.
- [Gradient accumulation](/reference/gradient-accumulation/). Orthogonal way to grow effective batch size.
