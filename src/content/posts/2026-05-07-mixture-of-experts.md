---
title: "Mixture of Experts (MoE)"
description: "Replace one large feed-forward block with N smaller experts and a router that activates only k of them per token. Trades parameter count for compute."
date: "2026-05-07"
draft: false
tags: ["reference"]
category: "reference"
---

## One-line definition

A Mixture-of-Experts layer replaces a single dense feed-forward network with $N$ parallel "expert" FFNs and a router that sends each token to the top-$k$ experts (typically $k=1$ or $k=2$). Total parameters scale with $N$; per-token compute scales with $k$.

## Why it matters

The defining tradeoff: a $k$-of-$N$ MoE has roughly the same per-token FLOPs as a dense model with $k/N$ of the parameters, but the *capacity* of all $N$ experts. Mixtral 8×7B [(Jiang et al., 2023)](https://arxiv.org/abs/2401.04088) has 47B total parameters and uses ~13B per token. Quality close to a 70B dense model at ~5× lower inference compute.

MoE is the dominant strategy for scaling parameter count beyond what dense training and inference can afford. GPT-4, Mixtral, DeepSeek-V3, Grok 1, and many other 2024-2026 frontier models are MoE.

## The mechanism

For each transformer block, replace the single FFN with:

1. **Router**: a small linear layer $W_r \in \mathbb{R}^{d \times N}$. For each token, compute logits $W_r x$, take top-$k$ experts, and softmax-normalize over those $k$.
2. **Experts**: $N$ independent FFN blocks $\{E_1, \dots, E_N\}$, each the same shape as the dense FFN it replaces.
3. **Combine**: output is $\sum_{i \in \text{top-}k} g_i \cdot E_i(x)$ where $g_i$ is the router weight.

Attention layers are typically *not* MoE (shared across all tokens).

## Load balancing

The router will collapse to a few favorite experts unless penalized. Standard fix: an **auxiliary load-balancing loss** that penalizes uneven expert usage within a batch. Alternatives include expert-choice routing (each expert picks its top tokens, [Zhou et al., 2022](https://arxiv.org/abs/2202.09368)) and noise injection.

If experts go unused for many steps their parameters drift; a few production systems "reset" dead experts.

## Capacity and expert parallelism

With $N=8$ experts, an intuitive serving setup is **expert parallelism**: each GPU holds one expert. Tokens are routed via all-to-all communication. This works but introduces:

- **Communication overhead**: all-to-all is bandwidth-bound and stalls when the routing is imbalanced.
- **Capacity factor**: each expert has a max number of tokens it can process per batch; overflow tokens are dropped (or sent to a fallback path). Capacity factor of 1.25 is common.

## Tradeoffs vs. dense

- **Memory**: MoE needs $N \times$ FFN parameters in HBM even though only $k$ are used per token. Inference VRAM is dominated by all experts being loaded, not just active ones.
- **Throughput**: MoE wins per-FLOP. Throughput per VRAM-byte is worse than dense.
- **Quality at fixed FLOPs**: MoE generally beats dense at matched per-token FLOPs.
- **Fine-tuning**: MoE is harder to fine-tune cleanly; routing can drift, and small datasets exacerbate load imbalance.

## Common pitfalls

- **Quoting "total parameters" as if they were active.** A 47B MoE with 13B active is a 13B-FLOPs model with 47B-VRAM cost.
- **Ignoring the routing loss.** Without it, training collapses to using a few experts.
- **Assuming MoE always wins.** At small scale or with limited compute for routing experimentation, dense is simpler and competitive.
