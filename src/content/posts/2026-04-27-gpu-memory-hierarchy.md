---
title: "GPU memory hierarchy: HBM, SRAM, and why I/O matters more than FLOPs"
description: "Modern GPUs are memory-bound for almost everything except big matmuls. Understanding HBM vs. SRAM bandwidth is the prerequisite for FlashAttention, KV-cache reasoning, and inference cost models."
date: "2026-04-27"
draft: false
tags: ["reference"]
category: "reference"
---

## One-line definition

A GPU has a small, fast on-chip SRAM and a large, slow off-chip HBM. Most LLM operations are bandwidth-bound between HBM and SRAM, not compute-bound. So the binding constraint is bytes moved, not FLOPs computed.

## Why it matters

Naive cost models count multiply-adds. They give the wrong answer for almost every modern LLM kernel. The correct first-order model is:

$$
\text{time} \approx \max\!\left(\frac{\text{FLOPs}}{\text{tensor TFLOPs/s}}, \frac{\text{bytes moved}}{\text{HBM TB/s}}\right)
$$

For most LLM ops at typical batch sizes, the second term dominates. This single fact explains:

- Why FlashAttention is faster despite doing the same FLOPs (reduces HBM traffic).
- Why decoding is slow with batch 1 even though the model "fits" (memory-bound).
- Why batching helps decoding so much (amortizes weight reads across many tokens).
- Why low-precision data types help even when the matmul is already fast (less HBM bandwidth).

## The hierarchy

| Tier | Capacity | Bandwidth | Latency | What lives here |
|------|----------|-----------|---------|----------------|
| **Registers** | KB per thread |. | <1 ns | per-thread scalars |
| **SRAM (shared memory / L1)** | ~100–200 KB per SM | ~10 TB/s aggregate | ~10 ns | tiles for matmul |
| **L2 cache** | ~50 MB (H100) | ~5 TB/s | ~100 ns | shared across SMs |
| **HBM** | 40–80 GB | 1.5–3 TB/s | ~500 ns | weights, activations, KV cache |
| **PCIe / NVLink to host or peer** |. | 50–900 GB/s | µs | inter-GPU |

H100 has 80 GB HBM3 at ~3 TB/s and ~989 BF16 TFLOPs. A100 is 40 or 80 GB HBM2e at ~2 TB/s and ~312 BF16 TFLOPs.

## Arithmetic intensity

For a kernel doing $F$ FLOPs and moving $B$ bytes between HBM and SRAM:

$$
\text{arithmetic intensity} = \frac{F}{B} \quad (\text{FLOPs per byte})
$$

A kernel is **compute-bound** when its intensity exceeds the GPU's peak FLOPs/byte ratio (the "ridge" in a roofline plot). Otherwise it's **memory-bound**.

H100 ridge: ~989 / 3 ≈ ~330 FLOPs/byte (BF16). For comparison:

- Large square matmul (n×n × n×n): ~n/2 FLOPs/byte → compute-bound for n ≥ ~660.
- Attention kernel (without FlashAttention): ~O(d) FLOPs/byte → memory-bound at common d=64–128.
- Single-token decode forward pass: ~2 FLOPs per weight byte (one multiply-add per weight) → severely memory-bound.

## Implications for LLMs

- **Training**: dominated by big matmuls and FlashAttention; mostly compute-bound at typical sequence lengths.
- **Decode**: weight bandwidth-bound. Batch size 1 wastes the GPU; batch size 32 amortizes weight reads across 32 outputs.
- **KV cache**: bandwidth-dominated read at every decode step. Cache size growth is a serving-throughput issue.

## Common pitfalls

- **Quoting FLOPs as a single-number cost.** Throughput on memory-bound kernels is dictated by bytes, not FLOPs.
- **Assuming faster GPUs scale equally.** H100 vs A100: ~3× FLOPs but ~1.5× HBM bandwidth. Memory-bound workloads see only the latter.
- **Ignoring SRAM size when designing kernels.** FlashAttention's tile sizes are determined by SRAM capacity per SM (~100 KB), not HBM size.
