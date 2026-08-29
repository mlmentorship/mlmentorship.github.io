---
title: "GPU memory hierarchy: HBM, SRAM, and roofline reasoning"
description: "Decide whether an accelerator operation is limited by compute or by data movement across HBM, caches, and on-chip memory."
date: "2026-04-27"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A GPU has small, fast on-chip memory and larger, slower HBM. Large matrix multiplications are often compute-bound. Decode, small matrix multiplications, and many elementwise operations are often memory-bound. The operation's arithmetic intensity determines which limit applies.

Counting only multiply-adds can give the wrong answer. A better first-order model is:

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
| **Register file** | hundreds of KB per SM | highest on-chip | very low | per-thread values |
| **SRAM (shared memory / L1)** | tens to hundreds of KB per SM | very high | low | tiles for matrix operations |
| **L2 cache** | tens to hundreds of MB per device | below local SRAM | medium | shared across SMs |
| **HBM** | tens to hundreds of GB | below on-chip memory | higher | weights, activations, KV cache |
| **PCIe or accelerator link** | external | below local HBM in many systems | µs scale | host or peer transfer |

These values are illustrative. Capacity, bandwidth, precision, sparsity mode, and power settings vary by accelerator and SKU. Use measured values from the target machine for a deployment plan.

## Arithmetic intensity

For a kernel doing $F$ FLOPs and moving $B$ bytes between HBM and SRAM:

$$
\text{arithmetic intensity} = \frac{F}{B} \quad (\text{FLOPs per byte})
$$

A kernel is **compute-bound** when its intensity exceeds the GPU's peak FLOPs/byte ratio (the "ridge" in a roofline plot). Otherwise it's **memory-bound**.

An accelerator with 989 BF16 TFLOPs/s and 3 TB/s of HBM bandwidth has a ridge near 330 FLOPs/byte. For comparison:

- Large square matmul ($n\times n$ by $n\times n$): up to about $n/2$ FLOPs/byte when counting two BF16 inputs and ignoring output traffic. Counting output reads and writes lowers the intensity.
- Attention kernel (without FlashAttention): ~O(d) FLOPs/byte → memory-bound at common d=64–128.
- Single-token decode with BF16 weights: about 1 FLOP per weight byte at batch 1, before other traffic. One multiply-add uses a two-byte weight. This is severely memory-bound.

## Implications for LLMs

- **Training**: large matrix multiplications can be compute-bound, while small batches, elementwise work, and communication can expose other limits.
- **Decode**: often weight-bandwidth-bound at small batch sizes. Batching amortizes weight reads until matrix compute or KV-cache traffic becomes the next limit.
- **KV cache**: bandwidth-dominated read at every decode step. Cache size growth is a serving-throughput issue.

## Common pitfalls

- **Quoting FLOPs as a single-number cost.** Throughput on memory-bound kernels is dictated by bytes, not FLOPs.
- **Assuming peak compute predicts every speedup.** A new accelerator can increase matrix throughput faster than memory bandwidth. Memory-bound work then sees a smaller gain.
- **Ignoring on-chip memory when designing kernels.** FlashAttention tile sizes are constrained by shared memory, registers, and the compiled kernel, not only HBM size.

## Related

- [Accelerator network topology](/concepts/accelerator-network-topology/). Extend the hierarchy across devices and nodes.
- [Transformer compute and memory accounting](/concepts/transformer-compute-memory-accounting/). Convert a model configuration into bytes and FLOPs.
- [Profiling distributed ML workloads](/concepts/profiling-distributed-ml-workloads/). Compare a trace with roofline limits.
