---
title: "Floating-point formats: FP32, FP16, BF16, FP8, TF32"
description: "How modern accelerators trade precision for speed. The bit layouts of every numeric format that appears in deep learning."
date: "2026-02-01"
draft: false
tags: ["concepts"]
category: "concepts"
---

## One-line definition

A floating-point number is encoded as $(-1)^s \cdot 1.m \cdot 2^{e - \text{bias}}$. One sign bit, several exponent bits, several mantissa bits. The choice of how many bits go where determines representable range, precision, and storage / bandwidth cost.

## Why it matters

Modern training and inference use a mix of formats: FP32 master weights, BF16 activations, FP8 matmuls, INT8 KV cache. Each format trades range vs. precision vs. throughput. Knowing the bit layouts saves hours of debugging numerical issues.

## The standard layouts

| Format | Bits | Sign | Exponent | Mantissa | Range | Smallest normal |
|--------|------|------|----------|----------|-------|----------------|
| **FP32** (single) | 32 | 1 | 8 | 23 | $\pm 3.4 \times 10^{38}$ | $\sim 1.2 \times 10^{-38}$ |
| **FP16** (half) | 16 | 1 | 5 | 10 | $\pm 6.5 \times 10^{4}$ | $\sim 6.1 \times 10^{-5}$ |
| **BF16** (brain float) | 16 | 1 | 8 | 7 | $\pm 3.4 \times 10^{38}$ | $\sim 1.2 \times 10^{-38}$ |
| **TF32** (TensorFloat) | 19 (in 32-bit reg) | 1 | 8 | 10 | $\pm 3.4 \times 10^{38}$ | $\sim 1.2 \times 10^{-38}$ |
| **FP8 E4M3** | 8 | 1 | 4 | 3 | $\pm 448$ | $\sim 0.0019$ |
| **FP8 E5M2** | 8 | 1 | 5 | 2 | $\pm 5.7 \times 10^{4}$ | $\sim 6.1 \times 10^{-5}$ |
| **INT8** | 8 | 1 |. | 7 (integer) | $-128$ to $+127$ | step = 1 |
| **FP64** (double) | 64 | 1 | 11 | 52 | $\pm 1.8 \times 10^{308}$ | $\sim 2.2 \times 10^{-308}$ |

## FP16 vs. BF16: range vs. precision

Same total bits (16) but different distributions:

- **FP16** (5 exponent + 10 mantissa): more precision, less range. Underflows easily on small gradients (e.g., during late-training small updates). Requires loss scaling (multiply loss by ~$2^{15}$ before backward, divide gradients afterward).
- **BF16** (8 exponent + 7 mantissa): same range as FP32, less precision. **Never underflows** in practice. No loss scaling needed.

In 2026, **BF16 has largely won for training**. Both Nvidia (Ampere+) and Google TPUs support it natively, and the "no loss scaling" simplicity is decisive.

## TF32: a hybrid for matmul accumulation

TF32 is **internal to Nvidia tensor cores**: matmul inputs are FP32, internally truncated to 19 bits (8 exponent + 10 mantissa) for the multiply, accumulated to FP32. Roughly 8× faster than full FP32 matmul with negligible accuracy impact. Default on Ampere+ for many PyTorch ops; controlled by `torch.backends.cuda.matmul.allow_tf32`.

## FP8: the new training format

H100, MI300X, and B100 support FP8 with two variants:

- **E4M3** (4 exponent, 3 mantissa): higher precision, smaller range. Used for forward pass and weights.
- **E5M2** (5 exponent, 2 mantissa): higher range, lower precision. Used for gradients (which can be large).

Mixed FP8 training (different formats for different tensors, plus dynamic per-tensor scaling) gives ~2× speedup over BF16 with minimal quality loss for many workloads.

Standardized by Open Compute Project; supported by NVIDIA Transformer Engine, AMD ROCm, JAX.

## INT8: the inference format

For inference (mostly), quantize weights and activations to 8-bit integers. **Asymmetric**: `int_value = round(float_value / scale + zero_point)`, where scale and zero_point are per-tensor or per-channel. **Symmetric** (zero_point = 0): more efficient on hardware, slightly less expressive.

**Smooth, INT8 weight + activation**: ~4× memory savings vs. BF16, ~2× throughput on tensor cores. Quality loss < 1% for most large models with proper calibration.

## INT4 / NF4 / GPTQ: aggressive inference quantization

For very large LLMs that exceed VRAM, weights can be quantized to 4 bits:

- **INT4**: 16 levels per tensor or per group.
- **NF4** (NormalFloat 4): non-uniform quantization optimized for normally-distributed weights (used in QLoRA).
- **GPTQ, AWQ**: post-training quantization with calibration data, minimizes layer-wise reconstruction error.

Typical quality loss: 0.5–2% on benchmarks; serving cost drops 2–4× on memory.

## Standard usage in 2026

| Task | Format |
|------|--------|
| Pretraining, large model | BF16 weights + activations, FP32 master copy + optimizer state |
| Pretraining, frontier scale | FP8 mixed (forward + backward), BF16 master, FP32 optimizer |
| Fine-tuning | BF16 with LoRA in FP32 |
| Inference, large LLM | INT8 weights / KV; BF16 activations |
| Inference, edge / mobile | INT8 throughout |
| Inference, extreme-scale model on consumer GPU | INT4 weights (GPTQ / AWQ / NF4) |

## Common pitfalls

- **Confusing FP16 and BF16.** Same bit count, very different range. FP16 needs loss scaling.
- **Storing optimizer state in BF16.** Adam moments need full FP32 precision; storing them in BF16 destroys training. Master weights and optimizer state stay FP32.
- **Forgetting per-channel quantization.** Per-tensor INT8 is cheaper but loses ~5% accuracy on transformer FFN; per-channel is the production default.
- **Not testing FP8 / INT4 on your specific model.** Quality drops are workload-dependent; always evaluate.
- **Reading peak TFLOPS without specifying format.** H100 BF16 ≠ FP8 ≠ FP32. Marketing numbers usually quote FP8 or sparse FP8.

## Related

- [Mixed precision training](/concepts/mixed-precision-training/). Practical training in FP16/BF16.
- [Quantization](/concepts/quantization/). Inference quantization in depth.
- [GPU memory hierarchy](/concepts/gpu-memory-hierarchy/). Why bandwidth matters.
