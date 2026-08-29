---
title: "Convolution as matrix multiplication (im2col)"
description: "A 2D convolution is a matmul in disguise. Unfold the input into columns, multiply by a flattened filter matrix. The reason CNNs run fast on the same hardware as transformers."
date: "2026-05-07"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

**im2col** rearranges a convolution input so each spatial location's receptive field becomes a column of a matrix. The convolution then reduces to a single matmul: $\text{Conv}(X, W) = W_{\text{flat}} \cdot \text{im2col}(X)$.

Modern hardware (GPUs, TPUs) is optimized for dense matmul. A naive 2D convolution loop is the wrong shape for that hardware: nested loops over spatial positions, channels, and kernel offsets, with poor memory locality. im2col turns the same arithmetic into a single GEMM call that lands on the highly tuned BLAS path.

Every major framework (cuDNN, MKL-DNN, XNNPACK) implements convolution as some variant of this idea. Understanding it explains why CNN inference cost scales like matmul, why grouped convolutions are cheap, and why depthwise-separable convolutions split into two matmuls.

## The mechanism

For input $X \in \mathbb{R}^{C_{in} \times H \times W}$, kernel $W \in \mathbb{R}^{C_{out} \times C_{in} \times k \times k}$, output $Y \in \mathbb{R}^{C_{out} \times H_{out} \times W_{out}}$:

1. **im2col**: for each output position $(i, j)$, extract the $C_{in} \cdot k \cdot k$ values in its receptive field and stack them as a column. The result is a matrix $X_{\text{col}} \in \mathbb{R}^{(C_{in} k^2) \times (H_{out} W_{out})}$.
2. **Flatten kernel**: reshape $W$ to $W_{\text{flat}} \in \mathbb{R}^{C_{out} \times (C_{in} k^2)}$.
3. **GEMM**: $Y_{\text{flat}} = W_{\text{flat}} \cdot X_{\text{col}}$, shape $C_{out} \times (H_{out} W_{out})$.
4. **col2im**: reshape $Y_{\text{flat}}$ back to $C_{out} \times H_{out} \times W_{out}$.

Memory cost: $X_{\text{col}}$ duplicates each input pixel up to $k^2$ times. For a $3 \times 3$ kernel, that is a 9x blowup of the activation tensor.

## Variants

- **Implicit GEMM**: avoid materializing $X_{\text{col}}$ in memory. Compute the matmul tile by tile, indexing back into $X$ on the fly. cuDNN's default for most conv shapes.
- **Winograd**: trade matmul FLOPs for additions via polynomial transforms. Faster for small kernels (e.g. $3 \times 3$) on certain hardware. Lower numerical precision.
- **FFT convolution**: $\mathcal{F}^{-1}(\mathcal{F}(X) \odot \mathcal{F}(W))$. Wins for large kernels (rare in modern CNNs).
- **Depthwise convolution**: each input channel has its own filter, so $W$ is block-diagonal. The matmul splits into $C_{in}$ tiny independent matmuls, much cheaper.

## Interview focus

If asked "how does convolution actually run on a GPU," the expected answer is: it is a matmul. Then walk through the im2col reshape, the GEMM call, and the memory blowup. Bonus points for noting that the flattened kernel ${W_{\text{flat}}}$ has shape $C_{out} \times C_{in} k^2$, so the FLOP count is $C_{out} \cdot C_{in} \cdot k^2 \cdot H_{out} \cdot W_{out}$. The same formula you see in every model card.

## Common pitfalls

- **Forgetting the memory cost**. im2col can be larger than the activation it came from. Implicit GEMM exists for this reason.
- **Conflating convolution with cross-correlation**. Deep learning frameworks implement cross-correlation; the kernel is not flipped. Mathematicians' convolution flips the kernel. Almost never matters in practice.
- **Treating depthwise and pointwise as a single op**. They are two distinct matmuls with very different shapes. Profile separately.

## Related

- [CNN architecture](/concepts/cnn-architecture/).
- [FlashAttention](/concepts/flashattention/). Same idea: rearrange for hardware.
