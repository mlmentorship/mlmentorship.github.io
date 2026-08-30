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

<!-- visual:im2col-patches-become-columns -->
<figure class="learning-figure plot-panel" aria-labelledby="im2col-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="im2col-title">Where do the columns in im2col come from?</p>
	<svg viewBox="0 0 360 500" role="img" aria-labelledby="im2col-svg-title im2col-svg-desc">
		<title id="im2col-svg-title">A convolution unfolded into matrix multiplication</title>
		<desc id="im2col-svg-desc">A three by three single-channel input is covered by four overlapping two by two patches at stride one. Flattening each patch creates one column of a four by four im2col matrix. The flattened filter one, zero, zero, negative one multiplies all four columns at once, producing four negative fours that reshape to a two by two output. Repeated values in overlapping columns show the memory cost of explicit im2col.</desc>
		<defs><marker id="im2col-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path class="viz-arrow-forward" d="M0,0 L7,3.5 L0,7 Z"></path></marker></defs>
		<text class="viz-axis-label" x="18" y="24">1 - SLIDE A 2 x 2 PATCH, STRIDE 1</text>
		<text class="viz-label" x="18" y="43">input X</text>
		<g class="viz-callout" text-anchor="middle">
			<rect class="viz-node viz-node--input" x="18" y="53" width="42" height="42"></rect><text x="39" y="79">1</text>
			<rect class="viz-node viz-node--input" x="60" y="53" width="42" height="42"></rect><text x="81" y="79">2</text>
			<rect class="viz-node" x="102" y="53" width="42" height="42"></rect><text x="123" y="79">3</text>
			<rect class="viz-node viz-node--input" x="18" y="95" width="42" height="42"></rect><text x="39" y="121">4</text>
			<rect class="viz-node viz-node--input" x="60" y="95" width="42" height="42"></rect><text x="81" y="121">5</text>
			<rect class="viz-node" x="102" y="95" width="42" height="42"></rect><text x="123" y="121">6</text>
			<rect class="viz-node" x="18" y="137" width="42" height="42"></rect><text x="39" y="163">7</text>
			<rect class="viz-node" x="60" y="137" width="42" height="42"></rect><text x="81" y="163">8</text>
			<rect class="viz-node" x="102" y="137" width="42" height="42"></rect><text x="123" y="163">9</text>
		</g>
		<rect x="16" y="51" width="88" height="88" rx="3" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:3"></rect>
		<text class="viz-label" x="162" y="61">four overlapping patches:</text>
		<text class="viz-callout" x="162" y="82">TL = [1, 2, 4, 5]</text>
		<text class="viz-callout" x="162" y="104">TR = [2, 3, 5, 6]</text>
		<text class="viz-callout" x="162" y="126">BL = [4, 5, 7, 8]</text>
		<text class="viz-callout" x="162" y="148">BR = [5, 6, 8, 9]</text>
		<text class="viz-label" x="162" y="171">5 occurs in every patch</text>
		<path d="M180 187V211" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#im2col-arrow)"></path>
		<text class="viz-axis-label" x="18" y="211">2 - FLATTEN EACH PATCH INTO ONE COLUMN</text>
		<text class="viz-label" x="18" y="232">X_col shape: 4 patch values x 4 output positions</text>
		<g class="viz-callout" text-anchor="middle">
			<text x="153" y="253">TL</text><text x="195" y="253">TR</text><text x="237" y="253">BL</text><text x="279" y="253">BR</text>
			<rect class="viz-node viz-node--focus" x="132" y="261" width="42" height="34"></rect><text x="153" y="283">1</text>
			<rect class="viz-node" x="174" y="261" width="42" height="34"></rect><text x="195" y="283">2</text>
			<rect class="viz-node" x="216" y="261" width="42" height="34"></rect><text x="237" y="283">4</text>
			<rect class="viz-node" x="258" y="261" width="42" height="34"></rect><text x="279" y="283">5</text>
			<rect class="viz-node" x="132" y="295" width="42" height="34"></rect><text x="153" y="317">2</text>
			<rect class="viz-node" x="174" y="295" width="42" height="34"></rect><text x="195" y="317">3</text>
			<rect class="viz-node" x="216" y="295" width="42" height="34"></rect><text x="237" y="317">5</text>
			<rect class="viz-node" x="258" y="295" width="42" height="34"></rect><text x="279" y="317">6</text>
			<rect class="viz-node" x="132" y="329" width="42" height="34"></rect><text x="153" y="351">4</text>
			<rect class="viz-node" x="174" y="329" width="42" height="34"></rect><text x="195" y="351">5</text>
			<rect class="viz-node" x="216" y="329" width="42" height="34"></rect><text x="237" y="351">7</text>
			<rect class="viz-node" x="258" y="329" width="42" height="34"></rect><text x="279" y="351">8</text>
			<rect class="viz-node" x="132" y="363" width="42" height="34"></rect><text x="153" y="385">5</text>
			<rect class="viz-node" x="174" y="363" width="42" height="34"></rect><text x="195" y="385">6</text>
			<rect class="viz-node" x="216" y="363" width="42" height="34"></rect><text x="237" y="385">8</text>
			<rect class="viz-node" x="258" y="363" width="42" height="34"></rect><text x="279" y="385">9</text>
		</g>
		<text class="viz-label" x="18" y="281">same values,</text>
		<text class="viz-label" x="18" y="298">new layout</text>
		<path d="M180 405V425" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#im2col-arrow)"></path>
		<text class="viz-axis-label" x="18" y="424">3 - ONE GEMM COMPUTES EVERY POSITION</text>
		<rect class="viz-node viz-node--input" x="18" y="439" width="132" height="35" rx="3"></rect>
		<text class="viz-callout" x="84" y="461" text-anchor="middle">W_flat = [1, 0, 0, -1]</text>
		<text class="viz-callout" x="160" y="461">x X_col =</text>
		<rect class="viz-node viz-node--output" x="227" y="439" width="115" height="35" rx="3"></rect>
		<text class="viz-callout" x="284.5" y="461" text-anchor="middle">[-4, -4, -4, -4]</text>
		<text class="viz-label" x="180" y="492" text-anchor="middle">reshape four outputs to 2 x 2</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> scan the four overlapping 2 x 2 patches in output order. Flattening does not change their values: it places each patch in one column, including repeated input values such as 5. The flattened filter then dot-products with all columns in one GEMM. Explicit im2col stores those repeats; implicit GEMM generates the same virtual columns tile by tile. Original example checked against <a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.Unfold.html">PyTorch Unfold</a> and <a href="https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html">NVIDIA's convolution performance guide</a>.</figcaption>
</figure>

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
