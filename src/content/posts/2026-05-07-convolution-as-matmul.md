---
title: "Convolution as matrix multiplication (im2col)"
description: "Lower convolution patches into matrix columns, verify the filter product, and distinguish explicit storage, implicit kernels and backward overlap accumulation."
date: "2026-05-07"
draft: false
tags: ["concepts"]
category: "concepts"
---

Convolution lowering puts each receptive field in a matrix column so flattened filters can score all output positions together.

## Summary

How do four overlapping patches become the four output values, and what must be stored?
Use the input rows $[1,2,3]$, $[4,5,6]$, $[7,8,9]$.
The filter is $[[1,0],[0,-1]]$, with stride one, no padding and no bias.
Use deep-learning cross-correlation: do not flip the filter.

## The mechanism

Flatten channel, kernel-row and kernel-column in that order; visit output positions row by row.

| Output location | Patch column | Dot product with $[1,0,0,-1]$ |
| --- | --- | --- |
| Top-left | $[1,2,4,5]$ | $1-5=-4$ |
| Top-right | $[2,3,5,6]$ | $2-6=-4$ |
| Bottom-left | $[4,5,7,8]$ | $4-8=-4$ |
| Bottom-right | $[5,6,8,9]$ | $5-9=-4$ |

The four column scores reshape to a two-by-two output; no input reconstruction happens during that reshape.
The patch matrix stores sixteen values from nine source values, ratio $16/9$.

<!-- visual:im2col-patches-become-columns -->
<figure class="learning-figure plot-panel" aria-labelledby="im2col-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="im2col-title">Each local patch becomes one column in the matrix product</p><div class="visual-argument" data-visual-argument="im2col-patches-become-columns" data-argument-kind="matrix"><p class="visual-argument-premise">Each patch becomes one matrix column</p><svg viewBox="0 0 360 248" role="img" aria-label="Each patch becomes one matrix column. Two adjacent 2x2 patches become these columns. Input rows are (1,2,3) and (4,5,6); shared pixels appear in both patches."><title>Each patch becomes one matrix column</title><desc>Two adjacent 2x2 patches become these columns. Input rows are (1,2,3) and (4,5,6); shared pixels appear in both patches.</desc><text x="128.5" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="128.5" dy="0">patch 0</tspan></text><text x="269.5" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="269.5" dy="0">patch 1</tspan></text><text x="46" y="54" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="46" dy="0">top</tspan><tspan x="46" dy="18">L</tspan></text><rect data-proof-cell="0,0" x="58" y="36" width="138" height="47" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke)"></rect><text x="127" y="65" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="127" dy="0">1</tspan></text><rect data-proof-cell="0,1" x="199" y="36" width="138" height="47" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke)"></rect><text x="268" y="65" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="268" dy="0">2</tspan></text><text x="46" y="104" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="46" dy="0">top</tspan><tspan x="46" dy="18">R</tspan></text><rect data-proof-cell="1,0" x="58" y="86" width="138" height="47" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke)"></rect><text x="127" y="115" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="127" dy="0">2</tspan></text><rect data-proof-cell="1,1" x="199" y="86" width="138" height="47" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke)"></rect><text x="268" y="115" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="268" dy="0">3</tspan></text><text x="46" y="154" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="46" dy="0">low</tspan><tspan x="46" dy="18">L</tspan></text><rect data-proof-cell="2,0" x="58" y="136" width="138" height="47" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke)"></rect><text x="127" y="165" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="127" dy="0">4</tspan></text><rect data-proof-cell="2,1" x="199" y="136" width="138" height="47" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke)"></rect><text x="268" y="165" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="268" dy="0">5</tspan></text><text x="46" y="204" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="46" dy="0">low</tspan><tspan x="46" dy="18">R</tspan></text><rect data-proof-cell="3,0" x="58" y="186" width="138" height="47" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke)"></rect><text x="127" y="215" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="127" dy="0">5</tspan></text><rect data-proof-cell="3,1" x="199" y="186" width="138" height="47" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke)"></rect><text x="268" y="215" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="268" dy="0">6</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
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
		<text class="viz-axis-label" x="18" y="211">2 - PATCHES TO COLUMNS</text>
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
		<text class="viz-axis-label" x="18" y="424">3 - ONE GEMM</text>
		<rect class="viz-node viz-node--input" x="18" y="439" width="132" height="35" rx="3"></rect>
		<text class="viz-callout" x="84" y="461" text-anchor="middle">W_flat = [1, 0, 0, -1]</text>
		<text class="viz-callout" x="160" y="461">x X_col =</text>
		<rect class="viz-node viz-node--output" x="227" y="439" width="115" height="35" rx="3"></rect>
		<text class="viz-callout" x="284.5" y="461" text-anchor="middle">[-4, -4, -4, -4]</text>
		<text class="viz-label" x="180" y="492" text-anchor="middle">reshape four outputs to 2 x 2</text>
	</svg>
	<div class="visual-derivation-caption">each patch becomes one column in output order. The center value 5 occurs in all four columns. Filter scores become four negative fours; forward only reshapes them. Explicit lowering stores the repeats. Original example checked against <a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.Unfold.html">PyTorch Unfold</a> and <a href="https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html">NVIDIA's convolution guide</a>.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="im2col-patches-become-columns"><strong>Check:</strong> Which input values produce the top-right output? Where do they appear in X_col?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="im2col-patches-become-columns">Two adjacent 2x2 patches become these columns. Input rows are (1,2,3) and (4,5,6); shared pixels appear in both patches.</span></figcaption>
</figure>

For one image with $C_{in}$ channels, let $D=C_{in}k_Hk_W$ and $P=H_{out}W_{out}$.
Unfold gives $U$ of shape $D\times P$; flattened filters $F$ have shape $C_{out}\times D$.
Compute $FU$ plus per-output-channel bias, then reshape to $C_{out}\times H_{out}\times W_{out}$.
For batch size $B$, explicit columns store $BDP$ entries.

Dense work is $BC_{out}DP$ multiply-accumulates, or approximately twice that many FLOPs under the two-FLOP convention.
The tiny example uses sixteen MACs even though its particular filter contains zeros.
Materialized storage and whole-operator memory traffic are different quantities.

## Change storage or channel structure

### Avoid full materialization

**New task:** preserve these outputs without storing every patch simultaneously.

**Picture and change:** process one four-value column at a time and emit its score.

**Check:** patch scratch falls from sixteen to four scalar values here, while all four outputs remain minus four.
Input, output and filter storage remain; repeated reads and indexing still cost work.

**Use when:** unfold storage or traffic is binding; benchmark an appropriate implicit/tiled kernel under the same operator.

### Separate depthwise and pointwise work

**New task:** apply per-channel spatial filters, then mix channels.

**Picture and change:** take two input channels, four output positions and two-by-two depthwise filters.
Depthwise work is $2\cdot4\cdot4=32$ MACs.
Mixing those channels into three outputs with a pointwise filter adds $3\cdot2\cdot4=24$ MACs.

**Check:** total work is 56 MACs versus 96 for a dense two-by-two convolution with three output channels.
These are different parameterizations, not generally interchangeable learned functions.

**Use when:** the architecture explicitly separates spatial filtering and channel mixing.

## Reconstruct backward accumulation

Why is folding patches not an inverse reshape?

<details>
<summary>Reasoned answer</summary>

Overlapping columns contribute to the same input coordinate, so backward sums their contributions.
Folding ones gives center coverage four in this example.
Actual input gradients also include filter coefficients; with this filter and unit output gradients, the center derivative is zero.

</details>

<!-- learning:core-end -->

<div data-learning-extension>

## Variants

For each axis, output length is $\lfloor(H+2p-d(k-1)-1)/s\rfloor+1$ for valid parameters.
Here $p$ is symmetric padding, $d$ dilation and $s$ stride; apply the formula independently to height and width.
Groups restrict each filter to $C_{in}/g$ input channels, changing the MAC count accordingly.
Depthwise is a grouped case; pointwise performs dense channel mixing at each location.

Implicit GEMM indexes virtual patch tiles instead of storing the complete unfold.
Direct kernels, Winograd and FFT implementations use different layouts or transforms.
FFT requires the correct padding/correlation convention; Winograd changes numerical behavior.
Shape, hardware, precision and workspace determine useful choices, not a universal framework implementation rule.

## Interview focus

Derive complete input, filter, column and output shapes, then state flatten order, bias and resource convention.
Algebraic lowering is not evidence that the actual kernel allocated the patch matrix.

## Common pitfalls

Flipping the example's filter changes all outputs to positive four; convolution versus cross-correlation can matter.
Explicit storage expansion depends on shape, stride, padding and dilation, not a constant nine-fold multiplier.
Forward output reshape and backward overlap-add solve different tasks.
Profile depthwise/pointwise work and their fusion boundary separately.

## Related

- [CNN architecture](/concepts/cnn-architecture/).
- [FlashAttention](/concepts/flashattention/). Same idea: rearrange for hardware.

</div>
