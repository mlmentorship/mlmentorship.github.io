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
	<p class="visual-title" id="im2col-title">Unfold patches, apply one filter row, then accumulate gradients at shared source pixels</p><div class="visual-argument" data-visual-argument="im2col-patches-become-columns" data-argument-kind="matrix" data-argument-composition="patch-product-and-backward"><svg viewBox="0 0 360 1980" role="img" aria-label="Unfold patches, apply one filter row, then accumulate gradients at shared source pixels. Unfold patches into columns and apply the same filter row to every column. This filter gives 37 and 47. During backward computation, repeated entries send gradients to the original pixel. Shared pixel 2 receives 2 and 1, giving 3 rather than an average. Explicit columns cost extra storage; an implicit kernel can generate the same entries as needed."><title>Unfold patches, apply one filter row, then accumulate gradients at shared source pixels</title><desc>Unfold patches into columns and apply the same filter row to every column. This filter gives 37 and 47. During backward computation, repeated entries send gradients to the original pixel. Shared pixel 2 receives 2 and 1, giving 3 rather than an average. Explicit columns cost extra storage; an implicit kernel can generate the same entries as needed.</desc><text x="20" y="27" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Overlapping patches reuse the same</tspan><tspan x="20" dy="18">pixels</tspan></text><rect x="54" y="102" width="80" height="70" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke)"></rect><text x="94" y="145" style="font-size:25px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="94" dy="0">1</tspan></text><rect x="134" y="102" width="80" height="70" style="fill:var(--viz-focus-bg);stroke:var(--viz-neutral-stroke)"></rect><text x="174" y="145" style="font-size:25px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="174" dy="0">2</tspan></text><rect x="214" y="102" width="80" height="70" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke)"></rect><text x="254" y="145" style="font-size:25px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="254" dy="0">3</tspan></text><rect x="54" y="172" width="80" height="70" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke)"></rect><text x="94" y="215" style="font-size:25px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="94" dy="0">4</tspan></text><rect x="134" y="172" width="80" height="70" style="fill:var(--viz-focus-bg);stroke:var(--viz-neutral-stroke)"></rect><text x="174" y="215" style="font-size:25px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="174" dy="0">5</tspan></text><rect x="214" y="172" width="80" height="70" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke)"></rect><text x="254" y="215" style="font-size:25px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="254" dy="0">6</tspan></text><rect x="51" y="99" width="166" height="146" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:3"></rect><text x="110" y="324" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-input-stroke)"><tspan x="110" dy="0">patch 0</tspan></text><rect data-picture-patch="0" data-picture-pixel="1" x="82" y="351" width="56" height="49" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="110" y="383" style="font-size:22px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="110" dy="0">1</tspan></text><rect data-picture-patch="0" data-picture-pixel="2" x="82" y="400" width="56" height="49" style="fill:var(--viz-focus-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="110" y="432" style="font-size:22px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="110" dy="0">2</tspan></text><rect data-picture-patch="0" data-picture-pixel="4" x="82" y="449" width="56" height="49" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="110" y="481" style="font-size:22px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="110" dy="0">4</tspan></text><rect data-picture-patch="0" data-picture-pixel="5" x="82" y="498" width="56" height="49" style="fill:var(--viz-focus-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="110" y="530" style="font-size:22px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="110" dy="0">5</tspan></text><rect x="131" y="105" width="166" height="146" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:3"></rect><text x="250" y="324" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-output-stroke)"><tspan x="250" dy="0">patch 1</tspan></text><rect data-picture-patch="1" data-picture-pixel="2" x="222" y="351" width="56" height="49" style="fill:var(--viz-focus-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><text x="250" y="383" style="font-size:22px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="250" dy="0">2</tspan></text><rect data-picture-patch="1" data-picture-pixel="3" x="222" y="400" width="56" height="49" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><text x="250" y="432" style="font-size:22px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="250" dy="0">3</tspan></text><rect data-picture-patch="1" data-picture-pixel="5" x="222" y="449" width="56" height="49" style="fill:var(--viz-focus-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><text x="250" y="481" style="font-size:22px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="250" dy="0">5</tspan></text><rect data-picture-patch="1" data-picture-pixel="6" x="222" y="498" width="56" height="49" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><text x="250" y="530" style="font-size:22px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="250" dy="0">6</tspan></text><text x="20" y="598" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">Pixels 2 and 5 belong to both</tspan><tspan x="20" dy="18">columns.</tspan></text><text x="20" y="670" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">One filter row is shared by both</tspan><tspan x="20" dy="18">columns</tspan></text><rect x="20" y="740" width="64" height="48" data-picture-im2col-filter="0" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.5"></rect><text x="52" y="771" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="52" dy="0">1</tspan></text><rect x="100" y="740" width="64" height="48" data-picture-im2col-filter="1" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.5"></rect><text x="132" y="771" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="132" dy="0">2</tspan></text><rect x="180" y="740" width="64" height="48" data-picture-im2col-filter="2" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.5"></rect><text x="212" y="771" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="212" dy="0">3</tspan></text><rect x="260" y="740" width="64" height="48" data-picture-im2col-filter="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.5"></rect><text x="292" y="771" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="292" dy="0">4</tspan></text><text x="20" y="853" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Take the matching products; add</tspan><tspan x="20" dy="18">within each column</tspan></text><path d="M90 896L90 933M86 925L90 933L94 925" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:2;stroke-linejoin:round"></path><circle cx="90" cy="970" r="27" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="90" y="976" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="90" dy="0">37</tspan></text><text x="90" y="1040" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="90" dy="0">Patch 0</tspan></text><path d="M270 896L270 933M266 925L270 933L274 925" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:2;stroke-linejoin:round"></path><circle cx="270" cy="970" r="27" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="270" y="976" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="270" dy="0">47</tspan></text><text x="270" y="1040" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="270" dy="0">Patch 1</tspan></text><text x="20" y="1135" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Backward: each response receives one</tspan><tspan x="20" dy="18">incoming unit</tspan></text><text x="20" y="1225" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">The two copies of source pixel 2 send</tspan><tspan x="20" dy="18">gradients back</tspan></text><text x="90" y="1300" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-input-stroke)"><tspan x="90" dy="0">Left column</tspan></text><circle cx="90" cy="1355" r="24" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="90" y="1361" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="90" dy="0">2</tspan></text><path d="M90 1393L165 1456M156.30159273599 1453.9172827677721L165 1456L161.4471294273762 1447.7916438494553" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2;stroke-linejoin:round"></path><text x="270" y="1300" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-output-stroke)"><tspan x="270" dy="0">Right column</tspan></text><circle cx="270" cy="1355" r="24" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="270" y="1361" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="270" dy="0">1</tspan></text><path d="M270 1393L195 1456M198.5528705726238 1447.7916438494553L195 1456L203.69840726401 1453.9172827677721" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:2;stroke-linejoin:round"></path><circle cx="180" cy="1490" r="27" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="180" y="1496" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="180" dy="0">3</tspan></text><text x="20" y="1560" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Shared gradient: 3. Sum</tspan><tspan x="20" dy="18">contributions; do not average.</tspan></text><text x="20" y="1655" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Accumulated image gradients</tspan></text><rect x="54" y="1690" width="80" height="64" data-picture-im2col-image-gradient="0,0" data-picture-im2col-gradient-value="1" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.5"></rect><text x="94" y="1730" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="94" dy="0">1</tspan></text><rect x="134" y="1690" width="80" height="64" data-picture-im2col-image-gradient="0,1" data-picture-im2col-gradient-value="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><text x="174" y="1730" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="174" dy="0">3</tspan></text><rect x="214" y="1690" width="80" height="64" data-picture-im2col-image-gradient="0,2" data-picture-im2col-gradient-value="2" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.5"></rect><text x="254" y="1730" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="254" dy="0">2</tspan></text><rect x="54" y="1754" width="80" height="64" data-picture-im2col-image-gradient="1,0" data-picture-im2col-gradient-value="3" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.5"></rect><text x="94" y="1794" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="94" dy="0">3</tspan></text><rect x="134" y="1754" width="80" height="64" data-picture-im2col-image-gradient="1,1" data-picture-im2col-gradient-value="7" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><text x="174" y="1794" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="174" dy="0">7</tspan></text><rect x="214" y="1754" width="80" height="64" data-picture-im2col-image-gradient="1,2" data-picture-im2col-gradient-value="4" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.5"></rect><text x="254" y="1794" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="254" dy="0">4</tspan></text><text x="20" y="1885" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">Implicit kernels avoid storing all</tspan><tspan x="20" dy="18">columns. Overlap still accumulates.</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
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
	<div class="visual-derivation-caption">each patch becomes one column in output order. The center value 5 occurs in all four columns. Filter scores become four negative fours; forward only reshapes them. Explicit lowering stores the repeats. Original example checked against <a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.Unfold.html">PyTorch Unfold</a> and <a href="https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html">NVIDIA's convolution guide</a>.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="im2col-patches-become-columns"><strong>Check:</strong> Which input values produce the top-right output? Where do they appear in X_col?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="im2col-patches-become-columns">Unfold patches into columns and apply the same filter row to every column. This filter gives 37 and 47. During backward computation, repeated entries send gradients to the original pixel. Shared pixel 2 receives 2 and 1, giving 3 rather than an average. Explicit columns cost extra storage; an implicit kernel can generate the same entries as needed. <strong>Why do the middle image pixels appear in both output columns?</strong></span></figcaption>
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
