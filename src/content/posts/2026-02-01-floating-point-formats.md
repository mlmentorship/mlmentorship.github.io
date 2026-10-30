---
title: "Floating-point formats: FP32, FP16, BF16, FP8, TF32"
description: "Separate exponent range, local rounding, accumulation and storage using checked FP16, BF16 and integer examples."
date: "2026-02-01"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A floating-point format spends bits on magnitude range and local precision.
More exponent bits do not preserve smaller relative updates.

Round $1.003$ to FP16 and BF16 using round-to-nearest-even.
Both use sixteen bits, but FP16 has ten fraction bits and BF16 has seven.
Which format retains more of this change from one?

## FP16 vs. BF16: range vs. precision

For normal values, decode sign $s$, fraction $F$ and unbiased exponent $E$ as $(-1)^s(1+F)2^E$.
With $f$ fraction bits, spacing within $[2^E,2^{E+1})$ is $2^{E-f}$.

<!-- visual:fp16-bf16-bit-budget -->
<figure class="learning-figure" aria-labelledby="fp16-bf16-budget-title">
	<p class="visual-kicker">Same storage, different rounded value</p>
	<p class="visual-title" id="fp16-bf16-budget-title">A broader numeric range leaves fewer bits for nearby distinctions</p><div class="visual-argument" data-visual-argument="fp16-bf16-bit-budget" data-argument-kind="parts"><p class="visual-argument-premise">The same 16 bits cannot maximize range and precision</p><svg viewBox="0 0 360 324" role="img" aria-label="The same 16 bits cannot maximize range and precision. BF16 allocates eight exponent bits and seven fraction bits; FP16 uses five and ten. Both reserve one sign bit."><title>The same 16 bits cannot maximize range and precision</title><desc>BF16 allocates eight exponent bits and seven fraction bits; FP16 uses five and ten. Both reserve one sign bit.</desc><text x="20" y="32" style="font-size:18px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">16 bits total</tspan></text><rect data-proof-part="BF16 exponent bits" data-proof-value="8" x="20" y="55" width="160" height="50" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><rect data-proof-part="BF16 fraction bits" data-proof-value="7" x="180" y="55" width="140" height="50" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-proof-part="sign bit" data-proof-value="1" x="320" y="55" width="20" height="50" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.5"></rect><circle cx="27" cy="143" r="5" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="45" y="147" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="45" dy="0">8: BF16 exponent bits</tspan></text><text x="45" y="170" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="45" dy="0">8 / 16 = 50.0%</tspan></text><circle cx="27" cy="201" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="45" y="205" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="45" dy="0">7: BF16 fraction bits</tspan></text><text x="45" y="228" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--viz-output-stroke)"><tspan x="45" dy="0">7 / 16 = 43.8%</tspan></text><circle cx="27" cy="259" r="5" style="fill:var(--viz-neutral-stroke);stroke:var(--viz-neutral-stroke);stroke-width:1.75"></circle><text x="45" y="263" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="45" dy="0">1: sign bit</tspan></text><text x="45" y="286" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="45" dy="0">1 / 16 = 6.3%</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
	<svg viewBox="0 0 360 144" role="img" aria-labelledby="fp-rounding-bits-title fp-rounding-bits-desc">
		<title id="fp-rounding-bits-title">Actual rounded bit fields for 1.003</title>
		<desc id="fp-rounding-bits-desc">Both rows use sixteen bits. FP16 has sign zero, biased exponent 01111 and fraction 0000000011, encoding 1.0029296875. BF16 has sign zero, biased exponent 01111111 and fraction 0000000, encoding exactly one. Field widths are proportional to their bit counts.</desc>
		<text class="viz-callout" x="12" y="18">FP16: 1.0029296875</text>
		<rect class="viz-node viz-node--input" x="12" y="28" width="20.75" height="28"></rect>
		<rect class="viz-node viz-node--focus" x="32.75" y="28" width="103.75" height="28"></rect>
		<rect class="viz-node viz-node--output" x="136.5" y="28" width="207.5" height="28"></rect>
		<text class="viz-node-label" x="22.375" y="47">0</text>
		<text class="viz-node-label" x="84.625" y="47">01111</text>
		<text class="viz-node-label" x="240.25" y="47">0000000011</text>
		<text class="viz-callout" x="12" y="82">BF16: 1</text>
		<rect class="viz-node viz-node--input" x="12" y="92" width="20.75" height="28"></rect>
		<rect class="viz-node viz-node--focus" x="32.75" y="92" width="166" height="28"></rect>
		<rect class="viz-node viz-node--output" x="198.75" y="92" width="145.25" height="28"></rect>
		<text class="viz-node-label" x="22.375" y="111">0</text>
		<text class="viz-node-label" x="115.75" y="111">01111111</text>
		<text class="viz-node-label" x="271.375" y="111">0000000</text>
		<text class="viz-label" x="180" y="139" text-anchor="middle">Fields from left: sign, exponent, fraction</text>
	</svg>
	<div class="visual-grid--two" role="group" aria-label="Side-by-side comparison of how FP16 and BF16 divide the same sixteen-bit budget and the numerical consequence of each division">
		<section class="visual-panel" aria-labelledby="fp16-budget-title">
			<h4 id="fp16-budget-title">FP16: 1 sign + 5 exponent + 10 fraction</h4>
			<p>More fraction bits make nearby values finer; fewer exponent bits narrow the normal range.</p>
			<table class="cm-grid" aria-label="FP16 bit allocation and numerical consequences">
				<thead><tr><th scope="col">Field</th><th scope="col">Bits</th><th scope="col">What those bits buy</th></tr></thead>
				<tbody>
					<tr><th scope="row">Exponent</th><td><strong>5</strong></td><td>normal magnitudes ≈ 6.10 × 10<sup>−5</sup> to 6.55 × 10<sup>4</sup></td></tr>
					<tr><th scope="row">Fraction</th><td class="cm-selected"><strong>10</strong></td><td>next value after 1 is 1 + 2<sup>−10</sup> ≈ 1.00098</td></tr>
				</tbody>
			</table>
			<p class="cm-equation">Failure mode: values below the narrow range become subnormal, then zero.</p>
		</section>
		<section class="visual-panel" aria-labelledby="bf16-budget-title">
			<h4 id="bf16-budget-title">BF16: 1 sign + 8 exponent + 7 fraction</h4>
			<p>More exponent bits preserve FP32-like range; fewer fraction bits make nearby values coarser.</p>
			<table class="cm-grid" aria-label="BF16 bit allocation and numerical consequences">
				<thead><tr><th scope="col">Field</th><th scope="col">Bits</th><th scope="col">What those bits buy</th></tr></thead>
				<tbody>
					<tr><th scope="row">Exponent</th><td class="cm-selected"><strong>8</strong></td><td>normal magnitudes ≈ 1.18 × 10<sup>−38</sup> to 3.39 × 10<sup>38</sup></td></tr>
					<tr><th scope="row">Fraction</th><td><strong>7</strong></td><td>next value after 1 is 1 + 2<sup>−7</sup> ≈ 1.00781</td></tr>
				</tbody>
			</table>
			<p class="cm-equation">Failure mode: updates can round away because adjacent values are farther apart.</p>
		</section>
	</div>
	<div class="visual-derivation-caption"><strong>Read it this way:</strong> each row has the same width. BF16 allocates more bits to the exponent, leaving fewer fraction bits. Its broader range does not preserve the 0.003 change near one.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="fp16-bf16-bit-budget"><strong>Check:</strong> Why can BF16 have broader range yet round away the 0.003 change retained by FP16?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="fp16-bf16-bit-budget">BF16 allocates eight exponent bits and seven fraction bits; FP16 uses five and ten. Both reserve one sign bit.</span></figcaption>
</figure>
<p class="diagram-source">References: <a href="https://docs.nvidia.com/cuda/floating-point/index.html">NVIDIA's IEEE 754 overview</a>, <a href="https://cloud.google.com/tpu/docs/bfloat16">Google Cloud's BF16 documentation</a>, and the <a href="https://arxiv.org/abs/1905.12322">BFLOAT16 study</a>.</p>

In $[1,2)$, FP16 spacing is $2^{-10}=0.0009765625$.
BF16 spacing is $2^{-7}=0.0078125$, eight times wider.

| Format | Rounded $1.003$ | Absolute error |
| --- | --- | --- |
| FP16 | $1.0029296875$ | $0.0000703125$ |
| BF16 | $1$ | $0.003$ |

These are rounding errors in a fixture, not measurements of model quality.

## Change the numerical requirement

### 1. Prioritize range

**New task:** store $70000$ with the same rounding rule.

**Picture and change:** inspect the exponent range rather than only spacing near one.

**Check:** FP16 overflows because its largest finite value is $65504$.
BF16 rounds to finite $70144$, not exact $70000$.

**Use when:** activations exceed FP16 range and BF16 rounding is acceptable for the application.

### 2. Store scaled integers

**New task:** store $x=0.26$ as signed INT8 with scale $a=0.1$ and zero point zero.

**Picture and change:** compute $q=\operatorname{clamp}(\operatorname{round}(x/a),-128,127)$, then reconstruct $\hat x=aq$.
This is scaled integer quantization, not an exponent/fraction format.

**Check:** $q=3$ reconstructs $0.3$, with error $0.04$.
Scale $0.01$ reconstructs $0.26$ exactly but narrows the representable range.
For 1024 raw values, INT8 uses 1024 bytes versus BF16's 2048 bytes, excluding metadata.

**Use when:** storage or bandwidth motivates quantization and calibration can establish a suitable scale.

## Saturation is not rounding noise

At scale $0.1$, $x=20$ exceeds INT8's largest code.
Clamping reconstructs $12.7$ instead. Finer rounding within the existing range cannot recover the clipped value.

## Try the next exponent interval

What is BF16 spacing within $[2,4)$?

<details>
<summary>Check your answer</summary>

It is $2^{1-7}=0.015625$.
Increasing the exponent doubles spacing while the fraction width stays fixed.

</details>

<!-- learning:core-end -->

<div data-learning-extension>

## The standard layouts

The table gives approximate finite endpoints and normal minima.
FP8 rows use the E4M3 finite-endpoint and E5M2 conventions described in [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433).

| Format | Stored bits | Exponent / fraction | Largest finite magnitude | Smallest positive normal |
| --- | --- | --- | --- | --- |
| FP32 | 32 | 8 / 23 | $3.4\times10^{38}$ | $1.18\times10^{-38}$ |
| FP16 | 16 | 5 / 10 | $65504$ | $2^{-14}$ |
| BF16 | 16 | 8 / 7 | $3.39\times10^{38}$ | $2^{-126}$ |
| TF32 multiply input precision | FP32 storage | 8 / 10 | Approximately FP32 range | Approximately FP32 normal range |
| FP8 E4M3 | 8 | 4 / 3 | $448$ | $2^{-6}$ |
| FP8 E5M2 | 8 | 5 / 2 | $57344$ | $2^{-14}$ |
| Signed INT8 | 8 | Neither | Codes $-128$ through $127$ | Scale-dependent step |
| FP64 | 64 | 11 / 52 | $1.8\times10^{308}$ | $2.23\times10^{-308}$ |

FP16's smallest positive subnormal is $2^{-24}$, below its normal minimum.
Actual operations may flush subnormals to zero; representable values and hardware behavior are separate questions.

## TF32: a hybrid for matmul accumulation

TF32 is an internal multiply-input precision, not a packed nineteen-bit tensor dtype.
Its reduced fraction precision and FP32 accumulation describe separate stages.
Check the actual operation, library version and enabled precision policy; defaults and speedups are not universal.

## FP8: the new training format

OFP8 E4M3 has more fraction precision and a smaller finite range than E5M2.
E4M3's finite-endpoint convention has no infinity encoding; E5M2 includes infinities.
Other FP8 variants have different endpoint conventions.
Scaling, tensor placement and accumulation precision are part of a training recipe, not implied by the bit count.

## INT8: the inference format

Affine quantization adds a zero point $z$: $q=\operatorname{clamp}(\operatorname{round}(x/a)+z,q_{\min},q_{\max})$.
Reconstruction is $a(q-z)$.
Per-tensor, per-channel and grouped scales trade metadata and kernel support against quantization error.

## INT4 / NF4 / GPTQ: aggressive inference quantization

INT4 stores sixteen codes; its reconstruction depends on scales and zero points.
NF4 uses nonuniform levels motivated by a normal-weight distribution.
GPTQ and AWQ are distinct quantization methods, not scalar floating-point formats.
Evaluate the actual model, calibration distribution, kernels and downstream quality.

<span id="standard-usage-in-2026"></span>

## Choosing precision by operation

| Task | Format |
|------|--------|
| Training | Select parameter, activation, reduction and optimizer-state precision separately |
| FP16 backward | Consider loss scaling; unscale before clipping and optimizer updates |
| BF16 backward | Broader range often avoids loss scaling, but coarse updates can still disappear |
| Inference | Check weight, activation and KV-cache quantization independently |
| Memory-limited deployment | Include scales, metadata, workspace and kernel support, not only raw weight bits |

## Common pitfalls

Loss scaling does not repair low-precision master-weight rounding or incorrect arithmetic.
Reduced-precision optimizer states need a validated recipe; FP32 is not a universal theorem or the only workable choice.
Throughput comparisons must state format, sparsity, operation and workload.

## Related

- [Mixed precision training](/concepts/mixed-precision-training/). Practical training in FP16/BF16.
- [Quantization](/concepts/quantization/). Inference quantization in depth.
- [GPU memory hierarchy](/concepts/gpu-memory-hierarchy/). Why bandwidth matters.

</div>
