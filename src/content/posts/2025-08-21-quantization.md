---
title: "Quantization: INT8, INT4, FP8, and the inference cost picture"
description: "Trace low-bit codes, rounding error, clipping, and scale overhead before measuring task quality or kernel-dependent speed."
date: "2025-08-21"
draft: false
tags: ["concepts"]
category: "concepts"
---


## Summary

Quantization stores or computes selected tensors with a reduced code set and an explicit reconstruction policy.
Rounding, clipping, scale granularity, metadata, and supported kernels determine numerical and systems effects separately.
Smaller weights do not guarantee faster execution or a universal bound on task-quality change.

## The mechanism

For positive scale $s$ and integer zero-point $z$, affine quantization uses:

$$
q=\operatorname{clip}(\operatorname{round}(x/s)+z,q_{min},q_{max}),\qquad
\hat x=s(q-z).
$$

The pictured symmetric INT4 fixture uses $z=0$, codes $-7$ through $7$, and half ties rounded away from zero.
It deliberately excludes code $-8$; other four-bit schemes need not use this convention.
With shared $s=8/7$, ordinary values $[-1,-0.5,0.5,1]$ become codes $[-1,0,0,1]$.
Reconstruction is $[-8/7,0,0,8/7]$, with squared error $2/49+1/2=53/98$.

Giving that channel $s=1/7$ yields codes $[-7,-4,4,7]$ and reconstruction $[-1,-4/7,4/7,1]$.
Its squared error becomes $1/98$.
The high-range row keeps $s=8/7$ under either policy, restoring $[-8,-32/7,32/7,8]$.

<!-- visual:quantization-scale-granularity -->
<figure class="learning-figure" aria-labelledby="quantization-scale-title" aria-describedby="quantization-scale-description">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="quantization-scale-title">One outlier can erase small values under a shared quantization scale</p>
	<p id="quantization-scale-description">An original symmetric INT4 example compares one scale shared by two channels with a separate scale for each channel. Values are rounded to integer codes from negative seven through seven and then dequantized.</p>
	<div class="visual-grid--two" role="group" aria-label="Shared-scale and per-channel symmetric INT4 quantization comparison">
		<section class="visual-panel" aria-labelledby="shared-quantization-title">
			<h4 id="shared-quantization-title">One shared scale</h4>
			<p>The high-range channel sets &Delta; = 8/7 &asymp; 1.14 for both rows.</p>
			<table class="cm-grid" aria-label="Symmetric INT4 quantization with one shared scale">
				<thead><tr><th scope="col">Channel</th><th scope="col">Original</th><th scope="col">Codes → restored</th></tr></thead>
				<tbody>
					<tr><th scope="row">ordinary</th><td>&minus;1, &minus;0.5, 0.5, 1</td><td class="cm-selected">&minus;1, 0, 0, 1<br>&rarr; &minus;1.14, 0, 0, 1.14</td></tr>
					<tr><th scope="row">high range</th><td>&minus;8, &minus;4, 4, 8</td><td>&minus;7, &minus;4, 4, 7<br>&rarr; &minus;8, &minus;4.57, 4.57, 8</td></tr>
				</tbody>
			</table>
			<p><strong>Lost detail:</strong> &minus;0.5 and 0.5 both collapse to code 0.</p>
		</section>
		<section class="visual-panel" aria-labelledby="channel-quantization-title">
			<h4 id="channel-quantization-title">One scale per channel</h4>
			<p>The ordinary row gets &Delta; = 1/7; the high-range row keeps &Delta; = 8/7.</p>
			<table class="cm-grid" aria-label="Symmetric INT4 quantization with a separate scale for each channel">
				<thead><tr><th scope="col">Channel</th><th scope="col">Original</th><th scope="col">Codes → restored</th></tr></thead>
				<tbody>
					<tr><th scope="row">ordinary</th><td>&minus;1, &minus;0.5, 0.5, 1</td><td class="cm-selected">&minus;7, &minus;4, 4, 7<br>&rarr; &minus;1, &minus;0.57, 0.57, 1</td></tr>
					<tr><th scope="row">high range</th><td>&minus;8, &minus;4, 4, 8</td><td>&minus;7, &minus;4, 4, 7<br>&rarr; &minus;8, &minus;4.57, 4.57, 8</td></tr>
				</tbody>
			</table>
			<p><strong>Recovered detail:</strong> each row uses the full signed code range.</p>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> compare the ordinary row across panels. Its shared-scale middle values collapse to zero; its separate scale preserves both signs. More scales add metadata and kernel requirements. This calculation is not a quality benchmark. Original example follows <a href="https://arxiv.org/abs/1712.05877">Jacob et al.</a> and <a href="https://arxiv.org/abs/2211.10438">SmoothQuant</a>. <span class="visual-reader-check" data-reader-check="quantization-scale-granularity"><strong>Check:</strong> Why does the shared scale collapse -0.5 and +0.5? Why does the channel-specific scale preserve their signs?</span></figcaption>
</figure>

### Calibrate for a Larger Outlier

**New task:** Deployment introduces value sixteen in the high-range channel.

**Picture and change:** Retain the old scale and clip, or recalibrate to $16/7$ and recompute all affected codes.

**Check:** The stale scale clips sixteen to code seven and restores eight, absolute error eight.
With a shared recalibrated scale, the entire ordinary row rounds to zero and its squared error becomes $2.5$.

**Use when:** Deployment ranges exceed calibration coverage; retaining separate scales can isolate this particular cross-channel effect.

### Budget Group-Scale Overhead

**New task:** Store 128 four-bit weights with one BF16 scale.

**Picture and change:** Pack sixty-four payload bytes and retain two scale bytes; the kernel must use the same grouping.

**Check:** Sixty-six bytes versus 256 BF16 bytes is $256/66\approx3.8788$ times smaller, excluding headers, alignment, and zero-point metadata.

**Use when:** Granularity improves reconstruction while memory remains constrained; benchmark actual dequantization and task quality separately.

<!-- learning:core-end -->
<div data-learning-extension>

## The flavors

### Weight-only quantization (most common)

Compress weights while declaring activation, compute, and accumulation dtypes separately.
The retained heading does not establish a deployment-frequency ranking.

### Activation quantization (less common, harder)

Declare W8A8's formats, scales, and accumulators.
FP8 E4M3 and E5M2 trade precision against range differently and are not affine integer codes.

### KV cache quantization

Compress cached K/V with a supported layout and evaluate long-context behavior, not only short prompts.

### Quantization-aware training (QAT)

Model quantization effects during training; compare its extra training cost and quality against suitable post-training baselines.

## What an interviewer expects you to say

Name the compressed tensor, rounding/scaling contract, target kernel, and quality evaluation before choosing a method.

## Where the wins come from

Profile weight bandwidth, arithmetic, cache capacity, and dequantization; the bottleneck changes with batching, context, hardware, and kernels.

## Common confusions

GPTQ uses calibrated second-order reconstruction approximations; AWQ uses activation-aware scaling, not mandatory higher-precision protection.
SmoothQuant redistributes scale across activations and weights; GGUF is a storage format supporting multiple schemes, not one quantization algorithm.

## Picking a quantization method (decision tree)

Measure supported candidates and sensitive slices; keep higher-precision tensors where justified rather than treating one method as universally best.

## Why interviewers ask

Connect byte arithmetic to a measured bottleneck and a task-quality acceptance criterion.

---

*Related: [Mixed precision training](/concepts/mixed-precision-training/), [KV cache](/concepts/kv-cache/), [How to think about LLM inference cost](/guides/llm-inference-cost/).*

</div>
