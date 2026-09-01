---
title: "Quantization: INT8, INT4, FP8, and the inference cost picture"
description: "Reduce model precision to shrink memory and speed up inference. The trade-offs are real but increasingly small with modern techniques."
date: "2025-08-21"
draft: false
tags: ["concepts"]
category: "concepts"
---


## Summary

Quantization reduces model weights (and sometimes activations) to lower precision than the trained representation, trading accuracy for memory and speed at inference.

LLM serving is dominated by weight-loading bandwidth. Reducing weight precision from 16 bits to 4 bits cuts that bandwidth by 4&times;, often the single largest serving cost reduction available. For GPU memory, it lets you fit much larger models on a given device.

INT8 quantization is standard in most inference stacks by 2026; many support INT4 or FP8. Quality cost is typically <1% on benchmarks for 4-bit when done well.

## The flavors

### Weight-only quantization (most common)

Compress only the weights; keep activations and computation in higher precision.

- **INT8 weights** with FP16/BF16 compute: 2&times; smaller weights, modest speedup, near-zero quality loss.
- **INT4 weights** (GPTQ, AWQ, GGML): 4&times; smaller weights, significant speedup for memory-bound decoding, &lt;1% quality loss in most cases.
- Standard for LLM inference. Used in vLLM, TGI, llama.cpp, all major serving systems.

### Activation quantization (less common, harder)

Quantize both weights and activations.

- **INT8 weights + INT8 activations** (W8A8): full INT8 inference. Tensor cores can run INT8 matmuls 2&times; faster than BF16. Quality cost is larger; needs more careful calibration.
- **FP8 (W8A8)**: H100 / B100 era. Uses FP8 tensor cores; fewer accuracy issues than INT8 W8A8 because FP8 has dynamic range.

### KV cache quantization

Quantize the KV cache to INT8 or INT4. Saves serving memory; lets you fit longer contexts or higher batch sizes. Quality cost is small if done with care (per-token or per-channel scaling).

### Quantization-aware training (QAT)

Quantize during training (with simulated quantization noise) so the model learns to be quantization-friendly. More accurate than post-training quantization but requires retraining. Used for INT4 / very low precision; less needed for INT8.

## The mechanism

Naive quantization: pick a scale `s` and zero-point `z`; quantize as `q = round((x - z) / s)`; dequantize as `x' = s * q + z`. Per-tensor scaling fails for LLMs because activation magnitudes vary wildly across channels.

<!-- visual:quantization-scale-granularity -->
<figure class="learning-figure" aria-labelledby="quantization-scale-title" aria-describedby="quantization-scale-description">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="quantization-scale-title">See how one high-range channel can waste low-bit levels for another channel.</p>
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
	<figcaption><strong>Read it this way:</strong> compare the ordinary row across panels. With one shared scale, the &plusmn;8 values determine the step for every value, so &minus;0.5 and 0.5 both round to zero. A separate scale lets the ordinary channel spend the same INT4 codes over its own smaller range and keeps the signs distinct. More scales reduce rounding error but add scale metadata and kernel complexity; calibration data and task-specific evaluation still determine whether the approximation is acceptable. This is an illustrative calculation, not a benchmark. Original example checked against <a href="https://arxiv.org/abs/1712.05877">Jacob et al.’s affine quantization formulation</a> and <a href="https://arxiv.org/abs/2211.10438">SmoothQuant’s analysis of channel-wise outliers</a>.</figcaption>
</figure>

Modern techniques:

- **Per-channel quantization**: separate scale per output channel. Standard for INT8.
- **Per-group quantization**: group columns/rows into small groups, scale per group. Used in GPTQ, AWQ for INT4.
- **GPTQ**: post-training quantization that uses second-order information (approximate Hessian) to choose quantized weights that minimize output error. Standard for INT4 LLMs.
- **AWQ (Activation-aware Weight Quantization)**: identifies "important" weight channels (those with large activation magnitudes) and protects them with higher precision or careful scaling. Often slightly better than GPTQ for INT4.
- **SmoothQuant**: handles activation outliers by mathematically shifting the scale from activations to weights.
- **GGUF / GGML quantization**: family of quantization methods used in llama.cpp; supports very flexible bit-widths (2-bit through 8-bit, often per-block).

## What an interviewer expects you to say

If asked about quantization:

1. Distinguish weight-only vs activation quantization.
2. Mention INT8 (easy, near-zero quality loss) vs INT4 (more accuracy concern, but workable with GPTQ/AWQ) vs FP8 (H100-era).
3. Mention per-channel / per-group scaling as the standard improvement over naive quantization.
4. Discuss when quantization helps most (memory-bound decoding) vs when it doesn't (compute-bound prefill).
5. Acknowledge quality measurement: quantized models should be A/B tested against the baseline on your own eval set, not just benchmarks.

## Where the wins come from

For LLM decoding (memory-bound):
- INT8 weights: ~1.5-2&times; speedup from less memory bandwidth.
- INT4 weights: ~3-4&times; speedup.
- FP8: similar to INT8, with better accuracy floor.

For LLM prefill (compute-bound on long inputs):
- Less benefit from weight quantization alone.
- W8A8 INT8 or FP8 helps: tensor core throughput is higher.

For GPU memory:
- INT8 cuts weight memory in half.
- INT4 quarters it.
- KV cache quantization is often the bigger memory win at long contexts.

## Common confusions

- **"Quantization always loses quality."** True but at INT8 the loss is usually negligible. At INT4 with GPTQ/AWQ, often still &lt;1% on standard benchmarks. Worth measuring on *your* eval, not just published benchmarks.
- **"Quantization is the same as distillation."** No. Distillation is training a smaller model to mimic a larger one. Quantization is reducing precision of the same model. They compose.
- **"Quantize everything."** No. LayerNorm parameters, the embedding lookup, and small layers often stay in higher precision because they cost little memory and are sensitive.
- **"FP8 is just smaller FP16."** Different exponent/mantissa split (E4M3 vs E5M2) and per-tensor scale management.

## Picking a quantization method (decision tree)

1. **Need to fit on smaller GPU?** Start with INT8 weight-only.
2. **Need maximum throughput?** INT4 weight-only or FP8 (H100+).
3. **Quality regression on eval?** Try AWQ instead of GPTQ; tune group size; consider mixed precision (sensitive layers in higher precision).
4. **Long context?** Add KV cache quantization on top.
5. **Have time for retraining?** Consider QAT for the most aggressive bit widths.

## Why interviewers ask

Quantization questions test:
1. Whether you know several quantization methods and their tradeoffs.
2. Whether you understand *why* it works (memory-bandwidth bottleneck).
3. Whether you've measured quality loss in production (vs trusting published numbers).
4. Whether you can prioritize: which quantization technique for which problem.

In senior LLM-team interviews, quantization comes up most often inside the cost-reduction question. See [How would you reduce LLM inference cost by 10x?](/questions/reduce-llm-inference-cost-10x/).

---

*Related: [Mixed precision training](/concepts/mixed-precision-training/), [KV cache](/concepts/kv-cache/), [How to think about LLM inference cost](/guides/llm-inference-cost/).*
