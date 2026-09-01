---
title: "Transformer compute and memory accounting"
description: "Estimate parameters, training FLOPs, activation memory, and KV-cache memory from a small set of model dimensions."
date: "2026-08-28"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Transformer accounting converts model dimensions into estimates for parameters, FLOPs, and memory. These estimates show whether a training or serving plan can fit and how long it may take.

## Why AI labs care

Large-model plans should start with arithmetic. A candidate should be able to estimate:

- model size;
- training compute;
- memory per device;
- KV-cache size;
- the effect of sequence length;
- the active compute of a mixture-of-experts model.

Exact framework code comes later.

## Symbols

Use these symbols for a decoder-only transformer:

| Symbol | Meaning |
| --- | --- |
| $B$ | number of sequences in a batch |
| $T$ | tokens per sequence |
| $D$ | model width |
| $F$ | feed-forward width |
| $L$ | number of layers |
| $N$ | number of query heads |
| $K$ | number of key/value heads |
| $H$ | size of each head |
| $V$ | vocabulary size |

Usually, $N H = D$. Grouped-query attention uses $K < N$.

The number of tokens in one batch is $B T$.

<!-- visual:transformer-training-inference-ledgers -->
<figure class="learning-figure" aria-labelledby="transformer-ledgers-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="transformer-ledgers-title">Which dimensions and multiplicities belong in each transformer budget?</p>
	<div class="visual-grid--two" role="group" aria-label="Training and inference accounting ledgers that separate compute from persistent and temporary memory">
		<section class="visual-panel" aria-labelledby="training-ledger-title">
			<h4 id="training-ledger-title">Training ledger</h4>
			<p>Compute repeats over tokens; memory combines persistent model state with workload-dependent activations.</p>
			<table class="cm-grid" aria-label="Training quantities, first estimates, and scaling drivers">
				<thead><tr><th scope="col">Quantity</th><th scope="col">First estimate</th><th scope="col">Count or driver</th></tr></thead>
				<tbody>
					<tr><th scope="row">Parameter matmuls</th><td>≈ 6 × <var>P</var><sub>active</sub> × training tokens</td><td>Repeats per token; use active MoE parameters</td></tr>
					<tr><th scope="row">Attention scores</th><td>≈ 12<var>BT</var><sup>2</sup><var>NHL</var></td><td>Quadratic in sequence length <var>T</var></td></tr>
					<tr><th scope="row">Model state</th><td>2<var>P</var> weights + 2<var>P</var> gradients + 8<var>P</var> Adam = 12<var>P</var> bytes</td><td>Uses total stored <var>P</var>; add 4<var>P</var> for FP32 master weights</td></tr>
					<tr><th scope="row">Saved activations</th><td>Add separately</td><td>Grows with <var>B</var>, <var>T</var>, <var>D</var>, <var>L</var>, and save policy</td></tr>
				</tbody>
			</table>
		</section>
		<section class="visual-panel" aria-labelledby="inference-ledger-title">
			<h4 id="inference-ledger-title">Inference ledger</h4>
			<p>No gradients or optimizer moments remain; weights and request-specific cache state dominate.</p>
			<table class="cm-grid" aria-label="Inference quantities, first estimates, and scaling drivers">
				<thead><tr><th scope="col">Quantity</th><th scope="col">First estimate</th><th scope="col">Count or driver</th></tr></thead>
				<tbody>
					<tr><th scope="row">Weights</th><td><var>P</var> × weight bytes</td><td>Uses total stored <var>P</var>, including all MoE experts</td></tr>
					<tr><th scope="row">KV cache</th><td>2<var>TLKHs</var><sub>KV</sub></td><td>Per sequence; multiply by active sequences</td></tr>
					<tr><th scope="row">Temporary activations</th><td>Add separately</td><td>Depends on batched tokens, scheduler, and kernels</td></tr>
					<tr><th scope="row">Paging</th><td>Allocation policy, not a new byte formula</td><td>Reduces reservation waste; present tokens still need their K/V values</td></tr>
				</tbody>
			</table>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> choose the ledger before multiplying. Training compute repeats active parameters over tokens, while training model-state memory stores total parameters plus gradients and optimizer moments. Inference drops those training-only states but adds one KV allocation per active sequence. Sequence length enters attention compute as <var>T</var><sup>2</sup> and KV memory as <var>T</var>; activations and temporary buffers must still be measured separately. Original accounting summary checked against the <a href="https://jax-ml.github.io/scaling-book/transformers/">JAX Scaling Book's Transformer derivation</a>.</figcaption>
</figure>

## Parameter count

### Gated feed-forward block

Many current models use three matrices in each feed-forward block:

- two projections from $D$ to $F$;
- one projection from $F$ back to $D$.

The count per layer is:

$$
P_{\text{ffn}} = 3 D F.
$$

A non-gated block has two large matrices and uses about $2DF$ parameters.

### Attention projections

Query and output projections each use about $D \times NH$ parameters. Key and value projections each use about $D \times KH$.

The count per layer is:

$$
P_{\text{attn}} = 2DNH + 2DKH = 2DH(N+K).
$$

When $NH=D$, this becomes:

$$
P_{\text{attn}} = 2D^2\left(1+\frac{K}{N}\right).
$$

Reducing $K$ saves key/value parameters and KV-cache memory.

### Embeddings and norms

A token embedding has $VD$ parameters. The output layer may share that matrix or add another $VD$ parameters. Normalization parameters are small compared with the large matrices.

A useful total is:

$$
P \approx L(P_{\text{ffn}} + P_{\text{attn}}) + P_{\text{vocab}}.
$$

For many dense models, the feed-forward blocks contain most parameters.

## FLOPs for training

A matrix multiply uses about two FLOPs per multiply-add. Its backward pass computes gradients for the input and the weight. The forward and backward passes together cost about three times the forward pass.

For the large parameter matrices, a common training estimate is:

$$
\text{training FLOPs} \approx 6 \times P \times \text{training tokens}.
$$

This estimate omits some attention work, normalization, routing, and other small operations. It is useful for a first estimate.

For a mixture-of-experts model, use **active parameters per token** for the compute estimate. Use total parameters for weight memory.

## Attention FLOPs

Attention scores and the weighted value sum add work that grows with $T^2$.

For standard self-attention, the training cost of these two matrix operations is roughly:

$$
12 B T^2 N H L.
$$

Under common model ratios, attention-score FLOPs become comparable to the other large matrix operations when the sequence length reaches several times the model width. The exact point depends on architecture, masking, and the attention kernel.

Long context can become expensive before this FLOP crossover because activation and KV memory also grow with sequence length.

## Training memory

Count each component separately:

| Component | Common storage |
| --- | ---: |
| BF16 parameters | 2 bytes per parameter |
| BF16 gradients | 2 bytes per parameter |
| Adam first moment | 4 bytes per parameter |
| Adam second moment | 4 bytes per parameter |
| Optional FP32 master weights | 4 bytes per parameter |

This gives about 12 bytes per parameter without FP32 master weights and about 16 with them.

Then add:

- saved activations;
- temporary kernel buffers;
- communication buffers;
- allocator overhead.

Activation memory depends on batch tokens, width, layers, and which intermediate values are saved. It can exceed model-state memory at long context. Activation checkpointing reduces saved values and repeats part of the forward work during backpropagation.

## Inference memory

Inference has no gradients or optimizer state. It stores weights, temporary activations, and a KV cache for each active sequence.

KV-cache bytes per sequence are approximately:

$$
M_{\text{KV}} = 2 T L K H s,
$$

where $s$ is bytes per stored value. The factor 2 stores both keys and values.

For a batch of active sequences, multiply by the number of sequences. Paged allocation reduces unused reserved space. It does not change the bytes needed for tokens that are present.

## Small example

A one-billion-parameter model in BF16 needs about 2 GB for weights.

During training with BF16 gradients and FP32 Adam moments, model state needs about 12 GB before activations and temporary buffers. If FP32 master weights are kept, the estimate becomes about 16 GB.

During inference, the same model may fit easily while long KV caches limit the number of active requests.

## In an interview

Use this order:

1. Write the model dimensions.
2. Estimate feed-forward, attention, and vocabulary parameters.
3. Use $6P$ FLOPs per training token as a first estimate.
4. Add attention-score FLOPs for long context.
5. Separate model state from activations and temporary buffers.
6. Compute KV bytes per token and per request.
7. State which assumptions may change the result.
8. Compare the estimate with measured utilization before making a cost claim.

## Common mistakes

- Using total MoE parameters to estimate per-token compute.
- Forgetting the backward pass.
- Counting weights while omitting gradients and optimizer state.
- Assuming every training stack keeps FP32 master weights.
- Ignoring activation memory.
- Using maximum context without multiplying KV memory by active requests.
- Treating the $6P$ rule as exact at long context.

*Related: [train a 100B parameter model](/questions/train-100b-model/), [KV cache](/concepts/kv-cache/), and [activation checkpointing](/concepts/activation-checkpointing/). Further practice: [Transformer math in the JAX Scaling Book](https://jax-ml.github.io/scaling-book/transformers).*