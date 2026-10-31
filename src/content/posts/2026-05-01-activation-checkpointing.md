---
title: "Activation checkpointing"
description: "Trade compute for memory: drop activations during the forward pass and recompute them during the backward pass. The cheapest way to fit a larger model on the same GPU."
date: "2026-05-01"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Activation checkpointing (also called gradient checkpointing) saves only a subset of activations during the forward pass and recomputes the rest from those saved checkpoints during the backward pass. Memory drops at the cost of one extra forward pass per checkpoint segment.

Backprop needs every layer's input activation to compute that layer's parameter gradient. For a deep model the activations dominate training memory. Often more than parameters and optimizer state combined. A 7B-parameter transformer with 32 layers, batch 1, sequence 4096 stores tens of GB of activations.

Checkpointing recovers this memory by repeating part of the forward work. Recomputing the full forward graph adds about one forward pass of FLOPs. Wall-time cost depends on kernels, memory traffic, and how much of the graph is recomputed.

## The mechanism

Partition a chain of $L$ layers into segments of $K$ consecutive layers. During forward:

1. Run the segment.
2. Save **only** its input (the checkpoint).
3. Discard intermediate activations.

During backward:

1. Recompute the segment's forward pass starting from the saved input.
2. Compute gradients normally for that segment.
3. Discard the recomputed activations.

For a transformer, the natural segment is one transformer block. PyTorch provides `torch.utils.checkpoint.checkpoint(...)` and `checkpoint_sequential(...)`; modern training stacks expose this as a single flag (e.g., `gradient_checkpointing=True` in HuggingFace `Trainer`).

## Cost model

- **Memory across a simple chain**: if each segment has $K$ layers, peak saved boundary state scales like $O(L/K)$ and peak temporary recomputation state like $O(K)$. The total $O(L/K + K)$ is minimized near $K=\sqrt{L}$.
- **Memory inside a transformer block**: checkpointing every block still stores block boundaries, but discards the larger internal matrix and attention intermediates. The reduction is a workload-dependent constant factor, not $L$×.
- **Compute**: full rematerialization adds roughly one forward pass to a training step. Since a forward and backward step is often estimated at three forward-pass units, this is about 33% more FLOPs. Wall-time overhead can differ.

<!-- visual:checkpoint-boundary-recompute-balance -->
<figure class="learning-figure plot-panel visual-wide" aria-labelledby="checkpoint-balance-title">
	<p class="visual-kicker">Memory at one backward step</p>
	<p class="visual-title" id="checkpoint-balance-title">Four-layer segments balance saved and rebuilt states in this model</p><div class="visual-argument" data-visual-argument="checkpoint-boundary-recompute-balance" data-argument-kind="curve" data-argument-composition="landmark"><svg viewBox="0 0 360 501" role="img" aria-label="Saving less requires rebuilding more. In this 16-layer model, boundary storage is 16/s and rebuilt segment storage is s. Their sum is minimized at s=4."><title>Saving less requires rebuilding more</title><desc>In this 16-layer model, boundary storage is 16/s and rebuilt segment storage is s. Their sum is minimized at s=4.</desc><g opacity=".35"><path d="M48 177.5L330 177.5" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path></g><g opacity=".35"><path d="M48 135L330 135" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path></g><g opacity=".35"><path d="M48 92.5L330 92.5" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path></g><path d="M48 40L48 220" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M48 220L334 220" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="48" y="245" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="48" dy="0">1</tspan></text><text x="330" y="245" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="330" dy="0">16</tspan></text><text x="40" y="54" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="40" dy="0">18</tspan></text><text x="40" y="220" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="40" dy="0">0</tspan></text><path data-proof-landmark="4" d="M104.4 40V220" style="stroke:var(--viz-focus-stroke);stroke-width:1.5;stroke-dasharray:2 4"></path><text x="104.4" y="245" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-focus-stroke)"><tspan x="104.4" dy="0">4</tspan></text><path data-proof-series="Saved plus rebuilt: 16/s + s" d="M48 59.44444444444446L52.40625 85.92288590014064L56.8125 103.24431146572104L61.21875 115.18914054536187L65.625 123.70855734767026L70.03125 129.9115082933653L74.4375 134.47488275613276L78.84375 137.83525435568706L83.25 140.28683574879227L87.65625 142.0351322585148L92.0625 143.22803089304256L96.46875 143.97467097282873L100.875 144.35735428051L104.4 144.44444444444446L105.28125 144.43931654869155L109.6875 144.269946269262L114.09375 143.88837826797385L118.5 143.32602339181287L122.90625 142.60838449146638L127.3125 141.75638306719895L131.71875 140.78734379974532L136.125 139.71573565323564L140.53125 138.5537369906186L144.9375 137.31167160180485L149.34375 135.99834878429772L153.75 134.62133123689728L158.15625 133.18714803214377L162.5625 131.70146536955457L166.96875 130.16922456171523L171.375 128.5947543617998L175.78125 126.9818630316188L180.1875 125.33391428880243L184.59375 123.65389033291325L189 121.94444444444443L193.40625 120.20794511528524L197.8125 118.44651325977546L202.21875 116.6620537398604L206.625 114.85628219278881L211.03125 113.03074795817628L215.4375 111.18685375043813L219.84375 109.32587260315015L224.25 107.4489625167336L228.65625 105.55717916462118L233.0625 103.65148695164906L237.46875 101.73276866870397L241.875 99.80183394720687L246.28125 97.8594266839573L250.6875 95.90623157972297L255.09375 93.94287991258487L259.5 91.96995464852608L263.90625 89.98799497635935L268.3125 87.99750034125034L272.71875 85.99893404034313L277.125 83.99272643496576L281.53125 81.97927782628378L285.9375 79.95896103483346L290.34375 77.93212371891013L294.75 75.89909046214356L299.15625 73.86016465663161L303.5625 71.81563020461576L307.96875 69.76575305877532L312.375 67.7107826187183L316.78125 65.65095299909203L321.1875 63.58648418287504L325.59375 61.51758307179824L330 59.44444444444446" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:3"></path><g data-proof-observation="0" data-proof-x="4" data-proof-y="8"><circle cx="104.4" cy="144.44444444444446" r="5" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><path data-proof-series="Saved boundaries: 16/s" d="M48 68.88888888888889L52.40625 97.58087201125176L56.8125 117.11583924349883L61.21875 131.2742099898063L65.625 142.00716845878136L70.03125 150.42366107114307L74.4375 157.2005772005772L78.84375 162.77449046679817L83.25 167.43961352657004L87.65625 171.40145170295924L92.0625 174.8078920041537L96.46875 177.7680737506065L100.875 180.36429872495447L104.4 182.22222222222223L105.28125 182.65980265980266L109.6875 184.70397404703974L114.09375 186.5359477124183L118.5 188.18713450292398L122.90625 189.68303726924418L127.3125 191.04457751164338L131.71875 192.2890799108564L136.125 193.43101343101344L140.53125 194.48255643506303L144.9375 195.45403271291596L149.34375 196.35425156207552L153.75 197.19077568134173L158.15625 197.97013414325488L162.5625 198.69799314733234L166.96875 199.37929400615968L171.375 200.01836547291094L175.78125 200.61901580939656L180.1875 201.18460873324688L184.59375 201.71812644402436L189 202.22222222222223L193.40625 202.69926455972967L197.8125 203.15137437088657L202.21875 203.5804565176382L206.625 203.98822663723325L211.03125 204.3762340692874L215.4375 204.7458815282159L219.84375 205.09844204759457L224.25 205.4350736278447L228.65625 205.75683194239895L233.0625 206.0646813960935L237.46875 206.35950477981507L241.875 206.64211172498466L246.28125 206.91324612840174L250.6875 207.17359269083408L255.09375 207.42378269036266L259.5 207.6643990929705L263.90625 207.89598108747046L268.3125 208.11902811902812L272.71875 208.33400348478756L277.125 208.54133754607687L281.53125 208.74143060406158L285.9375 208.9346554792779L290.34375 209.12135983002125L294.75 209.30186823992133L299.15625 209.47648410107604L303.5625 209.64549131572687L307.96875 209.8091558365531L312.375 209.96772706316276L316.78125 210.12143911020314L321.1875 210.2705119606528L325.59375 210.4151525162427L330 210.55555555555554" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:3;stroke-dasharray:7 4"></path><g data-proof-observation="1" data-proof-x="4" data-proof-y="4"><circle cx="104.4" cy="182.22222222222223" r="5" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><path data-proof-series="Rebuilt segment: s" d="M48 210.55555555555554L330 68.88888888888889" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:3;stroke-dasharray:7 4"></path><g data-proof-observation="2" data-proof-x="4" data-proof-y="4"><circle cx="104.4" cy="182.22222222222223" r="5" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle></g><text x="190" y="273" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="190" dy="0">segment length s</tspan></text><text x="20" y="20" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">relative stored states</tspan></text><path d="M20 306L48 306" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><text x="60" y="310" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="60" dy="0">Saved plus rebuilt: 16/s + s</tspan></text><text x="60" y="333" style="font-size:18px;font-weight:700;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="60" dy="0">8 at 4</tspan></text><path d="M20 368L48 368" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><text x="60" y="372" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="60" dy="0">Saved boundaries: 16/s</tspan></text><text x="60" y="395" style="font-size:18px;font-weight:700;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="60" dy="0">4 at 4</tspan></text><path d="M20 430L48 430" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><text x="60" y="434" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="60" dy="0">Rebuilt segment: s</tspan></text><text x="60" y="457" style="font-size:18px;font-weight:700;text-anchor:start;fill:var(--viz-warning-stroke)"><tspan x="60" dy="0">4 at 4</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
	<div class="visual-scroll">
		<svg viewBox="0 0 700 330" role="img" aria-labelledby="checkpoint-balance-svg-title checkpoint-balance-svg-desc">
			<title id="checkpoint-balance-svg-title">Activation checkpoint memory for a sixteen-layer chain split into four-layer segments</title>
			<desc id="checkpoint-balance-svg-desc">Sixteen numbered layer boxes are divided into four segments of four layers. Diamond checkpoints mark the four saved segment inputs. During backward, a dashed enclosure marks the four activations temporarily rebuilt for layers nine through twelve. Below, four checkpoint diamonds plus four rebuilt activation squares total eight illustrative activation-sized units. The diagram concludes that balancing L divided by K against K gives a segment length near the square root of L.</desc>
			<text class="viz-axis-label" x="22" y="24">EXAMPLE: L = 16 layers, K = 4 layers per segment</text>
			<text class="viz-label" x="22" y="47">Forward keeps only each segment input (diamond).</text>
			<path class="viz-axis" d="M42 92 H652"></path>
			<g aria-label="Sixteen layers in four segments">
				<rect class="viz-node" x="42" y="73" width="27" height="38" rx="3"></rect><text class="viz-node-value" x="55.5" y="96">1</text>
				<rect class="viz-node" x="80" y="73" width="27" height="38" rx="3"></rect><text class="viz-node-value" x="93.5" y="96">2</text>
				<rect class="viz-node" x="118" y="73" width="27" height="38" rx="3"></rect><text class="viz-node-value" x="131.5" y="96">3</text>
				<rect class="viz-node" x="156" y="73" width="27" height="38" rx="3"></rect><text class="viz-node-value" x="169.5" y="96">4</text>
				<rect class="viz-node" x="194" y="73" width="27" height="38" rx="3"></rect><text class="viz-node-value" x="207.5" y="96">5</text>
				<rect class="viz-node" x="232" y="73" width="27" height="38" rx="3"></rect><text class="viz-node-value" x="245.5" y="96">6</text>
				<rect class="viz-node" x="270" y="73" width="27" height="38" rx="3"></rect><text class="viz-node-value" x="283.5" y="96">7</text>
				<rect class="viz-node" x="308" y="73" width="27" height="38" rx="3"></rect><text class="viz-node-value" x="321.5" y="96">8</text>
				<rect class="viz-node viz-node--focus" x="346" y="73" width="27" height="38" rx="3"></rect><text class="viz-node-value" x="359.5" y="96">9</text>
				<rect class="viz-node viz-node--focus" x="384" y="73" width="27" height="38" rx="3"></rect><text class="viz-node-value" x="397.5" y="96">10</text>
				<rect class="viz-node viz-node--focus" x="422" y="73" width="27" height="38" rx="3"></rect><text class="viz-node-value" x="435.5" y="96">11</text>
				<rect class="viz-node viz-node--focus" x="460" y="73" width="27" height="38" rx="3"></rect><text class="viz-node-value" x="473.5" y="96">12</text>
				<rect class="viz-node" x="498" y="73" width="27" height="38" rx="3"></rect><text class="viz-node-value" x="511.5" y="96">13</text>
				<rect class="viz-node" x="536" y="73" width="27" height="38" rx="3"></rect><text class="viz-node-value" x="549.5" y="96">14</text>
				<rect class="viz-node" x="574" y="73" width="27" height="38" rx="3"></rect><text class="viz-node-value" x="587.5" y="96">15</text>
				<rect class="viz-node" x="612" y="73" width="27" height="38" rx="3"></rect><text class="viz-node-value" x="625.5" y="96">16</text>
			</g>
			<g aria-label="Four saved checkpoint boundaries">
				<path class="viz-node--output" d="M35 92 l7 -7 l7 7 l-7 7 Z"></path>
				<path class="viz-node--output" d="M187 92 l7 -7 l7 7 l-7 7 Z"></path>
				<path class="viz-node--output" d="M339 92 l7 -7 l7 7 l-7 7 Z"></path>
				<path class="viz-node--output" d="M491 92 l7 -7 l7 7 l-7 7 Z"></path>
			</g>
			<path class="viz-operating-guide" d="M340 64 H493 V120 H340 Z"></path>
			<text class="viz-callout" x="416.5" y="139" text-anchor="middle">one segment rebuilt during backward</text>
			<text class="viz-axis-label" x="22" y="178">PEAK ACTIVATION MEMORY IN THIS SIMPLE CHAIN</text>
			<text class="viz-label" x="22" y="201">persistent boundaries</text>
			<g aria-label="Four persistent boundary checkpoints">
				<path class="viz-node--output" d="M55 231 l10 -10 l10 10 l-10 10 Z"></path>
				<path class="viz-node--output" d="M87 231 l10 -10 l10 10 l-10 10 Z"></path>
				<path class="viz-node--output" d="M119 231 l10 -10 l10 10 l-10 10 Z"></path>
				<path class="viz-node--output" d="M151 231 l10 -10 l10 10 l-10 10 Z"></path>
			</g>
			<text class="viz-callout" x="183" y="235">L/K = 4</text>
			<text class="viz-callout" x="255" y="235">+</text>
			<text class="viz-label" x="292" y="201">temporary rebuilt activations</text>
			<g aria-label="Four temporary recomputed activations">
				<rect class="viz-node viz-node--focus" x="304" y="218" width="26" height="26" rx="3"></rect>
				<rect class="viz-node viz-node--focus" x="338" y="218" width="26" height="26" rx="3"></rect>
				<rect class="viz-node viz-node--focus" x="372" y="218" width="26" height="26" rx="3"></rect>
				<rect class="viz-node viz-node--focus" x="406" y="218" width="26" height="26" rx="3"></rect>
			</g>
			<text class="viz-callout" x="452" y="235">K = 4</text>
			<text class="viz-callout" x="515" y="235">= 8 units</text>
			<path class="viz-baseline" d="M22 266 H678"></path>
			<text class="viz-label" x="22" y="291">smaller K → more saved boundaries</text>
			<text class="viz-label" x="678" y="291" text-anchor="end">larger K → larger rebuilt window</text>
			<text class="viz-callout" x="350" y="317" text-anchor="middle">Balance L/K ≈ K  ⇒  K ≈ √L</text>
		</svg>
	</div>
	<div class="visual-derivation-caption">at one backward step, the four diamond checkpoints remain live while four discarded activations are rebuilt inside the dashed segment. For this illustrative 16-layer chain, memory is 16/4 + 4 = 8 activation-sized units; shortening segments saves fewer temporary activations but keeps more boundaries, so the terms balance near K = √L.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="checkpoint-boundary-recompute-balance"><strong>Check:</strong> Why do both very small and very large segments increase this simplified peak-memory estimate?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="checkpoint-boundary-recompute-balance">In this 16-layer model, boundary storage is 16/s and rebuilt segment storage is s. Their sum is minimized at s=4.</span></figcaption>
</figure>

## When to use

- **Always** when training would OOM otherwise.
- **Selectively** for the most memory-intensive blocks (FFN > attention typically). Selective checkpointing recovers most memory at lower compute cost.
- **Less useful** when peak memory is dominated by optimizer state (use [FSDP / ZeRO](/concepts/fsdp-and-zero/) instead).
- **Less useful** at inference (no backward pass).

## Combined with other techniques

- **FSDP**: orthogonal. FSDP shards parameters / gradients / optimizer state; checkpointing reduces activation memory. Most large training runs use both.
- **Mixed precision**: orthogonal; checkpointing saves activations in whatever precision they were computed.
- **CPU offload**: offload activations to CPU memory instead of recomputing. Saves GPU memory at higher communication cost.

## Common pitfalls

- **Recomputing through randomness.** Forward passes with dropout or other stochastic ops must use the same RNG state at recomputation; PyTorch's checkpoint utility handles this with `preserve_rng_state=True` (default).
- **Checkpointing too aggressively.** Larger rematerialized regions save more boundary state and repeat more work. Profile selective, per-block, and larger-segment choices under the real memory limit.
- **Forgetting that the recomputation runs inside the backward graph.** Custom forward hooks may fire twice; gradients stay correct.
- **Trying to checkpoint inference.** Checkpointing only helps when there is a backward pass to run.

## Related

- [FSDP and ZeRO](/concepts/fsdp-and-zero/). For sharding optimizer state and parameters.
- [Mixed precision training](/concepts/mixed-precision-training/). Independent memory reduction.
- [Gradient accumulation](/concepts/gradient-accumulation/). Simulate larger batches without growing per-step activation memory.
