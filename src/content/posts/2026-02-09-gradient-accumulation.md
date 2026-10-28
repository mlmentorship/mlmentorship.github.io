---
title: "Gradient accumulation"
description: "Hold parameters fixed, weight gradients by examples or valid tokens, flush partial windows, and distinguish accumulation from changing optimizer or batch statistics."
date: "2026-02-09"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Gradient accumulation holds parameters fixed while backward calls add contributions to one update.
Use the denominator of the intended objective: total examples or valid targets, not automatically the configured number of micro-batches.

**Task:** four equal-sized micro-batches have scalar mean gradients $[2,4,6,8]$ at the same parameter value.
Reproduce their full-window mean with one optimizer update and only one micro-batch's live activation graph.
Assume the loss decomposes over targets, with no batch-coupled operation changing the forward computation.

An optimizer-update clock differs from the forward/backward clock; scheduler and momentum state must follow the intended update boundary.

## The mechanism

Clear the gradient buffer before the window.
Equal-size contributions divided by four are $[0.5,1,1.5,2]$; the buffer becomes $[0.5,1.5,3,5]$.
Consume gradient $5$ once, with no parameter changes during the four gradient evaluations.

For unequal micro-batch sizes $n_i$ and mean gradients $g_i$, the intended mean is

$$
g=\frac{\sum_i n_i g_i}{\sum_i n_i}.
$$

Sizes $[3,1]$ with gradients $[2,8]$ give $(3\cdot2+1\cdot8)/4=3.5$, not the batch-means average $5$.
The picture's equal quarters are therefore a special case.

<!-- visual:gradient-accumulation-state-boundary -->
<figure class="learning-figure plot-panel visual-wide" aria-labelledby="gradient-accumulation-title">
	<p class="visual-kicker">One accumulation window · K = 4</p>
	<p class="visual-title" id="gradient-accumulation-title">The gradient buffer changes four times; the parameters change once.</p>
	<div class="visual-scroll">
		<svg viewBox="0 0 760 354" role="img" aria-labelledby="gradient-accumulation-svg-title gradient-accumulation-svg-desc">
			<title id="gradient-accumulation-svg-title">Gradient and parameter state across four accumulated micro-batches</title>
			<desc id="gradient-accumulation-svg-desc">Four numbered micro-batches run from left to right using the same parameter value theta t. Each backward pass adds one quarter of its gradient to the gradient buffer, which progresses from g1 over 4 to the sum of g1 through g4 over 4. A vertical boundary follows the fourth pass. Only then does optimizer step change the parameters from theta t to theta t plus 1, after which zero grad clears the buffer.</desc>
			<defs>
				<marker id="arrow-forward" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="viz-arrow-forward" d="M0,0 L8,4 L0,8 Z"></path></marker>
				<marker id="arrow-backward" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="viz-arrow-backward" d="M0,0 L8,4 L0,8 Z"></path></marker>
			</defs>
			<text class="viz-axis-label" x="18" y="24">FORWARD + BACKWARD ON FOUR EQUAL-SIZED MICRO-BATCHES</text>
			<g aria-label="Four sequential micro-batches">
				<rect class="viz-node viz-node--input" x="18" y="43" width="126" height="48" rx="5"></rect><text class="viz-node-label" x="81" y="64">1 · micro-batch B1</text><text class="viz-node-value" x="81" y="81">loss / 4, then backward</text>
				<rect class="viz-node viz-node--input" x="167" y="43" width="126" height="48" rx="5"></rect><text class="viz-node-label" x="230" y="64">2 · micro-batch B2</text><text class="viz-node-value" x="230" y="81">loss / 4, then backward</text>
				<rect class="viz-node viz-node--input" x="316" y="43" width="126" height="48" rx="5"></rect><text class="viz-node-label" x="379" y="64">3 · micro-batch B3</text><text class="viz-node-value" x="379" y="81">loss / 4, then backward</text>
				<rect class="viz-node viz-node--input" x="465" y="43" width="126" height="48" rx="5"></rect><text class="viz-node-label" x="528" y="64">4 · micro-batch B4</text><text class="viz-node-value" x="528" y="81">loss / 4, then backward</text>
			</g>
			<g aria-label="Backward calls add to the gradient buffer">
				<path class="viz-backward" d="M81 92 V122"></path><path class="viz-backward" d="M230 92 V122"></path><path class="viz-backward" d="M379 92 V122"></path><path class="viz-backward" d="M528 92 V122"></path>
				<rect class="viz-node viz-node--focus" x="18" y="124" width="126" height="54" rx="5"></rect><text class="viz-node-value" x="81" y="145">.grad after backward</text><text class="viz-node-label" x="81" y="166">g1 / 4</text>
				<rect class="viz-node viz-node--focus" x="167" y="124" width="126" height="54" rx="5"></rect><text class="viz-node-value" x="230" y="145">.grad after backward</text><text class="viz-node-label" x="230" y="166">(g1 + g2) / 4</text>
				<rect class="viz-node viz-node--focus" x="316" y="124" width="126" height="54" rx="5"></rect><text class="viz-node-value" x="379" y="145">.grad after backward</text><text class="viz-node-label" x="379" y="166">(g1 + g2 + g3) / 4</text>
				<rect class="viz-node viz-node--focus" x="465" y="124" width="126" height="54" rx="5"></rect><text class="viz-node-value" x="528" y="145">window mean gradient</text><text class="viz-node-label" x="528" y="166" style="font-size:12px">(g1 + g2 + g3 + g4) / 4</text>
			</g>
			<path class="viz-operating-guide" d="M607 34 V326"></path>
			<text class="viz-axis-label" x="617" y="49">BOUNDARY</text>
			<rect class="viz-node viz-node--output" x="617" y="68" width="125" height="54" rx="5"></rect><text class="viz-node-label" x="679.5" y="90">optimizer.step()</text><text class="viz-node-value" x="679.5" y="108">consume average once</text>
			<path class="viz-forward" d="M591 151 H607 V95 H615"></path>
			<text class="viz-axis-label" x="18" y="215">PARAMETER STATE SEEN BY EACH FORWARD PASS</text>
			<g aria-label="All four passes use unchanged parameters theta t">
				<rect class="viz-node" x="18" y="230" width="126" height="42" rx="5"></rect><text class="viz-node-label" x="81" y="256">parameters &theta;<tspan baseline-shift="sub" font-size="9">t</tspan></text>
				<rect class="viz-node" x="167" y="230" width="126" height="42" rx="5"></rect><text class="viz-node-label" x="230" y="256">parameters &theta;<tspan baseline-shift="sub" font-size="9">t</tspan></text>
				<rect class="viz-node" x="316" y="230" width="126" height="42" rx="5"></rect><text class="viz-node-label" x="379" y="256">parameters &theta;<tspan baseline-shift="sub" font-size="9">t</tspan></text>
				<rect class="viz-node" x="465" y="230" width="126" height="42" rx="5"></rect><text class="viz-node-label" x="528" y="256">parameters &theta;<tspan baseline-shift="sub" font-size="9">t</tspan></text>
			</g>
			<path class="viz-baseline" d="M144 251 H167 M293 251 H316 M442 251 H465"></path>
			<path class="viz-forward" d="M591 251 H615"></path>
			<rect class="viz-node viz-node--output" x="617" y="230" width="125" height="42" rx="5"></rect><text class="viz-node-label" x="679.5" y="256">parameters &theta;<tspan baseline-shift="sub" font-size="9">t+1</tspan></text>
			<path class="viz-baseline" d="M18 298 H591"></path>
			<text class="viz-callout" x="304.5" y="293" text-anchor="middle">no optimizer step · parameters stay fixed</text>
			<text class="viz-label" x="18" y="323">zero_grad() starts the window</text>
			<text class="viz-label" x="742" y="323" text-anchor="end">zero_grad() clears .grad for the next window</text>
			<text class="viz-callout" x="380" y="347" text-anchor="middle">effective batch = 4 micro-batches · parameter updates = 1</text>
		</svg>
	</div>
	<figcaption><strong>Read it this way:</strong> the gradient buffer changes after each backward call; every forward still uses the same parameters. Only the completed mean crosses the optimizer boundary. Clear the buffer when beginning the next window. Four equal-sized micro-batches justify the displayed quarters; unequal sizes require target-count weighting. <span class="visual-reader-check" data-reader-check="gradient-accumulation-state-boundary"><strong>Check:</strong> After the third backward pass, what has changed? What still equals the initial parameter state?</span></figcaption>
</figure>

### A complete weighted step

Here each input is a batch-by-one tensor; targets are one-dimensional int64 class IDs zero or one, with no ignored entries.
The model returns batch-by-two logits.
Summed cross-entropy divided by the whole window count makes every target contribute equally.

```python
import torch
from torch.nn import functional

def accumulated_step(model, optimizer, batches):
	batches = tuple(batches)
	count = sum(targets.numel() for inputs, targets in batches)
	if count == 0:
		raise ValueError('window must contain targets')
	optimizer.zero_grad(set_to_none=True)
	for inputs, targets in batches:
		loss = functional.cross_entropy(model(inputs), targets, reduction='sum')
		(loss / count).backward()
	optimizer.step()

model = torch.nn.Linear(1, 2, bias=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
batches = [(torch.tensor([[1.], [2.]]), torch.tensor([0, 1])), (torch.tensor([[3.]]), torch.tensor([1]))]
accumulated_step(model, optimizer, batches)
```

Materializing the window retains its input batches, not all their activation graphs.
Each call clears before accumulating; the next call must not reuse the previous gradient buffer.

### Change the window length

**New task:** three loader batches arrive with `accumulation_steps = 2`.
Replace the per-batch driver below; retain the function, model, optimizer, and dataloader.

```python
from itertools import islice
iterator = iter(dataloader)
while True:
	window = list(islice(iterator, accumulation_steps))
	if not window:
		break
	accumulated_step(model, optimizer, window)
```

**Check:** two batches form the first update; the final batch forms a second, using its own actual count.
Its loss is not divided by two merely because the configured window length is two.
**Use when:** the loader length is not divisible by the accumulation setting.

### Change which targets count

**New task:** ignore targets labeled `-100`.
Inside the function, replace its count and loss statements with these two nonadjacent replacements:

```python
count = sum(int((targets != -100).sum()) for inputs, targets in batches)
loss = functional.cross_entropy(model(inputs), targets, ignore_index=-100, reduction='sum')
```

Retain empty-count rejection, zeroing, normalization, backward, and the final step.
**Check:** three valid targets in one micro-batch and one in another receive weights $3/4$ and $1/4$.
An all-ignored window raises before stepping.
**Use when:** padding or supervision masks alter loss-bearing counts.

**Failure:** BatchNorm uses different statistics on separate forwards than on their concatenation.
Correct denominators cannot restore that lost equivalence.
SyncBatchNorm synchronizes concurrent ranks, not earlier serial micro-batches.

**Reconstruct:** eight devices each process four micro-batches of sixteen examples. Is the effective batch necessarily $512$?

<details><summary>Answer</summary>

The arithmetic gives $8\cdot4\cdot16=512$, but verify each example contributes once to one synchronized mean and update.
Specify rank reductions, tails, and batch-coupled operations before claiming equivalence.

</details>

<!-- learning:core-end -->
<div data-learning-extension>

## Cost model

Activation residency can stay near one micro-batch when graphs are released after backward.
Optimizer state and gradient buffers remain; retaining computation graphs defeats the intended activation saving.
Serial micro-batches require multiple forwards/backwards per update and may underutilize the device.
Measure throughput rather than assuming accumulation is faster or cheaper than distributed execution.

## Combined with other techniques

DDP `no_sync()` encloses forward and backward on intermediate micro-batches when that API is supported.
One final synchronization does not automatically normalize unequal valid counts across ranks; account for the reduction's averaging factor.
FSDP synchronization and memory behavior depend on its API and sharding strategy.

Under AMP, hold loss scale constant across the window, then unscale and clip the completed gradient once.
Advance optimizer and scheduler at their intended successful-update boundaries.
Checkpointing trades recomputation for retained activations independently of accumulation.

## Common pitfalls

Stepping per micro-batch changes parameter and optimizer trajectories.
A gradient scaled by $K$ is not universally equivalent to scaling learning rate by $K$, particularly with Adam or clipping.
Dropout masks, batch coupling, precision, and reduction order can also prevent exact full-batch equality.

The original sketches below are intentionally incomplete contrasts, not runnable classifier references.
They assume `model(batch)` returns a loss; the second also assumes equal counts and drops an unfinished tail.
Both omit an explicit initial buffer clear.

```
for batch in dataloader:
	loss = model(batch)
	loss.backward()
	optimizer.step()
	optimizer.zero_grad()
```

```
for step, batch in enumerate(dataloader):
	loss = model(batch) / accumulation_steps   # average over K micro-batches
	loss.backward()                             # gradients accumulate in .grad
	if (step + 1) % accumulation_steps == 0:
		optimizer.step()
		optimizer.zero_grad()
```

## Related

- [Activation checkpointing](/concepts/activation-checkpointing/). Independent memory reduction.
- [FSDP and ZeRO](/concepts/fsdp-and-zero/). Distributed memory reduction.
- [Mixed precision training](/concepts/mixed-precision-training/). Orthogonal memory reduction.

</div>
