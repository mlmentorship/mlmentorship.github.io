---
title: "Debug this training loop"
description: "Trace gradient buffers and module mode, separate demonstrated defects from missing contracts, then execute a corrected CPU training fixture."
date: "2025-12-01"
draft: false
tags: ["questions"]
category: "questions"
---


Inspect a faulty training loop and distinguish demonstrable state-management defects from questions that require missing model or data information.

## Problem

**Context.** The supplied exercise alternates optimization and evaluation inside an ordinary PyTorch training loop.
Model architecture, target semantics, and device setup are not supplied.

**Task.** Trace gradients and module mode across consecutive batches, identify supported defects, and propose a corrected execution sequence.
State which additional facts are needed before changing the loss or moving tensors.

**Constraints.** Treat the code below as deliberately broken exercise input, not a runnable reference.
Use ordinary optimizer and module semantics; do not infer a classification task merely from variable names.

Inspect the supplied loop. Assume ordinary PyTorch module and optimizer behavior.
Identify persistent-state defects, then separate them from questions requiring missing model, data or device information.
The following is deliberately broken exercise code, not a runnable reference.

<span id="example-a-bug-laden-pytorch-loop"></span>

```python
import torch
import torch.nn as nn
import torch.optim as optim

model = MyModel()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for epoch in range(10):
    for x, y in dataloader:
        # forward
        pred = model(x)
        loss = criterion(pred, y)

        # backward
        loss.backward()
        optimizer.step()

        # eval inside training loop
        if epoch % 2 == 0:
            model.eval()
            val_pred = model(x_val)
            val_loss = criterion(val_pred, y_val)
            print(f"epoch {epoch} val loss: {val_loss}")
```

## Picture

`optimizer.step()` does not clear `.grad` buffers.
`model.eval()` persists until the mode is changed again.
Gradient recording is a third, separate setting: `no_grad()` does not restore training mode.

<!-- visual:training-loop-persistent-state -->
<figure class="learning-figure editorial-illustration" aria-labelledby="training-loop-persistent-state-edition-title"><p class="visual-title" id="training-loop-persistent-state-edition-title">Clearing gradients and restoring training mode are separate operations</p><div class="visual-argument" data-visual-argument="training-loop-persistent-state" data-argument-kind="matrix" data-argument-composition="cells"><svg viewBox="0 0 360 304" role="img" aria-label="Clearing gradients does not restore training mode. A training iteration needs both the intended model mode and a clean gradient buffer. Each action changes a separate persistent state."><title>Clearing gradients does not restore training mode</title><desc>A training iteration needs both the intended model mode and a clean gradient buffer. Each action changes a separate persistent state.</desc><text x="147.835" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="147.835" dy="0">mode</tspan></text><text x="275.94500000000005" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="275.94500000000005" dy="0">gradient</tspan></text><text x="71.78" y="54" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="71.78" dy="0">before</tspan></text><g data-proof-status="value"><rect data-proof-cell="0,0" x="83.78" y="36" width="123.11000000000001" height="59" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1"></rect><text x="145.335" y="71" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="145.335" dy="0">train</tspan></text></g><g data-proof-status="value"><rect data-proof-cell="0,1" x="211.89000000000001" y="36" width="123.11000000000001" height="59" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1"></rect><text x="273.44500000000005" y="71" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="273.44500000000005" dy="0">old</tspan></text></g><text x="71.78" y="118" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="71.78" dy="0">eval()</tspan></text><g data-proof-status="value"><rect data-proof-cell="1,0" x="83.78" y="100" width="123.11000000000001" height="59" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1"></rect><text x="145.335" y="135" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="145.335" dy="0">eval</tspan></text></g><g data-proof-status="value"><rect data-proof-cell="1,1" x="211.89000000000001" y="100" width="123.11000000000001" height="59" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1"></rect><text x="273.44500000000005" y="135" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="273.44500000000005" dy="0">old</tspan></text></g><text x="71.78" y="182" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="71.78" dy="0">zero()</tspan></text><g data-proof-status="value"><rect data-proof-cell="2,0" x="83.78" y="164" width="123.11000000000001" height="59" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1"></rect><text x="145.335" y="199" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="145.335" dy="0">eval</tspan></text><path data-proof-warning="2,0" d="M87.78 172v43" style="stroke:var(--viz-warning-stroke);stroke-width:3;stroke-dasharray:3 3"></path></g><g data-proof-status="value"><rect data-proof-cell="2,1" x="211.89000000000001" y="164" width="123.11000000000001" height="59" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></rect><text x="273.44500000000005" y="199" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="273.44500000000005" dy="0">zero</tspan></text><path d="M216.89000000000001 217L330 217" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path></g><text x="71.78" y="246" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="71.78" dy="0">train()</tspan></text><g data-proof-status="value"><rect data-proof-cell="3,0" x="83.78" y="228" width="123.11000000000001" height="59" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></rect><text x="145.335" y="263" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="145.335" dy="0">train</tspan></text><path d="M88.78 281L201.89000000000001 281" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path></g><g data-proof-status="value"><rect data-proof-cell="3,1" x="211.89000000000001" y="228" width="123.11000000000001" height="59" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></rect><text x="273.44500000000005" y="263" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="273.44500000000005" dy="0">zero</tspan></text><path d="M216.89000000000001 281L330 281" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path></g></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content"><svg viewBox="0 0 360 435" role="img" aria-labelledby="training-loop-persistent-state-edition-svg-title training-loop-persistent-state-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="training-loop-persistent-state-edition-svg-title">Gradients and train/eval mode persist independently</title><desc id="training-loop-persistent-state-edition-svg-desc">Aligned uncorrected and corrected checkpoints distinguish gradient storage from module mode. Without clearing, batch two adds g2 to stored g1. Explicit zero_grad clears that state first. Separately, eval mode persists until train is called. Clearing gradients does not restore training mode.</desc><text x="180" y="22" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Two independent kinds of persistent state</tspan></text><text x="90" y="51" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">UNCORRECTED</tspan></text><text x="270" y="51" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">EXPLICIT RESET</tspan></text><path d="M15 72L345 72" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="180" y="92" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Before batch 2 backward</tspan></text><g data-editorial-node><rect x="15" y="106" width="150" height="67" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="90" y="144.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">.grad = g1</tspan></text></g><g data-editorial-node><rect x="195" y="106" width="150" height="67" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.3"></rect><text x="270" y="135" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">zero_grad() -&gt;</tspan><tspan x="270" dy="19">None</tspan></text></g><path d="M15 188L345 188" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="180" y="208" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">After batch 2 backward</tspan></text><g data-editorial-node><rect x="15" y="222" width="150" height="67" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="90" y="260.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">.grad = g1 + g2</tspan></text></g><g data-editorial-node><rect x="195" y="222" width="150" height="67" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.3"></rect><text x="270" y="260.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">.grad = g2</tspan></text></g><path d="M15 304L345 304" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="180" y="324" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">After validation calls eval()</tspan></text><g data-editorial-node><rect x="15" y="338" width="150" height="67" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="90" y="376.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">still eval mode</tspan></text></g><g data-editorial-node><rect x="195" y="338" width="150" height="67" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.3"></rect><text x="270" y="367" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">train() -&gt; train</tspan><tspan x="270" dy="19">mode</tspan></text></g></svg><div class="visual-derivation-caption">zero_grad clears gradient storage; train restores module mode. They repair different persistent states. The corrected batch-two gradient excludes stale g1.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="training-loop-persistent-state"><strong>Check:</strong> Does eval clear gradients? Which separate calls repair the two retained-state mistakes?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="training-loop-persistent-state">A training iteration needs both the intended model mode and a clean gradient buffer. Each action changes a separate persistent state.</span></figcaption></figure>

## Calculate the stale-gradient effect

Use a scalar SGD calculation, not Adam: $w_0=0$, rate $0.1$, loss $(w-t)^2/2$.
The first target $t=1$ gives gradient $-1$ and $w_1=0.1$.
For target $t=2$, the new gradient is $-1.9$.

| Buffer before the second update | Used gradient | Next weight |
| --- | --- | --- |
| Cleared before backward | $-1.9$ | $0.29$ |
| Old $-1$ retained | $-2.9$ | $0.39$ |

Stale accumulation changes the update. It is not generally an N-times learning-rate multiplier, especially with Adam's moments.
Intentional accumulation needs its own reduction and reset boundary.

## Code with an explicit task

This complete CPU fixture uses two-class logits and integer class targets.
Its tiny split tests execution, not convergence or evaluation validity.

```python
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(0)
device = torch.device("cpu")
train_data = TensorDataset(
	torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]]),
	torch.tensor([0, 1, 1, 0]),
)
val_data = TensorDataset(
	torch.tensor([[0.2, 0.1], [0.8, 0.1]]),
	torch.tensor([0, 1]),
)
dataloader = DataLoader(train_data, batch_size=2, shuffle=False)
val_loader = DataLoader(val_data, batch_size=2, shuffle=False)
model = nn.Sequential(
	nn.Linear(2, 4), nn.ReLU(), nn.Dropout(0.2), nn.Linear(4, 2)
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
for epoch in range(2):
	model.train()
	for batch_inputs, batch_targets in dataloader:
		batch_inputs = batch_inputs.to(device)
		batch_targets = batch_targets.to(device)
		optimizer.zero_grad(set_to_none=True)
		loss = criterion(model(batch_inputs), batch_targets)
		loss.backward()
		optimizer.step()
	model.eval()
	val_sum, val_count = 0.0, 0
	with torch.no_grad():
		for batch_inputs, batch_targets in val_loader:
			batch_inputs = batch_inputs.to(device)
			batch_targets = batch_targets.to(device)
			val_sum += criterion(model(batch_inputs), batch_targets).item() * batch_targets.numel()
			val_count += batch_targets.numel()
	print(epoch, val_sum / val_count)
	model.train()
```

## Change the loop requirement

### 1. Validate between training updates

**New task:** run one diagnostic validation batch immediately after an update.

**Picture and change:** replace the epoch validation block with this excerpt after `optimizer.step()`.
It uses the fixture's existing model, device, loss and validation loader.

```python
validation_inputs, validation_targets = next(iter(val_loader))
validation_inputs = validation_inputs.to(device)
validation_targets = validation_targets.to(device)
previous_mode = model.training
model.eval()
try:
	with torch.no_grad():
		val_loss = criterion(model(validation_inputs), validation_targets)
finally:
	model.train(previous_mode)
```

**Check:** entered from training mode, it exits in training mode and `val_loss.requires_grad` is false.

**Use when:** more training forwards follow the diagnostic in the same epoch.

### 2. Bound measured gradient growth

**New task:** require a maximum gradient norm of one after observing finite gradient growth.

**Picture and change:** insert this between backward and the optimizer step, retaining gradient clearing.

```python
torch.nn.utils.clip_grad_norm_(
	model.parameters(), max_norm=1.0, error_if_nonfinite=True
)
```

**Check:** gradient $[3,4]$ has norm five and becomes approximately $[0.6,0.8]$.
Nonfinite norms raise an error.

**Use when:** evidence motivates clipping. Its absence alone is not a demonstrated defect.

## Do not invent the task from the loss name

MSE can be correct for regression with matching floating targets and outputs.
The original excerpt does not establish classification, validation overlap or a GPU requirement.
Check those contracts before changing them.

## Try separating recording from mode

Validation uses `no_grad()`, but the next training batch behaves differently. Which state must be inspected?

<details>
<summary>Check your answer</summary>

Inspect `model.training` and mode-sensitive layers such as dropout and BatchNorm.
`no_grad()` controls recording, not persistent module mode.

</details>

<!-- learning:core-end -->

<div data-learning-extension>

<span id="senior-debugging-procedure-the-actual-signal"></span>
<span id="what-an-l4-candidate-misses"></span>
<span id="what-an-l5-candidate-finds"></span>
<span id="what-an-l6-candidate-adds"></span>
<span id="tells-that-get-you-a-strong-hire-vote"></span>
<span id="tells-that-get-you-down-leveled"></span>

## Choose the next discriminating check

| Suspected defect | Cheap check | Possible consequence |
| --- | --- | --- |
| Broken gradient flow | Inspect `requires_grad`, parameter gradients and parameter changes | No learning or an incorrect update |
| Wrong loss contract | Check output shape, target dtype, reduction and API inputs | Crash or silently wrong objective |
| Mode or recording leak | Inspect module mode and validation graph state | Changed dropout/BatchNorm or unnecessary graph memory |
| Device mismatch | Compare parameter, input and target devices | Runtime error or an unintended CPU run |
| Unstable numerics | Locate the first nonfinite value and inspect gradient scale | Corrupted update; clipping may not repair the cause |

Missing validation variables can crash; a fixed batch may instead mislead evaluation.
Neither consequence follows without checking what those variables contain.
Repeated validation inside every batch also changes cost and selection exposure.

## Common follow-up

"You've fixed all the bugs. The model still isn't training. What now?"

If the repaired loop still fails, try overfitting one batch and inspect predictions and gradients.
Then test a controlled rate change; see [model debugging](/questions/debug-model-not-learning/).

API references: [zero_grad](https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html), [module modes](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html), [no_grad](https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html), and [gradient clipping](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html).

Related: [attention implementation](/questions/implement-attention-from-scratch/) and [learning-rate selection](/questions/how-to-choose-learning-rate/).

</div>
