---
title: "Explain backprop in your own words"
description: "Backprop is reverse-mode automatic differentiation. It reuses forward-pass values to compute all parameter gradients at roughly one additional forward-pass cost."
date: "2026-04-13"
draft: false
tags: ["questions"]
category: "questions"
---


> *Asked in: ML breadth at every level.*

Explain backpropagation as a computation over a graph, including how gradients from multiple uses of a value combine.

## Problem

**Context.** A differentiable computation produces a scalar loss from parameters and intermediate values that may feed several later operations.
For a small illustration, take `loss = x * x + x` at `x = 2`.

**Task.** Trace the forward values and reverse derivatives, explain gradient accumulation, and relate the procedure to reverse-mode automatic differentiation.
Discuss computational cost, saved intermediates, and common ways gradient flow is interrupted.

**Constraints.** Distinguish the derivative calculation from the optimizer update.
Account for every path to a reused value, and state conventions for nondifferentiable operations rather than assuming all graphs are smooth.

## What an L4 answer sounds like

> "Backprop computes gradients of the loss with respect to every weight using the chain rule. Starting from the output, you propagate the error backwards through each layer."

Correct, but generic. You've memorized the textbook line, not internalized the algorithm.

## What an L5 answer sounds like

> "Backprop is reverse-mode automatic differentiation applied to the computation graph of a neural network.
>
> The forward pass computes the loss while caching intermediate activations. The backward pass starts at the loss with gradient 1, then walks backward through the cached graph. At each operation, it multiplies the upstream gradient by the local Jacobian of that op with respect to its inputs (the chain rule), and accumulates gradients at parameters.
>
> The key efficiency insight: reverse-mode evaluates one forward and one backward pass per output. A naive 'compute gradient by changing each parameter' approach would cost O(P) forward passes for P parameters. Reverse-mode is O(1) forward + O(1) backward to get *all* P gradients."

<!-- visual:explain-backprop-one-loss-all-gradients -->
<figure class="learning-figure backprop-visual" aria-labelledby="explain-backprop-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="explain-backprop-title">Save the forward graph; sum every reverse route into a reused input</p><div class="visual-argument" data-visual-argument="explain-backprop-one-loss-all-gradients" data-argument-kind="connections" data-argument-composition="reverse-fanout"><svg viewBox="0 0 360 1680" role="img" aria-label="Save the forward graph; sum every reverse route into a reused input. Forward reuses input two twice in multiplication and once in addition, saving product four and loss six. One reverse seed flows through these saved operations. Multiplication contributes two for each use; addition contributes one. All three routes meet at the same input, giving gradient five. One reverse sweep computes the scalar loss’s gradients; it does not apply an optimizer update."><title>Save the forward graph; sum every reverse route into a reused input</title><desc>Forward reuses input two twice in multiplication and once in addition, saving product four and loss six. One reverse seed flows through these saved operations. Multiplication contributes two for each use; addition contributes one. All three routes meet at the same input, giving gradient five. One reverse sweep computes the scalar loss’s gradients; it does not apply an optimizer update.</desc><text x="20" y="27" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Forward: save each value and each use</tspan></text><text x="20" y="90" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="20" dy="0">One shared input, value two</tspan></text><circle cx="180" cy="151" r="28" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="180" y="157" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="180" dy="0">2</tspan></text><path d="M153 166L77 312M77.14581114094358 303.0569166887937L77 312L84.2419533335312 306.7507989260311" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:3;stroke-linejoin:round"></path><path d="M177 182L121 310M120.54192133254891 301.0674659847037L121 310L127.87118001176641 304.2740166568613" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:3;stroke-linejoin:round"></path><text x="20" y="250" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="20" dy="0">2</tspan></text><text x="177" y="268" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="177" dy="0">2</tspan></text><path d="M212 151L305 151" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><path d="M305 151L305 526" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><path d="M305 526L215 526M223 522L215 526L223 530" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:3;stroke-linejoin:round"></path><text x="260" y="440" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-input-stroke)"><tspan x="260" dy="0">2</tspan></text><circle cx="100" cy="345" r="32" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.75"></circle><text x="100" y="351" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="100" dy="0">times</tspan></text><path d="M132 368L165 491M159.06360012236252 484.3097715664721L165 491L166.79034282023986 482.23674303777324" style="fill:none;stroke:var(--viz-state-stroke);stroke-width:3;stroke-linejoin:round"></path><text x="20" y="430" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-state-stroke)"><tspan x="20" dy="0">Product 4</tspan><tspan x="20" dy="18">saved</tspan></text><circle cx="180" cy="526" r="30" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="180" y="532" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="180" dy="0">add</tspan></text><path d="M180 560L180 633M176 625L180 633L184 625" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:3;stroke-linejoin:round"></path><circle cx="180" cy="667" r="30" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.75"></circle><text x="180" y="673" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="180" dy="0">loss</tspan></text><text x="245" y="673" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-focus-stroke)"><tspan x="245" dy="0">6 saved</tspan></text><text x="20" y="785" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Reverse: one seed, all routes</tspan><tspan x="20" dy="18">accumulate</tspan></text><circle cx="180" cy="887" r="30" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.75"></circle><text x="180" y="893" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="180" dy="0">loss</tspan></text><text x="20" y="893" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-focus-stroke)"><tspan x="20" dy="0">Seed 1</tspan></text><path d="M180 921L180 1003M176 995L180 1003L184 995" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:3;stroke-linejoin:round"></path><circle cx="180" cy="1037" r="30" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="180" y="1043" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="180" dy="0">add</tspan></text><path d="M161 1063L115 1169M114.5153648997983 1160.0688674391401L115 1169L121.85412498856692 1163.2536123833227" style="fill:none;stroke:var(--viz-state-stroke);stroke-width:3;stroke-linejoin:round"></path><text x="20" y="1109" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-state-stroke)"><tspan x="20" dy="0">1</tspan></text><circle cx="100" cy="1205" r="32" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.75"></circle><text x="100" y="1211" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="100" dy="0">times</tspan></text><path d="M212 1037L305 1037" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75"></path><path d="M305 1037L305 1422" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75"></path><path d="M305 1422L215 1422M223 1418L215 1422L223 1426" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:3;stroke-linejoin:round"></path><text x="260" y="1320" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-output-stroke)"><tspan x="260" dy="0">1</tspan></text><path d="M78 1232L151 1398M144.11798424924245 1392.2870446171596L151 1398L151.44115485581779 1389.0666141696897" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:3;stroke-linejoin:round"></path><path d="M109 1241L178 1387M170.9652362359756 1381.4762242275442L178 1387L178.19816235954997 1378.0579235253072" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:3;stroke-linejoin:round"></path><text x="45" y="1314" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-input-stroke)"><tspan x="45" dy="0">2</tspan></text><text x="205" y="1341" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-input-stroke)"><tspan x="205" dy="0">2</tspan></text><circle cx="180" cy="1422" r="30" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="180" y="1428" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="180" dy="0">input</tspan></text><text x="20" y="1518" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="20" dy="0">All three uses reach this same input.</tspan><tspan x="20" dy="18">Total gradient: five.</tspan></text><text x="20" y="1612" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">The optimizer update is a separate</tspan><tspan x="20" dy="18">operation.</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
	<div class="visual-scroll">
		<svg viewBox="0 0 700 320" role="img" aria-labelledby="explain-backprop-svg-title explain-backprop-svg-desc">
			<title id="explain-backprop-svg-title">One forward pass and one reverse sweep through a three-operation graph</title>
			<desc id="explain-backprop-svg-desc">A solid path carries input x through operations f1, f2, and f3 to scalar loss L. Parameters theta 1, theta 2, and theta 3 each enter one operation. Activations a1 and a2 are explicitly saved during the forward pass. A dashed reverse path starts at dL over dL equals 1, moves left through all three operations using the saved activations, and branches to produce dL over d theta 3, dL over d theta 2, and dL over d theta 1 in the same sweep.</desc>
			<defs>
				<marker id="arrow-forward" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="viz-arrow-forward" d="M0,0 L8,4 L0,8 Z"></path></marker>
				<marker id="arrow-backward" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="viz-arrow-backward" d="M0,0 L8,4 L0,8 Z"></path></marker>
			</defs>
			<text class="viz-axis-label" x="18" y="22">FORWARD · COMPUTE ONCE AND SAVE WHAT LOCAL DERIVATIVES NEED</text>
			<rect class="viz-node viz-node--input" x="18" y="92" width="58" height="44" rx="5"></rect>
			<text class="viz-node-label" x="47" y="119">x</text>
			<rect class="viz-node viz-node--focus" x="122" y="92" width="58" height="44" rx="5"></rect>
			<text class="viz-node-label" x="151" y="111">f₁</text><text class="viz-node-value" x="151" y="127">uses θ₁</text>
			<rect class="viz-node viz-node--state" x="226" y="92" width="66" height="44" rx="12"></rect>
			<text class="viz-node-label" x="259" y="111">a₁</text><text class="viz-node-value" x="259" y="127">saved</text>
			<rect class="viz-node viz-node--focus" x="338" y="92" width="58" height="44" rx="5"></rect>
			<text class="viz-node-label" x="367" y="111">f₂</text><text class="viz-node-value" x="367" y="127">uses θ₂</text>
			<rect class="viz-node viz-node--state" x="442" y="92" width="66" height="44" rx="12"></rect>
			<text class="viz-node-label" x="475" y="111">a₂</text><text class="viz-node-value" x="475" y="127">saved</text>
			<rect class="viz-node viz-node--focus" x="554" y="92" width="58" height="44" rx="5"></rect>
			<text class="viz-node-label" x="583" y="111">f₃</text><text class="viz-node-value" x="583" y="127">uses θ₃</text>
			<rect class="viz-node viz-node--output" x="642" y="92" width="40" height="44" rx="20"></rect>
			<text class="viz-node-label" x="662" y="119">L</text>
			<path class="viz-forward" d="M76 114H121M180 114H225M292 114H337M396 114H441M508 114H553M612 114H641"></path>
			<text class="viz-axis-label" x="18" y="177">BACKWARD · ONE REVERSE SWEEP, NOT ONE SWEEP PER PARAMETER</text>
			<path class="viz-backward" d="M662 145V205H584"></path>
			<path class="viz-backward" d="M554 205H368"></path>
			<path class="viz-backward" d="M338 205H152"></path>
			<path class="viz-backward" d="M122 205H48"></path>
			<text class="viz-gradient-label" x="642" y="194">seed ∂L/∂L = 1</text>
			<text class="viz-gradient-label" x="475" y="194">VJP uses saved a₂</text>
			<text class="viz-gradient-label" x="259" y="194">VJP uses saved a₁</text>
			<path class="viz-backward" d="M583 205V247"></path>
			<path class="viz-backward" d="M367 205V247"></path>
			<path class="viz-backward" d="M151 205V247"></path>
			<rect class="viz-node viz-node--output" x="501" y="248" width="164" height="46" rx="5"></rect>
			<text class="viz-node-label" x="583" y="267">∂L/∂θ₃</text><text class="viz-node-value" x="583" y="284">accumulate at parameter</text>
			<rect class="viz-node viz-node--output" x="285" y="248" width="164" height="46" rx="5"></rect>
			<text class="viz-node-label" x="367" y="267">∂L/∂θ₂</text><text class="viz-node-value" x="367" y="284">accumulate at parameter</text>
			<rect class="viz-node viz-node--output" x="69" y="248" width="164" height="46" rx="5"></rect>
			<text class="viz-node-label" x="151" y="267">∂L/∂θ₁</text><text class="viz-node-value" x="151" y="284">accumulate at parameter</text>
		</svg>
	</div>
	<div class="visual-derivation-caption">follow the solid arrows once to compute the scalar loss and save a₁ and a₂. Then start at ∂L/∂L = 1 and follow the dashed arrows left: each local vector-Jacobian product reuses a saved value and emits that operation's parameter gradient. The reverse path is traversed once, not once per θ. Original schematic checked against the <a href="https://jmlr.org/papers/v18/17-468.html">automatic-differentiation survey</a> and <a href="https://docs.pytorch.org/docs/stable/notes/autograd.html">PyTorch autograd mechanics</a>.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="explain-backprop-one-loss-all-gradients"><strong>Check:</strong> Why does adding more parameters inside these operations not require another full reverse sweep per parameter?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="explain-backprop-one-loss-all-gradients">Forward reuses input two twice in multiplication and once in addition, saving product four and loss six. One reverse seed flows through these saved operations. Multiplication contributes two for each use; addition contributes one. All three routes meet at the same input, giving gradient five. One reverse sweep computes the scalar loss’s gradients; it does not apply an optimizer update. <strong>If the input becomes three, which three contributions make its gradient seven?</strong></span></figcaption>
</figure>

This is L5. You've explained it as auto-differentiation, mentioned the activation caching, and quantified the efficiency win.

## What an L6 answer sounds like

> "...a few practical things worth adding:
>
> **Memory is the cost.** The backward pass needs the cached activations from the forward pass. For deep networks with large activations, this dominates GPU memory. Activation checkpointing trades compute for memory by recomputing activations during backward instead of storing them.
>
> **Backprop is exact, not approximate.** Unlike numerical differentiation, it has no truncation error. The only error is floating-point.
>
> **It composes through any differentiable op.** This is why frameworks (PyTorch, JAX) implement autograd as a graph of primitive ops with known local Jacobians. Custom ops just need to define forward and the local Jacobian (vector-Jacobian product); the framework composes the rest.
>
> **What breaks it in practice**: non-differentiable ops (argmax, hard threshold), vanishing gradients in deep networks (ReLU, residuals, normalization mitigate), exploding gradients (gradient clipping mitigates), and detached tensors silently breaking gradient flow."

## Tells that get you a strong-hire vote

- You frame it as **reverse-mode auto-diff**, not just "the chain rule."
- You mention **activation caching** and the memory cost.
- You connect to **activation checkpointing** as the standard memory-compute trade.
- You distinguish backprop (exact) from numerical differentiation.
- You mention **detached tensors** as a common bug source.

## Tells that get you down-leveled

- "It's the chain rule" with no further detail.
- Confusion about which gradients flow where (e.g., gradients flow through cached activations, not just weights).
- No awareness of the memory cost.
- Calling it "back-propagation of errors" without explaining what's actually being computed.

## Common follow-up

"Why is reverse-mode preferred over forward-mode auto-diff for neural networks?"

The L6 answer:

> "Forward-mode computes the derivative of *one input* with respect to *all outputs* in one pass. Reverse-mode computes the derivative of *one output* (the loss) with respect to *all inputs* (parameters) in one pass. Neural network training has many parameters and one scalar loss, so reverse-mode is the right shape. For small models with many outputs and few inputs, forward-mode can be faster."

---

*Related: [Adam and AdamW](/concepts/adam-and-adamw/) and [debug a model that is not learning](/questions/debug-model-not-learning/).*
