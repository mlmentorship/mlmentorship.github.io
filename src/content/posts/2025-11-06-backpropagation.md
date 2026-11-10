---
title: "Backpropagation"
description: "Reverse-mode automatic differentiation applied to a computation graph. The algorithm that computes gradients for every parameter in one backward pass."
date: "2025-11-06"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Backpropagation accumulates a scalar loss's sensitivity through every recorded route into each parameter.
It applies reverse-mode automatic differentiation to a computation graph, reusing forward values needed by local derivative rules.
For standard operations with comparable derivative costs, one scalar-output reverse pass costs $O(F)$ for forward work $F$.
It computes gradients, not a parameter update or a guarantee of optimization progress.

## The algorithm

Execute operations in dependency order and retain the values their backward rules need.
Initialize all accumulated sensitivities to zero and seed $\overline L=1$.
Visit operations in reverse dependency order, adding each contribution rather than overwriting earlier contributions:

$$
\overline z\mathrel{+}=J_f(z)^\top\overline y,\qquad y=f(z).
$$

The [matrix-calculus pullback](/concepts/matrix-calculus/) explains this transposed-Jacobian rule.
Local reverse products usually avoid constructing the full Jacobian.

For $x=2$, $w=3$, let $z_1=xw=6$, $z_2=z_1+1=7$, and $L=z_2^2=49$.
The reverse chain gives $\overline z_2=14$, $\overline z_1=14$, $\overline x=14w=42$, and $\overline w=14x=28$.
These are the exact values and arrows in the diagram.

<!-- visual:backprop-forward-reverse-trace -->
<figure class="learning-figure backprop-visual" aria-labelledby="backprop-visual-title">
	<p class="visual-kicker">Spatial intuition</p>
	<p class="visual-title" id="backprop-visual-title">The same upstream gradient meets different cached multipliers</p><div class="visual-argument" data-visual-argument="backprop-forward-reverse-trace" data-argument-kind="ledger" data-argument-composition="arithmetic-paths"><svg viewBox="0 0 360 824" role="img" aria-label="The same upstream gradient meets different local multipliers. The backward signal is scaled by how strongly each input affected the forward prediction. The same incoming signal can therefore produce different gradients. A larger cached input strengthens the coefficient gradient."><title>The same upstream gradient meets different local multipliers</title><desc>The backward signal is scaled by how strongly each input affected the forward prediction. The same incoming signal can therefore produce different gradients. A larger cached input strengthens the coefficient gradient.</desc><g data-picture-arithmetic="0" data-picture-result="6" data-picture-signal-operation="multiply" data-picture-signal-scale="4.333333333333333"><text x="20" y="30" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="20" dy="0">Forward prediction</tspan></text><path d="M20 160.5L129 160.5V173.5L20 173.5Z" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.25"></path><path d="M129 160.5L244 154V180L129 173.5Z" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.25"></path><text x="52" y="119" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="52" dy="0">3</tspan></text><circle cx="184" cy="167" r="27" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="184" y="173" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="184" dy="0">2</tspan></text><text x="52" y="244" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="52" dy="0">incoming</tspan></text><text x="184" y="244" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="184" dy="0">sensitivity</tspan></text><rect data-picture-signal-output="6" x="244" y="154" width="96" height="26" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.25"></rect><text x="298" y="119" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-input-stroke)"><tspan x="298" dy="0">6</tspan></text><text x="298" y="244" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="298" dy="0">outgoing</tspan></text></g><path d="M20 278L340 278" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><g data-picture-arithmetic="1" data-picture-result="8" data-picture-signal-operation="multiply" data-picture-signal-scale="4.333333333333333"><text x="20" y="298" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-output-stroke)"><tspan x="20" dy="0">Gradient to the coefficient</tspan></text><path d="M20 426.3333333333333L129 426.3333333333333V443.6666666666667L20 443.6666666666667Z" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.25"></path><path d="M129 426.3333333333333L244 417.6666666666667V452.3333333333333L129 443.6666666666667Z" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.25"></path><text x="52" y="387" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="52" dy="0">4</tspan></text><circle cx="184" cy="435" r="27" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="184" y="441" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="184" dy="0">2</tspan></text><text x="52" y="512" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="52" dy="0">incoming</tspan></text><text x="184" y="512" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="184" dy="0">sensitivity</tspan></text><rect data-picture-signal-output="8" x="244" y="417.6666666666667" width="96" height="34.666666666666664" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.25"></rect><text x="298" y="387" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-output-stroke)"><tspan x="298" dy="0">8</tspan></text><text x="298" y="512" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="298" dy="0">outgoing</tspan></text></g><path d="M20 546L340 546" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><g data-picture-arithmetic="2" data-picture-result="12" data-picture-signal-operation="multiply" data-picture-signal-scale="4.333333333333333"><text x="20" y="566" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-output-stroke)"><tspan x="20" dy="0">Gradient to the input</tspan></text><path d="M20 694.3333333333334L129 694.3333333333334V711.6666666666666L20 711.6666666666666Z" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.25"></path><path d="M129 694.3333333333334L244 677V729L129 711.6666666666666Z" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.25"></path><text x="52" y="655" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="52" dy="0">4</tspan></text><circle cx="184" cy="703" r="27" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="184" y="709" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="184" dy="0">3</tspan></text><text x="52" y="780" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="52" dy="0">incoming</tspan></text><text x="184" y="780" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="184" dy="0">sensitivity</tspan></text><rect data-picture-signal-output="12" x="244" y="677" width="96" height="52" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.25"></rect><text x="298" y="655" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-output-stroke)"><tspan x="298" dy="0">12</tspan></text><text x="298" y="780" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="298" dy="0">outgoing</tspan></text></g></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
	<div class="visual-scroll">
		<svg viewBox="0 0 700 300" overflow="hidden" role="img" aria-labelledby="backprop-svg-title backprop-svg-desc">
			<title id="backprop-svg-title">Forward and backward passes through a scalar computation graph</title>
			<desc id="backprop-svg-desc">Inputs x equals 2 and w equals 3 flow through multiplication, addition, and squaring to produce loss 49. A reverse pass starts with loss gradient 1 and applies local derivatives to produce gradients 42 for x and 28 for w.</desc>
			<defs>
				<marker id="arrow-forward" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="viz-arrow-forward" d="M0,0 L8,4 L0,8 Z"></path></marker>
				<marker id="arrow-backward" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="viz-arrow-backward" d="M0,0 L8,4 L0,8 Z"></path></marker>
			</defs>
			<text class="viz-axis-label" x="24" y="20">FORWARD: compute and cache values</text>
			<path class="viz-forward" d="M102 74 C145 74 157 105 193 122"></path>
			<path class="viz-forward" d="M102 184 C145 184 157 155 193 138"></path>
			<path class="viz-forward" d="M247 130 H353"></path>
			<path class="viz-forward" d="M407 130 H513"></path>
			<text class="viz-edge-label" x="145" y="72">multiply</text>
			<text class="viz-edge-label" x="300" y="118">+ 1</text>
			<text class="viz-edge-label" x="460" y="118">square</text>
			<circle class="viz-node viz-node--input" cx="75" cy="74" r="27"></circle>
			<text class="viz-node-label" x="75" y="69">x</text><text class="viz-node-value" x="75" y="86">value 2</text>
			<circle class="viz-node viz-node--input" cx="75" cy="184" r="27"></circle>
			<text class="viz-node-label" x="75" y="179">w</text><text class="viz-node-value" x="75" y="196">value 3</text>
			<circle class="viz-node viz-node--focus" cx="220" cy="130" r="27"></circle>
			<text class="viz-node-label" x="220" y="125">z₁</text><text class="viz-node-value" x="220" y="142">value 6</text>
			<circle class="viz-node viz-node--focus" cx="380" cy="130" r="27"></circle>
			<text class="viz-node-label" x="380" y="125">z₂</text><text class="viz-node-value" x="380" y="142">value 7</text>
			<circle class="viz-node viz-node--output" cx="540" cy="130" r="27"></circle>
			<text class="viz-node-label" x="540" y="125">L</text><text class="viz-node-value" x="540" y="142">value 49</text>
			<text class="viz-axis-label" x="24" y="250">BACKWARD: upstream gradient × local derivative</text>
			<path class="viz-backward" d="M520 153 C485 204 445 204 401 153"></path>
			<path class="viz-backward" d="M359 153 C325 204 286 204 241 153"></path>
			<path class="viz-backward" d="M199 153 C165 219 126 223 94 203"></path>
			<path class="viz-backward" d="M193 142 C150 153 124 124 94 96"></path>
			<text class="viz-gradient-label" x="460" y="211">1 × 2z₂ = 14</text>
			<text class="viz-gradient-label" x="300" y="211">14 × 1 = 14</text>
			<text class="viz-gradient-label" x="139" y="235">14 × x = 28</text>
			<text class="viz-gradient-label" x="137" y="132">14 × w = 42</text>
			<text class="viz-node-gradient" x="75" y="36">∂L/∂x = 42</text>
			<text class="viz-node-gradient" x="75" y="225">∂L/∂w = 28</text>
			<text class="viz-node-gradient" x="220" y="93">∂L/∂z₁ = 14</text>
			<text class="viz-node-gradient" x="380" y="93">∂L/∂z₂ = 14</text>
			<text class="viz-node-gradient" x="540" y="93">seed = 1</text>
		</svg>
	</div>
	<div class="visual-derivation-caption"><strong>Read it this way:</strong> each reverse arrow multiplies one upstream gradient by one local derivative. The graph is traversed once in each direction; no full Jacobian is built.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="backprop-forward-reverse-trace"><strong>Check:</strong> Why does w receive gradient 28 while x receives 42?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="backprop-forward-reverse-trace">The backward signal is scaled by how strongly each input affected the forward prediction. The same incoming signal can therefore produce different gradients. A larger cached input strengthens the coefficient gradient. <strong>If x doubles while the upstream signal stays fixed, which parameter gradient doubles?</strong></span></figcaption>
</figure>

## Vector-Jacobian products (VJPs)

For $y=f(z)$, an incoming output gradient $g_y$ produces $g_z=J_f^\top g_y$.
Primitive rules combine to implement this vector-Jacobian product without materializing a network-wide Jacobian.

### Reuse an Intermediate

**New task:** Differentiate $t=xw$ and $L=t+t$.

**Picture and change:** Two outgoing routes contribute to the same intermediate; sum them before propagating through multiplication.

**Check:** $\overline t=1+1=2$, hence at $x=2,w=3$, gradients are $\overline x=6$, $\overline w=4$.
Overwriting one contribution would incorrectly halve both answers.

**Use when:** Parameters or intermediate results feed multiple operations.
Repeated references are multiple derivative contributions even when the graph node is stored once.

## Memory cost

**New task:** Compute the same gradient with less retained activation memory.

**Picture and change:** Save selected boundaries and recompute needed forward values during backward.

**Check:** Recomputing $z_1=6$ and $z_2=7$ in the original example still gives gradients $42,28$.

**Use when:** [Activation checkpointing](/concepts/activation-checkpointing/) can trade extra work for a lower memory peak.
Recomputation must reproduce required random choices and state-dependent behavior.
Storing lower-precision activations is a separate numerical trade-off, not exact checkpointing.
Total memory depends on retained tensors, lifetimes, parameters, optimizer state, and temporary workspace, not depth alone.

## Connection to reverse-mode autodiff

For a full Jacobian with $P$ inputs and $M$ outputs, forward mode needs roughly $P$ directional sweeps.
Reverse mode needs roughly $M$ output-seeded sweeps.
One scalar loss therefore favors reverse mode for many parameters; a few requested directional derivatives may favor forward mode.

## What backprop does NOT do

- It does not choose a learning rate, regularizer, or optimizer.
- It is not specific to neural networks; differentiable programs can use the same accumulation rule.
- It does not differentiate through unrecorded operations or automatically supply a derivative for discrete choices.
- It does not prove convergence; even an exact gradient can be used in a non-descending finite update.

<!-- learning:core-end -->
<div data-learning-extension>

## Common pitfalls

- **Unintended accumulation.** Framework parameter gradients may persist across backward calls; reset them at the intended optimization boundary.
- **Saved-value reuse.** Backward may release needed saved tensors; a second backward through the same graph can require explicit retention.
- **Detached routes.** `detach()` stops gradient flow through that route.
- **Retained graphs.** Holding differentiable loss objects can retain graph metadata or saved values; log detached scalars when appropriate.
- **Leaf confusion.** Requiring gradients and being a leaf are different properties; retained intermediate gradients need their own policy.

## Reconstruct

<details>
<summary>Why must a shared node wait until every downstream contribution has arrived?</summary>

Its sensitivity is the sum over downstream routes.
Propagating a partial sum and then discarding the node loses later routes; reverse dependency order prevents that mistake.

</details>

## Related

- [Matrix calculus](/concepts/matrix-calculus/). The chain rule formalism.
- [Activation checkpointing](/concepts/activation-checkpointing/). Memory optimization for backprop.

</div>
