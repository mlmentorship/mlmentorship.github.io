---
title: "Implement reverse-mode autodiff for scalar expressions"
description: "Build the computation graph, local derivatives, gradient accumulation, and reverse topological pass behind neural-network training."
date: "2026-07-11"
draft: false
tags: ["questions"]
category: "questions"
---

> Implement a scalar `Value` type supporting addition, multiplication, powers, tanh, and `backward()`.

Implement scalar automatic differentiation and verify gradients when an intermediate value contributes through more than one path.

## Problem

**Context.** A scalar `Value` stores a numeric result and participates in a finite directed acyclic computation graph.
Required operations are addition, multiplication, constant powers, tanh, and a scalar-output `backward()` call.

**Task.** Produce correct forward values and input gradients, including repeated operands and shared intermediate nodes.
For `loss = x * x + x` at `x = 2`, the loss is 6 and the gradient is 5.

**Constraints.** Accumulate every path's contribution rather than overwriting gradients.
State how gradients are initialized or reset between backward calls, and constrain powers to their differentiable numeric domain.
The operation snippets below illustrate parts of the implementation, not a complete general-purpose tensor engine.

## Build graph nodes during the forward pass

For $z = x + y$:

$$
\frac{\partial z}{\partial x} = 1, \qquad \frac{\partial z}{\partial y} = 1.
$$

The local backward function adds the upstream gradient:

```python
out = Value(self.data + other.data, (self, other), "+")

def _backward():
    self.grad += out.grad
    other.grad += out.grad

out._backward = _backward
```

Use `+=`, not assignment. A node can influence the output through multiple paths. For $y = x^2 + 2x$, both paths contribute to $\partial y / \partial x$.

<p class="visual-kicker">Learning objective</p>
<!-- visual:autograd-shared-node-accumulate-then-propagate -->
<figure class="learning-figure editorial-illustration" aria-labelledby="autograd-shared-node-accumulate-then-propagate-edition-title"><p class="visual-title" id="autograd-shared-node-accumulate-then-propagate-edition-title">Add both contributions at the shared node before propagating its gradient</p><div class="visual-argument" data-visual-argument="autograd-shared-node-accumulate-then-propagate" data-argument-kind="ledger"><p class="visual-argument-premise">Collect both contributions before propagating through a shared node</p><svg viewBox="0 0 360 322" role="img" aria-label="Collect both contributions before propagating through a shared node. At shared u=6, the two branches contribute 12 and 2. Add them to 14, then multiply by du/dx=3 to obtain 42."><title>Collect both contributions before propagating through a shared node</title><desc>At shared u=6, the two branches contribute 12 and 2. Add them to 14, then multiply by du/dx=3 to obtain 42.</desc><text x="20" y="28" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Branch contributions at u</tspan></text><rect data-proof-token="12" x="20" y="58" width="156" height="34" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke)"></rect><text x="98" y="79" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="98" dy="0">12</tspan></text><rect data-proof-token="2" x="180" y="58" width="156" height="34" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke)"></rect><text x="258" y="79" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="258" dy="0">2</tspan></text><text x="20" y="126" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Accumulate once</tspan></text><rect data-proof-token="12" x="20" y="156" width="60" height="34" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke)"></rect><text x="50" y="177" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="50" dy="0">12</tspan></text><rect data-proof-token="+" x="84" y="156" width="60" height="34" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke)"></rect><text x="114" y="177" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="114" dy="0">+</tspan></text><rect data-proof-token="2" x="148" y="156" width="60" height="34" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke)"></rect><text x="178" y="177" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="178" dy="0">2</tspan></text><rect data-proof-token="=" x="212" y="156" width="60" height="34" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke)"></rect><text x="242" y="177" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="242" dy="0">=</tspan></text><rect data-proof-token="14" x="276" y="156" width="60" height="34" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke)"></rect><text x="306" y="177" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="306" dy="0">14</tspan></text><text x="20" y="224" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Propagate through u=3x</tspan></text><rect data-proof-token="14" x="20" y="254" width="60" height="34" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke)"></rect><text x="50" y="275" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="50" dy="0">14</tspan></text><rect data-proof-token="x" x="84" y="254" width="60" height="34" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke)"></rect><text x="114" y="275" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="114" dy="0">x</tspan></text><rect data-proof-token="3" x="148" y="254" width="60" height="34" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke)"></rect><text x="178" y="275" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="178" dy="0">3</tspan></text><rect data-proof-token="=" x="212" y="254" width="60" height="34" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke)"></rect><text x="242" y="275" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="242" dy="0">=</tspan></text><rect data-proof-token="42" x="276" y="254" width="60" height="34" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke)"></rect><text x="306" y="275" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="306" dy="0">42</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content"><svg viewBox="0 0 360 430" role="img" aria-labelledby="autograd-shared-node-accumulate-then-propagate-edition-svg-title autograd-shared-node-accumulate-then-propagate-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="autograd-shared-node-accumulate-then-propagate-edition-svg-title">Two branches add at the shared node before its gradient moves upstream</title><desc id="autograd-shared-node-accumulate-then-propagate-edition-svg-desc">The forward DAG has x equal to 2, shared u equal to 3x or 6, a equal to u squared or 36, b equal to 2u or 12, and L equal to 48. Dashed reverse edges contribute 12 and 2 at the same u. Their total 14 is multiplied by 3 once to yield x gradient 42.</desc><g data-editorial-node><rect x="125" y="25" width="110" height="53" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="180" y="56.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">x = 2</tspan></text></g><path d="M180 78L180 105" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M176.64402122977057,98.8569220667674L180,105L183.35597877022943,98.8569220667674" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><g data-editorial-node><rect x="100" y="109" width="160" height="62" rx="3" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.3"></rect><text x="180" y="135.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">u = 3x = 6</tspan><tspan x="180" dy="19">shared node</tspan></text></g><path d="M125 171L90 217" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M91.0489836050095,210.07904389578857L90,217L96.39055200942751,214.14328072523708" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M235 171L270 217" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M263.6094479905725,214.14328072523708L270,217L268.9510163949905,210.07904389578857" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><g data-editorial-node><rect x="15" y="221" width="150" height="61" rx="3" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.3"></rect><text x="90" y="256.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">a = u^2 = 36</tspan></text></g><g data-editorial-node><rect x="195" y="221" width="150" height="61" rx="3" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.3"></rect><text x="270" y="256.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">b = 2u = 12</tspan></text></g><path d="M90 282L145 325" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M138.09341813757402,323.8602074848427L145,325L142.22746022427023,318.57247923441736" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M270 282L215 325" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M217.77253977572977,318.57247923441736L215,325L221.90658186242598,323.8602074848427" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><g data-editorial-node><rect x="95" y="329" width="170" height="53" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.3"></rect><text x="180" y="360.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">L = a + b = 48</tspan></text></g><path d="M40 221L40 140" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75;stroke-dasharray:5 4"></path><path d="M40 140L100 140" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path><path d="M93.8569220667674,143.35597877022943L100,140L93.8569220667674,136.64402122977057" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path><path d="M320 221L320 140" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75;stroke-dasharray:5 4"></path><path d="M320 140L260 140" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path><path d="M266.14307793323263,136.64402122977057L260,140L266.14307793323263,143.35597877022943" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path><text x="50" y="200" style="font-size:13px;font-weight:700;fill:var(--viz-focus-stroke);text-anchor:middle"><tspan x="50" dy="0">+12</tspan></text><text x="310" y="200" style="font-size:13px;font-weight:700;fill:var(--viz-focus-stroke);text-anchor:middle"><tspan x="310" dy="0">+2</tspan></text><path d="M100 130L85 130" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75;stroke-dasharray:5 4"></path><path d="M85 130L85 51" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75;stroke-dasharray:5 4"></path><path d="M85 51L125 51" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path><path d="M118.8569220667674,54.35597877022942L125,51L118.8569220667674,47.64402122977058" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path><text x="43" y="82" style="font-size:12px;font-weight:700;fill:var(--viz-focus-stroke);text-anchor:middle"><tspan x="43" dy="0">14 x 3</tspan><tspan x="43" dy="19">= 42</tspan></text><text x="180" y="406" style="font-size:12px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">At u: 12 + 2 = 14; propagate once: 14 x 3 = 42</tspan></text></svg><div class="visual-derivation-caption">Solid arrows carry forward values. Dashed return paths contribute 12 and 2 to the same u. Accumulate 14, then propagate 14 x 3 once.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="autograd-shared-node-accumulate-then-propagate"><strong>Check:</strong> Why must both contributions reach u before its accumulated gradient is propagated to x?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="autograd-shared-node-accumulate-then-propagate">At shared u=6, the two branches contribute 12 and 2. Add them to 14, then multiply by du/dx=3 to obtain 42.</span></figcaption></figure>

For $z = xy$:

```python
self.grad += other.data * out.grad
other.grad += self.data * out.grad
```

The local derivative uses forward values and the upstream gradient.

## Reverse topological order

Before applying local backward functions, topologically order reachable nodes so every child contributes to a node before that node propagates further:

```python
ordered = []
visited = set()

def build(node):
    if node in visited:
        return
    visited.add(node)
    for parent in node._previous:
        build(parent)
    ordered.append(node)

build(self)
self.grad = 1.0
for node in reversed(ordered):
    node._backward()
```

Seeding the final scalar with gradient 1 encodes $dL/dL = 1$.

## What an L4 answer sounds like

The candidate implements operations but calls parent backward functions recursively as soon as each path is encountered. Shared nodes are processed in an unsafe order, gradients are overwritten, or the output is never seeded.

## What an L5 answer adds

An L5 candidate builds the graph in the forward pass, captures local derivatives, accumulates gradients, and executes one reverse topological traversal. They test shared subexpressions, not only chains.

Useful tests:

- $x^2 + 2x$ at a known point;
- a small neuron with tanh;
- one variable used three times;
- finite-difference comparison away from non-smooth points;
- repeated `backward()` semantics, either documented accumulation or explicit zeroing.

They distinguish leaf nodes from intermediate nodes and can explain why reverse mode fits many parameters and one scalar loss.

## What an L6 answer adds

An L6 candidate identifies what the scalar engine omits:

- tensors and broadcasting require reducing gradients over expanded axes;
- in-place mutation can invalidate saved forward values;
- dynamic graphs need lifecycle and memory management;
- custom operations need vector-Jacobian products, not full Jacobians;
- higher-order gradients require the backward computation itself to remain differentiable;
- checkpointing trades saved intermediates for recomputation;
- non-differentiable operations need a defined surrogate, subgradient, or stop-gradient behavior;
- parallel execution needs dependency-aware scheduling and accumulation.

They do not try to implement all of this in the timed baseline. They state the boundary after producing correct code.

## Tells that get you a strong-hire vote

- Operations create graph nodes during forward execution.
- Local derivatives multiply the upstream gradient.
- Gradients accumulate with `+=`.
- A reverse topological pass handles shared nodes.
- The output gradient is seeded to 1.
- Tests include branching and finite differences.
- You connect local scalar derivatives to vector-Jacobian products in real frameworks.

## Tells that get you down-leveled

- Treating backprop as numerical differentiation.
- Overwriting gradients at shared nodes.
- Recursing without dependency order.
- Constructing explicit Jacobians for every operation.
- Ignoring saved forward values.
- Explaining PyTorch internals instead of completing the scalar engine.

## Common follow-up

"Why does broadcasting require gradient reduction?"

If a value of shape `[d]` broadcasts across a batch of shape `[b, d]`, one original element influences $b$ output elements. The backward pass receives a `[b, d]` gradient and must sum over the broadcast batch axis to return shape `[d]`. Expansion in forward becomes reduction in backward.

Use the [autograd starter](/prep/labs/implementation/) for the timed attempt.

*Related: [explain backprop](/questions/explain-backprop/), [matrix calculus](/concepts/matrix-calculus/), and [activation checkpointing](/concepts/activation-checkpointing/).*
