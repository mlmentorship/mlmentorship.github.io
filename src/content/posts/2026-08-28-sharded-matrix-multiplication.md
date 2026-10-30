---
title: "Sharded matrix multiplication"
description: "Predict which collective communication a distributed matrix multiplication needs from its sharded axes."
date: "2026-08-28"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A sharded matrix multiplication splits matrix data across devices. The required communication follows from which matrix axes are split and which axis is summed during the multiplication.

## Why AI labs care

Transformer training is mostly matrix multiplication. Data, tensor, fully sharded, context, and expert parallelism all change where matrix pieces live.

A useful systems skill is predicting:

- which local multiplication each device can perform;
- whether the result is complete or only a partial sum;
- which collective communication completes the operation;
- how much data moves;
- whether compute can hide that communication.

## Global and local shapes

Suppose:

$$
A[I,J] B[J,K] = C[I,K].
$$

$J$ is the **contracting axis** because it appears in both inputs and is summed away. $I$ and $K$ are output axes.

A global shape describes the full matrix. A local shape describes the piece held by one device.

If $I$ is split across four devices, each device holds about one quarter of the rows of $A$ and $C$.

## A simple sharding notation

Write $A[I_X,J]$ when the $I$ axis is split across device-mesh axis $X$.

Examples:

- $A[I,J]$: replicated on $X$;
- $A[I_X,J]$: rows split across $X$;
- $A[I,J_X]$: columns split across $X$.

The notation is framework-neutral. A device mesh is only a named grid of devices.

## Case 1: the contracting axis is not split

$$
A[I_X,J] B[J,K_Y] \rightarrow C[I_X,K_Y].
$$

Each device has all values needed for its output block. It performs a local matrix multiplication. No communication is needed if this output layout is acceptable.

Communication may still be needed if the next operation requires a different layout.

## Case 2: one input is split on the contracting axis

$$
A[I,J_X] B[J,K] \rightarrow C[I,K].
$$

One device has only part of the $J$ values from $A$.

Two common choices are:

1. All-gather $A$ along $X$, then perform the full local multiplication.
2. Multiply local pieces, then reduce the partial outputs.

Choose by comparing the bytes in the gathered input with the bytes in the output reduction. The next operation's required layout also affects the choice.

## Case 3: both inputs share the split contracting axis

$$
A[I,J_X] B[J_X,K] \rightarrow C[I,K].
$$

Each device can multiply matching local pieces. Its result is only a partial sum over $J$.

Finish with:

- **all-reduce** if every device needs the full output;
- **reduce-scatter** if the output should remain split.

Keeping the output split often saves memory and avoids a later reshard.

## Case 4: two output axes use the same mesh axis

$$
A[I_X,J] B[J,K_X] \rightarrow C[I_X,K_X].
$$

This output layout is invalid. One mesh axis cannot independently select both output axes. The devices would hold only diagonal blocks of the full output.

With two devices, device 0 would compute the first row block with the first column block, while device 1 would compute the second row block with the second column block. No device computes the two off-diagonal blocks.

<!-- visual:same-mesh-axis-diagonal-gap -->
<figure class="learning-figure editorial-illustration" data-illustration-print-page aria-labelledby="same-mesh-axis-diagonal-gap-edition-title"><span class="visual-fragment-anchor" id="same-axis-gap-heading" aria-hidden="true"></span><span class="visual-fragment-anchor" id="same-axis-gap-title" aria-hidden="true"></span><span class="visual-fragment-anchor" id="same-axis-gap-desc" aria-hidden="true"></span><p class="visual-title" id="same-mesh-axis-diagonal-gap-edition-title">Adding diagonal blocks cannot create cross-block products that were never computed</p><div class="visual-argument" data-visual-argument="same-mesh-axis-diagonal-gap" data-argument-kind="matrix"><p class="visual-argument-premise">Diagonal products cannot manufacture the missing cross-block terms</p><svg viewBox="0 0 360 142" role="img" aria-label="Diagonal products cannot manufacture the missing cross-block terms. Computing A0B0 and A1B1 omits A0B1 and A1B0. Adding the two available blocks cannot reconstruct terms that were never computed."><title>Diagonal products cannot manufacture the missing cross-block terms</title><desc>Computing A0B0 and A1B1 omits A0B1 and A1B0. Adding the two available blocks cannot reconstruct terms that were never computed.</desc><text x="128.5" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="128.5" dy="0">B0</tspan></text><text x="269.5" y="18" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="269.5" dy="0">B1</tspan></text><text x="46" y="54" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="46" dy="0">A0</tspan></text><rect data-proof-cell="0,0" x="58" y="36" width="138" height="44" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke)"></rect><text x="127" y="63.5" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="127" dy="0">A0B0</tspan></text><rect data-proof-cell="0,1" x="199" y="36" width="138" height="44" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke)"></rect><text x="268" y="63.5" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="268" dy="0">absent</tspan></text><path d="M203 76L333 40" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><text x="46" y="101" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="46" dy="0">A1</tspan></text><rect data-proof-cell="1,0" x="58" y="83" width="138" height="44" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke)"></rect><text x="127" y="110.5" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="127" dy="0">absent</tspan></text><path d="M62 123L192 87" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><rect data-proof-cell="1,1" x="199" y="83" width="138" height="44" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke)"></rect><text x="268" y="110.5" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="268" dy="0">A1B1</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content"><svg viewBox="0 0 360 960" role="img" aria-labelledby="same-mesh-axis-diagonal-gap-edition-svg-title same-mesh-axis-diagonal-gap-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="same-mesh-axis-diagonal-gap-edition-svg-title">A reduction cannot invent the two products that no device computed</title><desc id="same-mesh-axis-diagonal-gap-edition-svg-desc">In a declared two-by-two toy, A has rows one-two and three-four, and B has columns five-seven and six-eight. Device zero initially owns only A row zero and B column zero, producing nineteen. Device one owns only A row one and B column one, producing fifty. The same-scale output grid leaves off-diagonal cells uncomputed, not zero. All-gather of B columns gives each device the missing input. New local dot products produce twenty-two and forty-three; a second grid fills those exact gaps while retaining nineteen and fifty.</desc><text x="180" y="22" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Toy 2x2 matrices; each output block is one</tspan><tspan x="180" dy="19">number</tspan></text><g data-editorial-node><rect x="15" y="65" width="150" height="110" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="90" y="106" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">D0 owns</tspan><tspan x="90" dy="19">A row 0 [1,2]</tspan><tspan x="90" dy="19">B col 0 [5,7]</tspan></text></g><g data-editorial-node><rect x="195" y="65" width="150" height="110" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.3"></rect><text x="270" y="106" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">D1 owns</tspan><tspan x="270" dy="19">A row 1 [3,4]</tspan><tspan x="270" dy="19">B col 1 [6,8]</tspan></text></g><text x="180" y="192" style="font-size:12px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">LOCAL WORK: TWO MISSING PRODUCTS</tspan></text><text x="120" y="218" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="120" dy="0">col 0</tspan></text><text x="240" y="218" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="240" dy="0">col 1</tspan></text><text x="35" y="279" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="35" dy="0">r0</tspan></text><g data-matmul-phase="before" data-row="0" data-column="0" data-value="19"><rect x="60" y="235" width="120" height="78" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="120" y="267" style="font-size:22px;font-weight:700;fill:var(--viz-input-stroke);text-anchor:middle"><tspan x="120" dy="0">19</tspan></text><text x="120" y="293" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="120" dy="0">D0</tspan></text></g><g data-matmul-phase="before" data-row="0" data-column="1" data-value="missing"><rect x="180" y="235" width="120" height="78" rx="3" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.3;stroke-dasharray:5 3"></rect><text x="240" y="267" style="font-size:22px;font-weight:700;fill:var(--c-text-soft);text-anchor:middle"><tspan x="240" dy="0">?</tspan></text><text x="240" y="293" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="240" dy="0">not computed</tspan></text></g><text x="35" y="357" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="35" dy="0">r1</tspan></text><g data-matmul-phase="before" data-row="1" data-column="0" data-value="missing"><rect x="60" y="313" width="120" height="78" rx="3" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.3;stroke-dasharray:5 3"></rect><text x="120" y="345" style="font-size:22px;font-weight:700;fill:var(--c-text-soft);text-anchor:middle"><tspan x="120" dy="0">?</tspan></text><text x="120" y="371" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="120" dy="0">not computed</tspan></text></g><g data-matmul-phase="before" data-row="1" data-column="1" data-value="50"><rect x="180" y="313" width="120" height="78" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.3"></rect><text x="240" y="345" style="font-size:22px;font-weight:700;fill:var(--viz-output-stroke);text-anchor:middle"><tspan x="240" dy="0">50</tspan></text><text x="240" y="371" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="240" dy="0">D1</tspan></text></g><text x="180" y="667" style="font-size:12px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">NEW PAIRS: ALL FOUR PRODUCTS EXIST</tspan></text><text x="120" y="693" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="120" dy="0">col 0</tspan></text><text x="240" y="693" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="240" dy="0">col 1</tspan></text><text x="35" y="754" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="35" dy="0">r0</tspan></text><g data-matmul-phase="after" data-row="0" data-column="0" data-value="19"><rect x="60" y="710" width="120" height="78" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="120" y="742" style="font-size:22px;font-weight:700;fill:var(--viz-input-stroke);text-anchor:middle"><tspan x="120" dy="0">19</tspan></text><text x="120" y="768" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="120" dy="0">D0</tspan></text></g><g data-matmul-phase="after" data-row="0" data-column="1" data-value="22"><rect x="180" y="710" width="120" height="78" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="240" y="742" style="font-size:22px;font-weight:700;fill:var(--viz-focus-stroke);text-anchor:middle"><tspan x="240" dy="0">22</tspan></text><text x="240" y="768" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="240" dy="0">D0</tspan></text></g><text x="35" y="832" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="35" dy="0">r1</tspan></text><g data-matmul-phase="after" data-row="1" data-column="0" data-value="43"><rect x="60" y="788" width="120" height="78" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="120" y="820" style="font-size:22px;font-weight:700;fill:var(--viz-focus-stroke);text-anchor:middle"><tspan x="120" dy="0">43</tspan></text><text x="120" y="846" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="120" dy="0">D1</tspan></text></g><g data-matmul-phase="after" data-row="1" data-column="1" data-value="50"><rect x="180" y="788" width="120" height="78" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.3"></rect><text x="240" y="820" style="font-size:22px;font-weight:700;fill:var(--viz-output-stroke);text-anchor:middle"><tspan x="240" dy="0">50</tspan></text><text x="240" y="846" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="240" dy="0">D1</tspan></text></g><text x="180" y="430" style="font-size:12px;font-weight:700;fill:var(--viz-warning-stroke);text-anchor:middle"><tspan x="180" dy="0">ALL-REDUCE ADDS; IT CANNOT CREATE THE ? CELLS</tspan></text><path d="M20 473L340 473" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="180" y="504" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">REPAIR: GATHER B, THEN MULTIPLY</tspan></text><text x="180" y="545" style="font-size:13px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">D0 gains [6,8]; D1 gains [5,7].</tspan></text><text x="180" y="593" style="font-size:13px;font-weight:400;fill:var(--viz-focus-stroke);text-anchor:middle"><tspan x="180" dy="0">D0: [1,2] dot [6,8] = 6 + 16 = 22</tspan></text><text x="180" y="635" style="font-size:13px;font-weight:400;fill:var(--viz-focus-stroke);text-anchor:middle"><tspan x="180" dy="0">D1: [3,4] dot [5,7] = 15 + 28 = 43</tspan></text><text x="180" y="913" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Gather inputs first. New multiplication, not</tspan><tspan x="180" dy="19">reduction, fills the gaps.</tspan></text></svg><div class="visual-derivation-caption">Compare the same output positions before and after the gather. The new input pairs produce 22 and 43; the existing 19 and 50 do not change. All-reduce alone cannot manufacture the absent cross-index multiplications.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="same-mesh-axis-diagonal-gap"><strong>Check:</strong> Why can reducing the two computed diagonal blocks not create C0,1 or C1,0?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="same-mesh-axis-diagonal-gap">Computing A0B0 and A1B1 omits A0B1 and A1B0. Adding the two available blocks cannot reconstruct terms that were never computed.</span></figcaption></figure>

All-gather or reshard one input before multiplying. Pick the smaller movement or the layout needed by the next operation.

## Collective operations

| Collective | Layout change |
| --- | --- |
| All-gather | remove a split and replicate the full axis |
| Reduce-scatter | sum partial values and keep a chosen output axis split |
| All-reduce | sum partial values and replicate the result |
| All-to-all | move a split from one tensor axis to another |

All-reduce can often be implemented as reduce-scatter followed by all-gather.

For large messages, a first communication estimate is:

$$
T_{\text{comm}} \approx \frac{\text{bytes moved per device}}{\text{effective link bandwidth}}.
$$

For small messages, fixed launch and hop latency also matters.

## Backpropagation

The backward pass uses more matrix multiplications:

$$
\frac{\partial L}{\partial B} = A^\top \frac{\partial L}{\partial C},
\qquad
\frac{\partial L}{\partial A} = \frac{\partial L}{\partial C} B^\top.
$$

Apply the same sharding rules to these operations.

An all-gather in the forward computation often corresponds to a reduce-scatter for its gradient. This is why fully sharded training naturally pairs those two collectives.

## Overlap communication with compute

A blocking all-gather moves all data before any multiplication begins. A collective matrix multiplication moves one block while computing another block.

Overlap helps when:

- the multiplication has enough independent work;
- blocks are large enough for efficient compute and communication;
- dependencies allow a pipeline;
- the implementation and hardware can run both at once.

Always compare with a simple blocking baseline. More complex overlap code is useful only when measured time improves.

## Small example

A transformer feed-forward projection uses activations $X[B,D]$ and weights $W[D,F]$.

If $D$ is split across devices in both $X$ and $W$, each device produces a partial $[B,F]$ output. Use all-reduce for a replicated output or reduce-scatter to keep $F$ split.

If only $B$ is split, each device processes different tokens with replicated weights. The forward multiplication needs no communication. Gradients must be combined later.

## In an interview

Use this order:

1. Name input, contracting, and output axes.
2. State global and local shapes.
3. Mark each split on the device mesh.
4. Decide whether local outputs are complete or partial.
5. Select all-gather, reduce-scatter, all-reduce, or all-to-all.
6. Estimate bytes and effective bandwidth.
7. Check the layout required by the next operation.
8. Discuss overlap only after the blocking plan is correct.

## Common mistakes

- Looking only at tensor size and not at the contracting axis.
- Treating a partial sum as a complete result.
- Replicating an output that could remain split.
- Assigning one mesh axis to two output axes.
- Ignoring the next operation's layout.
- Counting aggregate cluster bandwidth instead of the limiting link.
- Assuming communication always overlaps with compute.

*Related: [all-reduce and other collectives](/concepts/all-reduce-and-collectives/), [tensor parallelism](/concepts/tensor-parallelism/), and [FSDP and ZeRO](/concepts/fsdp-and-zero/). Further practice: [sharded matrices in the JAX Scaling Book](https://jax-ml.github.io/scaling-book/sharding).*