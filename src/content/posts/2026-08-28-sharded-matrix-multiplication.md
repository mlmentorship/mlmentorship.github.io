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