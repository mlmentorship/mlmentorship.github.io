---
title: "Plan and cost a 70B transformer training run"
description: "Turn a model configuration and cluster budget into parameters, memory, parallelism, time, and an experiment plan."
date: "2026-08-28"
draft: false
tags: ["questions"]
category: "questions"
---

Plan a dense language-model training run by turning the supplied model dimensions and cluster limits into a testable resource and parallelism plan.

## Problem

**Context.** Train a dense decoder-only Transformer on 300 billion tokens using the model and hardware specifications below.

**Task.** Estimate parameters, model and optimizer state, activation memory, compute, communication, and run duration.
Choose an initial parallel layout, batch and checkpoint strategy, then define measurements that could reject the plan.

**Constraints.** Distinguish GB from GiB, peak BF16 compute from achieved throughput, and effective cross-node bandwidth from an unconstrained fabric rate.
State missing sequence-length, precision, optimizer, utilization, and recovery assumptions; there is no single prescribed parallel layout.

| Model Specification | Given Value |
| --- | --- |
| Layers | 80 |
| Model width | 8,192 |
| Gated feed-forward width | 28,672 |
| Query heads / KV heads | 64 / 8 |
| Head size | 128 |
| Tied vocabulary | 128,000 tokens |

| Cluster Specification | Given Value |
| --- | --- |
| GPUs / nodes | 512 / 64, with eight GPUs per node |
| Memory per GPU | 80 GB |
| Peak BF16 compute per GPU | About 1 PFLOP/s |
| Intra-node connection | Fast switched link |
| Effective per-rank cross-node bandwidth | 45 GB/s for large collectives under load |

## Try it before reading the answer

Estimate:

1. total parameters;
2. model-state memory;
3. total training FLOPs;
4. wall time at 50% model FLOPs utilization;
5. accelerator-hours;
6. a first parallel layout;
7. global batch construction for about four million tokens per optimizer step;
8. the first experiments and failure checks.

Assume BF16 weights and gradients, FP32 Adam moments, and no separate FP32 master weights.

## Count the parameters

For each layer, gated feed-forward parameters are:

$$
3DF = 3(8192)(28672) \approx 704.6\text{ million}.
$$

Attention parameters with grouped-query attention are:

$$
2D^2 + 2DKH
= 2(8192)^2 + 2(8192)(8)(128)
\approx 151.0\text{ million}.
$$

Across 80 layers:

$$
80(704.6 + 151.0)\text{ million} \approx 68.4\text{ billion}.
$$

The tied token embedding adds:

$$
VD = 128000(8192) \approx 1.05\text{ billion}.
$$

Norms and small terms bring the estimate close to 70 billion parameters.

## Count the model state

Per parameter:

- BF16 weight: 2 bytes;
- BF16 gradient: 2 bytes;
- two FP32 Adam moments: 8 bytes.

Total model state is about:

$$
70\text{B} \times 12\text{ bytes} = 840\text{ GB}.
$$

This excludes activations, temporary kernel buffers, communication buffers, and allocator margin.

## Count the compute

Start with the parameter-matrix estimate:

$$
6 \times 70\text{B parameters} \times 300\text{B tokens}
= 1.26 \times 10^{23}\text{ FLOPs}.
$$

This estimate should be refined for long-context attention and the final architecture. It is sufficient for an early capacity plan.

At sequence length 8,192, the attention score and weighted-value operations are large enough to include. Their forward and backward cost is approximately:

$$
12 S T D L
= 12(300\text{B})(8192)(8192)(80)
\approx 1.93 \times 10^{22}\text{ FLOPs}.
$$

The refined total is about $1.45 \times 10^{23}$ FLOPs, before smaller operations.

The cluster peak is:

$$
512 \times 10^{15} = 5.12 \times 10^{17}\text{ FLOP/s}.
$$

At 50% MFU, useful model throughput is $2.56 \times 10^{17}$ FLOP/s. Using the refined compute estimate, the run time is:

$$
\frac{1.45 \times 10^{23}}{2.56 \times 10^{17}}
\approx 568{,}000\text{ seconds}
\approx 6.6\text{ days}.
$$

The run uses about:

$$
6.6 \times 24 \times 512 \approx 81{,}000
$$

accelerator-hours, before failed runs, evaluation, checkpoints, and recovery time.

## Choose a first layout

Start with eight-way tensor parallelism inside each node. This keeps frequent per-layer activation communication on the fast local links.

That leaves 64 data-parallel groups across nodes.

Before selecting full parameter sharding, check whether a lighter option fits. With eight-way tensor parallelism:

- weights per GPU: $140/8 = 17.5$ GB;
- gradients per GPU: another 17.5 GB;
- Adam moments sharded over the 64 data-parallel ranks: $560/(8 \times 64) \approx 1.1$ GB.

This ZeRO-1 style estimate uses about 36.1 GB per GPU for model state. It leaves about 44 GB for activations, temporary buffers, communication, and safety margin.

This may fit with activation checkpointing. If measured peak memory is safe, it avoids the parameter all-gathers required by full parameter sharding.

<!-- visual:plan-70b-parallel-accounting -->
<figure class="learning-figure" aria-labelledby="plan-70b-parallel-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="plan-70b-parallel-title">Count distinct token batches across data replicas, not every model shard</p><div class="visual-argument" data-visual-argument="plan-70b-parallel-accounting" data-argument-kind="ledger" data-argument-composition="distinct-examples"><svg viewBox="0 0 360 673" role="img" aria-label="Model shards do not multiply distinct training examples. Two data replicas, sixteen examples per microbatch, and four accumulation steps produce 128 distinct examples. Tensor and pipeline shards share those examples."><title>Model shards do not multiply distinct training examples</title><desc>Two data replicas, sixteen examples per microbatch, and four accumulation steps produce 128 distinct examples. Tensor and pipeline shards share those examples.</desc><text x="20" y="27" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Count each distinct example once</tspan><tspan x="20" dy="18">across the data replicas</tspan></text><text x="20" y="102" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Data replica 1: four batches of</tspan><tspan x="20" dy="18">sixteen</tspan></text><g data-picture-example-member="0"><circle cx="29" cy="144" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="1"><circle cx="49" cy="144" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="2"><circle cx="69" cy="144" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="3"><circle cx="89" cy="144" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="4"><circle cx="109" cy="144" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="5"><circle cx="129" cy="144" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="6"><circle cx="149" cy="144" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="7"><circle cx="169" cy="144" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="8"><circle cx="189" cy="144" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="9"><circle cx="209" cy="144" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="10"><circle cx="229" cy="144" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="11"><circle cx="249" cy="144" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="12"><circle cx="269" cy="144" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="13"><circle cx="289" cy="144" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="14"><circle cx="309" cy="144" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="15"><circle cx="329" cy="144" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="16"><circle cx="29" cy="180" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="17"><circle cx="49" cy="180" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="18"><circle cx="69" cy="180" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="19"><circle cx="89" cy="180" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="20"><circle cx="109" cy="180" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="21"><circle cx="129" cy="180" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="22"><circle cx="149" cy="180" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="23"><circle cx="169" cy="180" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="24"><circle cx="189" cy="180" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="25"><circle cx="209" cy="180" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="26"><circle cx="229" cy="180" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="27"><circle cx="249" cy="180" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="28"><circle cx="269" cy="180" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="29"><circle cx="289" cy="180" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="30"><circle cx="309" cy="180" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="31"><circle cx="329" cy="180" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="32"><circle cx="29" cy="216" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="33"><circle cx="49" cy="216" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="34"><circle cx="69" cy="216" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="35"><circle cx="89" cy="216" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="36"><circle cx="109" cy="216" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="37"><circle cx="129" cy="216" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="38"><circle cx="149" cy="216" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="39"><circle cx="169" cy="216" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="40"><circle cx="189" cy="216" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="41"><circle cx="209" cy="216" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="42"><circle cx="229" cy="216" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="43"><circle cx="249" cy="216" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="44"><circle cx="269" cy="216" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="45"><circle cx="289" cy="216" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="46"><circle cx="309" cy="216" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="47"><circle cx="329" cy="216" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="48"><circle cx="29" cy="252" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="49"><circle cx="49" cy="252" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="50"><circle cx="69" cy="252" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="51"><circle cx="89" cy="252" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="52"><circle cx="109" cy="252" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="53"><circle cx="129" cy="252" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="54"><circle cx="149" cy="252" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="55"><circle cx="169" cy="252" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="56"><circle cx="189" cy="252" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="57"><circle cx="209" cy="252" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="58"><circle cx="229" cy="252" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="59"><circle cx="249" cy="252" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="60"><circle cx="269" cy="252" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="61"><circle cx="289" cy="252" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="62"><circle cx="309" cy="252" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="63"><circle cx="329" cy="252" r="3" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><text x="20" y="345" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Data replica 2: four batches of</tspan><tspan x="20" dy="18">sixteen</tspan></text><g data-picture-example-member="64"><circle cx="29" cy="387" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="65"><circle cx="49" cy="387" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="66"><circle cx="69" cy="387" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="67"><circle cx="89" cy="387" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="68"><circle cx="109" cy="387" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="69"><circle cx="129" cy="387" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="70"><circle cx="149" cy="387" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="71"><circle cx="169" cy="387" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="72"><circle cx="189" cy="387" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="73"><circle cx="209" cy="387" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="74"><circle cx="229" cy="387" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="75"><circle cx="249" cy="387" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="76"><circle cx="269" cy="387" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="77"><circle cx="289" cy="387" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="78"><circle cx="309" cy="387" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="79"><circle cx="329" cy="387" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="80"><circle cx="29" cy="423" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="81"><circle cx="49" cy="423" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="82"><circle cx="69" cy="423" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="83"><circle cx="89" cy="423" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="84"><circle cx="109" cy="423" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="85"><circle cx="129" cy="423" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="86"><circle cx="149" cy="423" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="87"><circle cx="169" cy="423" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="88"><circle cx="189" cy="423" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="89"><circle cx="209" cy="423" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="90"><circle cx="229" cy="423" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="91"><circle cx="249" cy="423" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="92"><circle cx="269" cy="423" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="93"><circle cx="289" cy="423" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="94"><circle cx="309" cy="423" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="95"><circle cx="329" cy="423" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="96"><circle cx="29" cy="459" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="97"><circle cx="49" cy="459" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="98"><circle cx="69" cy="459" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="99"><circle cx="89" cy="459" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="100"><circle cx="109" cy="459" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="101"><circle cx="129" cy="459" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="102"><circle cx="149" cy="459" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="103"><circle cx="169" cy="459" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="104"><circle cx="189" cy="459" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="105"><circle cx="209" cy="459" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="106"><circle cx="229" cy="459" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="107"><circle cx="249" cy="459" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="108"><circle cx="269" cy="459" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="109"><circle cx="289" cy="459" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="110"><circle cx="309" cy="459" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="111"><circle cx="329" cy="459" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="112"><circle cx="29" cy="495" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="113"><circle cx="49" cy="495" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="114"><circle cx="69" cy="495" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="115"><circle cx="89" cy="495" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="116"><circle cx="109" cy="495" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="117"><circle cx="129" cy="495" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="118"><circle cx="149" cy="495" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="119"><circle cx="169" cy="495" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="120"><circle cx="189" cy="495" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="121"><circle cx="209" cy="495" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="122"><circle cx="229" cy="495" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="123"><circle cx="249" cy="495" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="124"><circle cx="269" cy="495" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="125"><circle cx="289" cy="495" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="126"><circle cx="309" cy="495" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-example-member="127"><circle cx="329" cy="495" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><text x="20" y="606" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">2 x 4 x 16 = 128. TP and PP reuse</tspan><tspan x="20" dy="18">these same identities.</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
	<div class="visual-grid--two" role="group" aria-label="A 512-GPU layout with tensor parallel degree eight inside each of 64 nodes and data parallel degree 64 across nodes, followed by its memory and batch accounting">
		<section class="visual-panel" aria-labelledby="plan-70b-ranks-title">
			<h4 id="plan-70b-ranks-title">512 ranks = TP 8 × DP 64</h4>
			<p>Each node is one tensor-sharded model replica; corresponding tensor ranks synchronize across nodes.</p>
			<table class="cm-grid" aria-label="Tensor and data parallel rank roles">
				<thead><tr><th scope="col">Axis</th><th scope="col">Placement</th><th scope="col">Data</th></tr></thead>
				<tbody>
					<tr><th scope="row">TP = 8</th><td class="cm-selected"><strong>within node</strong>fast links</td><td><strong>same</strong>sequence</td></tr>
					<tr><th scope="row">DP = 64</th><td><strong>across nodes</strong>64 replicas</td><td><strong>different</strong>sequences</td></tr>
				</tbody>
			</table>
		</section>
		<section class="visual-panel" aria-labelledby="plan-70b-ledger-title">
			<h4 id="plan-70b-ledger-title">Apply the right divisor</h4>
			<p>TP slices layer state. ZeRO-1 additionally slices Adam state over DP. Only DP creates extra data replicas.</p>
			<table class="cm-grid" aria-label="Per-GPU model-state and global-batch accounting">
				<thead><tr><th scope="col">Quantity</th><th scope="col">Divide or multiply by</th><th scope="col">Result</th></tr></thead>
				<tbody>
					<tr><th scope="row">Weights + gradients</th><td>÷ TP 8</td><td><strong>35 GB/GPU</strong></td></tr>
					<tr><th scope="row">Adam moments</th><td>÷ TP 8 ÷ DP 64</td><td><strong>1.1 GB/GPU</strong></td></tr>
					<tr><th scope="row">Tokens/step</th><td class="cm-selected">× DP 64, not TP</td><td><strong>4.194M</strong></td></tr>
				</tbody>
			</table>
		</section>
	</div>
	<div class="visual-derivation-caption">first form one eight-GPU tensor-parallel replica inside every node, then replicate that sharded model across 64 nodes. TP ranks cooperate on the same sequences, so the batch is <code>64 DP × 1 sequence × 8 accumulations × 8,192 tokens = 4,194,304</code>; multiplying by all 512 ranks would overcount it by eight.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="plan-70b-parallel-accounting"><strong>Check:</strong> Why does the token count multiply by DP 64 but not by all 512 ranks?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="plan-70b-parallel-accounting">Two data replicas, sixteen examples per microbatch, and four accumulation steps produce 128 distinct examples. Tensor and pipeline shards share those examples. <br><strong>Does multiplying by TP eight count any additional distinct example?</strong></span></figcaption>
</figure>

If activations still do not fit:

1. reduce the local micro-batch;
2. checkpoint activations;
3. use sequence or context parallelism for long sequences;
4. then consider parameter sharding or pipeline parallelism if the earlier steps are not enough.

Do not start with every parallelism dimension.

## Build the global batch

At sequence length 8,192, use one sequence per data-parallel replica per micro-batch and accumulate eight micro-batches.

$$
64 \times 1 \times 8 \times 8192
= 4{,}194{,}304\text{ tokens per optimizer step}.
$$

The 300-billion-token run needs about:

$$
300\text{B} / 4.194\text{M} \approx 71{,}500
$$

optimizer steps.

Tensor-parallel ranks work on the same sequences, so they do not multiply the global batch.

The target batch must also make sense for optimization. If it changes from the validated training recipe, retune learning rate, warmup, and possibly the token schedule.

## Check communication

Each tensor-parallel rank owns about 17.5 GB of gradients. Gradient synchronization may use an all-reduce or an equivalent reduce-scatter plus parameter all-gather for optimizer-state sharding. The ring traffic has this large-message lower bound:

$$
\frac{2(63/64)(17.5\text{ GB})}{45\text{ GB/s}} \approx 0.77\text{ seconds}.
$$

The real value depends on bucketing, topology, contention, and overlap with backpropagation.

Trace the last gradient buckets. A long reduction tail after backward compute ends is exposed communication and directly increases step time.

## Validate the plan in stages

### Stage 1: one-node correctness

- confirm loss and gradient behavior;
- verify parameter count and FLOP estimate;
- record peak memory by component;
- measure kernel shapes and numerical stability.

### Stage 2: multi-node scaling

Test a small grid, such as 8, 64, 128, and 512 GPUs when practical. Keep the global batch or report any change.

Record:

- tokens per second;
- MFU;
- median and tail step time;
- exposed collective time;
- per-rank imbalance;
- peak memory.

### Stage 3: reliability

- checkpoint model, optimizer, data position, and random state;
- test restart before the full run;
- validate checkpoint write time and storage load;
- define health checks for loss spikes, NaNs, stalled ranks, and slow nodes;
- keep a known-good checkpoint for rollback.

### Stage 4: convergence

Run enough tokens to test the learning curve, not only systems throughput. Compare with smaller-scale predictions and monitor held-out loss by data source.

## What an L4 answer sounds like

> "Use 512 GPUs, mixed precision, tensor parallelism, FSDP, activation checkpointing, and a distributed training framework."

This names tools without calculating fit, time, batch, or communication.

## What an L5 answer adds

An L5 candidate estimates 70 billion parameters, about 840 GB of unsharded model state, and the $1.26 \times 10^{23}$ parameter-matrix FLOP baseline. A stronger estimate adds about $1.93 \times 10^{22}$ FLOPs for long-context attention, giving about 6.6 days at 50% MFU. They place tensor parallelism inside each node and build the global batch correctly.

## What an L6 answer adds

An L6 candidate starts with the least expensive layout that fits. They notice that tensor parallelism plus optimizer-state sharding may fit without full parameter sharding. They reserve memory margin, estimate exposed gradient communication, preserve optimization behavior, and define a staged scaling, restart, and convergence plan.

They also report uncertainty. The 50% MFU and communication estimates are planning assumptions that traces must replace.

## Strong-hire signals

- Arithmetic comes before framework choice.
- Tensor-parallel ranks are not counted as extra data replicas.
- Model state and activations are budgeted separately.
- The topology determines group placement.
- Wall time and accelerator-hours are both reported.
- The candidate tests a simpler sharding stage before full parameter sharding.
- Communication overlap is measured, not assumed.
- Restart and convergence tests happen before the full run.

## Common follow-up

"Why not use full parameter sharding immediately?"

Full parameter sharding saves more memory. It also adds parameter all-gathers during forward and backward work. If a lighter sharding stage already leaves enough activation and buffer memory, the extra communication may not help. Measure both if the answer is close.

*Related: [Transformer compute and memory accounting](/concepts/transformer-compute-memory-accounting/), [strong scaling and parallelism selection](/concepts/strong-scaling-and-parallelism-selection/), [accelerator network topology](/concepts/accelerator-network-topology/), and [train a 100B parameter model](/questions/train-100b-model/).*