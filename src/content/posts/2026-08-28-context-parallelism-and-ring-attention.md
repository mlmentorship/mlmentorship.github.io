---
title: "Context parallelism and ring attention"
description: "Shard a long sequence across devices while preserving exact attention and controlling activation memory."
date: "2026-08-28"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Context parallelism splits sequence positions across devices. Each device stores and computes a local part of the sequence, then exchanges attention data so local queries can attend to all allowed keys and values.

## Why AI labs care

Long-context training can run out of memory even when model weights fit. The main pressure comes from activations and attention state that grow with sequence length.

Context parallelism can reduce memory per device without changing the model's attention rule. It is different from sparse or approximate attention.

## Sequence parallelism is not the same thing

The names are sometimes used differently across libraries. A useful distinction is:

- **Sequence parallelism** shards sequence positions for operations outside attention, such as normalization, dropout, and residual work. Tensor-parallel layers often already require layout changes that make this cheap.
- **Context parallelism** shards the sequence inside attention itself. It requires communication because each query may need keys and values from other devices.

Ask which definition a system uses before comparing results.

## The attention problem

For one head:

$$
O = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{H}} + M\right)V,
$$

where $M$ contains causal or other attention masks.

Split a sequence of length $T$ across $R$ devices. Each device holds about $T/R$ local query, key, and value positions.

Local queries still need remote keys and values. A local-only attention calculation would change the model.

## Ring attention

Ring attention keeps each device's query block local and sends key/value blocks around a ring.

At each step:

1. compute attention between local queries and the current key/value block;
2. update a running softmax result;
3. send that key/value block to the next device;
4. receive another block;
5. repeat until every local query has seen every allowed key/value block.

<!-- visual:ring-attention-fixed-query-kv-cycle -->
<figure class="learning-figure plot-panel visual-wide" aria-labelledby="ring-attention-cycle-title">
	<p class="visual-kicker">One device · four ring steps</p>
	<p class="visual-title" id="ring-attention-cycle-title">Keep queries local, rotate KV blocks, and merge one shared normalization</p><div class="visual-argument" data-visual-argument="ring-attention-fixed-query-kv-cycle" data-argument-kind="ledger" data-argument-composition="rotate-kv-and-normalize"><svg viewBox="0 0 360 1526" role="img" aria-label="Keep queries local, rotate KV blocks, and merge one shared normalization. Each query stays on its device while every KV block rotates past it. Accumulate weighted values with one running normalization; do not average separately normalized block answers. For Q0, the second block carries four times the normalization weight, so the combined answer is 8.4, not 6. Blocking future tokens changes eligible contributions, not the need to share normalization."><title>Keep queries local, rotate KV blocks, and merge one shared normalization</title><desc>Each query stays on its device while every KV block rotates past it. Accumulate weighted values with one running normalization; do not average separately normalized block answers. For Q0, the second block carries four times the normalization weight, so the combined answer is 8.4, not 6. Blocking future tokens changes eligible contributions, not the need to share normalization.</desc><text x="20" y="27" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Queries stay; KV blocks travel</tspan></text><text x="20" y="83" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">First visit: local block</tspan></text><rect x="20" y="128" width="145" height="192" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.5"></rect><g data-picture-ring-query="0" data-picture-ring-step="0"><circle cx="92.5" cy="173" r="22" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="92.5" y="179" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="92.5" dy="0">Q0</tspan></text></g><path d="M92.5 207L92.5 231M88.5 223L92.5 231L96.5 223" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2;stroke-linejoin:round"></path><rect x="45" y="245" width="95" height="48" data-picture-ring-kv="A" data-picture-ring-device="0" data-picture-ring-visit="0" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.5"></rect><text x="92.5" y="275" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-focus-stroke)"><tspan x="92.5" dy="0">KV A</tspan></text><rect x="195" y="128" width="145" height="192" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.5"></rect><g data-picture-ring-query="1" data-picture-ring-step="0"><circle cx="267.5" cy="173" r="22" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="267.5" y="179" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="267.5" dy="0">Q1</tspan></text></g><path d="M267.5 207L267.5 231M263.5 223L267.5 231L271.5 223" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2;stroke-linejoin:round"></path><rect x="220" y="245" width="95" height="48" data-picture-ring-kv="B" data-picture-ring-device="1" data-picture-ring-visit="0" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><text x="267.5" y="275" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-output-stroke)"><tspan x="267.5" dy="0">KV B</tspan></text><text x="20" y="503" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Second visit: received block</tspan></text><rect x="20" y="548" width="145" height="192" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.5"></rect><g data-picture-ring-query="0" data-picture-ring-step="1"><circle cx="92.5" cy="593" r="22" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="92.5" y="599" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="92.5" dy="0">Q0</tspan></text></g><path d="M92.5 627L92.5 651M88.5 643L92.5 651L96.5 643" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2;stroke-linejoin:round"></path><rect x="45" y="665" width="95" height="48" data-picture-ring-kv="B" data-picture-ring-device="0" data-picture-ring-visit="1" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><text x="92.5" y="695" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-output-stroke)"><tspan x="92.5" dy="0">KV B</tspan></text><rect x="195" y="548" width="145" height="192" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.5"></rect><g data-picture-ring-query="1" data-picture-ring-step="1"><circle cx="267.5" cy="593" r="22" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="267.5" y="599" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="267.5" dy="0">Q1</tspan></text></g><path d="M267.5 627L267.5 651M263.5 643L267.5 651L271.5 643" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2;stroke-linejoin:round"></path><rect x="220" y="665" width="95" height="48" data-picture-ring-kv="A" data-picture-ring-device="1" data-picture-ring-visit="1" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.5"></rect><text x="267.5" y="695" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-focus-stroke)"><tspan x="267.5" dy="0">KV A</tspan></text><path d="M92.5 303L92.5 350" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path><path d="M92.5 350L345 350" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path><path d="M345 350L345 689" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:1.75"></path><path d="M345 689L324 689M332 685L324 689L332 693" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2;stroke-linejoin:round"></path><path d="M267.5 303L267.5 370" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75"></path><path d="M267.5 370L15 370" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75"></path><path d="M15 370L15 689" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75"></path><path d="M15 689L37 689M29 693L37 689L29 685" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:2;stroke-linejoin:round"></path><text x="20" y="404" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">Exchange blocks, not queries</tspan></text><text x="20" y="810" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Q0: combine the two visits with their</tspan><tspan x="20" dy="18">unequal weights</tspan></text><g transform="translate(0 880)"><text x="20" y="26" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Width: weight. Height: value.</tspan></text><text x="20" y="66" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="20" dy="0">weight 0.2</tspan></text><rect data-picture-contribution="0" data-picture-source-member="0" x="20" y="224" width="26" height="20" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="20" y="278" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="20" dy="0">value 2</tspan></text><text x="20" y="313" style="font-size:20px;font-weight:700;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="20" dy="0">area 0.4</tspan></text><path d="M50 332L98 362" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><text x="210" y="66" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-output-stroke)"><tspan x="210" dy="0">weight 0.8</tspan></text><rect data-picture-contribution="1" data-picture-source-member="1" x="210" y="224" width="26" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-contribution="1" data-picture-source-member="2" x="236" y="224" width="26" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-contribution="1" data-picture-source-member="3" x="262" y="224" width="26" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-contribution="1" data-picture-source-member="4" x="288" y="224" width="26" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-contribution="1" data-picture-source-member="5" x="210" y="204" width="26" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-contribution="1" data-picture-source-member="6" x="236" y="204" width="26" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-contribution="1" data-picture-source-member="7" x="262" y="204" width="26" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-contribution="1" data-picture-source-member="8" x="288" y="204" width="26" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-contribution="1" data-picture-source-member="9" x="210" y="184" width="26" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-contribution="1" data-picture-source-member="10" x="236" y="184" width="26" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-contribution="1" data-picture-source-member="11" x="262" y="184" width="26" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-contribution="1" data-picture-source-member="12" x="288" y="184" width="26" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-contribution="1" data-picture-source-member="13" x="210" y="164" width="26" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-contribution="1" data-picture-source-member="14" x="236" y="164" width="26" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-contribution="1" data-picture-source-member="15" x="262" y="164" width="26" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-contribution="1" data-picture-source-member="16" x="288" y="164" width="26" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-contribution="1" data-picture-source-member="17" x="210" y="144" width="26" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-contribution="1" data-picture-source-member="18" x="236" y="144" width="26" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-contribution="1" data-picture-source-member="19" x="262" y="144" width="26" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-contribution="1" data-picture-source-member="20" x="288" y="144" width="26" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><text x="210" y="278" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--viz-output-stroke)"><tspan x="210" dy="0">value 10</tspan></text><text x="210" y="313" style="font-size:20px;font-weight:700;text-anchor:start;fill:var(--viz-output-stroke)"><tspan x="210" dy="0">area 8</tspan></text><path d="M240 332L242 362" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75"></path><rect data-picture-collected-member="0" x="20" y="376" width="26" height="20" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><rect data-picture-collected-member="1" x="46" y="376" width="26" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-collected-member="2" x="72" y="376" width="26" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-collected-member="3" x="98" y="376" width="26" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-collected-member="4" x="124" y="376" width="26" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-collected-member="5" x="150" y="376" width="26" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-collected-member="6" x="176" y="376" width="26" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-collected-member="7" x="202" y="376" width="26" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-collected-member="8" x="228" y="376" width="26" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-collected-member="9" x="254" y="376" width="26" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-collected-member="10" x="280" y="376" width="26" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-collected-member="11" x="306" y="376" width="26" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-collected-member="12" x="20" y="396" width="26" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-collected-member="13" x="46" y="396" width="26" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-collected-member="14" x="72" y="396" width="26" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-collected-member="15" x="98" y="396" width="26" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-collected-member="16" x="124" y="396" width="26" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-collected-member="17" x="150" y="396" width="26" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-collected-member="18" x="176" y="396" width="26" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-collected-member="19" x="202" y="396" width="26" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-picture-collected-member="20" x="228" y="396" width="26" height="20" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><text x="20" y="454" style="font-size:20px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Combined contribution: 8.4</tspan></text><text x="20" y="517" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">Same tiles, regrouped; each</tspan><tspan x="20" dy="18">contributes 0.4</tspan></text><text x="20" y="581" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">Normalizers 2 and 8 supply these</tspan><tspan x="20" dy="18">unequal weights.</tspan></text></g></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
	<div class="visual-scroll">
		<svg viewBox="0 0 820 350" role="img" aria-labelledby="ring-attention-cycle-svg-title ring-attention-cycle-svg-desc">
			<title id="ring-attention-cycle-svg-title">Device zero processes every key/value block in a four-device ring</title>
			<desc id="ring-attention-cycle-svg-desc">Across four ordered steps, query block Q zero remains on device zero. Device zero first processes its local key/value block zero, then receives blocks three, two, and one around the ring. After each visit, the block scores merge into the same running maximum m, denominator z, and weighted value sum u. Only after all four blocks have been seen does device zero produce output O zero equal to u divided by z. Every other device performs the same rotated process concurrently.</desc>
			<text class="viz-axis-label" x="20" y="24">TIME → · DEVICE 0 VIEW</text>
			<g aria-label="Step one, local key/value block zero">
				<rect class="viz-node" x="20" y="45" width="170" height="270" rx="5"></rect>
				<text class="viz-axis-label" x="35" y="68">STEP 1 · LOCAL</text>
				<rect class="viz-node viz-node--input" x="40" y="88" width="130" height="44" rx="4"></rect>
				<text class="viz-node-value" x="105" y="106">STAYS ON DEVICE 0</text>
				<text class="viz-node-label" x="105" y="124">Q₀</text>
				<text class="viz-node-label" x="105" y="153">+</text>
				<rect class="viz-node viz-node--focus" x="40" y="164" width="130" height="44" rx="4"></rect>
				<text class="viz-node-value" x="105" y="182">BLOCK ORIGIN 0</text>
				<text class="viz-node-label" x="105" y="200">K/V₀</text>
				<path class="viz-axis" d="M105 208 V229"></path><path class="viz-arrow-forward" d="M105 235 l-5 -9 h10 Z"></path>
				<rect class="viz-node viz-node--output" x="40" y="235" width="130" height="58" rx="4"></rect>
				<text class="viz-node-value" x="105" y="254">ONE ACCUMULATOR</text>
				<text class="viz-node-label" x="105" y="274">(m, z, u)</text>
				<text class="viz-label" x="105" y="307" text-anchor="middle">seen: 0</text>
			</g>
			<g aria-label="Step two, key/value block three">
				<rect class="viz-node" x="220" y="45" width="170" height="270" rx="5"></rect>
				<text class="viz-axis-label" x="235" y="68">STEP 2 · RECEIVE</text>
				<rect class="viz-node viz-node--input" x="240" y="88" width="130" height="44" rx="4"></rect>
				<text class="viz-node-value" x="305" y="106">STAYS ON DEVICE 0</text>
				<text class="viz-node-label" x="305" y="124">Q₀</text>
				<text class="viz-node-label" x="305" y="153">+</text>
				<rect class="viz-node viz-node--focus" x="240" y="164" width="130" height="44" rx="4"></rect>
				<text class="viz-node-value" x="305" y="182">FROM DEVICE 3</text>
				<text class="viz-node-label" x="305" y="200">K/V₃</text>
				<path class="viz-axis" d="M305 208 V229"></path><path class="viz-arrow-forward" d="M305 235 l-5 -9 h10 Z"></path>
				<rect class="viz-node viz-node--output" x="240" y="235" width="130" height="58" rx="4"></rect>
				<text class="viz-node-value" x="305" y="254">RESCALE + MERGE</text>
				<text class="viz-node-label" x="305" y="274">(m, z, u)</text>
				<text class="viz-label" x="305" y="307" text-anchor="middle">seen: 0, 3</text>
			</g>
			<g aria-label="Step three, key/value block two">
				<rect class="viz-node" x="420" y="45" width="170" height="270" rx="5"></rect>
				<text class="viz-axis-label" x="435" y="68">STEP 3 · RECEIVE</text>
				<rect class="viz-node viz-node--input" x="440" y="88" width="130" height="44" rx="4"></rect>
				<text class="viz-node-value" x="505" y="106">STAYS ON DEVICE 0</text>
				<text class="viz-node-label" x="505" y="124">Q₀</text>
				<text class="viz-node-label" x="505" y="153">+</text>
				<rect class="viz-node viz-node--focus" x="440" y="164" width="130" height="44" rx="4"></rect>
				<text class="viz-node-value" x="505" y="182">FROM DEVICE 2</text>
				<text class="viz-node-label" x="505" y="200">K/V₂</text>
				<path class="viz-axis" d="M505 208 V229"></path><path class="viz-arrow-forward" d="M505 235 l-5 -9 h10 Z"></path>
				<rect class="viz-node viz-node--output" x="440" y="235" width="130" height="58" rx="4"></rect>
				<text class="viz-node-value" x="505" y="254">RESCALE + MERGE</text>
				<text class="viz-node-label" x="505" y="274">(m, z, u)</text>
				<text class="viz-label" x="505" y="307" text-anchor="middle">seen: 0, 3, 2</text>
			</g>
			<g aria-label="Step four, key/value block one and final output">
				<rect class="viz-node" x="620" y="45" width="180" height="270" rx="5"></rect>
				<text class="viz-axis-label" x="635" y="68">STEP 4 · RECEIVE</text>
				<rect class="viz-node viz-node--input" x="645" y="88" width="130" height="44" rx="4"></rect>
				<text class="viz-node-value" x="710" y="106">STAYS ON DEVICE 0</text>
				<text class="viz-node-label" x="710" y="124">Q₀</text>
				<text class="viz-node-label" x="710" y="153">+</text>
				<rect class="viz-node viz-node--focus" x="645" y="164" width="130" height="44" rx="4"></rect>
				<text class="viz-node-value" x="710" y="182">FROM DEVICE 1</text>
				<text class="viz-node-label" x="710" y="200">K/V₁</text>
				<path class="viz-axis" d="M710 208 V229"></path><path class="viz-arrow-forward" d="M710 235 l-5 -9 h10 Z"></path>
				<rect class="viz-node viz-node--output" x="645" y="235" width="130" height="58" rx="4"></rect>
				<text class="viz-node-value" x="710" y="254">ALL BLOCKS SEEN</text>
				<text class="viz-node-label" x="710" y="274">O₀ = u / z</text>
				<text class="viz-label" x="710" y="307" text-anchor="middle">seen: 0, 3, 2, 1</text>
			</g>
			<path class="viz-baseline" d="M192 105 H216 M392 105 H416 M592 105 H616"></path>
			<path class="viz-arrow-forward" d="M218 105 l-9 -5 v10 Z"></path>
			<path class="viz-arrow-forward" d="M418 105 l-9 -5 v10 Z"></path>
			<path class="viz-arrow-forward" d="M618 105 l-9 -5 v10 Z"></path>
			<text class="viz-label" x="205" y="38" text-anchor="middle">KV →</text>
			<text class="viz-label" x="405" y="38" text-anchor="middle">KV →</text>
			<text class="viz-label" x="605" y="38" text-anchor="middle">KV →</text>
			<text class="viz-label" x="20" y="338">Every device runs the same rotated trace concurrently; only K/V blocks move around the ring.</text>
		</svg>
	</div>
	<div class="visual-derivation-caption">scan left to right. Q₀ never leaves device 0; K/V₀, K/V₃, K/V₂, and K/V₁ visit in turn. Each visit rescales and merges into the same (m, z, u), so the final O₀ = u/z is one softmax over all visible blocks, not four local softmaxes. Every device performs the same rotated process concurrently. Original schematic based on the <a href="https://arxiv.org/abs/2310.01889">Ring Attention paper</a> and <a href="https://docs.nvidia.com/megatron-core/developer-guide/0.16.1/user-guide/features/context_parallel.html">Megatron Core context-parallel documentation</a>.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="ring-attention-fixed-query-kv-cycle"><strong>Check:</strong> Why can independently normalized block outputs not simply be added?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="ring-attention-fixed-query-kv-cycle">Each query stays on its device while every KV block rotates past it. Accumulate weighted values with one running normalization; do not average separately normalized block answers. For Q0, the second block carries four times the normalization weight, so the combined answer is 8.4, not 6. Blocking future tokens changes eligible contributions, not the need to share normalization. <strong>Why is the answer not the equal average six?</strong></span></figcaption>
</figure>

The output is exact apart from normal floating-point effects.

## Online softmax

A device cannot normalize after each block independently. The final softmax denominator includes scores from all blocks.

For each query row, keep:

- the largest score seen so far;
- a running sum of exponentials adjusted to that maximum;
- a running weighted value sum.

Let $m$, $z$, and $u$ be those three running values. Let $m_b$, $z_b$, and $u_b$ be the values for a new block. Update them with:

$$
m' = \max(m,m_b),
$$

$$
z' = e^{m-m'}z + e^{m_b-m'}z_b,
\qquad
u' = e^{m-m'}u + e^{m_b-m'}u_b.
$$

After all blocks, the output is $u'/z'$. Rescaling prevents overflow and makes the result equal to one softmax over all visible blocks, apart from floating-point rounding.

## Causal attention

A token may attend only to its current or earlier positions. Devices need global position ranges for each query and key block.

A device can:

- skip a block that is entirely in the future;
- process a block that is entirely valid;
- apply a triangular mask when query and key ranges overlap.

Good block order can avoid work on future blocks.

## Memory and compute

With even sharding:

- local sequence activations fall by about $R$;
- local query positions fall by about $R$;
- total exact-attention work across the group is still based on the full sequence;
- each device communicates key/value blocks around the group.

The method helps memory and divides compute. It does not remove the quadratic total work of exact attention.

## Communication choices

A ring uses point-to-point transfers and can pipeline communication with attention compute.

Some systems use all-gather or all-to-all layouts instead. The best choice depends on:

- physical network topology;
- number of devices;
- local sequence length;
- number of key/value heads;
- whether tensor parallelism is also active;
- which layout the next operation needs.

Grouped-query or multi-query attention reduces key/value bytes and therefore reduces context-parallel communication.

## Load balance

Simple causal sharding can create uneven work. Early query blocks have fewer valid key blocks than late query blocks.

Balanced position assignment can pair early and late sequence pieces on each device. This improves utilization while preserving token order in the attention mask.

Padding and variable sequence lengths can create another imbalance. Packing and length-aware assignment help.

## When to use it

Use context parallelism when:

- activation memory is dominated by long sequences;
- reducing the local micro-batch is not enough;
- tensor or fully sharded parallelism does not split the needed sequence state;
- the network can carry key/value traffic efficiently.

Do not add it for short sequences that already fit. The communication and implementation complexity may reduce speed.

## In an interview

Use this order:

1. Separate model-state memory from sequence activation memory.
2. State the global and per-device sequence lengths.
3. Explain why local queries need remote keys and values.
4. Describe ring transfer and stable online softmax.
5. Handle causal masks and global positions.
6. Estimate key/value bytes per block.
7. Discuss overlap, topology, and load balance.
8. Compare with sparse attention only if changing model behavior is allowed.

## Common mistakes

- Calling context parallelism sparse attention.
- Computing a separate softmax for each remote block.
- Ignoring causal masking across device boundaries.
- Claiming the method removes quadratic total compute.
- Forgetting key/value communication.
- Combining tensor and context groups without checking the physical topology.
- Using one library's naming as a universal definition.

*Related: [FlashAttention](/concepts/flashattention/), [long-context LLMs](/concepts/long-context-llms/), [tensor parallelism](/concepts/tensor-parallelism/), and [accelerator network topology](/concepts/accelerator-network-topology/). Further reading: [training parallelism in the JAX Scaling Book](https://jax-ml.github.io/scaling-book/training).*