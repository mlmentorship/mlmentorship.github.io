---
title: "Decoding strategies: greedy, beam, top-k, top-p, temperature"
description: "Calculate eligible tokens, normalized probabilities and sequence scores under a declared decoding pipeline."
date: "2026-05-07"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Decoding turns next-token scores into eligible outputs using a declared selection or search policy.
Temperature changes probabilities; truncation removes tokens before the retained distribution is normalized.
Neither high model probability nor low temperature establishes factual correctness.

Specify vocabulary, allowed-token mask, processor order, tie policy, and EOS or budget stopping.
Our default calculation uses temperature $T=1$, then truncation, renormalization, and sampling.
An empty allowed set is an error requiring an explicit recovery policy, not an arbitrary token.

## The strategies

### Greedy

Pick $\arg\max_x p(x \mid x_{<t})$ at every step.

Declare tie-breaking among allowed maxima.
Local argmax need not maximize complete-sequence probability or task quality.

### Beam search

Retain a declared number of prefixes, expanding them by cumulative score $\sum_t\log p(x_t\mid x_{<t})$.
Finite beams prune possibilities and need not find the global optimum.
Specify completed-EOS handling, length normalization, and stopping together; these can change the selected sequence.

### Temperature sampling

Sample from $\text{softmax}(z / T)$ where $z$ is the logit vector and $T$ is the temperature.

Require $T>0$.
Lower values sharpen relative probabilities; $T\to0$ concentrates on maximizers, with ties requiring a policy.
Large $T$ approaches uniform over finite allowed logits, not masked tokens.

### Top-k sampling

Retain exactly $k$ highest-probability allowed tokens, resolving cutoff ties by a declared order ([Fan et al., 2018](https://arxiv.org/abs/1805.04833)).
Renormalize their probabilities; a fixed count does not fix retained mass.

### Top-p (nucleus) sampling

Retain the smallest sorted prefix whose mass reaches at least threshold $p$, including the crossing token.
Choose a threshold in $(0,1]$; declare cutoff ties and renormalize ([Holtzman et al., 2020](https://arxiv.org/abs/1904.09751)).

<!-- visual:decoding-fixed-count-adaptive-mass -->
<figure class="learning-figure" aria-labelledby="decoding-support-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="decoding-support-title">Filter named tokens, renormalize, and distinguish a local choice from a complete sequence</p><div class="visual-argument" data-visual-argument="decoding-fixed-count-adaptive-mass" data-argument-kind="parts" data-argument-composition="token-filter-and-sequence"><svg viewBox="0 0 360 2500" role="img" aria-label="Filter named tokens, renormalize, and distinguish a local choice from a complete sequence. Temperature changes probabilities before filtering; order matters. Top-k keeps a fixed count; top-p includes the crossing token. Renormalize retained tokens, then sample and append one. Repeat from the new prefix until EOS or the budget. Greedy picks locally; beams compare scored prefixes but may prune the best sequence. The toy shows why a weaker first token can win overall. Probability is not truth."><title>Filter named tokens, renormalize, and distinguish a local choice from a complete sequence</title><desc>Temperature changes probabilities before filtering; order matters. Top-k keeps a fixed count; top-p includes the crossing token. Renormalize retained tokens, then sample and append one. Repeat from the new prefix until EOS or the budget. Greedy picks locally; beams compare scored prefixes but may prune the best sequence. The toy shows why a weaker first token can win overall. Probability is not truth.</desc><text x="20" y="27" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Keep token identity through a</tspan><tspan x="20" dy="18">declared decoding pipeline</tspan></text><text x="20" y="96" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Temperature one: original</tspan><tspan x="20" dy="18">probabilities</tspan></text><rect data-picture-decoding-token="A" data-picture-decoding-share="0.45" x="20" y="150" width="144" height="42" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="92" y="177" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="92" dy="0">A</tspan></text><rect data-picture-decoding-token="B" data-picture-decoding-share="0.25" x="164" y="150" width="80" height="42" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.5"></rect><text x="204" y="177" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="204" dy="0">B</tspan></text><rect data-picture-decoding-token="C" data-picture-decoding-share="0.15" x="244" y="150" width="48" height="42" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><text x="268" y="177" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="268" dy="0">C</tspan></text><rect data-picture-decoding-token="D" data-picture-decoding-share="0.1" x="292" y="150" width="32" height="42" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.5"></rect><text x="308" y="177" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="308" dy="0">D</tspan></text><rect data-picture-decoding-token="E" data-picture-decoding-share="0.05" x="324" y="150" width="16" height="42" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.5"></rect><text x="332" y="177" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="332" dy="0">E</tspan></text><text x="22" y="246" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="22" dy="0">A</tspan><tspan x="22" dy="18">0.450</tspan></text><text x="87" y="246" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--viz-focus-stroke)"><tspan x="87" dy="0">B</tspan><tspan x="87" dy="18">0.250</tspan></text><text x="152" y="246" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--viz-output-stroke)"><tspan x="152" dy="0">C</tspan><tspan x="152" dy="18">0.150</tspan></text><text x="217" y="246" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--viz-state-stroke)"><tspan x="217" dy="0">D</tspan><tspan x="217" dy="18">0.100</tspan></text><text x="282" y="246" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--viz-warning-stroke)"><tspan x="282" dy="0">E</tspan><tspan x="282" dy="18">0.050</tspan></text><text x="20" y="356" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Temperature two: flatten the same</tspan><tspan x="20" dy="18">allowed tokens</tspan></text><rect data-picture-decoding-token="A" data-picture-decoding-share="0.31974991813394843" x="20" y="410" width="102.3199738028635" height="42" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="71.15998690143175" y="437" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="71.15998690143175" dy="0">A</tspan></text><rect data-picture-decoding-token="B" data-picture-decoding-share="0.23832751758250045" x="122.3199738028635" y="410" width="76.26480562640015" height="42" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.5"></rect><text x="160.4523766160636" y="437" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="160.4523766160636" dy="0">B</tspan></text><rect data-picture-decoding-token="C" data-picture-decoding-share="0.18460770130799592" x="198.58477942926365" y="410" width="59.07446441855869" height="42" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><text x="228.122011638543" y="437" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="228.122011638543" dy="0">C</tspan></text><rect data-picture-decoding-token="D" data-picture-decoding-share="0.15073155693090556" x="257.6592438478223" y="410" width="48.23409821788978" height="42" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.5"></rect><text x="281.77629295676724" y="437" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="281.77629295676724" dy="0">D</tspan></text><rect data-picture-decoding-token="E" data-picture-decoding-share="0.10658330604464947" x="305.8933420657121" y="410" width="34.10665793428783" height="42" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.5"></rect><text x="322.946671032856" y="437" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="322.946671032856" dy="0">E</tspan></text><text x="22" y="506" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="22" dy="0">A</tspan><tspan x="22" dy="18">0.320</tspan></text><text x="87" y="506" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--viz-focus-stroke)"><tspan x="87" dy="0">B</tspan><tspan x="87" dy="18">0.238</tspan></text><text x="152" y="506" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--viz-output-stroke)"><tspan x="152" dy="0">C</tspan><tspan x="152" dy="18">0.185</tspan></text><text x="217" y="506" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--viz-state-stroke)"><tspan x="217" dy="0">D</tspan><tspan x="217" dy="18">0.151</tspan></text><text x="282" y="506" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--viz-warning-stroke)"><tspan x="282" dy="0">E</tspan><tspan x="282" dy="18">0.107</tspan></text><text x="20" y="618" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Below, keep temperature one and</tspan><tspan x="20" dy="18">compare two filters.</tspan></text><text x="20" y="708" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Top-k two: keep A and B</tspan></text><rect data-picture-decoding-token="A" data-picture-decoding-share="0.45" x="20" y="776" width="144" height="42" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="92" y="803" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="92" dy="0">A</tspan></text><rect data-picture-decoding-token="B" data-picture-decoding-share="0.25" x="164" y="776" width="80" height="42" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.5"></rect><text x="204" y="803" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="204" dy="0">B</tspan></text><rect data-picture-decoding-token="C" data-picture-decoding-share="0.15" x="244" y="776" width="48" height="42" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.5"></rect><text x="268" y="803" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="268" dy="0">C</tspan></text><rect data-picture-decoding-token="D" data-picture-decoding-share="0.1" x="292" y="776" width="32" height="42" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.5"></rect><text x="308" y="803" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="308" dy="0">D</tspan></text><rect data-picture-decoding-token="E" data-picture-decoding-share="0.05" x="324" y="776" width="16" height="42" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.5"></rect><text x="332" y="803" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="332" dy="0">E</tspan></text><text x="20" y="870" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Retained mass 0.70; gray tokens</tspan><tspan x="20" dy="18">cannot be sampled</tspan></text><rect data-picture-decoding-token="A" data-picture-decoding-share="0.6428571428571429" x="20" y="932" width="205.71428571428572" height="42" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="122.85714285714286" y="959" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="122.85714285714286" dy="0">A</tspan></text><rect data-picture-decoding-token="B" data-picture-decoding-share="0.35714285714285715" x="225.71428571428572" y="932" width="114.28571428571429" height="42" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.5"></rect><text x="282.8571428571429" y="959" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="282.8571428571429" dy="0">B</tspan></text><text x="22" y="1022" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="22" dy="0">A 0.643</tspan></text><text x="127" y="1022" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--viz-focus-stroke)"><tspan x="127" dy="0">B 0.357</tspan></text><text x="20" y="1043" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Top-p 0.8: A, B, C cross the required</tspan><tspan x="20" dy="18">mass</tspan></text><rect data-picture-decoding-token="A" data-picture-decoding-share="0.45" x="20" y="1111" width="144" height="42" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="92" y="1138" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="92" dy="0">A</tspan></text><rect data-picture-decoding-token="B" data-picture-decoding-share="0.25" x="164" y="1111" width="80" height="42" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.5"></rect><text x="204" y="1138" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="204" dy="0">B</tspan></text><rect data-picture-decoding-token="C" data-picture-decoding-share="0.15" x="244" y="1111" width="48" height="42" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><text x="268" y="1138" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="268" dy="0">C</tspan></text><rect data-picture-decoding-token="D" data-picture-decoding-share="0.1" x="292" y="1111" width="32" height="42" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.5"></rect><text x="308" y="1138" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="308" dy="0">D</tspan></text><rect data-picture-decoding-token="E" data-picture-decoding-share="0.05" x="324" y="1111" width="16" height="42" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.5"></rect><text x="332" y="1138" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="332" dy="0">E</tspan></text><text x="20" y="1205" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Retained mass 0.85; gray tokens</tspan><tspan x="20" dy="18">cannot be sampled</tspan></text><rect data-picture-decoding-token="A" data-picture-decoding-share="0.5294117647058824" x="20" y="1267" width="169.41176470588235" height="42" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="104.70588235294117" y="1294" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="104.70588235294117" dy="0">A</tspan></text><rect data-picture-decoding-token="B" data-picture-decoding-share="0.29411764705882354" x="189.41176470588235" y="1267" width="94.11764705882354" height="42" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.5"></rect><text x="236.47058823529412" y="1294" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="236.47058823529412" dy="0">B</tspan></text><rect data-picture-decoding-token="C" data-picture-decoding-share="0.17647058823529413" x="283.52941176470586" y="1267" width="56.47058823529412" height="42" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><text x="311.7647058823529" y="1294" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="311.7647058823529" dy="0">C</tspan></text><text x="22" y="1357" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="22" dy="0">A 0.529</tspan></text><text x="127" y="1357" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--viz-focus-stroke)"><tspan x="127" dy="0">B 0.294</tspan></text><text x="232" y="1357" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--viz-output-stroke)"><tspan x="232" dy="0">C 0.176</tspan></text><text x="20" y="1478" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Sample one retained token; append it</tspan><tspan x="20" dy="18">to the prefix</tspan></text><circle cx="75" cy="1590" r="29" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="75" y="1596" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="75" dy="0">prefix</tspan></text><path d="M113 1590L235 1590M227 1594L235 1590L227 1586" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:3;stroke-linejoin:round"></path><circle cx="278" cy="1590" r="35" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="278" y="1596" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="278" dy="0">prefix</tspan><tspan x="278" dy="18">B</tspan></text><text x="20" y="1680" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Toy draw selects B. The next</tspan><tspan x="20" dy="18">distribution uses this new prefix.</tspan></text><text x="20" y="1820" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Separate two-token toy: stop after</tspan><tspan x="20" dy="18">two tokens</tspan></text><text x="20" y="1904" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">First A: chance 0.6</tspan></text><circle cx="55" cy="1965" r="24" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><text x="55" y="1971" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="55" dy="0">A</tspan></text><path d="M87 1965L272 1965M264 1969L272 1965L264 1961" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:2;stroke-linejoin:round"></path><text x="180" y="1938" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="180" dy="0">Best next X: 0.5</tspan></text><circle cx="305" cy="1965" r="24" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><text x="305" y="1971" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="305" dy="0">X</tspan></text><text x="20" y="2056" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-warning-stroke)"><tspan x="20" dy="0">AX complete score: 0.30</tspan></text><text x="20" y="2149" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">First B: chance 0.4</tspan></text><circle cx="55" cy="2210" r="24" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="55" y="2216" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="55" dy="0">B</tspan></text><path d="M87 2210L272 2210M264 2214L272 2210L264 2206" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:2;stroke-linejoin:round"></path><text x="180" y="2183" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="180" dy="0">Best next X: 0.9</tspan></text><circle cx="305" cy="2210" r="24" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="305" y="2216" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="305" dy="0">X</tspan></text><text x="20" y="2301" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-output-stroke)"><tspan x="20" dy="0">BX complete score: 0.36</tspan></text><text x="20" y="2395" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Greedy starts A; retaining B reveals</tspan><tspan x="20" dy="18">the better complete score. A finite</tspan><tspan x="20" dy="18">beam can still miss other paths.</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
	<div class="visual-grid--two" role="group" aria-label="Comparison of top-k and top-p candidate supports for confident and uncertain next-token distributions">
		<section class="visual-panel plot-panel">
			<svg viewBox="0 0 300 266" role="img" aria-labelledby="decoding-confident-title decoding-confident-desc">
				<title id="decoding-confident-title">Candidate supports for a confident distribution</title>
				<desc id="decoding-confident-desc">Seven sorted token probabilities are 72, 12, 6, 4, 3, 2, and 1 percent. Top-k with k equals 3 always keeps tokens A through C and captures 90 percent mass. Top-p with p equals 0.80 keeps only A and B, the smallest prefix reaching at least 80 percent, and captures 84 percent mass.</desc>
				<rect class="viz-plot-bg" x="8" y="25" width="284" height="232" rx="5"></rect>
				<text class="viz-axis-label" x="12" y="16">CONFIDENT NEXT TOKEN</text>
				<path class="viz-gridline" d="M18 160H282"></path>
				<g class="viz-label" text-anchor="middle">
					<rect class="viz-node viz-node--focus" x="20" y="52" width="26" height="108"></rect><text x="33" y="47">72%</text><text x="33" y="176">A</text>
					<rect class="viz-node" x="58" y="142" width="26" height="18"></rect><text x="71" y="137">12%</text><text x="71" y="176">B</text>
					<rect class="viz-node" x="96" y="151" width="26" height="9"></rect><text x="109" y="146">6%</text><text x="109" y="176">C</text>
					<rect class="viz-node" x="134" y="154" width="26" height="6"></rect><text x="147" y="149">4%</text><text x="147" y="176">D</text>
					<rect class="viz-node" x="172" y="155.5" width="26" height="4.5"></rect><text x="185" y="150.5">3%</text><text x="185" y="176">E</text>
					<rect class="viz-node" x="210" y="157" width="26" height="3"></rect><text x="223" y="152">2%</text><text x="223" y="176">F</text>
					<rect class="viz-node" x="248" y="158.5" width="26" height="1.5"></rect><text x="261" y="153.5">1%</text><text x="261" y="176">G</text>
				</g>
				<path d="M20 190V196H122V190" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2"></path>
				<text class="viz-callout" x="150" y="210" text-anchor="middle">top-k = 3 · 3 tokens · 90% mass</text>
				<path d="M20 220V226H84V220" style="fill:none;stroke:var(--viz-edge);stroke-width:2;stroke-dasharray:5 3"></path>
				<text class="viz-callout" x="20" y="243">top-p = .80 · 2 tokens · 84% mass</text>
			</svg>
		</section>
		<section class="visual-panel plot-panel">
			<svg viewBox="0 0 300 266" role="img" aria-labelledby="decoding-uncertain-title decoding-uncertain-desc">
				<title id="decoding-uncertain-title">Candidate supports for an uncertain distribution</title>
				<desc id="decoding-uncertain-desc">Seven sorted token probabilities are 24, 20, 17, 14, 11, 8, and 6 percent. Top-k with k equals 3 still keeps tokens A through C but now captures only 61 percent mass. Top-p with p equals 0.80 expands through token E, the smallest prefix reaching at least 80 percent, and captures 86 percent mass.</desc>
				<rect class="viz-plot-bg" x="8" y="25" width="284" height="232" rx="5"></rect>
				<text class="viz-axis-label" x="12" y="16">UNCERTAIN NEXT TOKEN</text>
				<path class="viz-gridline" d="M18 160H282"></path>
				<g class="viz-label" text-anchor="middle">
					<rect class="viz-node viz-node--focus" x="20" y="124" width="26" height="36"></rect><text x="33" y="119">24%</text><text x="33" y="176">A</text>
					<rect class="viz-node" x="58" y="130" width="26" height="30"></rect><text x="71" y="125">20%</text><text x="71" y="176">B</text>
					<rect class="viz-node" x="96" y="134.5" width="26" height="25.5"></rect><text x="109" y="129.5">17%</text><text x="109" y="176">C</text>
					<rect class="viz-node" x="134" y="139" width="26" height="21"></rect><text x="147" y="134">14%</text><text x="147" y="176">D</text>
					<rect class="viz-node" x="172" y="143.5" width="26" height="16.5"></rect><text x="185" y="138.5">11%</text><text x="185" y="176">E</text>
					<rect class="viz-node" x="210" y="148" width="26" height="12"></rect><text x="223" y="143">8%</text><text x="223" y="176">F</text>
					<rect class="viz-node" x="248" y="151" width="26" height="9"></rect><text x="261" y="146">6%</text><text x="261" y="176">G</text>
				</g>
				<path d="M20 190V196H122V190" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2"></path>
				<text class="viz-callout" x="150" y="210" text-anchor="middle">top-k = 3 · 3 tokens · 61% mass</text>
				<path d="M20 220V226H198V220" style="fill:none;stroke:var(--viz-edge);stroke-width:2;stroke-dasharray:5 3"></path>
				<text class="viz-callout" x="20" y="243">top-p = .80 · 5 tokens · 86% mass</text>
			</svg>
		</section>
	</div>
	<div class="visual-derivation-caption">both panels use the same vertical probability scale. Solid top-k brackets retain three tokens with masses 0.90 and 0.61. Dashed top-p brackets retain two and five tokens, reaching at least 0.80 with masses 0.84 and 0.86. Divide by the retained mass before sampling; do not sample from the original probabilities without renormalizing.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="decoding-fixed-count-adaptive-mass"><strong>Check:</strong> Why does the uncertain distribution need five tokens rather than four to reach 80% mass?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="decoding-fixed-count-adaptive-mass">Temperature changes probabilities before filtering; order matters. Top-k keeps a fixed count; top-p includes the crossing token. Renormalize retained tokens, then sample and append one. Repeat from the new prefix until EOS or the budget. Greedy picks locally; beams compare scored prefixes but may prune the best sequence. The toy shows why a weaker first token can win overall. Probability is not truth. <strong>Why must the 0.8 cutoff include a third token?</strong></span></figcaption>
</figure>

Confident top-k becomes $(0.8,2/15,1/15)$; confident top-p becomes $(6/7,1/7)$.
Uncertain top-k becomes $(24,20,17)/61$; uncertain top-p becomes $(24,20,17,14,11)/86$.

### Change processor order

**New task:** use $T=2$ and top-p $0.85$ on probabilities $(0.6,0.3,0.1)$.

**Picture and change:** temperature transforms probabilities proportionally to their square roots.

**Check:** temperature first gives approximately $(0.472734,0.334273,0.192993)$, requiring all three tokens.
Truncation first retains mass $0.9$, then temperature gives $(0.585786,0.414214)$.

**Use when:** reproducing a serving pipeline; processor order is part of the distribution.

### Search complete sequences

**New task:** maximize a two-token sequence's probability, then stop.

**Picture and change:** first-token probabilities are $p(A)=0.6,p(B)=0.4$.
Their best continuation probabilities are $0.5$ and $0.9$.

**Check:** greedy follows A, obtaining joint probability $0.30$.
A width-two search can retain B and its $0.36$ completion.

**Use when:** sequence-score search, rather than a single-step sample, matches the actual objective.

### Min-p sampling

Retain $p_i\geq p_{\min}\max_j p_j$ and renormalize.
This compares individual probabilities with the maximum, not cumulative mass.

<!-- learning:core-end -->

<div data-learning-extension>

## Repetition penalty and other modifiers

Real systems layer modifications on top of the base sampler:

- **Repetition penalty:** one sign-aware convention divides positive repeated-token logits by $c>1$ and multiplies negative ones by $c$.
- **Frequency / presence penalty**: linear adjustment based on how often a token has appeared.
- **No-repeat n-gram**: forbid repeating any n-gram already in the output.
- **Logit bias:** add token-specific offsets; use an explicit hard mask to forbid tokens.

Dividing a negative logit alone is wrong: $-2/2=-1$ increases its score, while $-2\times2=-4$ decreases it.
Grammar constraints restrict allowed tokens but do not establish semantic or factual correctness.

## Choosing per task

| Task | Policy to evaluate |
|---|---|
| Translation, QA, summarization | Compare search/sampling against task metrics, not likelihood alone |
| Code | Declare sampling and stopping; evaluate generated programs |
| Open chat | Evaluate quality and diversity under fixed settings |
| Creative writing | Choose support and temperature against a declared objective |
| Structured output | Enforce grammar, then validate the completed result |

## Common pitfalls

Compare models under declared policies, random states, masks, and stop rules.
An EOS mask can make a budget endpoint differ from a naturally completed sequence.
Zero temperature is an implementation's argmax policy, not a division operation.

What does min-p $0.2$ retain from the uncertain seven-token fixture?

<details><summary>Answer</summary>

All seven: the threshold is $0.2(0.24)=0.048$, below the smallest probability $0.06$.
Neither top-k's count nor top-p's cumulative threshold describes that support.

</details>

</div>

## Related

- [Speculative decoding](/concepts/speculative-decoding/). Faster decoding, same distribution.
- [Perplexity](/concepts/perplexity-and-bits-per-token/).
