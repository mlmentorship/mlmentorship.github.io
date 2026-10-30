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
	<p class="visual-title" id="decoding-support-title">Top-k fixes a count; top-p changes the count to reach a probability mass</p><div class="visual-argument" data-visual-argument="decoding-fixed-count-adaptive-mass" data-argument-kind="parts"><p class="visual-argument-premise">A probability threshold chooses a count only after seeing the distribution</p><svg viewBox="0 0 360 400" role="img" aria-label="A probability threshold chooses a count only after seeing the distribution. Top-k with k=2 keeps mass 0.70 here. Top-p with p=0.8 needs the third token too, reaching mass 0.85."><title>A probability threshold chooses a count only after seeing the distribution</title><desc>Top-k with k=2 keeps mass 0.70 here. Top-p with p=0.8 needs the third token too, reaching mass 0.85.</desc><text x="20" y="32" style="font-size:18px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">1 probability mass total</tspan></text><rect data-proof-part="first token" data-proof-value="0.45" x="20" y="55" width="144" height="50" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><rect data-proof-part="second token" data-proof-value="0.25" x="164" y="55" width="80" height="50" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><rect data-proof-part="third token: needed for p=.8" data-proof-value="0.15" x="244" y="55" width="48" height="50" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><rect data-proof-part="remaining two tokens" data-proof-value="0.15" x="292" y="55" width="48" height="50" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.5"></rect><circle cx="27" cy="143" r="5" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="45" y="147" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="45" dy="0">0.45: first token</tspan></text><text x="45" y="170" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="45" dy="0">0.45 / 1 = 45.0%</tspan></text><circle cx="27" cy="201" r="5" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="45" y="205" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="45" dy="0">0.25: second token</tspan></text><text x="45" y="228" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="45" dy="0">0.25 / 1 = 25.0%</tspan></text><circle cx="27" cy="259" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="45" y="263" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="45" dy="0">0.15: third token: needed for</tspan><tspan x="45" dy="18">p=.8</tspan></text><text x="45" y="304" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--viz-output-stroke)"><tspan x="45" dy="0">0.15 / 1 = 15.0%</tspan></text><circle cx="27" cy="335" r="5" style="fill:var(--viz-neutral-stroke);stroke:var(--viz-neutral-stroke);stroke-width:1.75"></circle><text x="45" y="339" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="45" dy="0">0.15: remaining two tokens</tspan></text><text x="45" y="362" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="45" dy="0">0.15 / 1 = 15.0%</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
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
	<div class="visual-derivation-caption">both panels use the same vertical probability scale. Solid top-k brackets retain three tokens with masses 0.90 and 0.61. Dashed top-p brackets retain two and five tokens, reaching at least 0.80 with masses 0.84 and 0.86. Divide by the retained mass before sampling; do not sample from the original probabilities without renormalizing.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="decoding-fixed-count-adaptive-mass"><strong>Check:</strong> Why does the uncertain distribution need five tokens rather than four to reach 80% mass?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="decoding-fixed-count-adaptive-mass">Top-k with k=2 keeps mass 0.70 here. Top-p with p=0.8 needs the third token too, reaching mass 0.85.</span></figcaption>
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
