---
title: "Decoding strategies: greedy, beam, top-k, top-p, temperature"
description: "Same model, different samplers, very different outputs. The choice of decoder is often more impactful than the last percent of training. Know the tradeoffs."
date: "2026-05-07"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

**Decoding** turns a language model's per-token distributions into actual text. Strategies differ in how they pick the next token from the distribution. Each makes a tradeoff between fidelity (high probability) and diversity (broader sampling).

A trained LLM produces a probability distribution $p(x_t \mid x_{<t})$ at every step. The text the user sees depends entirely on how you sample from those distributions. Bad decoding makes a strong model look weak: greedy can repeat, beam can be bland, pure sampling can be incoherent. Modern systems typically combine top-p with a moderate temperature, but the right choice depends on the task.

## The strategies

### Greedy

Pick $\arg\max_x p(x \mid x_{<t})$ at every step.

- **Pros**: deterministic, fast, optimal for tasks where the highest-probability completion is the right answer (translation with strong evidence, classification reformulated as generation).
- **Cons**: gets stuck in loops (the same token becomes most probable again because the previous step made it more likely). Bland and repetitive on open-ended generation.

### Beam search

Track the $k$ highest-probability sequences at every step. Expand each, keep the top $k$.

- **Pros**: better than greedy for tasks with a clear correct answer (machine translation, summarization).
- **Cons**: tends to produce short, bland outputs. The most likely sequence under the model is often boring or repetitive ([Holtzman et al., 2020](https://arxiv.org/abs/1904.09751)). Length-normalization tweaks help but do not fully fix this.

### Temperature sampling

Sample from $\text{softmax}(z / T)$ where $z$ is the logit vector and $T$ is the temperature.

- $T = 1$: model's native distribution.
- $T < 1$: sharper, more deterministic. $T \to 0$ recovers greedy.
- $T > 1$: flatter, more diverse. $T \to \infty$ recovers uniform.

Default for chat: $T \approx 0.7$. Default for code: lower ($T \approx 0.2$).

### Top-k sampling

Restrict the sampling pool to the $k$ most probable tokens, then sample with temperature ([Fan et al., 2018](https://arxiv.org/abs/1805.04833)).

- **Cons**: $k$ is a hyperparameter, but the right $k$ depends on the entropy of the distribution. When the model is very confident, $k = 50$ includes garbage; when it is uncertain, $k = 50$ may be too restrictive.

### Top-p (nucleus) sampling

Restrict the sampling pool to the smallest set of tokens whose cumulative probability exceeds $p$ ([Holtzman et al., 2020](https://arxiv.org/abs/1904.09751)).

- $p = 0.9$ is the modern default.
- Adapts to the entropy of each step: confident steps sample from a small set, uncertain steps from a larger one. The standard choice for open-ended generation.

<!-- visual:decoding-fixed-count-adaptive-mass -->
<figure class="learning-figure" aria-labelledby="decoding-support-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="decoding-support-title">Why does top-p adapt while top-k does not?</p>
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
					<rect class="viz-node" x="172" y="155" width="26" height="5"></rect><text x="185" y="150">3%</text><text x="185" y="176">E</text>
					<rect class="viz-node" x="210" y="156" width="26" height="4"></rect><text x="223" y="151">2%</text><text x="223" y="176">F</text>
					<rect class="viz-node" x="248" y="156" width="26" height="4"></rect><text x="261" y="151">1%</text><text x="261" y="176">G</text>
				</g>
				<path d="M20 190V196H122V190" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2"></path>
				<text class="viz-callout" x="71" y="210" text-anchor="middle">top-k = 3 · 3 tokens · 90% mass</text>
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
					<rect class="viz-node viz-node--focus" x="20" y="88" width="26" height="72"></rect><text x="33" y="83">24%</text><text x="33" y="176">A</text>
					<rect class="viz-node" x="58" y="100" width="26" height="60"></rect><text x="71" y="95">20%</text><text x="71" y="176">B</text>
					<rect class="viz-node" x="96" y="109" width="26" height="51"></rect><text x="109" y="104">17%</text><text x="109" y="176">C</text>
					<rect class="viz-node" x="134" y="118" width="26" height="42"></rect><text x="147" y="113">14%</text><text x="147" y="176">D</text>
					<rect class="viz-node" x="172" y="127" width="26" height="33"></rect><text x="185" y="122">11%</text><text x="185" y="176">E</text>
					<rect class="viz-node" x="210" y="136" width="26" height="24"></rect><text x="223" y="131">8%</text><text x="223" y="176">F</text>
					<rect class="viz-node" x="248" y="142" width="26" height="18"></rect><text x="261" y="137">6%</text><text x="261" y="176">G</text>
				</g>
				<path d="M20 190V196H122V190" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2"></path>
				<text class="viz-callout" x="71" y="210" text-anchor="middle">top-k = 3 · 3 tokens · 61% mass</text>
				<path d="M20 220V226H198V220" style="fill:none;stroke:var(--viz-edge);stroke-width:2;stroke-dasharray:5 3"></path>
				<text class="viz-callout" x="20" y="243">top-p = .80 · 5 tokens · 86% mass</text>
			</svg>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> compare the solid top-k brackets first: both keep exactly three tokens, although their captured probability mass falls from 90% to 61%. Then compare the dashed top-p brackets: the candidate pool expands from two tokens to five so that each pool reaches the 80% mass threshold. Probabilities are renormalized over the retained pool before sampling.</figcaption>
</figure>

### Min-p sampling

Restrict to tokens with probability at least $p_{\min} \cdot \max_x p(x)$. Closer to "filter out implausible options" than nucleus's "keep the top mass."

## Repetition penalty and other modifiers

Real systems layer modifications on top of the base sampler:

- **Repetition penalty**: divide logits of recently used tokens by some factor. Prevents loops.
- **Frequency / presence penalty**: linear adjustment based on how often a token has appeared.
- **No-repeat n-gram**: forbid repeating any n-gram already in the output.
- **Logit bias**: add a constant to specific token logits to nudge or forbid them.

## Choosing per task

| Task | Default |
|---|---|
| Translation, factual QA, summarization with reference | Beam (k=4 to 8) or low-temperature greedy |
| Code generation | Temperature 0.2 + top-p 0.95, or greedy with a stop-condition |
| Open chat | Temperature 0.7 + top-p 0.9 |
| Creative writing | Temperature 0.9 to 1.2 + top-p 0.95 |
| Constrained / structured output | Greedy + grammar-guided decoding (constrained decoding) |

## Common pitfalls

- **Comparing models with different decoders.** The decoding strategy is part of the system. State it.
- **Using beam search for open-ended generation.** Likelihood maximization is not the goal here.
- **Setting temperature to 0 and calling it "deterministic."** It is, modulo numerical ties at the argmax. With ties, behavior is library-dependent.
- **Mixing temperature and top-k/top-p naively.** The order matters: typical implementations apply top-p truncation first, then temperature, then sample. Verify your stack.

## Related

- [Speculative decoding](/concepts/speculative-decoding/). Faster decoding, same distribution.
- [Perplexity](/concepts/perplexity-and-bits-per-token/).
