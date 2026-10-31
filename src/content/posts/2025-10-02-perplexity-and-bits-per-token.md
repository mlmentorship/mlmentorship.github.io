---
title: "Perplexity and bits per token"
description: "The standard intrinsic metric for language models. What it measures, what units to use, and why it's a poor end-product evaluation."
date: "2025-10-02"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Perplexity is the exponentiated average per-token cross-entropy of a language model on a held-out corpus: $\text{PPL} = \exp(\bar{L})$ where $\bar L = -\frac{1}{N} \sum_{i=1}^{N} \log p(x_i \mid x_{<i})$. **Bits per token (BPT)** is the same quantity in $\log_2$ units: $\text{BPT} = \bar L / \ln 2$.

Perplexity is the dominant intrinsic metric used during pretraining: cheap to compute, smooth, monotonically related to next-token log-likelihood. Loss curves are usually reported as cross-entropy or perplexity. Scaling laws ([Kaplan et al., 2020](https://arxiv.org/abs/2001.08361); [Chinchilla, 2022](https://arxiv.org/abs/2203.15556)) are derived in perplexity / loss space.

But: perplexity is a **poor evaluation of model usefulness**. Lower perplexity does not reliably mean better summarization, instruction following, code, or chat. For these you need task evals (HumanEval, MMLU, MT-Bench, etc.).

## The formula

For a tokenized sequence $x_1, \dots, x_N$ and a model $p$:

$$
\bar L = -\frac{1}{N} \sum_{i=1}^{N} \log p(x_i \mid x_{<i})
$$

$\bar L$ is the average cross-entropy in nats (natural log) per token. Then:

- **Perplexity**: $\text{PPL} = \exp(\bar L)$
- **Bits per token**: $\text{BPT} = \bar L / \ln 2 = \log_2(\text{PPL})$
- **Bits per byte**: BPT divided by mean bytes per token (depends on tokenizer)

Interpretation of perplexity: the model is "as confused as if it had to choose uniformly among $\text{PPL}$ words." A perplexity of 20 means the model's predictions are as good as a uniform distribution over 20 next-token candidates.

## Tokenizer dependence

Perplexity is **not comparable across tokenizers**. Different tokenizers split the same text into different numbers of tokens, so the per-token cross-entropy differs even when the model perfectly captures the underlying language.

To compare across tokenizers, convert to **bits per byte (BPB)** or **bits per character**:

$$
\text{BPB} = \frac{\text{total bits to encode}}{\text{total bytes in corpus}}
$$

This is invariant to tokenization and is the proper cross-tokenizer comparison metric.

<!-- visual:perplexity-tokenizer-denominator -->
<figure class="learning-figure" aria-labelledby="perplexity-denominator-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="perplexity-denominator-title">Bits per token cannot compare models fairly when token units differ</p><div class="visual-argument" data-visual-argument="perplexity-tokenizer-denominator" data-argument-kind="ledger" data-argument-composition="sequence"><svg viewBox="0 0 360 376" role="img" aria-label="A token is not a fixed amount of text. Two tokenizations of the same four characters use different denominators. Compare total code length per shared text unit, not raw perplexity per token."><title>A token is not a fixed amount of text</title><desc>Two tokenizations of the same four characters use different denominators. Compare total code length per shared text unit, not raw perplexity per token.</desc><text x="20" y="28" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Tokenizer A; four text characters</tspan></text><g data-proof-token="a" data-proof-row="0" data-proof-position="0" data-proof-box="20,60,74,48"><rect x="20" y="60" width="74" height="48" rx="0" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="57" y="87" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="57" dy="0">a</tspan></text></g><path d="M97 84L101 84" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><g data-proof-token="b" data-proof-row="0" data-proof-position="1" data-proof-box="100,60,74,48"><rect x="100" y="60" width="74" height="48" rx="0" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="137" y="87" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="137" dy="0">b</tspan></text></g><path d="M177 84L181 84" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><g data-proof-token="c" data-proof-row="0" data-proof-position="2" data-proof-box="180,60,74,48"><rect x="180" y="60" width="74" height="48" rx="0" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="217" y="87" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="217" dy="0">c</tspan></text></g><path d="M257 84L261 84" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><g data-proof-token="d" data-proof-row="0" data-proof-position="3" data-proof-box="260,60,74,48"><rect x="260" y="60" width="74" height="48" rx="0" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="297" y="87" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="297" dy="0">d</tspan></text></g><text x="20" y="144" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Tokenizer B; same characters</tspan></text><g data-proof-token="ab" data-proof-row="1" data-proof-position="0" data-proof-box="20,176,154,48"><rect x="20" y="176" width="154" height="48" rx="0" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.5"></rect><text x="97" y="203" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="97" dy="0">ab</tspan></text></g><path d="M177 200L181 200" style="fill:none;stroke:var(--viz-state-stroke);stroke-width:1.75"></path><g data-proof-token="cd" data-proof-row="1" data-proof-position="1" data-proof-box="180,176,154,48"><rect x="180" y="176" width="154" height="48" rx="0" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.5"></rect><text x="257" y="203" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="257" dy="0">cd</tspan></text></g><text x="20" y="260" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Shared comparison unit</tspan></text><g data-proof-token="bits" data-proof-row="2" data-proof-position="0" data-proof-box="20,292,100.66666666666667,48"><rect x="20" y="292" width="100.66666666666667" height="48" rx="0" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><text x="70.33333333333334" y="319" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="70.33333333333334" dy="0">bits</tspan></text></g><path d="M123.66666666666667 316L127.66666666666667 316" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75"></path><g data-proof-token="/" data-proof-row="2" data-proof-position="1" data-proof-box="126.66666666666667,292,100.66666666666667,48"><text x="177" y="319" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="177" dy="0">/</tspan></text></g><path d="M230.33333333333334 316L234.33333333333334 316" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75"></path><g data-proof-token="character" data-proof-row="2" data-proof-position="2" data-proof-box="233.33333333333334,292,100.66666666666667,48"><rect x="233.33333333333334" y="292" width="100.66666666666667" height="48" rx="0" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><text x="283.6666666666667" y="319" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="283.6666666666667" dy="0">character</tspan></text></g></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
	<div class="visual-grid--two" role="group" aria-label="Two tokenizations of the six-byte ASCII string sunset, each assigned five total bits of surprisal">
		<section class="visual-panel" aria-labelledby="perplexity-two-token-title">
			<h4 id="perplexity-two-token-title">Tokenizer A: 2 tokens</h4>
			<p><code>[sun] [set]</code> with surprisals 3 + 2 bits</p>
			<table class="cm-grid" aria-label="Metrics for the two-token segmentation">
				<thead><tr><th scope="col">Total</th><th scope="col">BPT</th><th scope="col">BPB</th></tr></thead>
				<tbody><tr><td><strong>5 bits</strong></td><td class="cm-selected"><strong>2.50</strong>5 / 2 tokens</td><td><strong>0.83</strong>5 / 6 bytes</td></tr></tbody>
			</table>
		</section>
		<section class="visual-panel" aria-labelledby="perplexity-three-token-title">
			<h4 id="perplexity-three-token-title">Tokenizer B: 3 tokens</h4>
			<p><code>[s] [un] [set]</code> with surprisals 1 + 1 + 3 bits</p>
			<table class="cm-grid" aria-label="Metrics for the three-token segmentation">
				<thead><tr><th scope="col">Total</th><th scope="col">BPT</th><th scope="col">BPB</th></tr></thead>
				<tbody><tr><td><strong>5 bits</strong></td><td class="cm-selected"><strong>1.67</strong>5 / 3 tokens</td><td><strong>0.83</strong>5 / 6 bytes</td></tr></tbody>
			</table>
		</section>
	</div>
	<div class="visual-derivation-caption"><strong>Read it this way:</strong> both rows encode the same six UTF-8 bytes with the same five bits of total surprisal. BPT changes from 2.50 to 1.67 only because each tokenizer chooses a different denominator; BPB divides by the fixed byte count and stays 0.83. Real models need not assign equal total surprisal, but byte normalization makes their reported unit comparable.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="perplexity-tokenizer-denominator"><strong>Check:</strong> Why does 1.67 bits per token not establish a better model than 2.50 in the illustrated comparison?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="perplexity-tokenizer-denominator">Two tokenizations of the same four characters use different denominators. Compare total code length per shared text unit, not raw perplexity per token.</span></figcaption>
</figure>

## Typical numbers (for context)

On a clean English text corpus:

| Model | Perplexity | BPB |
|-------|-----------|-----|
| Trigram (Brown corpus 1992) | ~150 | ~1.5 |
| LSTM ([Mikolov 2010](https://www.fit.vut.cz/research/group/speech/public/publi/2010/mikolov_interspeech2010_IS100722.pdf), PTB) | ~80 |. |
| GPT-2 small ([Radford 2019](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), WebText) | ~30 | ~1.0 |
| GPT-3 ([Brown 2020](https://arxiv.org/abs/2005.14165), books) | ~10–20 | ~0.7 |
| Frontier 2024–2026 LLM | ~5–10 | ~0.5 |

The lower bound (compressed bits per byte of natural English) is estimated at ~0.6 bits/byte (Shannon).

## Why perplexity ≠ usefulness

A model that has memorized its training data has perplexity 1 on it but doesn't generalize. A model with low perplexity on web text may still:

- Generate fluent nonsense in long-form generation.
- Fail at instruction following (perplexity isn't measured on that distribution).
- Fail at tool use, reasoning, or code (a small perplexity gap can hide a huge capability gap).

Production model evaluation always requires task evals.

## Common pitfalls

- **Comparing perplexities across tokenizers.** Use BPB.
- **Reporting perplexity on the training set.** Always use a held-out test corpus.
- **Reporting perplexity on a domain very different from your eval target.** A model with low perplexity on Wikipedia may be terrible at code; pick a test corpus that matches the deployment distribution.
- **Treating a perplexity drop as a quality win.** Sub-half-bit improvements are common during fine-tuning but don't translate to user-facing quality.

## Related

- [Tokenization](/concepts/tokenization/). Affects how perplexity is computed.
- [LLM evals: the hardest part](/guides/llm-evals-the-hardest-part/). Why intrinsic metrics fail for production LLMs.
