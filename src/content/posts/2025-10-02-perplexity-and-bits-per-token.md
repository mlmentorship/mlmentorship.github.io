---
title: "Perplexity and bits per token"
description: "The standard intrinsic metric for language models. What it measures, what units to use, and why it's a poor end-product evaluation."
date: "2025-10-02"
draft: false
tags: ["reference"]
category: "reference"
---

## One-line definition

Perplexity is the exponentiated average per-token cross-entropy of a language model on a held-out corpus: $\text{PPL} = \exp(\bar{L})$ where $\bar L = -\frac{1}{N} \sum_{i=1}^{N} \log p(x_i \mid x_{<i})$. **Bits per token (BPT)** is the same quantity in $\log_2$ units: $\text{BPT} = \bar L / \ln 2$.

## Why it matters

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

- [Tokenization](/reference/tokenization/). Affects how perplexity is computed.
- [LLM evals: the hardest part](/essays/llm-evals-the-hardest-part/). Why intrinsic metrics fail for production LLMs.
