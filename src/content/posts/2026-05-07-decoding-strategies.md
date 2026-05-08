---
title: "Decoding strategies: greedy, beam, top-k, top-p, temperature"
description: "Same model, different samplers, very different outputs. The choice of decoder is often more impactful than the last percent of training. Know the tradeoffs."
date: "2026-05-07"
draft: false
tags: ["concepts"]
category: "concepts"
---

## One-line definition

**Decoding** turns a language model's per-token distributions into actual text. Strategies differ in how they pick the next token from the distribution. Each makes a tradeoff between fidelity (high probability) and diversity (broader sampling).

## Why it matters

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
