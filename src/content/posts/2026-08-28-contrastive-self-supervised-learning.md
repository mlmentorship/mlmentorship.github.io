---
title: "Contrastive and self-supervised learning"
description: "Learn useful representations from unlabeled data by defining which views should agree and which examples should stay apart."
date: "2026-08-28"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Self-supervised learning creates training targets from the data itself. Contrastive learning is one form: it brings related examples closer in an embedding space and pushes unrelated examples apart.

## Why AI labs care

Most available data has no human label. Self-supervised objectives let models learn from text, images, audio, video, code, and interaction logs at large scale.

Examples include:

- next-token prediction for language models;
- masked-token prediction for BERT;
- masked audio prediction for speech models;
- image reconstruction for masked autoencoders;
- image-text matching for CLIP;
- two augmented views of one image for SimCLR.

The hard part is choosing a task that teaches information useful for later work.

## Contrastive learning

Let $z_i$ be the embedding of an example. For an anchor $i$, choose a positive example $j$ that should have a similar representation. Other examples in the batch act as negatives.

A common loss is InfoNCE:

$$
\mathcal{L}_i = -\log
\frac{\exp(\operatorname{sim}(z_i,z_j)/\tau)}
{\sum_k \exp(\operatorname{sim}(z_i,z_k)/\tau)}.
$$

- $\operatorname{sim}$ is often cosine similarity.
- $\tau$ is the temperature. A lower value makes the model focus more on the hardest comparisons.
- The denominator contains the positive and candidate negatives.

The loss teaches the model to identify the positive among the candidates.

## The positive pair defines the representation

A positive pair tells the model which changes should not alter meaning.

Examples:

| Task | Positive pair | Invariance learned |
| --- | --- | --- |
| Image learning | Two crops of one image | Crop, color, and small viewpoint changes |
| Image-text | Image and its caption | Cross-modal meaning |
| Search | Query and relevant document | Relevance |
| Recommendation | User context and engaged item | Preference under the logging policy |
| Speech | Two views of one utterance | Noise or channel changes |

A bad positive definition teaches the wrong invariance. If two crops remove the object, the model is asked to match unrelated content. If clicks define positives, position bias becomes part of the representation.

## Choosing negatives

Useful negatives are plausible alternatives. Very easy negatives add little signal.

Risks:

- **False negatives:** two examples are treated as unrelated even though both are valid matches.
- **Sampling bias:** in-batch negatives come from a distribution that may differ from serving traffic.
- **Popularity bias:** common items appear as negatives more often and receive different training pressure.
- **Shortcut features:** the model separates examples using source, formatting, or language instead of meaning.

Hard-negative mining can help. It can also select mislabeled false negatives. Review mined examples and track performance by slice.

## Representation collapse

A collapsed model maps every input to the same vector. It satisfies some similarity goals without learning useful information.

Contrastive negatives prevent the simplest collapse. Other self-supervised methods use stop-gradient paths, predictors, variance constraints, or decorrelation terms instead of explicit negatives.

The goal is not only to avoid identical vectors. A representation can keep enough variance while encoding the wrong features. Evaluate it on the target use.

## How to evaluate representations

Use several checks:

1. **Linear probe:** freeze embeddings and train a simple linear model.
2. **Retrieval:** test whether relevant examples appear near each other.
3. **Transfer:** fine-tune with limited labels on a new task.
4. **Robustness:** test noise, domain, language, or viewpoint shifts.
5. **Slice analysis:** measure rare groups and long-tail items.
6. **Efficiency:** measure embedding size, index cost, and serving latency.

A two-dimensional plot is useful for inspection. It is not enough to prove representation quality.

## Contrastive versus generative objectives

Contrastive objectives focus on relationships among examples. Generative objectives model the input or missing parts of it.

Use contrastive learning when matching and retrieval are central. Use generative or masked objectives when detailed content and generation matter. Many systems combine both.

## In an interview

Use this order:

1. Define the downstream task.
2. Define the positive pair and intended invariance.
3. Explain negative sampling and false-negative risk.
4. State the loss and role of temperature.
5. Discuss collapse and shortcut learning.
6. Evaluate transfer, retrieval, slices, and serving cost.

## Common mistakes

- Saying "use contrastive loss" without defining positives.
- Treating every other batch item as a true negative.
- Choosing augmentations that remove task-relevant information.
- Evaluating only on the pretraining dataset.
- Assuming larger batches always help.
- Reading a t-SNE plot as proof of useful clusters.

*Related: [embedding spaces and similarity](/concepts/embedding-spaces-and-similarity/), [two-tower retrieval](/concepts/two-tower-retrieval/), and [multimodal foundation models](/concepts/multimodal-foundation-models/).*