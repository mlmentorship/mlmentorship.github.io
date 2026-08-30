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

<!-- visual:infonce-positive-candidate-set -->
<figure class="learning-figure plot-panel" aria-labelledby="infonce-candidates-visual-title">
	<p class="visual-kicker">Candidate-set intuition</p>
	<p class="visual-title" id="infonce-candidates-visual-title">InfoNCE rewards one declared positive while every other candidate competes with it.</p>
	<div class="visual-scroll">
		<svg viewBox="0 0 360 370" role="img" aria-labelledby="infonce-candidates-svg-title infonce-candidates-svg-desc">
			<title id="infonce-candidates-svg-title">One anchor, one declared positive, and three candidate negatives in InfoNCE</title>
			<desc id="infonce-candidates-svg-desc">An anchor embedding from crop A of one image sits at the left. A solid line connects it to crop B of the same image, the sole declared positive and numerator term. Dashed lines connect the anchor to three other candidates, all included with the positive in the denominator. Two diamond candidates are true negatives. A third diamond is close to the anchor and depicts the same semantic class, but because it came from another source item the sampler labels it negative. A warning states that InfoNCE pushes this false negative away. A summary panel says the positive-pair rule chooses invariance and the candidate sampler chooses repulsion.</desc>
			<text class="viz-axis-label" x="18" y="20">ONE ANCHOR'S INFONCE CLASSIFICATION TASK</text>
			<circle class="viz-node viz-node--focus" cx="78" cy="169" r="31"></circle>
			<text class="viz-callout" x="78" y="165" text-anchor="middle">ANCHOR i</text>
			<text class="viz-label" x="78" y="181" text-anchor="middle">source A · crop 1</text>
			<path class="viz-axis" d="M109 156L232 105"></path>
			<text class="viz-callout" x="165" y="116" text-anchor="middle">PULL TOGETHER</text>
			<circle class="viz-node viz-node--output" cx="266" cy="91" r="31"></circle>
			<text class="viz-callout" x="266" y="87" text-anchor="middle">POSITIVE j</text>
			<text class="viz-label" x="266" y="103" text-anchor="middle">source A · crop 2</text>
			<text class="viz-label" x="266" y="132" text-anchor="middle">the numerator</text>
			<path class="viz-operating-guide" d="M108 177L213 199"></path>
			<polygon class="viz-node viz-node--focus" points="246,176 272,202 246,228 220,202"></polygon>
			<text class="viz-callout" x="246" y="198" text-anchor="middle">FALSE</text>
			<text class="viz-callout" x="246" y="212" text-anchor="middle">NEGATIVE</text>
			<text class="viz-label" x="246" y="244" text-anchor="middle">source B · same class</text>
			<path class="viz-operating-guide" d="M98 194L143 272"></path>
			<polygon class="viz-node" points="160,270 181,291 160,312 139,291"></polygon>
			<text class="viz-callout" x="160" y="295" text-anchor="middle">NEG k₁</text>
			<path class="viz-operating-guide" d="M101 190L275 277"></path>
			<polygon class="viz-node" points="292,274 313,295 292,316 271,295"></polygon>
			<text class="viz-callout" x="292" y="299" text-anchor="middle">NEG k₂</text>
			<text class="viz-label" x="226" y="332" text-anchor="middle">all three sampled items enter the denominator</text>
			<rect class="viz-node viz-node--focus" x="18" y="259" width="105" height="72" rx="3"></rect>
			<text class="viz-callout" x="70" y="279" text-anchor="middle">SAMPLER SAYS</text>
			<text class="viz-label" x="70" y="296" text-anchor="middle">“different item”</text>
			<text class="viz-callout" x="70" y="314" text-anchor="middle">LOSS SAYS</text>
			<text class="viz-label" x="70" y="326" text-anchor="middle">push away</text>
			<text class="viz-callout" x="180" y="354" text-anchor="middle">positive rule chooses invariance · sampler chooses repulsion</text>
		</svg>
	</div>
	<figcaption><strong>Read it this way:</strong> for anchor <em>i</em>, only the second view of source A is positive <em>j</em> and enters the numerator; that view plus every sampled candidate enters the denominator. If source B is semantically a valid match, the batch still calls it negative and pushes it away; this is the false-negative risk hidden inside the sampling rule. Original schematic informed by <a href="https://proceedings.mlr.press/v119/chen20j.html">SimCLR</a>, <a href="https://arxiv.org/abs/1807.03748">CPC</a>, and <a href="https://arxiv.org/abs/2007.00224">Debiased Contrastive Learning</a>.</figcaption>
</figure>

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