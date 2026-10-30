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
<figure class="learning-figure editorial-illustration" data-illustration-print-page aria-labelledby="infonce-positive-candidate-set-edition-title"><span class="visual-fragment-anchor" id="infonce-candidates-visual-title" aria-hidden="true"></span><span class="visual-fragment-anchor" id="infonce-candidates-svg-title" aria-hidden="true"></span><span class="visual-fragment-anchor" id="infonce-candidates-svg-desc" aria-hidden="true"></span><p class="visual-title" id="infonce-positive-candidate-set-edition-title">The declared positive enters the denominator; an undeclared positive remains a negative</p><div class="visual-argument" data-visual-argument="infonce-positive-candidate-set" data-argument-kind="ledger"><p class="visual-argument-premise">The positive participates in its own denominator</p><svg viewBox="0 0 360 224" role="img" aria-label="The positive participates in its own denominator. The loss compares the declared positive against the whole candidate set. A relevant but undeclared item remains a negative under this labeling."><title>The positive participates in its own denominator</title><desc>The loss compares the declared positive against the whole candidate set. A relevant but undeclared item remains a negative under this labeling.</desc><text x="20" y="28" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Numerator</tspan></text><rect data-proof-token="positive" x="20" y="58" width="316" height="34" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke)"></rect><text x="178" y="79" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="178" dy="0">positive</tspan></text><text x="20" y="126" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Denominator candidates</tspan></text><rect data-proof-token="positive" x="20" y="156" width="102.66666666666667" height="34" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke)"></rect><text x="71.33333333333334" y="177" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="71.33333333333334" dy="0">positive</tspan></text><rect data-proof-token="negative 1" x="126.66666666666667" y="156" width="102.66666666666667" height="34" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke)"></rect><text x="178" y="177" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="178" dy="0">negative 1</tspan></text><rect data-proof-token="negative 2" x="233.33333333333334" y="156" width="102.66666666666667" height="34" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke)"></rect><text x="284.6666666666667" y="177" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="284.6666666666667" dy="0">negative 2</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content"><svg viewBox="0 0 360 774" role="img" aria-labelledby="infonce-positive-candidate-set-edition-svg-title infonce-positive-candidate-set-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="infonce-positive-candidate-set-edition-svg-title">The positive belongs to both numerator and denominator; a false negative stays in the denominator</title><desc id="infonce-positive-candidate-set-edition-svg-desc">A toy InfoNCE example uses unnormalized exponentiated similarity weights four, three, two, and one. One denominator enclosure contains all four candidates. Its positive row has a smaller numerator enclosure and four equal units. The same-class different-source item is still a sampler-negative with three units. The denominator totals ten and the positive probability is four tenths, drawn with ten equal units. Weights are illustrative, not observed similarities.</desc><text x="180" y="22" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Toy exp(similarity / temperature) weights</tspan></text><circle cx="70" cy="82" r="23" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:2"></circle><text x="70" y="87" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="70" dy="0">i</tspan></text><circle cx="290" cy="82" r="23" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:2"></circle><text x="290" y="87" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="290" dy="0">j</tspan></text><path d="M100 82L260 82" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><path d="M253.8569220667674,85.35597877022943L260,82L253.8569220667674,78.64402122977057" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><text x="180" y="65" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">same item A</tspan></text><text x="70" y="124" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="70" dy="0">anchor crop</tspan></text><text x="290" y="124" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="290" dy="0">positive crop</tspan></text><text x="180" y="158" style="font-size:12px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">DENOMINATOR: ALL FOUR CANDIDATES</tspan></text><g data-infonce-denominator><rect x="12" y="174" width="336" height="343" rx="3" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.3"></rect><g data-infonce-candidate="positive" data-weight="4" data-infonce-numerator><rect x="21" y="183" width="318" height="68" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="33" y="203" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="33" dy="0">positive j</tspan></text><text x="33" y="227" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:start"><tspan x="33" dy="0">same source A</tspan></text><rect data-infonce-unit x="218" y="195" width="23" height="24" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke)"></rect><rect data-infonce-unit x="243" y="195" width="23" height="24" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke)"></rect><rect data-infonce-unit x="268" y="195" width="23" height="24" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke)"></rect><rect data-infonce-unit x="293" y="195" width="23" height="24" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke)"></rect><text x="273" y="240" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="273" dy="0">4 in numerator</tspan></text></g><g data-infonce-candidate="false-negative" data-weight="3"><text x="33" y="283" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="33" dy="0">sampler-negative</tspan></text><text x="33" y="307" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:start"><tspan x="33" dy="0">source B; same class</tspan></text><rect data-infonce-unit x="218" y="275" width="23" height="24" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke)"></rect><rect data-infonce-unit x="243" y="275" width="23" height="24" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke)"></rect><rect data-infonce-unit x="268" y="275" width="23" height="24" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke)"></rect><text x="273" y="320" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="273" dy="0">3</tspan></text></g><g data-infonce-candidate="negative-1" data-weight="2"><text x="33" y="363" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="33" dy="0">negative k1</tspan></text><text x="33" y="387" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:start"><tspan x="33" dy="0">different source</tspan></text><rect data-infonce-unit x="218" y="355" width="23" height="24" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke)"></rect><rect data-infonce-unit x="243" y="355" width="23" height="24" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke)"></rect><text x="273" y="400" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="273" dy="0">2</tspan></text></g><g data-infonce-candidate="negative-2" data-weight="1"><text x="33" y="443" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="33" dy="0">negative k2</tspan></text><text x="33" y="467" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:start"><tspan x="33" dy="0">different source</tspan></text><rect data-infonce-unit x="218" y="435" width="23" height="24" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke)"></rect><text x="273" y="480" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="273" dy="0">1</tspan></text></g></g><text x="180" y="554" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Denominator = 4 + 3 + 2 + 1 = 10</tspan></text><text x="180" y="598" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Positive probability = 4 / 10 = 0.4</tspan></text><rect data-infonce-probability-unit="0" x="30" y="623" width="28" height="30" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke)"></rect><rect data-infonce-probability-unit="1" x="60" y="623" width="28" height="30" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke)"></rect><rect data-infonce-probability-unit="2" x="90" y="623" width="28" height="30" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke)"></rect><rect data-infonce-probability-unit="3" x="120" y="623" width="28" height="30" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke)"></rect><rect data-infonce-probability-unit="4" x="150" y="623" width="28" height="30" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke)"></rect><rect data-infonce-probability-unit="5" x="180" y="623" width="28" height="30" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke)"></rect><rect data-infonce-probability-unit="6" x="210" y="623" width="28" height="30" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke)"></rect><rect data-infonce-probability-unit="7" x="240" y="623" width="28" height="30" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke)"></rect><rect data-infonce-probability-unit="8" x="270" y="623" width="28" height="30" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke)"></rect><rect data-infonce-probability-unit="9" x="300" y="623" width="28" height="30" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke)"></rect><text x="180" y="686" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Four of ten equal units belong to the positive.</tspan></text><text x="180" y="733" style="font-size:13px;font-weight:400;fill:var(--viz-warning-stroke);text-anchor:middle"><tspan x="180" dy="0">Same-class B stays in the denominator: the</tspan><tspan x="180" dy="19">loss still repels it.</tspan></text></svg><div class="visual-derivation-caption">The nested positive row belongs to both enclosures. Its four units are part of the denominator total of ten. Source B is semantically similar but remains a sampler-negative, so the loss still pushes it away. The weights are illustrative.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="infonce-positive-candidate-set"><strong>Check:</strong> Is the positive also in the denominator? Why is a same-class item still repelled?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="infonce-positive-candidate-set">The loss compares the declared positive against the whole candidate set. A relevant but undeclared item remains a negative under this labeling.</span></figcaption></figure>

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