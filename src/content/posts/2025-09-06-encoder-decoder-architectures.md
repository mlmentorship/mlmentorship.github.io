---
title: "Encoder-decoder architectures"
description: "An encoder summarizes the input into a representation; a decoder generates the output conditioned on it. The structure behind translation, T5, summarization, and many multimodal models."
date: "2025-09-06"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

An attention-based autoregressive encoder-decoder routes source representations and a growing target prefix through different attention paths.
Its source memory can remain fixed during generation while target state grows; this requires fixed source input and parameters.
Other encoder-decoder systems, including nonautoregressive decoders, need not share this factorization.

## The structure

For one decoder layer, let $H=Encoder(source)$ and use:

$$
Q=target\_states\,W_Q,\qquad K=HW_K,\qquad V=HW_V.
$$

Source K/V projections belong to that layer; different layers need not share them.
Target self-attention reads the causal target prefix, whereas cross-attention reads the permitted source memory.
Padding and source-availability masks remain separate from the target causal mask.

Consider three source positions producing $H\in\mathbb R^{3\times2}$, then output token y1 followed by EOS.
One encoder pass produces H and the layer's source K/V.
BOS supplies the first incremental target query: three source score pairs precede y1.
The next query, after appending y1, contributes three more pairs before EOS.
Total: one encoder pass, two incremental decoder steps, and six cross-score pairs per head and layer.

<p class="visual-kicker">Source and target state</p>
<!-- visual:encoder-once-decoder-loop -->
<figure class="learning-figure editorial-illustration" aria-labelledby="encoder-once-decoder-loop-edition-title"><p class="visual-title" id="encoder-once-decoder-loop-edition-title">The source memory stays fixed while the generated prefix grows</p><svg viewBox="0 0 360 461" role="img" aria-labelledby="encoder-once-decoder-loop-edition-svg-title encoder-once-decoder-loop-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="encoder-once-decoder-loop-edition-svg-title">Three growing prefixes share one fixed encoder memory</title><desc id="encoder-once-decoder-loop-edition-svg-desc">One encoder-memory block feeds three successive decoder iterations. Prefixes grow from start to start-A to start-A-B. Cross-attention source K/V remain fixed while the causal decoder prefix and its cache grow. The shared input lines distinguish reuse from re-encoding.</desc><g data-editorial-node><rect x="25" y="24" width="310" height="62" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="180" y="50.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Encode the complete source ONCE</tspan><tspan x="180" dy="19">cross-attention K/V stay fixed</tspan></text></g><path d="M38 86L38 159" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><path d="M38 159L82 159" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><path d="M75.8569220667674,162.35597877022943L82,159L75.8569220667674,155.64402122977057" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><g data-editorial-node><rect x="86" y="134" width="248" height="53" rx="3" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.3"></rect><text x="210" y="165.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="210" dy="0">decode 1: &lt;start&gt;</tspan></text></g><path d="M38 86L38 241" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><path d="M38 241L82 241" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><path d="M75.8569220667674,244.35597877022943L82,241L75.8569220667674,237.64402122977057" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><g data-editorial-node><rect x="86" y="216" width="248" height="53" rx="3" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.3"></rect><text x="210" y="247.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="210" dy="0">decode 2: &lt;start&gt; A</tspan></text></g><path d="M38 86L38 323" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><path d="M38 323L82 323" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><path d="M75.8569220667674,326.3559787702294L82,323L75.8569220667674,319.6440212297706" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><g data-editorial-node><rect x="86" y="298" width="248" height="53" rx="3" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.3"></rect><text x="210" y="329.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="210" dy="0">decode 3: &lt;start&gt; A B</tspan></text></g><text x="180" y="425" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">The output prefix and decoder cache grow;</tspan><tspan x="180" dy="19">source K/V do not.</tspan></text></svg><figcaption><strong>Read it this way:</strong> Each decode step reads the same encoded source. Only the output prefix and decoder-side cache grow. Fixed source K/V do not mean fixed decoder state. <span class="visual-reader-check" data-reader-check="encoder-once-decoder-loop"><strong>Check:</strong> Which representation remains fixed while the generated prefix grows?</span></figcaption></figure>

### Switch to Teacher-Forced Scoring

**New task:** Score observed labels y1, EOS rather than sampling them.

**Picture and change:** Decoder inputs are BOS, y1; labels are y1, EOS.
Suitable causal-transformer implementations evaluate both target positions in parallel, with source access retained except padding.

**Check:** The first prediction cannot read the second input position containing y1.
Two target positions still make six source score pairs.
Unshifted inputs y1, EOS would leak each answer through its own position despite a causal mask.

**Use when:** Computing training loss or conditional sequence likelihood.

### Change the Source

**New task:** A fourth token arrives for a bidirectional encoder.

**Picture and change:** Recompute or explicitly invalidate affected source encodings and derived caches; older source states can change too.

**Check:** New H has shape $4\times2$ and four cross keys, not three old keys plus an independently encoded token.
Previously decoded target states may also be incompatible with the changed source.

**Use when:** The input changes; streaming needs a separately specified information-flow and cache policy.

## Cross-attention complexity

For source length $n$ and evaluated target length $m$, dense cross-attention makes $mn$ score pairs per head and layer.
Score and value arithmetic also depend on their dimensions; projection, encoder, and target-self-attention costs are additional.
The incremental trace evaluates only new target queries; repeatedly recomputing the full prefix performs different work.

<!-- learning:core-end -->
<div data-learning-extension>

## Encoder-only vs. decoder-only vs. encoder-decoder

| Model class | Use cases | Examples |
|------------|-----------|----------|
| **Encoder-only** | Embeddings, classification, retrieval | BERT, RoBERTa, sentence-T5 |
| **Decoder-only** | Generation, chat, code | GPT-2/3/4, Llama, Mistral, Claude |
| **Encoder-decoder** | Translation, summarization, structured output | T5, BART, mT5, FLAN-T5 |

## Why decoder-only models took over chat

The retained historical heading is not an architectural superiority theorem.
Compare training recipe, source/target lengths, reuse, quality, and deployment costs rather than asserting that a clear split guarantees victory.

## Variants and their distinctions

- **Original transformer** [(Vaswani 2017)](https://arxiv.org/abs/1706.03762): encoder-decoder for NMT.
- **BERT**: encoder-only, masked-language-model objective.
- **GPT**: decoder-only, autoregressive next-token.
- **T5** [(Raffel 2019)](https://arxiv.org/abs/1910.10683): encoder-decoder, span-corruption objective; everything is text-to-text.
- **BART**: encoder-decoder, denoising autoencoder.
- **FLAN-T5**: T5 + instruction fine-tuning.

## Common pitfalls

- **Unshifted targets:** causality alone does not hide the current input token.
- **Wrong cache lifetime:** source memory is reusable only under its fixed-input/parameter contract.
- **Universal dominance:** neither a separate encoder nor a single stack guarantees better cost or quality.

## Related

- [Transformer architecture](/concepts/transformer-architecture/). Block-level structure.
- [Attention mechanism](/concepts/attention-mechanism/). Both self- and cross-attention.

</div>
