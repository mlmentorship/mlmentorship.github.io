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
<figure class="learning-figure editorial-illustration" aria-labelledby="encoder-once-decoder-loop-edition-title"><p class="visual-title" id="encoder-once-decoder-loop-edition-title">Keep source keys fixed while new target queries read permitted source memory</p><div class="visual-argument" data-visual-argument="encoder-once-decoder-loop" data-argument-kind="matrix" data-argument-composition="fixed-source-growing-prefix"><svg viewBox="0 0 360 1725" role="img" aria-label="Keep source keys fixed while new target queries read permitted source memory. One encoder pass supplies layer-specific source keys and values while input and parameters stay fixed. Two target queries each score three source positions: six pairs per head and layer. Target self-attention is causal; cross-attention uses separate source masks. Teacher forcing shifts inputs against labels and may score positions in parallel without future access. A changed bidirectional source invalidates affected encodings, caches, and possibly target states; streaming requires its own policy."><title>Keep source keys fixed while new target queries read permitted source memory</title><desc>One encoder pass supplies layer-specific source keys and values while input and parameters stay fixed. Two target queries each score three source positions: six pairs per head and layer. Target self-attention is causal; cross-attention uses separate source masks. Teacher forcing shifts inputs against labels and may score positions in parallel without future access. A changed bidirectional source invalidates affected encodings, caches, and possibly target states; streaming requires its own policy.</desc><text x="20" y="27" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Fixed source memory; new target</tspan><tspan x="20" dy="18">queries</tspan></text><text x="20" y="105" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">One encoder pass; three source</tspan><tspan x="20" dy="18">positions</tspan></text><circle cx="90" cy="190" r="25" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="90" y="196" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="90" dy="0">s1</tspan></text><circle cx="190" cy="190" r="25" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="190" y="196" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="190" dy="0">s2</tspan></text><circle cx="290" cy="190" r="25" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="290" y="196" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="290" dy="0">s3</tspan></text><text x="20" y="265" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Layer-specific keys and values stay</tspan><tspan x="20" dy="18">cached while source and parameters</tspan><tspan x="20" dy="18">stay fixed.</tspan></text><text x="20" y="370" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Step 1: BOS query predicts y1</tspan></text><rect data-picture-cross-score="0-0" x="55" y="450" width="70" height="45" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.5"></rect><text x="90" y="478" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="90" dy="0">key 1</tspan></text><rect data-picture-cross-score="0-1" x="155" y="450" width="70" height="45" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.5"></rect><text x="190" y="478" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="190" dy="0">key 2</tspan></text><rect data-picture-cross-score="0-2" x="255" y="450" width="70" height="45" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.5"></rect><text x="290" y="478" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="290" dy="0">key 3</tspan></text><text x="20" y="545" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Three scores; normalize and mix</tspan><tspan x="20" dy="18">source values.</tspan></text><text x="20" y="615" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Step 2: y1 query predicts EOS</tspan></text><rect data-picture-cross-score="1-0" x="55" y="695" width="70" height="45" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.5"></rect><text x="90" y="723" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="90" dy="0">key 1</tspan></text><rect data-picture-cross-score="1-1" x="155" y="695" width="70" height="45" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.5"></rect><text x="190" y="723" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="190" dy="0">key 2</tspan></text><rect data-picture-cross-score="1-2" x="255" y="695" width="70" height="45" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.5"></rect><text x="290" y="723" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="290" dy="0">key 3</tspan></text><text x="20" y="790" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Three scores; normalize and mix</tspan><tspan x="20" dy="18">source values.</tspan></text><text x="20" y="880" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Six score pairs per head and layer;</tspan><tspan x="20" dy="18">one encoder pass, two incremental</tspan><tspan x="20" dy="18">queries.</tspan></text><text x="20" y="990" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Teacher forcing: shifted inputs BOS,</tspan><tspan x="20" dy="18">y1; labels y1, EOS. Positions may run</tspan><tspan x="20" dy="18">in parallel.</tspan></text><text x="140" y="1120" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="140" dy="0">BOS</tspan></text><text x="240" y="1120" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="240" dy="0">y1</tspan></text><text x="20" y="1181" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Predict y1</tspan></text><rect x="110" y="1150" width="60" height="53" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><text x="140" y="1183" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-output-stroke)"><tspan x="140" dy="0">read</tspan></text><rect x="210" y="1150" width="60" height="53" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.5"></rect><text x="240" y="1183" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="240" dy="0">no</tspan></text><text x="20" y="1276" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Predict</tspan><tspan x="20" dy="18">EOS</tspan></text><rect x="110" y="1245" width="60" height="53" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><text x="140" y="1278" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-output-stroke)"><tspan x="140" dy="0">read</tspan></text><rect x="210" y="1245" width="60" height="53" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><text x="240" y="1278" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-output-stroke)"><tspan x="240" dy="0">read</tspan></text><text x="20" y="1380" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-warning-stroke)"><tspan x="20" dy="0">Unshifted labels leak their own</tspan><tspan x="20" dy="18">answer, even with a causal mask.</tspan></text><text x="20" y="1485" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Cross-attention has separate</tspan><tspan x="20" dy="18">source/padding masks; target</tspan><tspan x="20" dy="18">causality does not mask future source</tspan><tspan x="20" dy="18">positions.</tspan></text><text x="20" y="1600" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-warning-stroke)"><tspan x="20" dy="0">A fourth bidirectional source token</tspan><tspan x="20" dy="18">changes the cache identity: recompute</tspan><tspan x="20" dy="18">or invalidate affected source and</tspan><tspan x="20" dy="18">target state.</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content"><svg viewBox="0 0 360 461" role="img" aria-labelledby="encoder-once-decoder-loop-edition-svg-title encoder-once-decoder-loop-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="encoder-once-decoder-loop-edition-svg-title">Three growing prefixes share one fixed encoder memory</title><desc id="encoder-once-decoder-loop-edition-svg-desc">One encoder-memory block feeds three successive decoder iterations. Prefixes grow from start to start-A to start-A-B. Cross-attention source K/V remain fixed while the causal decoder prefix and its cache grow. The shared input lines distinguish reuse from re-encoding.</desc><g data-editorial-node><rect x="25" y="24" width="310" height="62" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="180" y="50.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Encode the complete source ONCE</tspan><tspan x="180" dy="19">cross-attention K/V stay fixed</tspan></text></g><path d="M38 86L38 159" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><path d="M38 159L82 159" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><path d="M75.8569220667674,162.35597877022943L82,159L75.8569220667674,155.64402122977057" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><g data-editorial-node><rect x="86" y="134" width="248" height="53" rx="3" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.3"></rect><text x="210" y="165.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="210" dy="0">decode 1: &lt;start&gt;</tspan></text></g><path d="M38 86L38 241" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><path d="M38 241L82 241" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><path d="M75.8569220667674,244.35597877022943L82,241L75.8569220667674,237.64402122977057" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><g data-editorial-node><rect x="86" y="216" width="248" height="53" rx="3" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.3"></rect><text x="210" y="247.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="210" dy="0">decode 2: &lt;start&gt; A</tspan></text></g><path d="M38 86L38 323" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><path d="M38 323L82 323" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><path d="M75.8569220667674,326.3559787702294L82,323L75.8569220667674,319.6440212297706" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><g data-editorial-node><rect x="86" y="298" width="248" height="53" rx="3" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.3"></rect><text x="210" y="329.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="210" dy="0">decode 3: &lt;start&gt; A B</tspan></text></g><text x="180" y="425" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">The output prefix and decoder cache grow;</tspan><tspan x="180" dy="19">source K/V do not.</tspan></text></svg><div class="visual-derivation-caption">Each decode step reads the same encoded source. Only the output prefix and decoder-side cache grow. Fixed source K/V do not mean fixed decoder state.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="encoder-once-decoder-loop"><strong>Check:</strong> Which representation remains fixed while the generated prefix grows?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="encoder-once-decoder-loop">One encoder pass supplies layer-specific source keys and values while input and parameters stay fixed. Two target queries each score three source positions: six pairs per head and layer. Target self-attention is causal; cross-attention uses separate source masks. Teacher forcing shifts inputs against labels and may score positions in parallel without future access. A changed bidirectional source invalidates affected encodings, caches, and possibly target states; streaming requires its own policy. <strong>May the first teacher-forced prediction read the y1 input?</strong></span></figcaption></figure>

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
