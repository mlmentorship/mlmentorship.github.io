---
title: "Multimodal foundation models"
description: "Align modality encoders, token budgets, fusion, objectives, and evaluation without pretending text, image, audio, and video share one natural representation."
date: "2026-07-11"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A multimodal foundation model learns representations or generation across two or more modalities, such as text, image, audio, or video, using modality-specific encoders plus a shared alignment, fusion, or autoregressive model.

Modern assistants observe screens, images, speech, documents, and video rather than isolated text. The design problem is not simply adding more tokens. Modalities have different sampling rates, spatial or temporal structure, noise, and evaluation.

A 30-second video can contain thousands of visual frames and audio events. Turning all of it into dense tokens can exhaust context before reasoning starts.

## Three architecture patterns

### Dual encoder

Encode each modality separately and train matched pairs to have similar embeddings, often with a contrastive objective. CLIP is the canonical image-text example.

**Strengths:** efficient retrieval, independent indexing, scalable negatives.

**Limit:** shallow cross-modal interaction because each side is encoded before seeing the other.

### Encoder plus language model

Use a vision, audio, or video encoder, project its outputs into the language-model representation, and let a decoder generate text or actions. The connector may be a linear projection, resampler, query transformer, or cross-attention module.

**Strengths:** reuses a strong language model and supports open-ended output.

**Limit:** alignment and token compression can bottleneck modality detail.

### Unified token model

Represent modalities as discrete or continuous tokens and train one model over mixed sequences. This supports joint generation but creates difficult tokenization, scale, and objective-balancing problems.

<!-- visual:multimodal-architectures-meeting-point -->
<figure class="learning-figure visual-wide plot-panel" aria-labelledby="multimodal-patterns-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="multimodal-patterns-title">Locate where image and text information first meet in each architecture pattern.</p>
	<div class="visual-scroll">
		<svg viewBox="0 0 760 470" role="img" aria-labelledby="multimodal-patterns-svg-title multimodal-patterns-svg-desc">
			<title id="multimodal-patterns-svg-title">Three multimodal architecture patterns compared by meeting point</title>
			<desc id="multimodal-patterns-svg-desc">Three horizontal rows compare multimodal architectures. In a dual encoder, image and text pass through separate encoders and meet only when their final embeddings are compared, producing retrieval or ranking. In an encoder plus language model, image features pass through a connector and meet text tokens inside the language model, which generates text or actions. In a unified token model, modality-specific tokenizers create image and text tokens that are interleaved before entering one shared transformer, which can generate either modality.</desc>
			<defs>
				<marker id="arrow-forward" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path class="viz-arrow-forward" d="M0,0 L8,4 L0,8 Z"></path></marker>
			</defs>
			<text class="viz-axis-label" x="20" y="27">1 · DUAL ENCODER</text>
			<rect class="viz-node viz-node--input" x="20" y="43" width="90" height="44" rx="22"></rect><text class="viz-node-label" x="65" y="70">image</text>
			<rect class="viz-node" x="155" y="43" width="110" height="44" rx="8"></rect><text class="viz-node-label" x="210" y="62">image</text><text class="viz-node-value" x="210" y="78">encoder</text>
			<rect class="viz-node viz-node--input" x="20" y="99" width="90" height="44" rx="8"></rect><text class="viz-node-label" x="65" y="126">text</text>
			<rect class="viz-node" x="155" y="99" width="110" height="44" rx="8"></rect><text class="viz-node-label" x="210" y="118">text</text><text class="viz-node-value" x="210" y="134">encoder</text>
			<path class="viz-forward" d="M110 65 H155"></path><path class="viz-forward" d="M110 121 H155"></path>
			<rect class="viz-node viz-node--focus" x="325" y="64" width="155" height="58" rx="29"></rect><text class="viz-node-label" x="402" y="88">embedding</text><text class="viz-node-value" x="402" y="106">similarity is first meeting</text>
			<path class="viz-forward" d="M265 65 C295 65 295 82 325 87"></path><path class="viz-forward" d="M265 121 C295 121 295 104 325 99"></path>
			<rect class="viz-node viz-node--output" x="540" y="64" width="180" height="58" rx="8"></rect><text class="viz-node-label" x="630" y="88">retrieve or rank</text><text class="viz-node-value" x="630" y="106">no joint token reasoning</text>
			<path class="viz-forward" d="M480 93 H540"></path>
			<line class="viz-gridline" x1="20" y1="166" x2="740" y2="166"></line>
			<text class="viz-axis-label" x="20" y="196">2 · ENCODER + LANGUAGE MODEL</text>
			<rect class="viz-node viz-node--input" x="20" y="216" width="90" height="44" rx="22"></rect><text class="viz-node-label" x="65" y="243">image</text>
			<rect class="viz-node" x="145" y="216" width="105" height="44" rx="8"></rect><text class="viz-node-label" x="197" y="235">vision</text><text class="viz-node-value" x="197" y="251">encoder</text>
			<rect class="viz-node" x="285" y="216" width="105" height="44" rx="8"></rect><text class="viz-node-label" x="337" y="235">connector</text><text class="viz-node-value" x="337" y="251">compress + project</text>
			<path class="viz-forward" d="M110 238 H145"></path><path class="viz-forward" d="M250 238 H285"></path>
			<rect class="viz-node viz-node--input" x="285" y="280" width="105" height="44" rx="8"></rect><text class="viz-node-label" x="337" y="299">text</text><text class="viz-node-value" x="337" y="315">tokens</text>
			<rect class="viz-node viz-node--focus" x="450" y="235" width="125" height="70" rx="10"></rect><text class="viz-node-label" x="512" y="262">language model</text><text class="viz-node-value" x="512" y="280">features + text meet</text>
			<path class="viz-forward" d="M390 238 C420 238 420 256 450 262"></path><path class="viz-forward" d="M390 302 C420 302 420 284 450 278"></path>
			<rect class="viz-node viz-node--output" x="635" y="241" width="95" height="58" rx="8"></rect><text class="viz-node-label" x="682" y="265">generate</text><text class="viz-node-value" x="682" y="283">text or action</text>
			<path class="viz-forward" d="M575 270 H635"></path>
			<line class="viz-gridline" x1="20" y1="347" x2="740" y2="347"></line>
			<text class="viz-axis-label" x="20" y="377">3 · UNIFIED TOKEN MODEL</text>
			<rect class="viz-node viz-node--input" x="20" y="394" width="75" height="44" rx="22"></rect><text class="viz-node-label" x="57" y="421">image</text>
			<rect class="viz-node viz-node--input" x="115" y="394" width="75" height="44" rx="8"></rect><text class="viz-node-label" x="152" y="421">text</text>
			<rect class="viz-node" x="225" y="388" width="115" height="56" rx="8"></rect><text class="viz-node-label" x="282" y="410">modality</text><text class="viz-node-value" x="282" y="428">tokenizers</text>
			<path class="viz-forward" d="M95 409 C165 378 190 398 225 407"></path><path class="viz-forward" d="M190 421 H225"></path>
			<rect class="viz-node viz-node--focus" x="390" y="388" width="135" height="56" rx="8"></rect><text class="viz-node-label" x="457" y="410">interleaved tokens</text><text class="viz-node-value" x="457" y="428">meet before model</text>
			<path class="viz-forward" d="M340 416 H390"></path>
			<rect class="viz-node viz-node--focus" x="575" y="388" width="95" height="56" rx="8"></rect><text class="viz-node-label" x="622" y="410">one shared</text><text class="viz-node-value" x="622" y="428">transformer</text>
			<path class="viz-forward" d="M525 416 H575"></path><text class="viz-node-label" x="716" y="410">image</text><text class="viz-node-value" x="716" y="428">or text</text><path class="viz-forward" d="M670 416 H688"></path>
		</svg>
	</div>
	<figcaption><strong>Read it this way:</strong> follow each row until the two input paths first converge. Dual encoders meet only at embedding comparison, so they retrieve efficiently but do not jointly reason over tokens. Encoder-plus-LM systems meet after a connector inside the language model. Unified models interleave modality tokens before one shared transformer, enabling mixed generation at the cost of harder tokenization and training. Original schematic checked against <a href="https://arxiv.org/abs/2103.00020">CLIP</a>, <a href="https://arxiv.org/abs/2204.14198">Flamingo</a>, and <a href="https://arxiv.org/abs/2405.09818">Chameleon</a>.</figcaption>
</figure>

## Fusion choices

- **Early fusion:** combine modality tokens before deep processing. Rich interaction, high cost.
- **Cross-attention:** one stream queries another. More controlled compute and asymmetric roles.
- **Late fusion:** combine independent predictions or embeddings. Cheap but misses fine interaction.
- **Hierarchical fusion:** compress local spatial or temporal information before global reasoning.

Choose based on the interaction the task requires. Cross-modal retrieval does not need the same fusion as video question answering.

## Training objectives

Common objectives include:

- contrastive alignment;
- captioning or conditional generation;
- masked prediction within and across modalities;
- matching or ranking paired inputs;
- reconstruction or diffusion objectives;
- instruction tuning on multimodal conversations;
- preference optimization for response quality;
- temporal prediction for video or action.

Objective mixture can create interference. A model that improves caption fluency may rely less on the image. Include tests where text priors conflict with visual evidence.

## Data

Paired data is noisy and uneven. Captions omit details, audio transcripts miss non-speech events, and internet video has weak temporal labels. Synthetic captions or questions scale supervision but inherit generator bias.

Track provenance, consent, modality quality, alignment timing, language, duplication, and whether train and evaluation share source artifacts.

## Evaluation

Measure the actual capability:

- retrieval recall for dual encoders;
- grounded question answering with adversarial text priors;
- spatial, temporal, and counting accuracy;
- OCR and document structure;
- audio event and speaker understanding;
- video consistency and event order;
- hallucination grounded against modality evidence;
- robustness to missing or corrupted modalities;
- latency and token cost.

Text-only benchmarks cannot establish multimodal grounding.

## Common confusions

- **"Concatenate image and text embeddings."** Shapes can align while semantics do not.
- **"More frames mean better video understanding."** Redundant frames consume tokens and compute.
- **"Caption quality proves grounding."** Language priors can produce plausible captions without using the input.
- **"CLIP is a generative VLM."** It is a contrastive dual encoder.
- **"One tokenizer should treat every modality identically."** Modality structure and bitrate differ.
- **"Multimodal evaluation is text evaluation with images attached."** Grounding and cross-modal conflict require dedicated tests.

## In an interview

Start with task and interaction, then modality encoders, token budget, fusion, objectives, data alignment, grounding evaluation, failure slices, and serving cost.

*Related: [design a real-time multimodal assistant](/questions/design-real-time-multimodal-assistant/), [vision transformers](/concepts/vision-transformers/), [automatic speech recognition](/concepts/automatic-speech-recognition/), and [embedding spaces](/concepts/embedding-spaces-and-similarity/).*
