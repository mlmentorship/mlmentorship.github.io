---
title: "Multimodal foundation models"
description: "Align modality encoders, token budgets, fusion, objectives, and evaluation without pretending text, image, audio, and video share one natural representation."
date: "2026-07-11"
draft: false
tags: ["concepts"]
category: "concepts"
---

## One-line definition

A multimodal foundation model learns representations or generation across two or more modalities, such as text, image, audio, or video, using modality-specific encoders plus a shared alignment, fusion, or autoregressive model.

## Why it matters

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

*Related: [vision transformers](/concepts/vision-transformers/), [automatic speech recognition](/concepts/automatic-speech-recognition/), and [embedding spaces](/concepts/embedding-spaces-and-similarity/).*
