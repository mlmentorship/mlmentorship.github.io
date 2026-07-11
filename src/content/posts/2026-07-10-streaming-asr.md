---
title: "Streaming automatic speech recognition"
description: "Emit transcripts with bounded latency using chunked encoders, monotonic alignment, endpointing, and stability-aware evaluation."
date: "2026-07-10"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Definition

Streaming ASR produces partial transcripts while audio is arriving. Unlike offline ASR, the model cannot attend to unlimited future context. It must trade recognition quality against emission latency, compute, and transcript stability.

## Architecture choices

- **Chunked encoder:** processes bounded windows with cached state or limited lookahead.
- **RNN-T / transducer:** learns monotonic alignment and can emit tokens incrementally.
- **CTC:** simple monotonic objective, often paired with streaming beam search and a language model.
- **Streaming attention:** constrains or chunks encoder–decoder attention.

## Latency components

End-to-end latency includes audio chunk size, feature extraction, model compute, lookahead, decoding, endpointing, network, and client rendering. Reporting model runtime alone hides user-perceived delay.

## Endpointing and partial stability

The system must decide when speech has ended. Aggressive endpointing lowers latency but truncates pauses; conservative endpointing feels slow. Partial hypotheses may revise earlier words, so evaluate flicker or edit overhead in addition to final word error rate.

## Evaluation

- Final and streaming word error rate
- Time to first token and finalization latency
- Partial stability / revision rate
- Real-time factor and peak compute
- Slice quality by accent, noise, device, language, and speech rate

## Interview answer

1. Clarify latency and quality targets.
2. Choose monotonic or chunked architecture.
3. Account for every latency component.
4. Discuss endpointing and partial transcript stability.
5. Design fallback, offline rescoring, and slice monitoring.

## Common confusions

- **“Streaming means batch size one.”** It means bounded future context and incremental output; batching may still occur across streams.
- **“WER captures latency.”** It does not measure when words appear or how often they change.
- **“More lookahead is free quality.”** It directly adds user-visible latency.

*Related: [automatic speech recognition](/concepts/automatic-speech-recognition/), [RNN-T](/concepts/rnn-transducer/), and [CTC](/concepts/connectionist-temporal-classification/).*
