---
title: "Streaming automatic speech recognition"
description: "Emit transcripts with bounded latency using chunked encoders, monotonic alignment, endpointing, and stability-aware evaluation."
date: "2026-07-10"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Why it matters

Streaming ASR is the difference between a voice assistant that feels responsive and one that feels broken: it has to emit transcript while the user is still speaking, with bounded latency. The catch is that the model cannot attend to unlimited future context, so it trades recognition quality against emission latency, compute, and transcript stability. Offline ASR has none of those constraints, which is why a system that scores well offline can still feel terrible live.

## Architecture choices

- **Chunked encoder:** processes bounded windows with cached state or limited lookahead.
- **RNN-T / transducer:** learns a monotonic alignment and emits tokens incrementally.
- **CTC:** a simple monotonic objective, often paired with streaming beam search and a language model.
- **Streaming attention:** constrains or chunks encoder-decoder attention so it does not need the whole utterance.

## Latency components

End-to-end latency is more than model runtime: audio chunk size, feature extraction, model compute, lookahead, decoding, endpointing, network, and client rendering all add up. Reporting model runtime alone hides the delay the user actually feels.

## Endpointing and partial stability

The system has to decide when speech has ended. Aggressive endpointing lowers latency but clips pauses; conservative endpointing feels sluggish. Partial hypotheses may also revise earlier words, so measure flicker or edit overhead alongside final word error rate.

## Evaluation

- Final and streaming word error rate
- Time to first token and finalization latency
- Partial stability / revision rate
- Real-time factor and peak compute
- Slice quality by accent, noise, device, language, and speech rate

## In an interview

1. Clarify the latency and quality targets.
2. Choose a monotonic or chunked architecture.
3. Account for every latency component, not just the model.
4. Discuss endpointing and partial-transcript stability.
5. Design fallback, offline rescoring, and slice monitoring.

## Common confusions

- **"Streaming means batch size one."** It means bounded future context and incremental output; you can still batch across streams.
- **"WER captures latency."** It says nothing about when words appear or how often they change.
- **"More lookahead is free quality."** Every frame of lookahead is user-visible latency.

*Related: [automatic speech recognition](/concepts/automatic-speech-recognition/), [RNN-T](/concepts/rnn-transducer/), and [CTC](/concepts/connectionist-temporal-classification/).*
