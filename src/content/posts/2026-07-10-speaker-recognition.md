---
title: "Speaker recognition"
description: "Speaker verification and identification using embeddings, metric learning, calibration, anti-spoofing, and operating-point evaluation."
date: "2026-07-10"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Why it matters

Voice is an attractive identity signal (hands-free, no extra hardware) and a dangerous one: it is observable, hard to revoke, and easy to replay or synthesize. Speaker recognition predicts identity from voice, and the real engineering difficulty is less the embedding than calibrating a decision threshold to the risk and defending it against spoofing. Two tasks sit underneath: **verification** asks whether an utterance matches a claimed speaker, and **identification** chooses among enrolled speakers. Both differ from speech recognition, which predicts words.

## System shape

1. Voice activity detection and acoustic preprocessing
2. An encoder producing a fixed-dimensional speaker embedding
3. Enrollment aggregation from one or more reference utterances
4. Similarity scoring, usually cosine or PLDA
5. Thresholding calibrated to the operating risk
6. Anti-spoofing and liveness checks in adversarial settings

## Training

Classifying over training speakers can learn embeddings, but metric losses align training more directly with verification. Triplet and contrastive losses need informative sampling. Additive angular-margin losses (AAM-Softmax) create well-separated directions on the embedding sphere.

## Evaluation

- False accept and false reject rates
- Equal error rate for a single summary number
- Detection cost at the real operating point
- Calibration across channels, devices, languages, and demographics
- Spoof and replay performance

One threshold rarely serves every risk tier: account recovery and low-stakes personalization have very different false-accept costs.

## In an interview

1. Separate verification from identification.
2. Describe enrollment, embedding, scoring, and thresholding.
3. Explain channel and session variability.
4. Choose metrics at the deployment operating point.
5. Cover spoofing, privacy, consent, and fallback authentication.

## Common confusions

- **"Low EER means secure authentication."** Security depends on the attack conditions and the chosen threshold.
- **"A voiceprint is a password."** Voice is observable, hard to revoke, and vulnerable to replay or synthesis.
- **"Cosine similarity needs no calibration."** Raw scores shift across domains and populations.

*Related: [word embeddings](/concepts/word-embeddings/), [calibration](/concepts/calibration/), and [automatic speech recognition](/concepts/automatic-speech-recognition/).*
