---
title: "Speaker recognition"
description: "Speaker verification and identification using embeddings, metric learning, calibration, anti-spoofing, and operating-point evaluation."
date: "2026-07-10"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Definition

Speaker recognition predicts identity from voice. **Verification** asks whether an utterance matches a claimed speaker; **identification** chooses among enrolled speakers. Both differ from speech recognition, which predicts words.

## System shape

1. Voice activity detection and acoustic preprocessing
2. Encoder producing a fixed-dimensional speaker embedding
3. Enrollment aggregation from one or more reference utterances
4. Similarity scoring, often cosine or PLDA
5. Thresholding calibrated to the operating risk
6. Anti-spoofing and liveness checks for adversarial settings

## Training

Classification over training speakers can learn embeddings, but metric losses align training more directly with verification. Triplet and contrastive losses require informative sampling. Additive angular-margin losses such as AAM-Softmax create separated directions on the embedding sphere.

## Evaluation

- False accept and false reject rates
- Equal error rate for summary comparison
- Detection cost at the real operating point
- Calibration across channels, devices, languages, and demographics
- Spoof and replay performance

One threshold rarely serves every risk tier. Account recovery and low-stakes personalization have different false-accept costs.

## Interview answer

1. Distinguish verification from identification.
2. Describe enrollment, embedding, scoring, and thresholding.
3. Explain channel and session variability.
4. Choose metrics at the deployment operating point.
5. Cover spoofing, privacy, consent, and fallback authentication.

## Common confusions

- **“Low EER means secure authentication.”** Security depends on attack conditions and the selected threshold.
- **“A voiceprint is a password.”** Voice is observable, difficult to revoke, and vulnerable to replay or synthesis.
- **“Cosine similarity needs no calibration.”** Raw scores shift across domains and populations.

*Related: [word embeddings](/concepts/word-embeddings/), [calibration](/concepts/calibration/), and [automatic speech recognition](/concepts/automatic-speech-recognition/).*
