---
title: "Speaker recognition"
description: "Speaker verification and identification using embeddings, metric learning, calibration, anti-spoofing, and operating-point evaluation."
date: "2026-07-10"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

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

**Learning objective:** trace how moving one speaker-verification threshold changes false rejects and false accepts on the same set of target and non-target trials.

<!-- visual:speaker-verification-threshold-tradeoff -->
<figure class="learning-figure plot-panel" aria-labelledby="speaker-threshold-heading">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="speaker-threshold-heading">Why does a safer operating point reject more genuine speakers?</p>
	<div class="visual-scroll">
		<svg viewBox="0 0 360 330" role="img" aria-labelledby="speaker-threshold-title speaker-threshold-desc">
			<title id="speaker-threshold-title">The false-reject and false-accept tradeoff from moving a speaker-verification threshold</title>
			<desc id="speaker-threshold-desc">Two panels apply different thresholds to the same eight illustrative similarity scores. Squares labeled N are non-target trials and circles labeled T are target-speaker trials. In the strict upper panel, the threshold is far right: no illustrated non-target is accepted, but two target trials fall in the reject region and are false rejects. In the lenient lower panel, the threshold moves left: only one target trial is falsely rejected, but one non-target trial now falls in the accept region and is a false accept. Position, shape, letters, boundary lines, and direct error labels make the tradeoff independent of color.</desc>
			<defs>
				<marker id="speaker-score-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" class="viz-arrow-forward"></path></marker>
			</defs>
			<text class="viz-label" x="20" y="18">N = non-target trial (square)</text>
			<text class="viz-label" x="340" y="18" text-anchor="end">T = target trial (circle)</text>
			<rect class="viz-plot-bg" x="8" y="31" width="344" height="128" rx="5"></rect>
			<text class="viz-axis-label" x="20" y="51">STRICT THRESHOLD · HIGH FALSE-ACCEPT COST</text>
			<text class="viz-label" x="20" y="70">reject</text>
			<text class="viz-label" x="340" y="70" text-anchor="end">accept claimed speaker</text>
			<path class="viz-axis" d="M30 102H330" marker-end="url(#speaker-score-arrow)"></path>
			<path class="viz-operating-guide" d="M248 66V139"></path>
			<text class="viz-callout" x="340" y="151" text-anchor="end">strict threshold</text>
			<rect class="viz-node viz-node--input" x="54" y="77" width="24" height="24" rx="2"></rect>
			<rect class="viz-node viz-node--input" x="89" y="77" width="24" height="24" rx="2"></rect>
			<rect class="viz-node viz-node--input" x="131" y="77" width="24" height="24" rx="2"></rect>
			<rect class="viz-node viz-node--input" x="176" y="77" width="24" height="24" rx="2"></rect>
			<circle class="viz-node viz-node--output" cx="166" cy="117" r="12"></circle>
			<circle class="viz-node viz-node--output" cx="222" cy="117" r="12"></circle>
			<circle class="viz-node viz-node--output" cx="278" cy="117" r="12"></circle>
			<circle class="viz-node viz-node--output" cx="316" cy="117" r="12"></circle>
			<g class="viz-node-value"><text x="66" y="93" text-anchor="middle">N</text><text x="101" y="93" text-anchor="middle">N</text><text x="143" y="93" text-anchor="middle">N</text><text x="188" y="93" text-anchor="middle">N</text><text x="166" y="121" text-anchor="middle">T</text><text x="222" y="121" text-anchor="middle">T</text><text x="278" y="121" text-anchor="middle">T</text><text x="316" y="121" text-anchor="middle">T</text></g>
			<path d="M158 131V138H230V131" fill="none" stroke="var(--viz-warning-stroke)" stroke-width="2"></path>
			<text class="viz-callout" x="194" y="151" text-anchor="middle">2 false rejects</text>
			<rect class="viz-plot-bg" x="8" y="177" width="344" height="128" rx="5"></rect>
			<text class="viz-axis-label" x="20" y="197">LENIENT THRESHOLD · LOWER FALSE-REJECT COST</text>
			<text class="viz-label" x="20" y="216">reject</text>
			<text class="viz-label" x="340" y="216" text-anchor="end">accept claimed speaker</text>
			<path class="viz-axis" d="M30 248H330" marker-end="url(#speaker-score-arrow)"></path>
			<path class="viz-operating-guide" d="M180 212V285"></path>
			<text class="viz-callout" x="250" y="298" text-anchor="middle">lenient threshold</text>
			<rect class="viz-node viz-node--input" x="54" y="223" width="24" height="24" rx="2"></rect>
			<rect class="viz-node viz-node--input" x="89" y="223" width="24" height="24" rx="2"></rect>
			<rect class="viz-node viz-node--input" x="131" y="223" width="24" height="24" rx="2"></rect>
			<rect class="viz-node viz-node--input" x="176" y="223" width="24" height="24" rx="2"></rect>
			<circle class="viz-node viz-node--output" cx="166" cy="263" r="12"></circle>
			<circle class="viz-node viz-node--output" cx="222" cy="263" r="12"></circle>
			<circle class="viz-node viz-node--output" cx="278" cy="263" r="12"></circle>
			<circle class="viz-node viz-node--output" cx="316" cy="263" r="12"></circle>
			<g class="viz-node-value"><text x="66" y="239" text-anchor="middle">N</text><text x="101" y="239" text-anchor="middle">N</text><text x="143" y="239" text-anchor="middle">N</text><text x="188" y="239" text-anchor="middle">N</text><text x="166" y="267" text-anchor="middle">T</text><text x="222" y="267" text-anchor="middle">T</text><text x="278" y="267" text-anchor="middle">T</text><text x="316" y="267" text-anchor="middle">T</text></g>
			<path d="M158 277V284H174V277" fill="none" stroke="var(--viz-warning-stroke)" stroke-width="2"></path>
			<text class="viz-callout" x="92" y="293">1 false reject</text>
			<path d="M180 226H202V234" fill="none" stroke="var(--viz-warning-stroke)" stroke-width="2"></path>
			<text class="viz-callout" x="206" y="228">1 false accept</text>
			<text class="viz-axis-label" x="180" y="324" text-anchor="middle">higher similarity score → stronger claimed-speaker match</text>
		</svg>
	</div>
	<figcaption><strong>Read it this way:</strong> compare the same T and N trials in both rows. Moving the boundary left recovers one genuine target, but also admits one impostor. Choose the threshold from deployment costs and calibrated validation data, not from EER alone. Trial positions and counts are illustrative, not measured.</figcaption>
</figure>

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
