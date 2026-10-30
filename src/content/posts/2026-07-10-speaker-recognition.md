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
	<p class="visual-title" id="speaker-threshold-heading">Moving one threshold exchanges false rejections for false acceptances</p><div class="visual-argument" data-visual-argument="speaker-verification-threshold-tradeoff" data-argument-kind="curve"><p class="visual-argument-premise">One threshold trades false acceptance for false rejection</p><svg viewBox="0 0 360 405" role="img" aria-label="One threshold trades false acceptance for false rejection. These illustrative score distributions overlap. Raising the threshold rejects more impostors but also rejects more genuine speakers."><title>One threshold trades false acceptance for false rejection</title><desc>These illustrative score distributions overlap. Raising the threshold rejects more impostors but also rejects more genuine speakers.</desc><path d="M48 40L48 220" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M48 220L334 220" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="48" y="245" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="48" dy="0">-3</tspan></text><text x="330" y="245" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="330" dy="0">3</tspan></text><text x="40" y="54" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="40" dy="0">0.65</tspan></text><text x="40" y="220" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="40" dy="0">0</tspan></text><path d="M198.4 40L198.4 220" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><path data-proof-series="Impostor scores" d="M48 217.48396283343047L52.40625 216.34399968073728L56.8125 214.7819826654611L61.21875 212.6849899059156L65.625 209.92756527290274L70.03125 206.37727112098315L74.4375 201.90311257970671L78.84375 196.38685816670682L83.25 189.73687396799906L87.65625 181.9036018500296L92.0625 172.89530814506782L96.46875 162.79229349216138L100.875 151.75748457992586L105.28125 140.04131633803377L109.6875 127.97912366810273L114.09375 115.97991280135477L118.5 104.5063300236455L122.90625 94.04678137646395L127.3125 85.08181884568361L131.71875 78.04790537633687L136.125 73.30231766905231L140.53125 71.09310102922302L144.9375 71.53759005900847L149.34375 74.61208367675124L153.75 80.15393895616961L158.15625 87.87582634590729L162.5625 97.3904073548049L166.96875 108.24248544069219L171.375 119.94492092572797L175.78125 132.01438701143422L180.1875 144.00337454853224L184.59375 155.52563518304578L189 166.27332600591737L193.40625 176.0252925095947L197.8125 184.64701553671858L202.21875 192.08360561342758L206.625 198.34776721228144L211.03125 203.50485377433608L215.4375 207.65702720680503L219.84375 210.92819945171172L224.25 213.4509646507546L228.65625 215.35622364804192L233.0625 216.76573667347748L237.46875 217.7874679181623L241.875 218.5133323309212L246.28125 219.01882079184315L250.6875 219.36394759927566L255.09375 219.59500694538573L259.5 219.74671330665373L263.90625 219.84440798937976L268.3125 219.90612016204454L272.71875 219.94436251511033L277.125 219.96761284743567L281.53125 219.9814822522188L285.9375 219.98960047072754L290.34375 219.99426346934706L294.75 219.9968918993431L299.15625 219.99834594117254L303.5625 219.9991353965834L307.96875 219.99955609190675L312.375 219.99977613855722L316.78125 219.99988911427164L321.1875 219.99994605115785L325.59375 219.9999742190615L330 219.99998789888363" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:2.5"></path><path data-proof-series="Genuine scores" d="M48 219.99998789888363L52.40625 219.9999742190615L56.8125 219.99994605115785L61.21875 219.99988911427164L65.625 219.99977613855722L70.03125 219.99955609190675L74.4375 219.9991353965834L78.84375 219.99834594117254L83.25 219.9968918993431L87.65625 219.99426346934706L92.0625 219.98960047072754L96.46875 219.9814822522188L100.875 219.96761284743567L105.28125 219.94436251511033L109.6875 219.90612016204454L114.09375 219.84440798937976L118.5 219.74671330665373L122.90625 219.59500694538573L127.3125 219.36394759927566L131.71875 219.01882079184315L136.125 218.5133323309212L140.53125 217.7874679181623L144.9375 216.76573667347748L149.34375 215.35622364804192L153.75 213.4509646507546L158.15625 210.92819945171172L162.5625 207.65702720680503L166.96875 203.50485377433608L171.375 198.34776721228144L175.78125 192.08360561342758L180.1875 184.64701553671858L184.59375 176.0252925095947L189 166.27332600591737L193.40625 155.52563518304578L197.8125 144.00337454853224L202.21875 132.01438701143422L206.625 119.94492092572797L211.03125 108.24248544069219L215.4375 97.3904073548049L219.84375 87.87582634590729L224.25 80.15393895616961L228.65625 74.61208367675124L233.0625 71.53759005900847L237.46875 71.09310102922302L241.875 73.30231766905231L246.28125 78.04790537633687L250.6875 85.08181884568361L255.09375 94.04678137646395L259.5 104.5063300236455L263.90625 115.97991280135477L268.3125 127.97912366810273L272.71875 140.04131633803377L277.125 151.75748457992586L281.53125 162.79229349216138L285.9375 172.89530814506782L290.34375 181.9036018500296L294.75 189.73687396799906L299.15625 196.38685816670682L303.5625 201.90311257970671L307.96875 206.37727112098315L312.375 209.92756527290274L316.78125 212.6849899059156L321.1875 214.7819826654611L325.59375 216.34399968073728L330 217.48396283343047" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:2.5;stroke-dasharray:6 3"></path><text x="190" y="273" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="190" dy="0">verification score</tspan></text><text x="20" y="20" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">illustrative score density</tspan></text><path d="M20 306L48 306" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75"></path><text x="60" y="310" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="60" dy="0">Impostor scores</tspan></text><path d="M20 336L48 336" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><text x="60" y="340" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="60" dy="0">Genuine scores</tspan></text><path d="M20 366L48 366" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><text x="60" y="370" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--viz-warning-stroke)"><tspan x="60" dy="0">Decision threshold = 0.2</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
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
	<div class="visual-derivation-caption">compare the same T and N trials in both rows. Moving the boundary left recovers one genuine target, but also admits one impostor. Choose the threshold from deployment costs and calibrated validation data, not from EER alone. Trial positions and counts are illustrative, not measured.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="speaker-verification-threshold-tradeoff"><strong>Check:</strong> Which exact trials change decisions when the threshold moves? What cost tradeoff determines whether that change is desirable?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="speaker-verification-threshold-tradeoff">These illustrative score distributions overlap. Raising the threshold rejects more impostors but also rejects more genuine speakers.</span></figcaption>
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
