---
title: "Expectation, variance, covariance, and correlation"
description: "Use moments to describe location, uncertainty, and dependence. Know what covariance measures, how transformations change it, and why correlation does not establish causation."
date: "2026-08-28"
draft: false
tags: ["concepts"]
category: "concepts"
aliases: ["expected value", "variance covariance", "covariance matrix", "pearson correlation"]
roles: ["Applied Scientist", "Research Scientist", "Research Engineer", "Machine Learning Engineer"]
rounds: ["Math", "Statistics", "ML breadth"]
difficulty: "Foundation"
priority: "Core"
prerequisites: []
---

## Summary

Expectation averages values under a probability law.
Variance averages squared deviations from a mean. Covariance averages products of two variables' deviations.

Suppose a hypothetical population gives probability $1/4$ to each pair $(-1,-1)$, $(-1,1)$, $(1,1)$ and $(1,3)$.
How does their co-movement affect the spread of $X+Y$?
These are population probabilities, not four observations used to estimate an unknown population.

## Covariance

Center both coordinates: the mean of $X$ is zero and the mean of $Y$ is one.
Multiply each point's horizontal and vertical deviations, then average those products.

<!-- visual:covariance-centered-product-signs -->
<figure class="learning-figure plot-panel" aria-labelledby="covariance-signs-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="covariance-signs-title">Center the population to see spread, co-movement and the limits of correlation</p><div class="visual-argument" data-visual-argument="covariance-centered-product-signs" data-argument-kind="matrix" data-argument-composition="population-moments-and-dependence"><svg viewBox="0 0 360 3120" role="img" aria-label="Center the population to see spread, co-movement and the limits of correlation. Expectation locates the mean. Variance measures squared spread around it. Covariance averages signed products of joint deviations; same-side points add and opposite-side points subtract. Their pairing changes the spread of a sum even when marginal spreads stay fixed. Changing units alters covariance, not standardized correlation. A curved deterministic relationship can have zero covariance. None of these associations proves causation."><title>Center the population to see spread, co-movement and the limits of correlation</title><desc>Expectation locates the mean. Variance measures squared spread around it. Covariance averages signed products of joint deviations; same-side points add and opposite-side points subtract. Their pairing changes the spread of a sum even when marginal spreads stay fixed. Changing units alters covariance, not standardized correlation. A curved deterministic relationship can have zero covariance. None of these associations proves causation.</desc><text x="20" y="27" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Four equally likely outcomes; center</tspan><tspan x="20" dy="18">on the means</tspan></text><text x="20" y="100" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Actual population: two products are</tspan><tspan x="20" dy="18">positive, two zero</tspan></text><path d="M60 240L320 240" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M180 130L180 360" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="326" y="245" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="326" dy="0">X</tspan></text><text x="160" y="133" style="font-size:14px;font-weight:700;text-anchor:end;fill:var(--c-text)"><tspan x="160" dy="0">Y</tspan></text><rect x="90" y="240" width="90" height="100" data-picture-centered-product="2" style="fill:var(--viz-output-bg);opacity:.7"></rect><text x="135" y="296" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-output-stroke)"><tspan x="135" dy="0">positive 2</tspan></text><g data-picture-population-location="0,-1,-1" data-picture-probability="0.25"><circle cx="90" cy="340" r="8.5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-population-location="0,-1,1" data-picture-probability="0.25"><circle cx="90" cy="240" r="8.5" style="fill:var(--viz-neutral-stroke);stroke:var(--viz-neutral-stroke);stroke-width:1.75"></circle></g><g data-picture-population-location="0,1,1" data-picture-probability="0.25"><circle cx="270" cy="240" r="8.5" style="fill:var(--viz-neutral-stroke);stroke:var(--viz-neutral-stroke);stroke-width:1.75"></circle></g><rect x="180" y="140" width="90" height="100" data-picture-centered-product="2" style="fill:var(--viz-output-bg);opacity:.7"></rect><text x="225" y="196" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-output-stroke)"><tspan x="225" dy="0">positive 2</tspan></text><g data-picture-population-location="0,1,3" data-picture-probability="0.25"><circle cx="270" cy="140" r="8.5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><text x="90" y="391" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="90" dy="0">-1</tspan></text><text x="180" y="391" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="180" dy="0">0</tspan></text><text x="270" y="391" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="270" dy="0">1</tspan></text><text x="52" y="346" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text)"><tspan x="52" dy="0">-1</tspan></text><text x="52" y="246" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text)"><tspan x="52" dy="0">1</tspan></text><text x="52" y="146" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text)"><tspan x="52" dy="0">3</tspan></text><circle cx="180" cy="240" r="8" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.75"></circle><text x="20" y="438" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Means: X zero, Y one. Center here</tspan><tspan x="20" dy="18">before comparing.</tspan></text><text x="20" y="500" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">Rectangles show signed products;</tspan><tspan x="20" dy="18">circle area shows probability,</tspan><tspan x="20" dy="18">including repeated locations.</tspan></text><text x="20" y="579" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Covariance: 1; squared spreads: X 1,</tspan><tspan x="20" dy="18">Y 2.</tspan></text><text x="20" y="677" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Keep the marginals; pair X and Y</tspan><tspan x="20" dy="18">independently</tspan></text><text x="20" y="750" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Same X and Y probabilities; different</tspan><tspan x="20" dy="18">pairings</tspan></text><path d="M60 890L320 890" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M180 780L180 1010" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="326" y="895" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="326" dy="0">X</tspan></text><text x="160" y="783" style="font-size:14px;font-weight:700;text-anchor:end;fill:var(--c-text)"><tspan x="160" dy="0">Y</tspan></text><rect x="90" y="890" width="90" height="100" data-picture-centered-product="2" style="fill:var(--viz-output-bg);opacity:.7"></rect><text x="135" y="946" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-output-stroke)"><tspan x="135" dy="0">positive 2</tspan></text><g data-picture-population-location="1,-1,-1" data-picture-probability="0.125"><circle cx="90" cy="990" r="6.0104076400856545" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-population-location="1,-1,1" data-picture-probability="0.25"><circle cx="90" cy="890" r="8.5" style="fill:var(--viz-neutral-stroke);stroke:var(--viz-neutral-stroke);stroke-width:1.75"></circle></g><rect x="90" y="790" width="90" height="100" data-picture-centered-product="-2" style="fill:var(--viz-warning-bg);opacity:.7"></rect><text x="135" y="846" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="135" dy="0">negative 2</tspan></text><g data-picture-population-location="1,-1,3" data-picture-probability="0.125"><circle cx="90" cy="790" r="6.0104076400856545" style="fill:var(--viz-warning-stroke);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle></g><rect x="180" y="890" width="90" height="100" data-picture-centered-product="-2" style="fill:var(--viz-warning-bg);opacity:.7"></rect><text x="225" y="946" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="225" dy="0">negative 2</tspan></text><g data-picture-population-location="1,1,-1" data-picture-probability="0.125"><circle cx="270" cy="990" r="6.0104076400856545" style="fill:var(--viz-warning-stroke);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle></g><g data-picture-population-location="1,1,1" data-picture-probability="0.25"><circle cx="270" cy="890" r="8.5" style="fill:var(--viz-neutral-stroke);stroke:var(--viz-neutral-stroke);stroke-width:1.75"></circle></g><rect x="180" y="790" width="90" height="100" data-picture-centered-product="2" style="fill:var(--viz-output-bg);opacity:.7"></rect><text x="225" y="846" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-output-stroke)"><tspan x="225" dy="0">positive 2</tspan></text><g data-picture-population-location="1,1,3" data-picture-probability="0.125"><circle cx="270" cy="790" r="6.0104076400856545" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><text x="90" y="1041" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="90" dy="0">-1</tspan></text><text x="180" y="1041" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="180" dy="0">0</tspan></text><text x="270" y="1041" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="270" dy="0">1</tspan></text><text x="52" y="996" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text)"><tspan x="52" dy="0">-1</tspan></text><text x="52" y="896" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text)"><tspan x="52" dy="0">1</tspan></text><text x="52" y="796" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text)"><tspan x="52" dy="0">3</tspan></text><circle cx="180" cy="890" r="8" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.75"></circle><text x="20" y="1088" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Means: X zero, Y one. Center here</tspan><tspan x="20" dy="18">before comparing.</tspan></text><text x="20" y="1150" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">Rectangles show signed products;</tspan><tspan x="20" dy="18">circle area shows probability,</tspan><tspan x="20" dy="18">including repeated locations.</tspan></text><text x="20" y="1229" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Covariance: 0; squared spreads: X 1,</tspan><tspan x="20" dy="18">Y 2.</tspan></text><text x="20" y="1377" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Pairing changes the spread of the sum</tspan></text><text x="20" y="1437" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Original sum variance: 5</tspan></text><rect x="40" y="1480" width="250" height="32" data-picture-sum-variance="5" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.5"></rect><text x="20" y="1582" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Independent sum variance: 3</tspan></text><rect x="40" y="1625" width="150" height="32" data-picture-sum-variance="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="20" y="1697" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">Shared variance scale. Positive</tspan><tspan x="20" dy="18">co-movement increases sum spread</tspan><tspan x="20" dy="18">without changing either marginal</tspan><tspan x="20" dy="18">spread.</tspan></text><text x="20" y="1807" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Change only Y units: values become</tspan><tspan x="20" dy="18">one hundred times larger</tspan></text><text x="20" y="1890" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Covariance: 1 becomes 100. Y</tspan><tspan x="20" dy="18">variance: 2 becomes 20000.</tspan></text><text x="20" y="1980" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Correlation: 0.7071 in either unit</tspan><tspan x="20" dy="18">system</tspan></text><path d="M55 2170L320 2170" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M180 2065L180 2260" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="326" y="2175" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="326" dy="0">X</tspan></text><text x="160" y="2075" style="font-size:14px;font-weight:700;text-anchor:end;fill:var(--c-text)"><tspan x="160" dy="0">Y</tspan></text><circle cx="90" cy="2240.710678118655" r="9" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><g data-picture-standardized-rescaled="0" data-picture-vector="-1,-1.414213562373095"><circle cx="90" cy="2240.710678118655" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><circle cx="90" cy="2170" r="9" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><g data-picture-standardized-rescaled="1" data-picture-vector="-1,0"><circle cx="90" cy="2170" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><circle cx="270" cy="2170" r="9" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><g data-picture-standardized-rescaled="2" data-picture-vector="1,0"><circle cx="270" cy="2170" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><circle cx="270" cy="2099.289321881345" r="9" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><g data-picture-standardized-rescaled="3" data-picture-vector="1,1.414213562373095"><circle cx="270" cy="2099.289321881345" r="3" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><text x="20" y="2317" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">In standard-deviation units, old</tspan><tspan x="20" dy="18">rings and rescaled dots coincide at</tspan><tspan x="20" dy="18">every point.</tspan></text><text x="20" y="2437" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Zero covariance can still hide curved</tspan><tspan x="20" dy="18">dependence</tspan></text><text x="20" y="2517" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">X: minus one, zero, one. Y: one,</tspan><tspan x="20" dy="18">zero, one.</tspan></text><text x="20" y="2590" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">Mean Y: two thirds; signed centered</tspan><tspan x="20" dy="18">products cancel.</tspan></text><path d="M55 2800L320 2800" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M180 2680L180 2830" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M90 2710Q180 2890 270 2710" data-picture-curved-dependence="true" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:2"></path><text x="326" y="2805" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="326" dy="0">X</tspan></text><text x="160" y="2690" style="font-size:14px;font-weight:700;text-anchor:end;fill:var(--c-text)"><tspan x="160" dy="0">Y</tspan></text><circle cx="90" cy="2710" r="7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="90" y="2860" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="90" dy="0">-1</tspan></text><circle cx="180" cy="2800" r="7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="180" y="2860" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="180" dy="0">0</tspan></text><circle cx="270" cy="2710" r="7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="270" y="2860" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="270" dy="0">1</tspan></text><text x="20" y="2937" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Covariance is zero, but X still</tspan><tspan x="20" dy="18">determines Y.</tspan></text><text x="20" y="3015" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">These are declared population</tspan><tspan x="20" dy="18">probabilities, not empirical</tspan><tspan x="20" dy="18">estimates. Correlation does not prove</tspan><tspan x="20" dy="18">causation.</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
	<svg viewBox="0 0 480 330" role="img" aria-labelledby="covariance-signs-svg-title covariance-signs-svg-desc">
		<title id="covariance-signs-svg-title">Signs of centered-product contributions to covariance</title>
		<desc id="covariance-signs-svg-desc">Four equally likely outcomes have X mean zero and Y mean one. Points (-1, -1) and (1, 3) each contribute a centered product of two. Points (-1, 1) and (1, 1) lie on the Y-mean line and contribute zero. Their average product is one. Empty quadrants show where opposite-sign deviations would contribute negatively.</desc>
		<rect x="55" y="30" width="370" height="240" rx="8" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.5"></rect>
		<path class="viz-axis" d="M55 150H425M240 30V270"></path>
		<path d="M235 38L240 30L245 38M417 145L425 150L417 155" style="fill:none;stroke:var(--c-text);stroke-width:1.8"></path>
		<text class="viz-axis-label" x="438" y="155" style="font-size:15px">X</text>
		<text class="viz-axis-label" x="246" y="20" style="font-size:15px">Y</text>
		<text class="viz-axis-label" x="240" y="291" text-anchor="middle" style="font-size:15px">mean of X = 0</text>
		<text class="viz-axis-label" x="62" y="143" style="font-size:15px">mean of Y = 1</text>
		<text class="viz-callout" x="332" y="54" text-anchor="middle" style="font-size:15px">same signs: (+)(+) = +</text>
		<text class="viz-callout" x="148" y="54" text-anchor="middle" style="font-size:15px">opposite signs: (-)(+) = -</text>
		<text class="viz-callout" x="148" y="254" text-anchor="middle" style="font-size:15px">same signs: (-)(-) = +</text>
		<text class="viz-callout" x="332" y="254" text-anchor="middle" style="font-size:15px">opposite signs: (+)(-) = -</text>
		<circle cx="305" cy="80" r="12" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></circle>
		<text class="viz-axis-label" x="305" y="85" text-anchor="middle" style="font-size:15px">2</text>
		<text class="viz-axis-label" x="323" y="85" style="font-size:15px">(1, 3)</text>
		<circle cx="175" cy="220" r="12" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></circle>
		<text class="viz-axis-label" x="175" y="225" text-anchor="middle" style="font-size:15px">2</text>
		<text class="viz-axis-label" x="91" y="225" style="font-size:15px">(-1, -1)</text>
		<path d="M175 138L187 150L175 162L163 150Z" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:2"></path>
		<text class="viz-axis-label" x="175" y="155" text-anchor="middle" style="font-size:15px">0</text>
		<text class="viz-axis-label" x="130" y="181" style="font-size:15px">(-1, 1)</text>
		<path d="M305 138L317 150L305 162L293 150Z" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:2"></path>
		<text class="viz-axis-label" x="305" y="155" text-anchor="middle" style="font-size:15px">0</text>
		<text class="viz-axis-label" x="283" y="181" style="font-size:15px">(1, 1)</text>
		<text class="viz-callout" x="240" y="317" text-anchor="middle" style="font-size:15px">Cov(X, Y) averages every signed centered product.</text>
	</svg>
	<div class="visual-derivation-caption"><strong>Read it this way:</strong> each point has probability one quarter. Two centered products equal two; two equal zero. Their average is one. Opposite-sign deviations would contribute negatively.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="covariance-centered-product-signs"><strong>Check:</strong> Why do points with Y equal to its mean contribute zero even when X differs from its mean?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="covariance-centered-product-signs">Expectation locates the mean. Variance measures squared spread around it. Covariance averages signed products of joint deviations; same-side points add and opposite-side points subtract. Their pairing changes the spread of a sum even when marginal spreads stay fixed. Changing units alters covariance, not standardized correlation. A curved deterministic relationship can have zero covariance. None of these associations proves causation. <strong>Does multiplying Y's units by one hundred change its correlation?</strong></span></figcaption>
</figure>

## Expectation

For probability mass $p(x)$, $\mathbb E[X]=\sum_x xp(x)$.
For a density $p(x)$, replace the sum by $\int xp(x)\,dx$.
Assume finite expectations. For constants $a,b$, linearity gives

$$
\mathbb E[aX+bY]=a\mathbb E[X]+b\mathbb E[Y].
$$

No independence is needed. An expectation need not be a possible outcome: a fair die has expectation $3.5$.

## Variance

Let $\mu_X=\mathbb E[X]$ and $\mu_Y=\mathbb E[Y]$. Assume finite second moments.
Variance squares one deviation; covariance multiplies two:

$$
\operatorname{Var}(X)=\mathbb E[(X-\mu_X)^2],\qquad
\operatorname{Cov}(X,Y)=\mathbb E[(X-\mu_X)(Y-\mu_Y)].
$$

Expanding these expressions gives $\operatorname{Var}(X)=\mathbb E[X^2]-\mu_X^2$ and $\operatorname{Cov}(X,Y)=\mathbb E[XY]-\mu_X\mu_Y$.

## Worked example

| Outcome | Probability | $X-0$ | $Y-1$ | Centered product |
| --- | --- | --- | --- | --- |
| $(-1,-1)$ | $1/4$ | $-1$ | $-2$ | $2$ |
| $(-1,1)$ | $1/4$ | $-1$ | $0$ | $0$ |
| $(1,1)$ | $1/4$ | $1$ | $0$ | $0$ |
| $(1,3)$ | $1/4$ | $1$ | $2$ | $2$ |

The means are $0$ and $1$. The second moments are $1$ and $3$.
Thus $\operatorname{Var}(X)=1$, $\operatorname{Var}(Y)=3-1^2=2$ and $\operatorname{Cov}(X,Y)=1$.

## Variance of a sum

Square $(X-\mu_X)+(Y-\mu_Y)$, then average. The two cross-products produce the covariance term:

$$
\operatorname{Var}(X+Y) = \operatorname{Var}(X) + \operatorname{Var}(Y) + 2\operatorname{Cov}(X,Y).
$$

Here the variance is $1+2+2(1)=5$.
Direct check: $X+Y$ takes $-2,0,2,4$, with mean $1$.
Its squared deviations are $9,1,1,9$, averaging $5$.
Removing dependence while retaining the marginals would instead give sum variance $3$.

## Correlation

Pearson correlation divides covariance by positive standard deviations $\sigma_X$ and $\sigma_Y$:

$$
\rho_{X,Y} = \frac{\operatorname{Cov}(X,Y)}{\sigma_X\sigma_Y}.
$$

Here $\rho=1/\sqrt2\approx0.707107$. It is undefined if either variance is zero.
Correlation lies between $-1$ and $1$ and does not establish an intervention effect.

## Change the measurement

### 1. Convert the units of Y

**New task:** replace $Y$ by $100Y$ without changing the joint observations.

**Picture and change:** each vertical deviation grows by $100$.
Covariance scales once; $Y$ variance scales twice.

**Check:** covariance becomes $100$ and variance becomes $20{,}000$.
Correlation stays $1/\sqrt2$. A negative rescaling would reverse its sign.

**Use when:** units change but the underlying association does not.

### 2. Average correlated measurements

**New task:** average four variables, each with variance $4$ and pairwise correlation $0.5$.

**Picture and change:** retain all six pairwise covariances, each equal to $2$.
Divide the full sum variance by $4^2$.

**Check:** average variance is $(16+2\times12)/16=2.5$, not the independent value $1$.

**Use when:** measurements share a user, environment or disturbance.

## Zero covariance can hide dependence

Let $X$ be equally likely to be $-1,0,1$, and let $Y=X^2$.
Both $\mathbb E[X]$ and $\mathbb E[XY]$ are zero, so covariance is zero.
Yet $Y$ is determined by $X$. Independence implies zero covariance under these moment conditions; the reverse implication fails.

## Try averaging two independent quantities

Independent variables have variances $9$ and $16$. Find the variance and standard deviation of their average.

<details>
<summary>Check your answer</summary>

The covariance term is zero. Variance is $(9+16)/4=6.25$, and standard deviation is $2.5$.

</details>

<!-- learning:core-end -->

<div data-learning-extension>

## Scaling and repeated measurements

For constants $a,b$, $\operatorname{Var}(aX+b)=a^2\operatorname{Var}(X)$.
Translation changes the mean but not spread. Standard deviation scales by $|a|$.
More generally,

$$
\operatorname{Var}(aX+bY)=a^2\operatorname{Var}(X)+b^2\operatorname{Var}(Y)+2ab\operatorname{Cov}(X,Y).
$$

For independent measurements with common variance $\sigma^2$, their mean has variance $\sigma^2/n$.
Its standard error falls as $1/\sqrt n$, not $1/n$.
As another check, variances $4,4$ and covariance $3$ give sum variance $4+4+2(3)=14$.

## Covariance matrices

For a random vector $x \in \mathbb{R}^d$ with mean $\mu$, the covariance matrix is

$$
\Sigma = \mathbb{E}[(x-\mu)(x-\mu)^\top].
$$

The diagonal contains feature variances. Entry $(i,j)$ contains the covariance between features $i$ and $j$.

A covariance matrix is symmetric and positive semidefinite. For any vector $v$,

$$
v^\top\Sigma v = \operatorname{Var}(v^\top x) \ge 0.
$$

This supports PCA, Gaussian models, whitening and uncertainty ellipsoids.
Use a scatter plot alongside Pearson correlation: outliers can change it sharply.
Spearman correlation instead measures association between ranks; it answers a different question about monotonic relationships.

## In an interview

Reconstruct the sum-variance formula by centering and squaring.
State the observation law, moment conditions and independence assumptions before simplifying it.

## Common mistakes

- Saying zero correlation means independence.
- Treating correlation as evidence of intervention effects.
- Forgetting the covariance term in the variance of a sum.
- Saying variance scales linearly with the units of $X$.
- Comparing covariances across variables with very different units.
- Using Pearson correlation without checking for nonlinearity or outliers.

## Practice next

Use these quantities in [bias and variance of estimators](/concepts/bias-variance-of-estimators/), [SVD and PCA](/concepts/svd-and-pca/), and the [ML math oral](/prep/labs/math-oral/).

</div>
