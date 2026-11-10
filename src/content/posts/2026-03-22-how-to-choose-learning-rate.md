---
title: "How do you choose a learning rate?"
description: "Check update stability, propose a learning-rate peak, then test its schedule from a clean starting state."
date: "2026-03-22"
draft: false
tags: ["questions"]
category: "questions"
---


Choose a learning-rate policy by comparing update behavior and validating a complete training schedule rather than a single sweep result.

## Problem

**Context.** The exact one-dimensional SGD fixture below isolates learning-rate behavior before considering noisy, scheduled training.

**Task.** Compare the three constant rates, explain convergence or divergence, and propose a controlled learning-rate selection procedure.
Describe how warmup, decay, optimizer choice, and intended run length affect the final comparison.

**Constraints.** Use the stated loss, gradient, and starting point for the fixture; do not add momentum or stochastic noise.
A short range test proposes candidates rather than proving the best full-run schedule.

Use plain SGD on $L(w)=5w^2$, starting at $w_0=1$ with exact gradient $10w$.
Compare constant rates $0.1$, $0.2$ and $0.25$.
Which rates reduce the loss, and what changes when the intended run uses a schedule?

<span id="what-an-l4-answer-sounds-like"></span>
<span id="what-an-l5-answer-sounds-like"></span>

<!-- visual:learning-rate-range-to-schedule -->
<figure class="learning-figure" aria-labelledby="learning-rate-procedure-title">
	<p class="visual-kicker">Candidate first, controlled run second</p>
	<p class="visual-title" id="learning-rate-procedure-title">Check the actual update and loss, then test a complete schedule from a clean starting state</p><div class="visual-argument" data-visual-argument="learning-rate-range-to-schedule" data-argument-kind="curve" data-argument-composition="iterative-step"><svg viewBox="0 0 360 2630" role="img" aria-label="Check the actual update and loss, then test a complete schedule from a clean starting state. One plain-SGD rate reaches the minimum. Another oscillates without progress; the third grows the error. Finite coordinates do not prove improvement. Doubling curvature changes which rate is stable. A disposable range test proposes candidates, not independent full-run comparisons. Restore model, optimizer and schedule state before testing the intended warmup and decay at a controlled budget. Optimizer, reduction and gradient scale matter; these bounds are not universal."><title>Check the actual update and loss, then test a complete schedule from a clean starting state</title><desc>One plain-SGD rate reaches the minimum. Another oscillates without progress; the third grows the error. Finite coordinates do not prove improvement. Doubling curvature changes which rate is stable. A disposable range test proposes candidates, not independent full-run comparisons. Restore model, optimizer and schedule state before testing the intended warmup and decay at a controlled budget. Optimizer, reduction and gradient scale matter; these bounds are not universal.</desc><text x="20" y="27" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Same exact SGD fixture; three</tspan><tspan x="20" dy="18">different outcomes</tspan></text><text x="20" y="115" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Start at coordinate one. Loss is five</tspan><tspan x="20" dy="18">times its square; gradient ten times</tspan><tspan x="20" dy="18">the coordinate. No momentum or noise.</tspan></text><text x="20" y="250" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="20" dy="0">Rate 0.1: reaches the minimum</tspan></text><path d="M35 330L325 330" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M180 285L180 395" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><g data-picture-update-step="0,0" data-picture-coordinate="1"><circle cx="240" cy="300" r="5" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><g data-picture-update-step="0,1" data-picture-coordinate="0"><circle cx="180" cy="340" r="5" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><path d="M240 300L180 340M184.43760156980184 332.2341972528468L180 340L188.87520313960366 338.8905996075496" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2;stroke-linejoin:round"></path><g data-picture-update-step="0,2" data-picture-coordinate="0"><circle cx="180" cy="380" r="5" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><path d="M180 340L180 380M176 372L180 380L184 372" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2;stroke-linejoin:round"></path><text x="20" y="450" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="20" dy="0">Coordinates: 1, 0, 0. Losses: 5, 0,</tspan><tspan x="20" dy="18">0.</tspan></text><text x="20" y="590" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-output-stroke)"><tspan x="20" dy="0">Rate 0.2: finite oscillation, no</tspan><tspan x="20" dy="18">progress</tspan></text><path d="M35 670L325 670" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M180 625L180 735" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><g data-picture-update-step="1,0" data-picture-coordinate="1"><circle cx="240" cy="640" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-update-step="1,1" data-picture-coordinate="-1"><circle cx="120" cy="680" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><path d="M240 640L120 680M126.32455532033676 673.6754446796632L120 680L128.85437744847147 681.2649110640674" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:2;stroke-linejoin:round"></path><g data-picture-update-step="1,2" data-picture-coordinate="1"><circle cx="240" cy="720" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><path d="M120 680L240 720M231.14562255152853 721.2649110640674L240 720L233.67544467966323 713.6754446796632" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:2;stroke-linejoin:round"></path><text x="20" y="790" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-output-stroke)"><tspan x="20" dy="0">Coordinates: 1, -1, 1. Losses: 5, 5,</tspan><tspan x="20" dy="18">5.</tspan></text><text x="20" y="930" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-warning-stroke)"><tspan x="20" dy="0">Rate 0.25: crosses and moves farther</tspan></text><path d="M35 1010L325 1010" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M180 965L180 1075" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><g data-picture-update-step="2,0" data-picture-coordinate="1"><circle cx="240" cy="980" r="5" style="fill:var(--viz-warning-stroke);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle></g><g data-picture-update-step="2,1" data-picture-coordinate="-1.5"><circle cx="90" cy="1020" r="5" style="fill:var(--viz-warning-stroke);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle></g><path d="M240 980L90 1020M96.69922891456865 1014.0737590371124L90 1020L98.7605301190513 1021.8036385539223" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:2;stroke-linejoin:round"></path><g data-picture-update-step="2,2" data-picture-coordinate="2.25"><circle cx="315" cy="1060" r="5" style="fill:var(--viz-warning-stroke);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle></g><path d="M90 1020L315 1060M306.42336659884273 1062.5379833534037L315 1060L307.8236332765827 1054.6614832911166" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:2;stroke-linejoin:round"></path><text x="20" y="1130" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-warning-stroke)"><tspan x="20" dy="0">Coordinates: 1, -1.5, 2.25. Losses:</tspan><tspan x="20" dy="18">5, 11.25, 25.3125.</tspan></text><text x="20" y="1270" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Double the curvature: the old rate</tspan><tspan x="20" dy="18">0.1 now oscillates without progress.</tspan><tspan x="20" dy="18">Recompute the update; do not transfer</tspan><tspan x="20" dy="18">the old bound.</tspan></text><text x="20" y="1435" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Separate schedule fixture: one</tspan><tspan x="20" dy="18">thousand steps, one hundred warmup,</tspan><tspan x="20" dy="18">peak 0.001, final 0.0001.</tspan></text><path d="M35 1830L325 1830" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M35 1830L37.9 1808L40.8 1786L43.7 1764L46.6 1742L49.5 1720L52.4 1698L55.3 1676L58.2 1654L61.099999999999994 1632L64 1610L66.9 1610.0603081251095L69.8 1610.2411590242775L72.7 1610.542332358541L75.6 1610.9634611945846L78.5 1611.5040324517913L81.4 1612.1633875273533L84.30000000000001 1612.9407230986765L87.19999999999999 1613.8350921021065L90.1 1614.8454048867798L93 1615.9704305421951L95.9 1617.2087983978881L98.8 1618.5589996933825L101.7 1620.0193894163824L104.6 1621.5881883069662L107.5 1623.2634850253405L110.4 1625.0432384805138L113.30000000000001 1626.925280317051L116.2 1628.9073175568801L119.1 1630.9869353929346L122 1633.1616001312211L124.9 1635.4286622777379L127.8 1637.7853597664734L130.7 1640.2288213245592L133.60000000000002 1642.756069970473L136.5 1645.3640266410325L139.39999999999998 1648.0495139427599L142.3 1650.8092600230452L145.2 1653.639902556396L148.10000000000002 1656.5379928409127L151 1659.5L153.89999999999998 1662.5223152841968L156.8 1665.6012564678813L159.7 1668.7330723354958L162.6 1671.9139472518248L165.5 1675.1400058107588L168.4 1678.4073175568801L171.29999999999998 1681.711901774117L174.2 1685.0497323356328L177.1 1688.4167426090419L180 1691.8088304109738L182.9 1695.2218630049535L185.8 1698.6516821365024L188.70000000000002 1702.0941090993315L191.60000000000002 1705.5449498264525L194.5 1709L197.4 1712.4550501735475L200.29999999999998 1715.9058909006685L203.2 1719.3483178634976L206.1 1722.7781369950465L209 1726.191169589026L211.9 1729.5832573909581L214.8 1732.9502676643672L217.7 1736.288098225883L220.6 1739.5926824431199L223.5 1742.8599941892412L226.4 1746.0860527481752L229.3 1749.2669276645042L232.20000000000002 1752.3987435321187L235.1 1755.4776847158032L238 1758.5L240.89999999999998 1761.4620071590873L243.79999999999998 1764.360097443604L246.7 1767.1907399769548L249.6 1769.9504860572401L252.5 1772.6359733589675L255.4 1775.2439300295268L258.3 1777.7711786754408L261.20000000000005 1780.2146402335266L264.1 1782.5713377222621L267 1784.8383998687789L269.9 1787.0130646070654L272.79999999999995 1789.0926824431199L275.7 1791.074719682949L278.6 1792.9567615194862L281.5 1794.7365149746595L284.4 1796.4118116930338L287.3 1797.9806105836176L290.2 1799.4410003066175L293.1 1800.7912016021119L296 1802.0295694578049L298.90000000000003 1803.1545951132202L301.8 1804.1649078978935L304.7 1805.0592769013238L307.59999999999997 1805.8366124726467L310.5 1806.4959675482087L313.4 1807.0365388054154L316.3 1807.457667641459L319.2 1807.7588409757227L322.1 1807.9396918748905L325 1808" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2.5"></path><text x="35" y="1880" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="35" dy="0">0</tspan></text><text x="180" y="1880" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="180" dy="0">500</tspan></text><text x="325" y="1880" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="325" dy="0">1000</tspan></text><text x="20" y="1980" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Position 50: rate 0.00050</tspan></text><text x="20" y="2055" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Position 100: rate 0.00100</tspan></text><text x="20" y="2130" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Position 550: rate 0.00055</tspan></text><text x="20" y="2205" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Position 1000: rate 0.00010</tspan></text><text x="20" y="2320" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">A sweep changes weights along its</tspan><tspan x="20" dy="18">path. Reload model, optimizer and</tspan><tspan x="20" dy="18">schedule state; control data order,</tspan><tspan x="20" dy="18">randomness and total budget.</tspan></text><text x="20" y="2470" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--viz-warning-stroke)"><tspan x="20" dy="0">Mean versus summed loss changes</tspan><tspan x="20" dy="18">gradient scale. Adam moments change</tspan><tspan x="20" dy="18">the update. A smaller rate cannot</tspan><tspan x="20" dy="18">repair wrong labels or a broken</tspan><tspan x="20" dy="18">backward pass.</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
	<div class="visual-grid--two" role="group" aria-label="Two-stage learning-rate selection procedure">
		<section class="visual-panel plot-panel">
			<svg viewBox="0 0 300 238" role="img" aria-labelledby="lr-range-title lr-range-desc">
				<title id="lr-range-title">Read a learning-rate range test from too small through useful descent to instability</title>
				<desc id="lr-range-desc">An original qualitative plot shows training loss against learning rate on a logarithmic horizontal axis. At very small rates the loss is nearly flat. It then falls, with a hatched candidate band around the sustained steep descent. Farther right the curve bottoms and turns sharply upward in an instability region marked with crosses. The selected candidate is before the minimum and blow-up, not at either one.</desc>
				<defs>
					<pattern id="lr-candidate-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
						<line x1="0" y1="0" x2="0" y2="6" style="stroke:var(--viz-focus-stroke);stroke-width:2"></line>
					</pattern>
				</defs>
				<rect class="viz-plot-bg" x="38" y="28" width="248" height="158" rx="4"></rect>
				<rect x="125" y="28" width="64" height="158" style="fill:url(#lr-candidate-hatch);opacity:.24"></rect>
				<path class="viz-gridline" d="M38 81H286M38 133H286M100 28V186M162 28V186M224 28V186"></path>
				<path class="viz-axis" d="M38 28V186H286"></path>
				<path class="viz-pr-curve" d="M45 61C76 62 94 60 112 70C134 83 151 112 171 139C187 160 205 169 218 160C235 149 246 103 259 57C267 35 275 30 282 34"></path>
				<circle class="viz-operating-point" cx="158" cy="121" r="5"></circle>
				<path class="viz-operating-guide" d="M158 121V186"></path>
				<path d="M252 47L262 57M262 47L252 57M267 31L277 41M277 31L267 41" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:2"></path>
				<text class="viz-axis-label" x="42" y="18">1 · SWEEP LR UP; WATCH LOSS</text>
				<text class="viz-label" x="69" y="48" text-anchor="middle">too small</text>
				<text class="viz-callout" x="157" y="102" text-anchor="middle">candidate</text>
				<text class="viz-callout" x="157" y="115" text-anchor="middle">steep descent</text>
				<text class="viz-label" x="247" y="82" text-anchor="middle">unstable</text>
				<text class="viz-label" x="38" y="204">10⁻⁷</text>
				<text class="viz-label" x="151" y="204">10⁻⁴</text>
				<text class="viz-label" x="268" y="204">10⁻¹</text>
				<text class="viz-axis-label" x="162" y="228" text-anchor="middle">learning rate (log scale) →</text>
				<text class="viz-axis-label" transform="translate(14 132) rotate(-90)">training loss</text>
			</svg>
		</section>
		<section class="visual-panel plot-panel">
			<svg viewBox="0 0 300 238" role="img" aria-labelledby="lr-schedule-title lr-schedule-desc">
				<title id="lr-schedule-title">Use the selected candidate as the peak of a warmup and decay schedule</title>
				<desc id="lr-schedule-desc">A learning-rate schedule begins low, rises linearly during a short warmup to the candidate peak selected by the range test, then follows a curved decay over the remaining training steps to a lower final rate. A vertical dotted guide marks the end of warmup, and the peak has a double-ring marker. The figure emphasizes that the selected value is a peak, not a constant rate used for every step.</desc>
				<rect class="viz-plot-bg" x="38" y="28" width="248" height="158" rx="4"></rect>
				<path class="viz-gridline" d="M38 81H286M38 133H286M100 28V186M162 28V186M224 28V186"></path>
				<path class="viz-axis" d="M38 28V186H286"></path>
				<path d="M45 174L89 58C132 60 174 70 211 92C241 110 263 133 282 154" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2.5"></path>
				<path class="viz-operating-guide" d="M89 42V186"></path>
				<circle class="viz-node viz-node--focus" cx="89" cy="58" r="8"></circle>
				<circle class="viz-node" cx="89" cy="58" r="4"></circle>
				<text class="viz-axis-label" x="42" y="18">2 · USE IT AS THE PEAK</text>
				<text class="viz-callout" x="65" y="104" text-anchor="middle">warmup</text>
				<text class="viz-callout" x="89" y="43" text-anchor="middle">selected peak</text>
				<text class="viz-callout" x="205" y="78" text-anchor="middle">decay</text>
				<text class="viz-label" x="57" y="204" text-anchor="middle">start</text>
				<text class="viz-label" x="89" y="204" text-anchor="middle">warmup end</text>
				<text class="viz-label" x="274" y="204" text-anchor="middle">finish</text>
				<text class="viz-axis-label" x="162" y="228" text-anchor="middle">training step →</text>
				<text class="viz-axis-label" transform="translate(14 132) rotate(-90)">learning rate</text>
			</svg>
		</section>
	</div>
	<div class="visual-derivation-caption">the first plot proposes a rate before instability. The second assigns that candidate to a schedule. These qualitative curves are not measured results or proof of an optimal peak.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="learning-rate-range-to-schedule"><strong>Check:</strong> Which LR from the sweep becomes the schedule peak? Why is the unstable endpoint not the choice?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="learning-rate-range-to-schedule">One plain-SGD rate reaches the minimum. Another oscillates without progress; the third grows the error. Finite coordinates do not prove improvement. Doubling curvature changes which rate is stable. A disposable range test proposes candidates, not independent full-run comparisons. Restore model, optimizer and schedule state before testing the intended warmup and decay at a controlled budget. Optimizer, reduction and gradient scale matter; these bounds are not universal. <strong>Does finite oscillation at rate 0.2 establish progress?</strong></span></figcaption>
</figure>
<p class="diagram-source">Range-test references: <a href="https://arxiv.org/abs/1506.01186">Smith (2017)</a> and the <a href="https://docs.fast.ai/callback.schedule.html#lrfind">fastai learning-rate finder documentation</a>.</p>

## Calculate the updates

For plain SGD, $w_{t+1}=w_t-\eta\nabla L(w_t)=(1-10\eta)w_t$.
Contraction requires $|1-10\eta|<1$, so $0<\eta<0.2$.

| Rate | $w_0,w_1,w_2$ | $L(w_0),L(w_1),L(w_2)$ | Decision |
| --- | --- | --- | --- |
| $0.1$ | $1,0,0$ | $5,0,0$ | Reaches the minimum |
| $0.2$ | $1,-1,1$ | $5,5,5$ | Oscillates without progress |
| $0.25$ | $1,-1.5,2.25$ | $5,11.25,25.3125$ | Diverges |

Finite values alone do not establish progress. The boundary rate $0.2$ never improves this loss.
This calculation is not an Adam benchmark or a universal neural-network bound.

## Test a candidate

Record the optimizer, loss reduction, batch, initialization, precision and intended training length.
Use a disposable run to increase rates while recording loss and update behavior.
For $N\ge2$ sweep steps, a geometric sweep can use

$$
\eta_j=\eta_{\mathrm{low}}\left(\frac{\eta_{\mathrm{high}}}{\eta_{\mathrm{low}}}\right)^{j/(N-1)},\qquad j=0,\ldots,N-1.
$$

Choose a candidate in sustained descent, with margin before instability.
Restore the starting model and optimizer state, then test the intended schedule on controlled runs.
Compare representative validation results and stability at the same budget.
A sweep changes weights along its path; its points are not independent rate trials.

## Change the run

### 1. Double the curvature

**New task:** replace the loss by $10w^2$.

**Picture and change:** the gradient becomes $20w$. Recompute contraction from $|1-20\eta|<1$.

**Check:** the interval becomes $0<\eta<0.1$. The old rate $0.1$ now oscillates without progress.

**Use when:** loss scaling, parameterization or optimizer changes the effective update.

### 2. Specify a shorter training horizon

**New task:** use $S=1000$ steps, $W=100$ warmup steps, peak $0.001$ and final rate $0.0001$.

**Picture and change:** choose the schedule separately from the peak.
At schedule position $t\in[0,S]$, use

$$
\eta(t)=\begin{cases}
0.001t/W,&0\le t\le W,\\
0.0001+\frac{0.0009}{2}\left[1+\cos\left(\pi\frac{t-W}{S-W}\right)\right],&W<t\le S.
\end{cases}
$$

**Check:** positions $50$, $100$, $550$ and $1000$ give rates $0.0005$, $0.001$, $0.00055$ and $0.0001$.

**Use when:** the step budget changes. These fractions describe this fixture, not a universal schedule.

## Do not continue from a failed sweep

Diverged weights and accumulated Adam moments are not a clean starting state.
Reload model, optimizer and schedule state; control data order and randomness for the comparison.
Otherwise a failed sweep can invalidate the candidate test.

## Try changing the batch

Doubling batch size also changes loss reduction from mean to sum. What must be checked first?

<details>
<summary>Check your answer</summary>

Restore the intended reduction or account for its doubled gradient scale.
Only then test a batch-scaling heuristic under controlled conditions.

</details>

<!-- learning:core-end -->

<div data-learning-extension>

<span id="what-an-l6-answer-sounds-like"></span>
<span id="the-tells-that-get-you-a-strong-hire-vote"></span>
<span id="the-tells-that-get-you-down-leveled"></span>

## Optimizer and training regime

Adam uses accumulated, coordinate-dependent moments, so its effective update is not simply plain SGD's gradient times a rate.
Batch-scaling rules require assumptions. Neither linear scaling for SGD nor square-root scaling for Adam is universal.
The [large-minibatch SGD recipe](https://arxiv.org/abs/1706.02677) combines scaling with specific training conditions and warmup.

Fine-tuning must balance adaptation against preserving useful behavior.
Compare layer updates and held-out behavior instead of assuming pretrained weights are well-conditioned.
Fixed fine-tuning, SFT or preference-training rate ranges do not transfer automatically between models and objectives.

## Schedules and diagnostics

Warmup limits early updates; its usefulness and mechanism depend on optimizer, parameterization and model.
The [Transformer schedule](https://arxiv.org/abs/1706.03762), cosine decay, step decay and one-cycle use different time profiles.
Compare them at a stated training budget rather than declaring one always superior.

If every reasonable rate fails, inspect data, initialization, gradient scale, loss implementation and numerical precision.
Reducing the rate does not repair incorrect labels or a broken backward pass.

Related: [debug a model that is not learning](/questions/debug-model-not-learning/) and [bias and variance](/questions/bias-variance-tradeoff/).

</div>
