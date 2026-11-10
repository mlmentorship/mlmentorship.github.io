---
title: "Expectation-Maximization (EM)"
description: "Compute posterior expected complete-data statistics, then improve a touching likelihood bound. Trace one exact two-coin EM iteration."
date: "2025-11-09"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

[Expectation-Maximization](https://www.jstor.org/stable/2984875) alternates posterior computation with parameter improvement for a latent-variable model.
An exact posterior makes a likelihood lower bound touch the current objective; improving that fixed bound cannot decrease observed likelihood.
This guarantee is not a global-optimum or component-identification guarantee.

## The two steps

Let x be observations, z latent variables, and theta parameters.
Require available exact posteriors, common support, finite required expectations, and an M-step that does not reduce its target.

### E-step

Given current parameters $\theta_t$, compute the posterior over latents:

$$
q_t(z) = p(z \mid x; \theta_t).
$$

Current theta stays fixed while q is computed.
Calculate expected complete-data sufficient statistics, not merely a substituted mean latent value.
For example, $E[zz^\top]$ generally differs from $E[z]E[z]^\top$.

### M-step

Update $\theta$ to maximize the **expected complete-data log-likelihood**:

$$
\theta_{t+1} = \arg\max_\theta \mathbb{E}_{z \sim q_t}[\log p(x, z; \theta)].
$$

Here q stays fixed while theta changes.
Write $Q(\theta\mid\theta_t)=E_{q_t}\log p(x,z;\theta)$.

For a teaching mixture, choose a coin independently for each two-toss group.
Initial weights are $(0.5,0.5)$ and head probabilities $(0.75,0.25)$; observe head counts two and one.
The binomial probability includes $\binom{2}{h}p^h(1-p)^{2-h}$.
Count two has likelihoods $(0.5625,0.0625)$, giving responsibilities $(0.9,0.1)$.
Count one has likelihoods $(0.375,0.375)$, giving responsibilities $(0.5,0.5)$.

Effective group counts are $(1.4,0.6)$; weighted heads are $(2.3,0.7)$.
The M-step sets weights to $(0.7,0.3)$ and head probabilities to $(2.3/2.8,0.7/1.2)=(23/28,7/12)$.
The denominators count expected tosses, not expected groups.
Observed likelihood rises from $0.3125\times0.375=0.1171875$ to approximately $0.20172548$.
Its log rises from $-2.143980$ to $-1.600848$.

## Why it works

EM maximizes a lower bound on $\log p(x; \theta)$:

$$
\log p(x; \theta) = \mathbb{E}_q[\log p(x, z; \theta)] - \mathbb{E}_q[\log q(z)] + \mathrm{KL}(q \| p_\theta(z \mid x)).
$$

Write $F(q,\theta)=Q(\theta\mid\theta_t)+H(q)$ when q is the current posterior.
The identity says $\ell(\theta)=F(q,\theta)+\mathrm{KL}(q\|p_\theta)$, with a nonnegative KL gap.
Exact E sets the gap to zero at theta_t.
Improving Q with q fixed then yields:

$$
\ell(\theta_{t+1})\ge F(q_t,\theta_{t+1})\ge F(q_t,\theta_t)=\ell(\theta_t).
$$

<!-- visual:em-lower-bound-climb -->
<figure class="learning-figure plot-panel" aria-labelledby="em-bound-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="em-bound-title">Posterior memberships feed expected-count updates; improving their fixed bound raises likelihood</p><div class="visual-argument" data-visual-argument="em-lower-bound-climb" data-argument-kind="curve" data-argument-composition="coin-posteriors-and-fixed-bound"><svg viewBox="0 0 360 3030" role="img" aria-label="Posterior memberships feed expected-count updates; improving their fixed bound raises likelihood. E fixes parameters and splits each group membership by its posterior. M freezes those memberships and updates expected group counts, heads and tosses. Tosses, not groups, normalize head chances. Exact E makes the likelihood bound touch; improving that fixed bound cannot lower observed likelihood. Repeat with new posteriors. Approximate updates need separate guarantees; this is neither global optimality nor component identification."><title>Posterior memberships feed expected-count updates; improving their fixed bound raises likelihood</title><desc>E fixes parameters and splits each group membership by its posterior. M freezes those memberships and updates expected group counts, heads and tosses. Tosses, not groups, normalize head chances. Exact E makes the likelihood bound touch; improving that fixed bound cannot lower observed likelihood. Repeat with new posteriors. Approximate updates need separate guarantees; this is neither global optimality nor component identification.</desc><text x="20" y="27" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">E-step: keep current coin parameters</tspan><tspan x="20" dy="18">fixed</tspan></text><text x="20" y="111" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Equal mixture shares; head chances A</tspan><tspan x="20" dy="18">75%, B 25%</tspan></text><text x="20" y="186" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Group 1: 2 heads in two tosses</tspan></text><circle cx="65" cy="257" r="20" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.75"></circle><text x="65" y="263" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="65" dy="0">H</tspan></text><circle cx="131" cy="257" r="20" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.75"></circle><text x="131" y="263" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="131" dy="0">H</tspan></text><rect x="25" y="332" width="279" height="38" data-picture-em-responsibility="0,A,0.9" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:2"></rect><rect x="304" y="332" width="31" height="38" data-picture-em-responsibility="0,B,0.1" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:2"></rect><text x="20" y="419" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="20" dy="0">A: 0.9</tspan></text><text x="340" y="419" style="font-size:14px;font-weight:700;text-anchor:end;fill:var(--viz-output-stroke)"><tspan x="340" dy="0">B: 0.1</tspan></text><text x="20" y="516" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Group 2: 1 heads in two tosses</tspan></text><circle cx="65" cy="587" r="20" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.75"></circle><text x="65" y="593" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="65" dy="0">H</tspan></text><circle cx="131" cy="587" r="20" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.75"></circle><text x="131" y="593" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="131" dy="0">T</tspan></text><rect x="25" y="662" width="155" height="38" data-picture-em-responsibility="1,A,0.5" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:2"></rect><rect x="180" y="662" width="155" height="38" data-picture-em-responsibility="1,B,0.5" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:2"></rect><text x="20" y="749" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="20" dy="0">A: 0.5</tspan></text><text x="340" y="749" style="font-size:14px;font-weight:700;text-anchor:end;fill:var(--viz-output-stroke)"><tspan x="340" dy="0">B: 0.5</tspan></text><text x="20" y="851" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">Bars split one group membership. Toss</tspan><tspan x="20" dy="18">order is ignored; no group is counted</tspan><tspan x="20" dy="18">twice.</tspan></text><text x="20" y="973" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">M-step: freeze those memberships</tspan></text><text x="20" y="1097" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Expected groups</tspan></text><text x="20" y="1155" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="20" dy="0">A: 1.4</tspan></text><text x="340" y="1155" style="font-size:14px;font-weight:700;text-anchor:end;fill:var(--viz-output-stroke)"><tspan x="340" dy="0">B: 0.6</tspan></text><text x="20" y="1252" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Expected heads</tspan></text><text x="20" y="1310" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="20" dy="0">A: 2.3</tspan></text><text x="340" y="1310" style="font-size:14px;font-weight:700;text-anchor:end;fill:var(--viz-output-stroke)"><tspan x="340" dy="0">B: 0.7</tspan></text><text x="20" y="1407" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Expected tosses</tspan></text><text x="20" y="1465" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="20" dy="0">A: 2.8</tspan></text><text x="340" y="1465" style="font-size:14px;font-weight:700;text-anchor:end;fill:var(--viz-output-stroke)"><tspan x="340" dy="0">B: 1.2</tspan></text><text x="20" y="1527" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">A: 2.3 heads among 2.8 tosses</tspan></text><rect x="25" y="1589" width="289.3333333333333" height="40" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2"></rect><rect x="25" y="1589" width="237.66666666666666" height="40" data-picture-em-expected-heads="2.3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:2"></rect><text x="20" y="1687" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="20" dy="0">Updated head chance: 0.8214</tspan></text><text x="20" y="1734" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">B: 0.7 heads among 1.2 tosses</tspan></text><rect x="25" y="1796" width="123.99999999999999" height="40" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:2"></rect><rect x="25" y="1796" width="72.33333333333333" height="40" data-picture-em-expected-heads="0.7" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:2"></rect><text x="20" y="1894" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-output-stroke)"><tspan x="20" dy="0">Updated head chance: 0.5833</tspan></text><text x="20" y="1977" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Updated mixture shares: A 70%, B 30%</tspan></text><rect x="25" y="2023" width="217" height="38" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:2"></rect><rect x="242" y="2023" width="93" height="38" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:2"></rect><text x="20" y="2157" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Actual update: solid observed</tspan><tspan x="20" dy="18">likelihood</tspan></text><text x="20" y="2227" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Dashed bound keeps the old posterior</tspan><tspan x="20" dy="18">fixed</tspan></text><path d="M35 2578L325 2578" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M35 2578L35 2320" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M35 2550L39.833333333333336 2543.4283613043085L44.666666666666664 2536.9992885481056L49.5 2530.710211640946L54.33333333333333 2524.5586787397506L59.166666666666664 2518.542348802132L64 2512.6589847472824L68.83333333333334 2506.906447166582L73.66666666666666 2501.2826885322493L78.5 2495.7857478577553L83.33333333333333 2490.4137457685442L88.16666666666666 2485.1648799458276L93 2480.0374209099946L97.83333333333334 2475.0297081135136L102.66666666666667 2470.14014631618L107.5 2465.367202218205L112.33333333333333 2460.709401329003L117.16666666666667 2456.1653250516424L122 2451.733607964808L126.83333333333333 2447.4129352858013L131.66666666666666 2443.202040499634L136.5 2439.0997031406046L141.33333333333331 2435.1047467139774L146.16666666666669 2431.2160367464726L151 2427.432478955263L155.83333333333334 2423.753017526054L160.66666666666669 2420.17663349163L165.5 2416.7023432029637L170.33333333333334 2413.329196885634L175.16666666666666 2410.056277274898L180 2406.8826983232793L184.83333333333334 2403.8076039750135L189.66666666666666 2400.8301670021424L194.5 2397.9495878974253L199.33333333333334 2395.1650938195908L204.16666666666669 2392.4759375867884L209 2389.881396714368L213.83333333333334 2387.380772493402L218.66666666666666 2384.9733891065794L223.5 2382.65859277833L228.33333333333331 2380.435750956233L233.16666666666666 2378.3042515209186L238 2376.2635020218536L242.83333333333334 2374.3129289365206L247.66666666666666 2372.451976950646L252.5 2370.6801082572206L257.33333333333337 2368.9968018721793L262.16666666666663 2367.4015529646827L267 2365.8938722000134L271.83333333333337 2364.473285093185L276.6666666666667 2363.139331371403L281.5 2361.891564343569L286.33333333333337 2360.729550275067L291.1666666666667 2359.652867766097L296 2358.661107131836L300.8333333333333 2357.7538697827486L305.6666666666667 2356.930767603359L310.5 2356.191422327804L315.3333333333333 2355.535464910495L320.16666666666663 2354.962534890201L325 2354.4722797458426" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2.5;"></path><path d="M35 2550L39.833333333333336 2543.4415299875977L44.666666666666664 2537.0502402204875L49.5 2530.8211410072713L54.33333333333333 2524.7495654090426L59.166666666666664 2518.831144202641L64 2513.061783339351L68.83333333333334 2507.4376436097346L73.66666666666666 2501.9551222638975L78.5 2496.6108363693684L83.33333333333333 2491.4016077167653L88.16666666666666 2486.324449107459L93 2481.3765518780215L97.83333333333334 2476.5552745339915L102.66666666666667 2471.8581323808094L107.5 2467.28278805304L112.33333333333333 2462.8270428545075L117.16666666666667 2458.488828831994L122 2454.2662015139003L126.83333333333333 2450.1573332529047L131.66666666666666 2446.160507118377L136.5 2442.274111290184L141.33333333333331 2438.496633910723L146.16666666666669 2434.8266583566024L151 2431.262858895437L155.83333333333334 2427.8039966968367L160.66666666666669 2424.4489161698484L165.5 2421.1965416019743L170.33333333333334 2418.0458740774275L175.16666666666666 2414.995988654561L180 2412.046031784465L184.83333333333334 2409.195218954544L189.66666666666666 2406.442832542576L194.5 2403.7882198682246L199.33333333333334 2401.230791430371L204.16666666666669 2398.770019319853L209 2396.4054357983428L213.83333333333334 2394.1366320351367L218.66666666666666 2391.9632569945948L223.5 2389.885016467844L228.33333333333331 2387.9016722432043L233.16666666666666 2386.0130414105583L238 2384.2189957956166L242.83333333333334 2382.519461520725L247.66666666666666 2380.914418689521L252.5 2379.4039011933583L257.33333333333337 2377.9879966380668L262.16666666666663 2376.6668463901815L267 2375.4406457423747L271.83333333333337 2374.309644198413L276.6666666666667 2373.274145878539L281.5 2372.3345100467704L286.33333333333337 2371.4911517622136L291.1666666666667 2370.7445426570944L296 2370.0952118448636L300.8333333333333 2369.543746962386L305.6666666666667 2369.090795350928L310.5 2368.7370653813896L315.3333333333333 2368.4833279300187L320.16666666666663 2368.3304180116797L325 2368.2792365786454" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:2.5;stroke-dasharray:7 5"></path><circle cx="35" cy="2550" r="5" style="fill:var(--viz-state-stroke);stroke:var(--viz-state-stroke);stroke-width:1.75"></circle><circle cx="325" cy="2354.4722797458426" r="5" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><path d="M325 2368.2792365786454L325 2354.4722797458426" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><text x="20" y="2632" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Old parameters</tspan></text><text x="340" y="2632" style="font-size:14px;font-weight:700;text-anchor:end;fill:var(--c-text)"><tspan x="340" dy="0">Updated</tspan></text><text x="20" y="2720" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Old log likelihood: -2.144</tspan></text><text x="20" y="2780" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-warning-stroke)"><tspan x="20" dy="0">New fixed bound: -1.639</tspan></text><text x="20" y="2840" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="20" dy="0">New log likelihood: -1.601</tspan></text><text x="20" y="2920" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">Next exact E closes the new gap.</tspan><tspan x="20" dy="18">Nondecrease does not identify</tspan><tspan x="20" dy="18">components or prove a global optimum.</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
	<svg viewBox="0 0 360 420" role="img" aria-labelledby="em-bound-svg-title em-bound-svg-desc">
		<title id="em-bound-svg-title">EM climbs a lower bound on the observed-data log-likelihood</title>
		<desc id="em-bound-svg-desc">A solid curve is the observed-data log-likelihood. A dashed lower-bound curve touches it at the current parameter theta t because the E-step sets q t to the exact posterior and makes the KL gap zero. Holding q t fixed, the M-step moves right to theta t plus 1, where the lower bound is higher. The solid likelihood at the new parameter is at least as high as that bound. The next E-step closes the remaining gap.</desc>
		<rect class="viz-plot-bg" x="4" y="4" width="352" height="218" rx="6"></rect>
		<path class="viz-axis" d="M40 190H338M40 190V24"></path>
		<text class="viz-axis-label" x="338" y="207" text-anchor="end">parameter θ</text>
		<text class="viz-axis-label" x="45" y="20">objective</text>
		<path d="M42 170Q74 148 106 132Q154 112 201 78Q259 37 332 76" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:3;stroke-linecap:round"></path>
		<text class="viz-callout" x="326" y="64" text-anchor="end">log p(x; θ)</text>
		<text class="viz-label" x="326" y="79" text-anchor="end">observed likelihood · solid</text>
		<path d="M42 184Q76 151 106 132Q167 99 226 104Q283 109 332 146" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2.5;stroke-dasharray:7 5;stroke-linecap:round"></path>
		<text class="viz-callout" x="326" y="131" text-anchor="end">F(q<tspan baseline-shift="sub" font-size="8">t</tspan>, θ)</text>
		<text class="viz-label" x="326" y="146" text-anchor="end">fixed lower bound · dashed</text>
		<path d="M106 132V190M226 104V190" style="fill:none;stroke:var(--c-text-soft);stroke-width:1.2;stroke-dasharray:3 3"></path>
		<circle cx="106" cy="132" r="5" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:2.5"></circle>
		<circle cx="226" cy="104" r="5" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2.5"></circle>
		<text class="viz-axis-label" x="106" y="207" text-anchor="middle">θ<tspan baseline-shift="sub" font-size="8">t</tspan></text>
		<text class="viz-axis-label" x="226" y="207" text-anchor="middle">θ<tspan baseline-shift="sub" font-size="8">t+1</tspan></text>
		<path d="M116 119Q164 91 216 99" style="fill:none;stroke:var(--viz-edge);stroke-width:1.8"></path>
		<path d="M210 93L220 100L209 104Z" style="fill:var(--viz-edge)"></path>
		<text class="viz-callout" x="158" y="157" text-anchor="middle">M-step raises bound →</text>
		<rect class="viz-node viz-node--input" x="12" y="240" width="336" height="48" rx="5"></rect>
		<text class="viz-node-label" x="180" y="259">1 · E-step: make the bound tight</text>
		<text class="viz-node-value" x="180" y="277">q<tspan baseline-shift="sub" font-size="8">t</tspan> = p(z|x; θ<tspan baseline-shift="sub" font-size="8">t</tspan>) ⇒ KL gap = 0</text>
		<path d="M180 288V302" style="stroke:var(--viz-edge);stroke-width:2"></path>
		<path d="M174 298L180 306L186 298Z" style="fill:var(--viz-edge)"></path>
		<rect class="viz-node viz-node--focus" x="12" y="306" width="336" height="48" rx="5" style="stroke-dasharray:7 4"></rect>
		<text class="viz-node-label" x="180" y="325">2 · M-step: improve the fixed bound</text>
		<text class="viz-node-value" x="180" y="343">F(q<tspan baseline-shift="sub" font-size="8">t</tspan>, θ<tspan baseline-shift="sub" font-size="8">t+1</tspan>) ≥ F(q<tspan baseline-shift="sub" font-size="8">t</tspan>, θ<tspan baseline-shift="sub" font-size="8">t</tspan>) = log p(x; θ<tspan baseline-shift="sub" font-size="8">t</tspan>)</text>
		<path d="M180 354V368" style="stroke:var(--viz-edge);stroke-width:2"></path>
		<path d="M174 364L180 372L186 364Z" style="fill:var(--viz-edge)"></path>
		<rect class="viz-node viz-node--output" x="12" y="372" width="336" height="40" rx="20"></rect>
		<text class="viz-node-label" x="180" y="389">3 · Likelihood stays above its bound</text>
		<text class="viz-node-value" x="180" y="404">log p(x; θ<tspan baseline-shift="sub" font-size="8">t+1</tspan>) ≥ F(q<tspan baseline-shift="sub" font-size="8">t</tspan>, θ<tspan baseline-shift="sub" font-size="8">t+1</tspan>)</text>
	</svg>
	<div class="visual-derivation-caption">exact E makes the dashed bound touch observed log likelihood at the current parameter. M improves that fixed bound. Likelihood remains above it, proving nondecrease under the stated assumptions. The next exact E-step closes the new gap.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="em-lower-bound-climb"><strong>Check:</strong> Which two inequalities ensure the new likelihood cannot fall after the bound is improved?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="em-lower-bound-climb">E fixes parameters and splits each group membership by its posterior. M freezes those memberships and updates expected group counts, heads and tosses. Tosses, not groups, normalize head chances. Exact E makes the likelihood bound touch; improving that fixed bound cannot lower observed likelihood. Repeat with new posteriors. Approximate updates need separate guarantees; this is neither global optimality nor component identification. <strong>Which state stays fixed during M?</strong></span></figcaption>
</figure>

### Add a Parameter Prior

**New task:** Prefer interior coin probabilities with independent Beta(2,2) priors; leave mixture weights unregularized.

**Picture and change:** Keep posterior responsibilities, but maximize Q plus log prior.

**Check:** Probability updates become $3.3/4.8=0.6875$ and $1.7/3.2=0.53125$.

**Use when:** A MAP estimator is required.
The protected objective is log posterior, not bare observed likelihood.

### Improve Only Part of M

**New task:** Update mixture weights while keeping both coin probabilities at their initial values.

**Picture and change:** This generalized M-step improves Q without jointly maximizing every parameter.

**Check:** Weights $(0.7,0.3)$ give observed likelihood $0.4125\times0.375=0.1546875$, above the initial value.

**Use when:** A verified improving parameter block is available but joint optimization is expensive.

Identical components can remain identical; nondecreasing likelihood does not distinguish them or identify a global solution.
Under appropriate regularity and convergence conditions, limit points can be stationary, including saddles.
Unconstrained Gaussian mixtures can instead diverge through covariance collapse.

<!-- learning:core-end -->
<div data-learning-extension>

## Canonical example: GMM

[Gaussian mixtures](/concepts/gaussian-mixture-models/) apply this owner to posterior component responsibilities and weighted Gaussian sufficient statistics.
Detailed covariance updates belong to that application, not a second generic EM derivation.

## Variants

- **Hard assignment:** k-means relates to restricted mixtures through hard-assignment or small-variance limits, not ordinary finite-variance exact EM.
- **Stochastic approximations:** sampling noise need not preserve likelihood nondecrease at every iteration.
- **Variational E:** improving a restricted lower bound does not automatically increase observed likelihood; the touching-bound step may be missing.
- **MAP-EM:** an explicit prior changes the optimized objective.

## Limitations

Initialization, latent-model specification, missing information, and numerical optimization affect convergence and usefulness.
Comparing restarts does not turn a nonconvex model into a globally solved problem.

## Common pitfalls

Ad hoc covariance stabilization and approximate E/M updates require their own objective/monotonicity analysis.
Alternating two convenient steps is not enough to establish the exact EM proof.

## Related

- [Gaussian mixture models](/concepts/gaussian-mixture-models/). The canonical EM application.
- [Hidden Markov models](/concepts/hidden-markov-models/). Sequential model trained by EM (Baum-Welch).

</div>
