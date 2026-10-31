---
title: "Random forests"
description: "Bag deep decision trees plus random feature subsets per split. Variance averaging beats any single tree; the dominant out-of-the-box ensemble before GBDT."
date: "2026-01-27"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A **random forest** [(Breiman, 2001)](https://link.springer.com/article/10.1023/A:1010933404324) trains an ensemble of decision trees on bootstrap samples of the data ("bagging") with each split restricted to a random subset of features, then averages predictions (regression) or takes a majority vote (classification).

Single decision trees are high-variance: small data shifts cause large changes in splits. Random forests average over many trees, shrinking the uncorrelated part of their variance roughly as $1/\text{ensemble size}$. They are:

- A reliable **out-of-the-box baseline** for tabular data. Minimal tuning, strong performance.
- **Hard to overfit** with enough trees and reasonable depth limits.
- **Parallel-friendly**: trees train independently.

GBDT (xgboost, lightgbm, catboost) usually beats RF in modern tabular benchmarks, but RF remains useful as a fast baseline and for problems where you want low variance with minimal tuning.

## The two randomizations

1. **Bagging** (Bootstrap Aggregating): each tree trains on a bootstrap sample (sample with replacement, same size as original). Each tree sees ~63% of the unique training rows.
2. **Random feature subsets per split**: at each split, only $m$ randomly chosen features are considered as candidates. Classic heuristics are $m = \sqrt{p}$ for classification and $m = p/3$ for regression.

Both reduce correlation between trees. Bagging alone gives "tree bagging"; the second randomization is what makes it a *random forest*.

## Out-of-bag (OOB) estimation

Each tree's bootstrap sample omits ~37% of the data. Predict each row using only trees that didn't see it (its **out-of-bag** trees) and average. This gives a free held-out estimate of generalization error. No separate validation set needed.

OOB estimates are typically very close to k-fold CV estimates. Useful for hyperparameter search without a separate split.

## Variance reduction analysis

For an ensemble of $T$ trees with pairwise correlation $\rho$ and per-tree variance $\sigma^2$:

$$
\mathrm{Var}(\bar f) = \rho \sigma^2 + \frac{1 - \rho}{T} \sigma^2.
$$

**Learning objective:** separate ensemble variance into the correlated floor that remains as trees are added and the uncorrelated share that averaging can remove.

<!-- visual:random-forest-correlation-floor -->
<figure class="learning-figure" aria-labelledby="random-forest-variance-title">
	<p class="visual-kicker">Why decorrelation matters</p>
	<p class="visual-title" id="random-forest-variance-title">In this model, correlated tree variation leaves a nonzero variance floor</p><div class="visual-argument" data-visual-argument="random-forest-correlation-floor" data-argument-kind="curve" data-argument-composition="landmark"><svg viewBox="0 0 360 439" role="img" aria-label="Correlated trees leave a variance floor. With unit tree variance and pairwise correlation 0.2, ensemble variance is 0.2+0.8/T. More trees cannot push this model below 0.2."><title>Correlated trees leave a variance floor</title><desc>With unit tree variance and pairwise correlation 0.2, ensemble variance is 0.2+0.8/T. More trees cannot push this model below 0.2.</desc><g opacity=".35"><path d="M48 177.5L330 177.5" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path></g><g opacity=".35"><path d="M48 135L330 135" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path></g><g opacity=".35"><path d="M48 92.5L330 92.5" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path></g><path d="M48 40L48 220" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M48 220L334 220" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><text x="48" y="245" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="48" dy="0">1</tspan></text><text x="330" y="245" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="330" dy="0">100</tspan></text><text x="40" y="54" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="40" dy="0">1</tspan></text><text x="40" y="220" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text-soft)"><tspan x="40" dy="0">0</tspan></text><path data-proof-landmark="100" d="M330 40V220" style="stroke:var(--viz-focus-stroke);stroke-width:1.5;stroke-dasharray:2 4"></path><path data-proof-series="Correlation 0.2" d="M48 50L52.40625 132.60122699386503L56.8125 152.77862595419845L61.21875 161.88919667590028L65.625 167.0782608695652L70.03125 170.42933810375672L74.4375 172.77203647416414L78.84375 174.5019815059445L83.25 175.83177570093457L87.65625 176.88586387434555L92.0625 177.74193548387098L96.46875 178.45099739809194L100.875 179.04792332268372L105.28125 179.55736491487787L109.6875 179.99724137931034L114.09375 180.38089089735314L118.5 180.71844660194176L122.90625 181.01774470520894L127.3125 181.28494041170097L131.71875 181.5249357326478L136.125 181.7416829745597L140.53125 181.9384041063929L144.9375 182.1177520071365L149.34375 182.28193079880393L153.75 182.4327868852459L158.15625 182.57187869239857L162.5625 182.70053070507961L166.96875 182.8198757763975L171.375 182.9308885754584L175.78125 183.03441226575808L180.1875 183.13117996044826L184.59375 183.22183210979892L189 183.30693069306932L193.40625 183.38697087961572L197.8125 183.46239067055393L202.21875 183.53357891754038L206.625 183.60088202866592L211.03125 183.6646096055809L215.4375 183.72503920543647L219.84375 183.7824203821656L224.25 183.83697813121273L228.65625 183.88891583798204L233.0625 183.93841781146375L237.46875 183.98565146956724L241.875 184.03076923076924L246.28125 184.0739101571144L250.6875 184.11520138588133L255.09375 184.15475938096247L259.5 184.19269102990035L263.90625 184.22909460834182L268.3125 184.26406063023535L272.71875 184.2976725992568L277.125 184.33000767459708L281.53125 184.3611372622858L285.9375 184.39112754158964L290.34375 184.42003993465238L294.75 184.44793152639087L299.15625 184.47485544068687L303.5625 184.5008611780916L307.96875 184.5259949195597L312.375 184.55029980013325L316.78125 184.57381615598885L321.1875 184.59658174782328L325.59375 184.61863196318046L330 184.64" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:3"></path><g data-proof-observation="0" data-proof-x="100" data-proof-y="0.20800000000000002"><circle cx="330" cy="184.64" r="5" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle></g><path data-proof-series="Independent trees" d="M48 50L52.40625 153.25153374233128L56.8125 178.4732824427481L61.21875 189.86149584487535L65.625 196.34782608695653L70.03125 200.5366726296959L74.4375 203.46504559270517L78.84375 205.62747688243064L83.25 207.28971962616822L87.65625 208.60732984293193L92.0625 209.67741935483872L96.46875 210.5637467476149L100.875 211.30990415335464L105.28125 211.94670614359734L109.6875 212.49655172413793L114.09375 212.9761136216914L118.5 213.39805825242718L122.90625 213.77218088151116L127.3125 214.10617551462622L131.71875 214.40616966580976L136.125 214.67710371819962L140.53125 214.92300513299114L144.9375 215.1471900089206L149.34375 215.35241349850492L153.75 215.54098360655738L158.15625 215.71484836549823L162.5625 215.8756633813495L166.96875 216.0248447204969L171.375 216.163610719323L175.78125 216.29301533219763L180.1875 216.41397495056032L184.59375 216.52729013724866L189 216.63366336633663L193.40625 216.73371359951966L197.8125 216.82798833819243L202.21875 216.91697364692547L206.625 217.00110253583242L211.03125 217.08076200697613L215.4375 217.15629900679562L219.84375 217.228025477707L224.25 217.29622266401591L228.65625 217.36114479747758L233.0625 217.4230222643297L237.46875 217.48206433695904L241.875 217.53846153846155L246.28125 217.59238769639302L250.6875 217.64400173235165L255.09375 217.6934492262031L259.5 217.74086378737542L263.90625 217.78636826042725L268.3125 217.8300757877942L272.71875 217.872090749071L277.125 217.91250959324634L281.53125 217.95142157785727L285.9375 217.98890942698705L290.34375 218.0250499183155L294.75 218.0599144079886L299.15625 218.0935693008586L303.5625 218.12607647261453L307.96875 218.1574936494496L312.375 218.18787475016654L316.78125 218.21727019498607L321.1875 218.2457271847791L325.59375 218.27328995397556L330 218.3" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:3;stroke-dasharray:7 4"></path><g data-proof-observation="1" data-proof-x="100" data-proof-y="0.01"><circle cx="330" cy="218.3" r="5" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><text x="190" y="273" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text-soft)"><tspan x="190" dy="0">number of trees T</tspan></text><text x="20" y="20" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">ensemble variance</tspan></text><path d="M20 306L48 306" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:1.75"></path><text x="60" y="310" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="60" dy="0">Correlation 0.2</tspan></text><text x="60" y="333" style="font-size:18px;font-weight:700;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="60" dy="0">0.208 at 100</tspan></text><path d="M20 368L48 368" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><text x="60" y="372" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="60" dy="0">Independent trees</tspan></text><text x="60" y="395" style="font-size:18px;font-weight:700;text-anchor:start;fill:var(--viz-output-stroke)"><tspan x="60" dy="0">0.01 at 100</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
	<div class="visual-panel plot-panel">
		<svg viewBox="0 0 360 330" role="img" aria-labelledby="random-forest-variance-svg-title random-forest-variance-svg-desc">
			<title id="random-forest-variance-svg-title">Random forest variance at one, four, and infinitely many trees</title>
			<desc id="random-forest-variance-svg-desc">A worked example sets pairwise tree correlation rho to one quarter and each tree's variance to one. Three stacked bars show total ensemble variance. At one tree, correlated variance is one quarter and uncorrelated variance is three quarters, totaling one. At four trees, the correlated floor remains one quarter while the uncorrelated share shrinks to three sixteenths, totaling seven sixteenths or about point four four. At infinitely many trees, the uncorrelated share vanishes but the one-quarter correlation floor remains. A note says bootstrap rows and random feature candidates lower rho and therefore lower the floor.</desc>
			<defs>
				<pattern id="random-forest-shared-hatch" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
					<rect width="7" height="7" style="fill:var(--viz-state-bg)"></rect>
					<path d="M0 0V7" style="stroke:var(--viz-state-stroke);stroke-width:2"></path>
				</pattern>
			</defs>
			<text class="viz-axis-label" x="15" y="24">VARIANCE, NORMALIZED BY &#963;&#178;</text>
			<path class="viz-gridline" d="M42 50H330M42 92.5H330M42 135H330M42 177.5H330"></path>
			<path class="viz-axis" d="M42 42V220H334"></path>
			<text class="viz-label" x="34" y="54" text-anchor="end">1.00</text>
			<text class="viz-label" x="34" y="181" text-anchor="end">0.25</text>
			<rect x="66" y="50" width="54" height="127.5" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></rect>
			<rect x="66" y="177.5" width="54" height="42.5" rx="3" style="fill:url(#random-forest-shared-hatch);stroke:var(--viz-state-stroke);stroke-width:2"></rect>
			<rect x="157" y="145.6" width="54" height="31.9" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:2"></rect>
			<rect x="157" y="177.5" width="54" height="42.5" rx="3" style="fill:url(#random-forest-shared-hatch);stroke:var(--viz-state-stroke);stroke-width:2"></rect>
			<rect x="248" y="177.5" width="54" height="42.5" rx="3" style="fill:url(#random-forest-shared-hatch);stroke:var(--viz-state-stroke);stroke-width:2"></rect>
			<text class="viz-callout" x="93" y="40" text-anchor="middle">1.00</text>
			<text class="viz-callout" x="184" y="136" text-anchor="middle">0.44</text>
			<text class="viz-callout" x="275" y="168" text-anchor="middle">0.25</text>
			<text class="viz-axis-label" x="93" y="242" text-anchor="middle">T = 1</text>
			<text class="viz-axis-label" x="184" y="242" text-anchor="middle">T = 4</text>
			<text class="viz-axis-label" x="275" y="242" text-anchor="middle">T &#8594; &#8734;</text>
			<path d="M118 108H306" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:2"></path>
			<text class="viz-label" x="310" y="104" text-anchor="end">uncorrelated share</text>
			<text class="viz-label" x="310" y="119" text-anchor="end">(1 &#8722; &#961;) / T shrinks</text>
			<path d="M118 201H306" style="fill:none;stroke:var(--viz-state-stroke);stroke-width:2;stroke-dasharray:5 4"></path>
			<text class="viz-label" x="310" y="198" text-anchor="end" style="paint-order:stroke;stroke:var(--viz-canvas);stroke-width:4;stroke-linejoin:round">fixed floor &#961; = 0.25</text>
			<rect class="viz-node" x="28" y="264" width="304" height="48" rx="4"></rect>
			<text class="viz-callout" x="180" y="284" text-anchor="middle">bootstrap rows + random feature candidates</text>
			<text class="viz-label" x="180" y="302" text-anchor="middle">&#8594; less-aligned tree errors &#8594; lower &#961; and a lower floor</text>
		</svg>
	</div>
	<div class="visual-derivation-caption"><strong>Read it this way:</strong> with &#961; = 0.25 and &#963;&#178; = 1, one tree has variance 0.25 + 0.75 = 1. Four trees keep the hatched 0.25 shared-error floor but shrink the remaining variance to 0.75 / 4, for 0.4375 total. More trees can erase the uncorrelated share, never the floor; bootstrap sampling and random feature candidates matter because they lower &#961; itself. Original schematic checked against <a href="https://doi.org/10.1023/A:1010933404324">Breiman (2001)</a> and <a href="https://hastie.su.domains/ElemStatLearn/">The Elements of Statistical Learning</a>.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="random-forest-correlation-floor"><strong>Check:</strong> Why does adding infinitely many trees not remove the 0.25 variance share in this model?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="random-forest-correlation-floor">With unit tree variance and pairwise correlation 0.2, ensemble variance is 0.2+0.8/T. More trees cannot push this model below 0.2.</span></figcaption>
</figure>

As $T \to \infty$, variance approaches $\rho \sigma^2$. The two randomizations work by *reducing $\rho$*; without them the trees are too similar and the correlated first term dominates.

## Bias-variance picture

- Each tree is grown deep (low bias, high variance).
- Averaging cuts variance.
- Bias of the ensemble = bias of a single tree (averaging unbiased-ish quantities).
- Bigger forests almost never overfit; you might get diminishing returns past 200–500 trees but rarely get worse.

## Hyperparameters that matter

| Parameter | Default | Notes |
|-----------|---------|-------|
| n_estimators | 100–500 | More is usually better; saturates around 200–500. |
| max_features | $\sqrt{p}$ (clf), $p/3$ (reg) | Smaller = more decorrelation, more bias. |
| max_depth / min_samples_leaf | None / 1 | Often left unbounded; constrain on huge data. |
| bootstrap | True | False = random subspace method, lighter on memory. |

## When to use vs. alternatives

- **Tabular baseline**: RF is the fastest "good enough" model.
- **GBDT** (xgboost, lightgbm): usually 1–3% better but needs more tuning.
- **Logistic regression**: better when the truth is approximately linear or interpretability matters.
- **Neural nets**: rarely beat tree-based models on small/medium tabular data.

## Common pitfalls

- **Comparing RF against a single tree.** Trivial; that's not the comparison.
- **Treating feature importances as causal.** They are correlational; correlated features split the importance among themselves.
- **Tuning on training accuracy.** Use OOB or CV; trees can fit training perfectly.
- **Skipping permutation importance.** Built-in importance favors high-cardinality features; permutation importance is more honest.

## Related

- [Decision trees](/concepts/decision-trees/). The base learner.
- [Gradient boosting](/concepts/gradient-boosting/). Sequential boosting alternative.
