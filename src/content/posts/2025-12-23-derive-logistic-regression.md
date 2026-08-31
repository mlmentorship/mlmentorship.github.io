---
title: "Derive logistic regression from MLE"
description: "Standard math-screen question. The senior signal is whether you can derive it cleanly and connect MLE to cross-entropy."
date: "2025-12-23"
draft: false
tags: ["questions"]
category: "questions"
---


> *Asked in: ML breadth and math-screen rounds.*

The L4 candidate states the logistic regression formula. The L6 candidate derives it from first principles and connects it to cross-entropy.

## The setup

Binary classification: `y` in {0, 1}. We model `P(y=1 | x) = sigma(w^T x)` where `sigma(z) = 1 / (1 + exp(-z))` is the sigmoid.

## The derivation

For a single example `(x, y)`, the likelihood under our model:

```
P(y | x; w) = sigma(w^T x)^y * (1 - sigma(w^T x))^(1-y)
```

Equivalent to a Bernoulli likelihood with parameter `sigma(w^T x)`.

For N i.i.d. examples, the joint likelihood is the product:

```
L(w) = prod_i P(y_i | x_i; w)
```

The log-likelihood:

```
log L(w) = sum_i [ y_i * log sigma(w^T x_i) + (1 - y_i) * log (1 - sigma(w^T x_i)) ]
```

MLE picks `w` to maximize this, equivalently minimizes the negative log-likelihood:

```
NLL(w) = -sum_i [ y_i * log sigma(w^T x_i) + (1 - y_i) * log (1 - sigma(w^T x_i)) ]
```

This is exactly **binary cross-entropy** between `y_i` (the true label) and `sigma(w^T x_i)` (the predicted probability). Logistic regression's standard loss is the MLE under a Bernoulli noise model.

## The gradient

```
d NLL / d w = sum_i (sigma(w^T x_i) - y_i) * x_i
```

Notice: the gradient is `(predicted - true) * input`. Same form as linear regression's gradient under MSE, except the prediction is now passed through a sigmoid. This is *not* a coincidence; both are MLE under different exponential-family noise models (Bernoulli for logistic, Gaussian for linear).

<!-- visual:logistic-gradient-cancellation -->
<figure class="learning-figure plot-panel" aria-labelledby="logistic-gradient-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="logistic-gradient-title">Trace why sigmoid plus Bernoulli NLL reduces to the residual <code>p - y</code>.</p>
	<svg viewBox="0 0 360 430" role="img" aria-labelledby="logistic-gradient-svg-title logistic-gradient-svg-desc">
		<title id="logistic-gradient-svg-title">Chain-rule derivation of the logistic-regression gradient</title>
		<desc id="logistic-gradient-svg-desc">Starting with logit z equal to w transpose x and probability p equal to sigmoid z, the Bernoulli negative log-likelihood is differentiated by the chain rule. Its probability derivative, negative y over p plus one minus y over one minus p, multiplies the sigmoid derivative p times one minus p. The denominators cancel to negative y times one minus p plus one minus y times p, which collects to p minus y. Multiplying by dz over dw equal to x gives dL over dw equal to p minus y times x. For label one the negative residual raises the logit under gradient descent; for label zero the positive residual lowers it.</desc>
		<defs><marker id="logistic-gradient-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path class="viz-arrow-forward" d="M0,0 L7,3.5 L0,7 Z"></path></marker></defs>
		<text class="viz-axis-label" x="18" y="22">ONE EXAMPLE: z = w^T x, p = sigmoid(z)</text>
		<rect class="viz-node viz-node--input" x="18" y="38" width="324" height="52" rx="4"></rect>
		<text class="viz-label" x="30" y="57">BERNOULLI NEGATIVE LOG-LIKELIHOOD</text>
		<text class="viz-callout" x="180" y="78" text-anchor="middle">L = -[y log(p) + (1-y) log(1-p)]</text>
		<path d="M180 94V112" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#logistic-gradient-arrow)"></path>
		<text class="viz-axis-label" x="18" y="128">1 - APPLY THE CHAIN RULE</text>
		<rect class="viz-node" x="18" y="140" width="201" height="58" rx="4"></rect>
		<text class="viz-label" x="30" y="159">LOSS CONTRIBUTION</text>
		<text class="viz-callout" x="118.5" y="184" text-anchor="middle">dL/dp = -y/p + (1-y)/(1-p)</text>
		<text class="viz-callout" x="228" y="174">x</text>
		<rect class="viz-node viz-node--focus" x="244" y="140" width="98" height="58" rx="4"></rect>
		<text class="viz-label" x="293" y="159" text-anchor="middle">SIGMOID</text>
		<text class="viz-callout" x="293" y="184" text-anchor="middle">dp/dz = p(1-p)</text>
		<path d="M180 202V220" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#logistic-gradient-arrow)"></path>
		<text class="viz-axis-label" x="18" y="236">2 - MULTIPLY; THE RECIPROCALS CANCEL</text>
		<rect class="viz-node viz-node--focus" x="18" y="248" width="324" height="87" rx="4"></rect>
		<text class="viz-callout" x="180" y="271" text-anchor="middle">dL/dz = [-y/p + (1-y)/(1-p)] p(1-p)</text>
		<path class="viz-operating-guide" d="M74 279H286"></path>
		<text class="viz-callout" x="180" y="303" text-anchor="middle">= -y(1-p) + (1-y)p</text>
		<text class="viz-callout" x="180" y="325" text-anchor="middle">= -y + yp + p - yp = p - y</text>
		<path d="M180 339V357" style="fill:none;stroke:var(--viz-edge);stroke-width:2;marker-end:url(#logistic-gradient-arrow)"></path>
		<rect class="viz-node viz-node--output" x="18" y="366" width="324" height="45" rx="4"></rect>
		<text class="viz-label" x="30" y="384">SINCE dz/dw = x</text>
		<text class="viz-callout" x="330" y="395" text-anchor="end">dL/dw = (p - y)x</text>
		<text class="viz-label" x="18" y="427">y=1: p-1 &lt; 0, raise z</text>
		<text class="viz-label" x="342" y="427" text-anchor="end">y=0: p &gt; 0, lower z</text>
	</svg>
	<figcaption><strong>Read it this way:</strong> follow the two chain-rule factors. Differentiating the log loss introduces <code>1/p</code> and <code>1/(1-p)</code>; differentiating the sigmoid supplies <code>p(1-p)</code>. Those factors cancel, leaving the probability residual <code>p-y</code>. The remaining derivative of <code>w^T x</code> contributes <code>x</code>, so each example updates the weights by <code>(p-y)x</code>. Original derivation checked against the <a href="https://cs229.stanford.edu/notes2022fall/main_notes.pdf?forcedefault=true">Stanford CS229 notes</a>.</figcaption>
</figure>

## The L6 connections

> "...two things worth noting:
>
> **The sigmoid + binary cross-entropy gradient simplifies to (p - y).** The same simplification holds for softmax + categorical cross-entropy. This isn't algebraic coincidence; it's a property of generalized linear models under the canonical link function. The numerical stability and ease of implementation come from this simplification.
>
> **MLE assumes the model is correct.** If the true relationship is not log-linear in the features, MLE gives the best parameters under the wrong model. Diagnostic: if predictions don't fit the data well, the issue is model misspecification, not optimization.
>
> **Regularization fits naturally**: L2 regularization is MAP estimation with a Gaussian prior on `w`; L1 is MAP with a Laplace prior. The Bayesian framing makes regularization a derivation rather than an ad-hoc add-on."

## Tells that get you a strong-hire vote

- You **derive cleanly** without skipping steps.
- You **identify NLL with binary cross-entropy** explicitly.
- You **simplify the gradient** to `(p - y) * x`.
- You connect to **GLMs** and the **canonical link function**.
- You discuss **regularization as MAP** with a prior.

## Tells that get you down-leveled

- Stating the formula without derivation.
- Confusing logistic regression with linear regression.
- Not knowing the gradient form.
- Treating cross-entropy as separate from MLE.

## Common follow-up

"Why use sigmoid and not another function that maps to (0, 1)?"

The L6 answer:

> "The logit is the canonical link for the Bernoulli distribution, and sigmoid is its inverse: it maps the natural parameter (the log-odds) to a probability. This choice gives the gradient simplification we just showed, gives a convex loss in the parameters, and corresponds to maximum-entropy modeling subject to the constraint of matching feature expectations under the data. Other inverse links (e.g., probit, which uses the Gaussian CDF) work but don't have the same algebraic and computational properties. Probit is occasionally preferred in econometrics for theoretical reasons; logit dominates in ML for the practical reasons above."

---

*Related: [cross-entropy and softmax](/concepts/cross-entropy-softmax/), [regularization](/concepts/regularization/), and [Bayesian versus frequentist](/questions/bayesian-vs-frequentist/).*
