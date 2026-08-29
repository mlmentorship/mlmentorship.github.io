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

> "Sigmoid is the canonical link function for the Bernoulli distribution. Specifically, it's the inverse of the natural parameter of the Bernoulli (the log-odds). This choice gives the gradient simplification we just showed, gives a convex loss in the parameters, and corresponds to maximum-entropy modeling subject to the constraint of matching feature expectations under the data. Other functions (e.g., probit, which uses the Gaussian CDF) work but don't have the same algebraic and computational properties. Probit is occasionally preferred in econometrics for theoretical reasons; logit dominates in ML for the practical reasons above."

---

*Related: [cross-entropy and softmax](/concepts/cross-entropy-softmax/), [regularization](/concepts/regularization/), and [Bayesian versus frequentist](/questions/bayesian-vs-frequentist/).*
