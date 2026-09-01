---
title: "Naive Bayes"
description: "A trivially simple generative classifier that assumes features are conditionally independent given the class. Fast, parameter-light, surprisingly hard to beat on text."
date: "2026-04-04"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

Naive Bayes models $p(y \mid x) \propto p(y) \prod_j p(x_j \mid y)$, assuming features $x_j$ are **conditionally independent given the class $y$**. Trained by counting (closed-form MLE for each conditional).

Naive Bayes is the cheapest possible probabilistic classifier. Closed-form MLE, $O(nd)$ training, $O(d)$ prediction, no hyperparameter tuning. Despite the obviously wrong independence assumption, it works remarkably well as a baseline on:

- **Text classification** (spam filtering, topic categorization, sentiment): bag-of-words features.
- **Tiny-data classification** where logistic regression overfits.
- **Initial baselines** that should be beaten before claiming victory with a fancier model.

It is also conceptually important. The canonical example of a **generative classifier** (model $p(x, y)$) versus the **discriminative** logistic regression (model $p(y \mid x)$ directly).

## The model

By Bayes' rule:

$$
p(y \mid x) = \frac{p(y) p(x \mid y)}{p(x)}.
$$

The "naive" assumption: $p(x \mid y) = \prod_j p(x_j \mid y)$. Then for prediction:

$$
\hat y = \arg\max_y p(y) \prod_j p(x_j \mid y) = \arg\max_y \log p(y) + \sum_j \log p(x_j \mid y).
$$

Sum logs to avoid underflow.

## Variants

| Variant | $p(x_j \mid y)$ | Use case |
|---------|----------------|----------|
| **Multinomial** | Multinomial over token counts | Text (bag-of-words) |
| **Bernoulli** | Bernoulli (binary) per feature | Text (presence/absence) |
| **Gaussian** | $\mathcal{N}(\mu_{j,y}, \sigma_{j,y}^2)$ | Continuous features |
| **Categorical** | Categorical (multinoulli) | Discrete features |

## Training

Just count.

For multinomial NB on text:

- $p(y = c) = \text{count}(c) / N$.
- $p(\text{word } w \mid y = c) = \frac{\text{count}(w \text{ in class } c) + \alpha}{\text{total tokens in class } c + \alpha V}$ (with Laplace / additive smoothing $\alpha$, typically 1.0; $V$ = vocabulary size).

Without smoothing, any unseen word in a class gives $p(x \mid y) = 0$ and $p(y \mid x) = 0$, breaking inference.

## Why the independence assumption isn't fatal

Even though words are clearly correlated, naive Bayes can still rank classes correctly. The independence assumption gives biased probability estimates (overconfident. Predicted probabilities tend to be near 0 or 1) but the **argmax is often right**.

For pure classification accuracy, NB is competitive. For calibrated probabilities, prefer logistic regression with proper regularization.

<!-- visual:naive-bayes-evidence-ledger -->
<figure class="learning-figure" aria-labelledby="naive-bayes-ledger-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="naive-bayes-ledger-title">How can naive Bayes choose the right class but be too confident?</p>
	<div class="visual-grid--two" role="group" aria-label="An illustrative Naive Bayes evidence ledger beside the conditional-independence caveat">
		<section class="visual-panel" aria-labelledby="nb-score-title">
			<h4 id="nb-score-title">Model score: count every token separately</h4>
			<p>For two equally likely classes, suppose a text model learned these illustrative likelihood ratios for a document containing “free prize.”</p>
			<table class="cm-grid" aria-label="Illustrative spam to ham odds calculation">
				<tbody>
					<tr><th scope="row">Prior odds</th><td>1 : 1</td></tr>
					<tr><th scope="row">“free”</th><td>× 8</td></tr>
					<tr><th scope="row">“prize”</th><td>× 7</td></tr>
					<tr><th scope="row">NB odds</th><td class="cm-selected"><strong>56 : 1</strong> spam</td></tr>
				</tbody>
			</table>
			<p class="cm-equation">1 × 8 × 7 = 56 → choose spam</p>
		</section>
		<section class="visual-panel" aria-labelledby="nb-caveat-title">
			<h4 id="nb-caveat-title">Reality check: evidence can overlap</h4>
			<p>“Free” and “prize” may come from the same promotional phrase, so their occurrences can remain correlated even after the class is known.</p>
			<table class="cm-grid" aria-label="Difference between the Naive Bayes assumption and correlated text">
				<tbody>
					<tr><th scope="row">NB assumes</th><td>two conditionally independent contributions</td></tr>
					<tr><th scope="row">Text may contain</th><td>one underlying signal expressed twice</td></tr>
					<tr><th scope="row">Class decision</th><td class="cm-selected"><strong>spam can still win</strong></td></tr>
					<tr><th scope="row">Probability</th><td>56 : 1 can overstate certainty</td></tr>
				</tbody>
			</table>
			<p class="cm-equation">use the argmax; distrust uncalibrated probability</p>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> multiply down the left ledger first: naive Bayes gives each observed token its own likelihood-ratio contribution, so the class score can separate sharply. Then read the right panel: if correlated tokens repeat one underlying signal, multiplying them as independent evidence exaggerates the odds. The winning class can remain correct even when the reported probability is overconfident. The numbers are an original illustrative calculation, checked against the <a href="https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html"><cite>Introduction to Information Retrieval</cite> treatment of multinomial naive Bayes</a> and <a href="https://scikit-learn.org/stable/modules/naive_bayes.html">scikit-learn's official model guidance</a>.</figcaption>
</figure>

## Generative vs. discriminative

Naive Bayes models the joint $p(x, y)$. Logistic regression models $p(y \mid x)$ directly. Asymptotic results [(Ng & Jordan, 2002)](https://papers.nips.cc/paper/2001/hash/7b7a53e239400a13bd6be6c91c4f6c4e-Abstract.html):

- For **small $n$**, NB usually wins (less variance from the strong assumption).
- For **large $n$**, logistic catches up and surpasses NB (the assumption hurts at large scale).

## Where it shows up in 2026

- **Spam filters** in low-resource embedded systems.
- **Quick text baselines** before training a transformer.
- **Document filtering** in retrieval pipelines (cheap pre-filter).

For most modern NLP, neural classifiers dominate. NB persists in resource-constrained settings and as a reliable benchmark.

## Common pitfalls

- **Forgetting to smooth.** Without Laplace smoothing, any test document with a vocabulary token never seen in a class gets that class's posterior set to 0.
- **Using NB on highly correlated features.** Probabilities become very poorly calibrated; predicted class can still be okay but never trust the probability.
- **Mixing variants.** Multinomial NB on continuous features is wrong; use Gaussian NB (or discretize first).
- **Comparing NB on raw counts vs. tfidf vs. binarized.** Different preprocessing changes the model class; compare apples to apples.
