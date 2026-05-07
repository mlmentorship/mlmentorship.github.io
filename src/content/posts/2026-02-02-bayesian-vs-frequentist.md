---
title: "Bayesian vs frequentist: a practitioner's framing"
description: "The textbook distinction is philosophical. The practitioner distinction is whether you can sample from a posterior cheaply, and whether you need uncertainty for downstream decisions."
date: "2026-02-02"
draft: false
tags: ["interviews"]
category: "interviews"
---


> *Asked in: ML breadth, especially in research-heavy or stats-heavy roles.*

The interviewer is checking whether you can use the Bayesian framing as a *tool*, not just describe it as a viewpoint. The L6 answer maps the philosophy onto specific practical decisions.

## What an L4 answer sounds like

> "Frequentists treat probability as long-run frequency; Bayesians treat probability as belief. Bayesians use priors and update them with data; frequentists don't."

True but useless. The interviewer learns the textbook line, not whether you can apply it.

## What an L5 answer sounds like

> "The two frameworks differ in what 'probability' means and what they estimate.
>
> - **Frequentist** estimates parameters as fixed unknowns. Confidence intervals describe the procedure (95% of CIs of this form contain the true parameter), not the parameter. Common tools: MLE, hypothesis tests, bootstrap.
> - **Bayesian** treats parameters as random variables with a prior; data updates the prior to a posterior. Credible intervals describe the parameter's distribution conditional on the data. Common tools: MCMC, variational inference, conjugate priors.
>
> Practical implications:
> - Bayesian methods give you a full distribution over predictions, useful when downstream decisions need uncertainty (active learning, Bayesian optimization, exploration in RL).
> - Frequentist methods are simpler and computationally cheaper for point estimates with confidence intervals.
> - With enough data, the prior washes out and the two converge."

This is L5. You've explained both, given examples, and noted when each is the right tool.

## What an L6 answer sounds like

> "...a few practical things that change the picture:
>
> **In deep learning, almost everything is implicitly Bayesian.** Dropout is approximate variational inference (Yarin Gal). Weight decay is a Gaussian prior. Mini-batch SGD has noise that approximates a posterior sample. Most 'frequentist' deep models are doing approximate Bayesian inference, just not labeled that way.
>
> **The expensive part of Bayes is sampling from the posterior.** MCMC is slow; variational inference is fast but biased. Modern alternatives: Monte Carlo dropout, deep ensembles (which give similar uncertainty without the Bayesian framing), Laplace approximation around a MAP estimate. For LLMs, no real Bayesian treatment exists; we use sampling at inference (varying temperature) as a poor man's posterior.
>
> **Priors are useful when data is small.** With abundant data, the prior is irrelevant. With sparse data, a thoughtful prior (regularization is one) is the difference between a model that works and one that doesn't.
>
> **The interviewer is usually checking whether you can use Bayes, not whether you prefer it.** A practical answer: 'I'd reach for Bayesian methods when uncertainty is the product (active learning, BO, A/B testing with sequential analysis), and reach for frequentist methods when point estimates with CIs are sufficient and compute matters.'"

## Tells that get you a strong-hire vote

- You connect Bayesian inference to **specific deep learning techniques** (dropout, weight decay, ensembles).
- You distinguish **credible intervals** from **confidence intervals** correctly.
- You discuss the **computational cost** of full Bayesian inference and the practical alternatives.
- You give a **decision rule** for when to use which.

## Tells that get you down-leveled

- "Bayesian uses priors, frequentist doesn't" with no application.
- Misstating what a confidence interval means (the most common stats error).
- Treating the two as opposing teams instead of complementary tools.
- No awareness that most deep learning is implicitly Bayesian.

## Common follow-up

"What's a confidence interval, exactly?"

The L6 answer:

> "A 95% confidence interval is a *procedure* such that 95% of intervals constructed by that procedure (over repeated experiments) would contain the true parameter. It is *not* 'a 95% probability that the parameter is in this interval' (that's a credible interval, the Bayesian object). The frequentist parameter is fixed; the interval is random. The Bayesian parameter is random; the interval is fixed. Confusing these is the most common stats mistake in ML interviews."

---

*Related: [Calibration](/reference/calibration/), [A/B testing for ML systems](/reference/ab-testing-for-ml/).*
