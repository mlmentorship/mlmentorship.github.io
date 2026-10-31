---
title: "Matrix factorization for recsys"
description: "Predict missing user-item ratings from shared factors, while separating observed targets, implicit-feedback assumptions, and cold-item support."
date: "2026-02-16"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A missing rating supplies two indices: a user row and an item row.
Matrix factorization shares learned rows across many observed ratings rather than storing an independent prediction per cell.
Its low-rank assumption is a modeling choice, not a guarantee of matrix completion.

Let $R\in\mathbb R^{m\times n}$, $U\in\mathbb R^{m\times k}$, and $V\in\mathbb R^{n\times k}$.
The shared coordinate count is $k$, and $R\approx UV^\top$.
With zero biases, fixture rows $u_A=(2,1)$ and $v_3=(1.5,1)$ predict $2(1.5)+1(1)=4$.
These are chosen vectors, not a fitted solution for all displayed ratings.

<!-- visual:mf-missing-rating-dot-product -->
<figure class="learning-figure editorial-illustration" aria-labelledby="mf-missing-rating-dot-product-edition-title"><span class="visual-fragment-anchor" id="mf-rating-visual-title" aria-hidden="true"></span><span class="visual-fragment-anchor" id="mf-rating-svg-title" aria-hidden="true"></span><span class="visual-fragment-anchor" id="mf-rating-svg-desc" aria-hidden="true"></span><span class="visual-fragment-anchor" id="arrow-forward" aria-hidden="true"></span><p class="visual-title" id="mf-missing-rating-dot-product-edition-title">Matching user and item coordinates multiply, then sum to the missing rating</p><div class="visual-argument" data-visual-argument="mf-missing-rating-dot-product" data-argument-kind="ledger" data-argument-composition="arithmetic"><svg viewBox="0 0 360 376" role="img" aria-label="A missing rating is a sum of matched coordinate products. The example user vector (1,2) and item vector (3,1) predict 1x3+2x1=5. This is a model estimate, not an observed rating."><title>A missing rating is a sum of matched coordinate products</title><desc>The example user vector (1,2) and item vector (3,1) predict 1x3+2x1=5. This is a model estimate, not an observed rating.</desc><text x="20" y="28" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">User factors</tspan></text><g data-proof-token="1" data-proof-row="0" data-proof-position="0" data-proof-box="20,60,154,48"><text x="97" y="87" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="97" dy="0">1</tspan></text></g><g data-proof-token="2" data-proof-row="0" data-proof-position="1" data-proof-box="180,60,154,48"><text x="257" y="87" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="257" dy="0">2</tspan></text></g><text x="20" y="144" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Item factors</tspan></text><g data-proof-token="3" data-proof-row="1" data-proof-position="0" data-proof-box="20,176,154,48"><text x="97" y="203" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="97" dy="0">3</tspan></text></g><g data-proof-token="1" data-proof-row="1" data-proof-position="1" data-proof-box="180,176,154,48"><text x="257" y="203" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="257" dy="0">1</tspan></text></g><text x="20" y="260" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Predicted missing rating</tspan></text><g data-proof-token="3" data-proof-row="2" data-proof-position="0" data-proof-box="20,292,58,48"><text x="49" y="319" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="49" dy="0">3</tspan></text></g><g data-proof-token="+" data-proof-row="2" data-proof-position="1" data-proof-box="84,292,58,48"><text x="113" y="319" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="113" dy="0">+</tspan></text></g><g data-proof-token="2" data-proof-row="2" data-proof-position="2" data-proof-box="148,292,58,48"><text x="177" y="319" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="177" dy="0">2</tspan></text></g><g data-proof-token="=" data-proof-row="2" data-proof-position="3" data-proof-box="212,292,58,48"><text x="241" y="319" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="241" dy="0">=</tspan></text></g><g data-proof-token="5" data-proof-row="2" data-proof-position="4" data-proof-box="276,292,58,48"><text x="305" y="319" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="305" dy="0">5</tspan></text><path d="M280 337L330 337" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75"></path></g></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content"><svg viewBox="0 0 360 634" role="img" aria-labelledby="mf-missing-rating-dot-product-edition-svg-title mf-missing-rating-dot-product-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="mf-missing-rating-dot-product-edition-svg-title">One missing entry selects two matching rows, then sums their coordinate products</title><desc id="mf-missing-rating-dot-product-edition-svg-desc">The original toy sparse ratings matrix highlights user A and item three. Matching factors u_A=[2,1] and v_3=[1.5,1] contribute three and one, totaling predicted rating four. The complete selection and calculation remain visible in a vertical phone-width composition.</desc><text x="180" y="22" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Original toy factors; one predicted missing</tspan><tspan x="180" dy="19">rating</tspan></text><text x="180" y="64" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">1. SELECT user A, item 3 in sparse R</tspan></text><text x="85" y="96" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="85" dy="0">1</tspan></text><text x="140" y="96" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="140" dy="0">2</tspan></text><text x="195" y="96" style="font-size:12px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="195" dy="0">3</tspan></text><text x="250" y="96" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="250" dy="0">4</tspan></text><text x="38" y="132" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="38" dy="0">A</tspan></text><rect x="58" y="109" width="55" height="40" rx="3" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.3"></rect><text x="85" y="135" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="85" dy="0">5</tspan></text><rect x="113" y="109" width="55" height="40" rx="3" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.3"></rect><text x="140" y="135" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="140" dy="0">3</tspan></text><rect x="168" y="109" width="55" height="40" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="195" y="135" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="195" dy="0">?</tspan></text><rect x="223" y="109" width="55" height="40" rx="3" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.3"></rect><text x="250" y="135" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="250" dy="0">1</tspan></text><text x="38" y="172" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="38" dy="0">B</tspan></text><rect x="58" y="149" width="55" height="40" rx="3" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.3"></rect><text x="85" y="175" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="85" dy="0">4</tspan></text><rect x="113" y="149" width="55" height="40" rx="3" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.3"></rect><text x="140" y="175" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="140" dy="0"></tspan></text><rect x="168" y="149" width="55" height="40" rx="3" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.3"></rect><text x="195" y="175" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="195" dy="0">2</tspan></text><rect x="223" y="149" width="55" height="40" rx="3" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.3"></rect><text x="250" y="175" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="250" dy="0"></tspan></text><text x="38" y="212" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="38" dy="0">C</tspan></text><rect x="58" y="189" width="55" height="40" rx="3" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.3"></rect><text x="85" y="215" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="85" dy="0"></tspan></text><rect x="113" y="189" width="55" height="40" rx="3" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.3"></rect><text x="140" y="215" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="140" dy="0">1</tspan></text><rect x="168" y="189" width="55" height="40" rx="3" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.3"></rect><text x="195" y="215" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="195" dy="0"></tspan></text><rect x="223" y="189" width="55" height="40" rx="3" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.3"></rect><text x="250" y="215" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="250" dy="0">5</tspan></text><text x="38" y="252" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="38" dy="0">D</tspan></text><rect x="58" y="229" width="55" height="40" rx="3" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.3"></rect><text x="85" y="255" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="85" dy="0">2</tspan></text><rect x="113" y="229" width="55" height="40" rx="3" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.3"></rect><text x="140" y="255" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="140" dy="0"></tspan></text><rect x="168" y="229" width="55" height="40" rx="3" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.3"></rect><text x="195" y="255" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="195" dy="0">4</tspan></text><rect x="223" y="229" width="55" height="40" rx="3" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.3"></rect><text x="250" y="255" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="250" dy="0"></tspan></text><text x="180" y="312" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">2. MATCH user and item factor rows</tspan></text><g data-editorial-node><rect x="15" y="337" width="150" height="73" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="90" y="369" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">user A</tspan><tspan x="90" dy="19">u_A = [2, 1]</tspan></text></g><g data-editorial-node><rect x="195" y="337" width="150" height="73" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.3"></rect><text x="270" y="369" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">item 3</tspan><tspan x="270" dy="19">v_3 = [1.5, 1]</tspan></text></g><path d="M90 410L90 444" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M86.64402122977057,437.85692206676737L90,444L93.35597877022943,437.85692206676737" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M270 410L270 444" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M266.6440212297706,437.85692206676737L270,444L273.3559787702294,437.85692206676737" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><g data-editorial-node><rect x="15" y="448" width="150" height="61" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="90" y="474" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="90" dy="0">factor 1</tspan><tspan x="90" dy="19">2 * 1.5 = 3</tspan></text></g><g data-editorial-node><rect x="195" y="448" width="150" height="61" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="270" y="474" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="270" dy="0">factor 2</tspan><tspan x="270" dy="19">1 * 1 = 1</tspan></text></g><text x="180" y="551" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">3. SUM: predicted R[A,3] = 3 + 1 = 4</tspan></text><text x="180" y="598" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">A prediction fills the missing cell; it is not</tspan><tspan x="180" dy="19">an observed rating.</tspan></text></svg><div class="visual-derivation-caption"><strong>Read it this way:</strong> Keep user A and item 3 fixed through every stage. Their matching factor rows contribute 3 and 1. The predicted missing rating is 4, not a newly observed preference.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="mf-missing-rating-dot-product"><strong>Check:</strong> Which two factor rows predict the missing A/item-3 entry? How do their matching coordinates produce 4?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="mf-missing-rating-dot-product">The example user vector (1,2) and item vector (3,1) predict 1x3+2x1=5. This is a model estimate, not an observed rating.</span></figcaption></figure>

| Entry in user A's row | Observation | Explicit-rating loss |
| --- | --- | --- |
| Item 1 | Rating 5 | Included |
| Item 3 | Unknown | Excluded |
| Item 4 | Rating 1 | Included |

## The objective

Minimize regularized squared error on observed ratings $\Omega = \{(u, i) : r_{ui} \text{ observed}\}$:

$$
\min_{U, V} \sum_{(u, i) \in \Omega} (r_{ui} - u_u^\top v_i)^2 + \lambda (\|U\|_F^2 + \|V\|_F^2).
$$

The observed set $\Omega$ excludes unknown entries, and $\lambda\ge0$ controls the stated L2 penalty.
With biases, use $\hat r_{ui}=\mu+b_u+b_i+u_u^\top v_i$, defining global mean and user/item offsets.
Fit these together; adding $\mu=3,b_A=0.2,b_3=-0.1$ afterward changes four to $7.1$.
That exceeds a 1-5 rating scale and is not automatically a calibrated prediction.

### Change observed signal

**New task:** replace explicit ratings with clicks whose exposure is not fully observed.

**Picture and change:** a confidence-weighted implicit model uses

$$
\min_{U, V} \sum_{u, i} c_{ui} (p_{ui} - u_u^\top v_i)^2 + \lambda (\|U\|_F^2 + \|V\|_F^2)
$$

Here $p_{ui}$ is an interaction indicator and $c_{ui}\ge0$ is modeling confidence, not an identified exposure probability.
Unlike explicit-rating loss, this objective includes all pairs, usually with lower confidence for unobserved interactions.

**Check:** a separate two-item fixture has scores $(0.8,0.2)$, preferences $(1,0)$, and confidences $(5,1)$.
Its unregularized loss is $5(1-0.8)^2+(0-0.2)^2=0.24$, not the positive-only value $0.20$.
A never-shown item's weak zero preference remains an assumption, not evidence of dislike.

**Use when:** revise the objective and evaluation when observation semantics change.

## Cold start

### Change item availability

**New task:** score a new item without a trained ID row.

**Picture and change:** use a separately validated feature encoder or content/context fallback, rather than a nonexistent lookup row.

**Check:** if an encoder supplies $(1,0)$, user A's zero-bias score is $2(1)+1(0)=2$.
Increasing $k$ alone cannot produce a trained row for an unseen ID.
An encoder can compute a vector without proving cold-item accuracy.

**Use when:** evaluate the cold-item slice and fallback separately from seen-ID performance.

<!-- learning:core-end -->

<div data-learning-extension>

## Training

Squared-error factorization is nonconvex jointly, but convex in either factor when the other is fixed.
[ALS](/concepts/alternating-least-squares/) solves conditional ridge problems; SGD updates selected rows.
For one pair's squared residual and two-row L2 penalty, hold biases fixed and save both old rows.
Set $e=r-\mu-b_u-b_i-u_{\rm old}^\top v_{\rm old}$, then compute simultaneously

$$
u_{\rm new}=(1-2\eta\lambda)u_{\rm old}+2\eta e v_{\rm old},
\qquad
v_{\rm new}=(1-2\eta\lambda)v_{\rm old}+2\eta e u_{\rm old}.
$$

Using the updated user row for the item gradient changes this step.
Sampling and regularization scaling must match the full objective.

## Properties

For orthogonal $Q$, replacing $(U,V)$ with $(UQ,VQ)$ preserves predictions and L2 norms.
The fixture rows can become $(1,-2)$ and $(1,-1.5)$ while retaining dot product four.
Tune $k$ and regularization against matched baselines; coordinates are not uniquely identifiable tastes.

## Connection to two-tower models

[Two-tower retrieval](/concepts/two-tower-retrieval/) replaces free rows with feature encoders $f_\theta,g_\phi$.
An ID-only tower still lacks trained parameters for new IDs.
Sampled retrieval losses differ from squared-error ratings and all-pair implicit fitting.

## When to use plain MF in 2026

MF can be a rating baseline, prototype, or embedding initializer.
Choose architecture from task, latency, features, and measured validation, not a universal deployment-prevalence claim.

## Common pitfalls

Dot-product prediction does not validate within-type cosine; normalization changes scores.
Compare bias baselines and ranking metrics when top-K quality matters; RMSE answers a different question.

Can a PCA rotation identify unique causal tastes?

<details><summary>Answer</summary>

No. It chooses coordinates without removing equivalent factorizations or establishing causal meanings.

</details>

References: [Koren et al. (2009)](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf), [Hu et al. (2008)](http://yifanhu.net/PUB/cf.pdf).

</div>

## Related

- [Two-tower retrieval](/concepts/two-tower-retrieval/). Neural generalization.
- [Embedding spaces](/concepts/embedding-spaces-and-similarity/). How the latent factors are used.
