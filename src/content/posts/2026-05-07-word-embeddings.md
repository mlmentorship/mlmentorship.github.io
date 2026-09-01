---
title: "Word embeddings: Word2Vec, GloVe, and the geometry of meaning"
description: "Map words to dense vectors so that similar words land near each other. The breakthrough that proved meaning lives in geometry, not symbols."
date: "2026-05-07"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

**Word embeddings** assign each word a dense vector (typically 100 to 300 dimensions) such that distributional similarity in text corresponds to geometric proximity in the embedding space. Trained from co-occurrence patterns, no explicit supervision.

Pre-2013 NLP represented words as one-hot vectors. The vocabulary was the dimension; "king" and "queen" were as far apart as "king" and "table." Word2Vec ([Mikolov et al., 2013](https://arxiv.org/abs/1301.3781)) showed that learned dense vectors satisfy famous analogies like $\text{vec}(\text{king}) - \text{vec}(\text{man}) + \text{vec}(\text{woman}) \approx \text{vec}(\text{queen})$. The geometry encodes meaning.

Modern transformers learn embeddings end-to-end as part of training. Pretrained Word2Vec / GloVe vectors are mostly historical, but the conceptual frame (meaning as geometry, training from distributional signal) is still the foundation of every embedding-based retrieval system.

**Learning objective:** interpret a word analogy as an approximately shared displacement between two pairs, rather than as four words that merely occupy nearby points.

## Word2Vec: skip-gram

Predict context words from a target word. For corpus $w_1, \dots, w_T$ and window size $c$:

$$
\mathcal{L} = -\sum_{t=1}^{T} \sum_{-c \le j \le c, j \ne 0} \log p(w_{t+j} \mid w_t).
$$

The probability $p(w_{t+j} \mid w_t)$ uses two embeddings per word: a target embedding $v_w$ and a context embedding $u_w$. The score is $u_{w_{t+j}}^\top v_{w_t}$, normalized over the vocabulary.

Computing the softmax over a 100k-vocabulary at every step is infeasible. Two tricks:

- **Hierarchical softmax**: arrange the vocabulary as a binary tree. Predicting a word becomes a sequence of binary decisions, $O(\log V)$ per step.
- **Negative sampling**: instead of normalizing over the full vocabulary, sample a few negative examples (words sampled from a noise distribution) and treat the prediction as binary classification (positive context vs. sampled negatives). $O(k)$ per step where $k$ is the number of negatives. The dominant choice in practice.

## CBOW

The mirror image of skip-gram: predict the target from the average of context embeddings. Faster but slightly worse on rare words.

## GloVe

GloVe ([Pennington et al., 2014](https://aclanthology.org/D14-1162/)) takes a different angle: factorize the global co-occurrence matrix.

Build a matrix $X$ where $X_{ij}$ counts how often word $j$ appears in the context of word $i$. The training objective:

$$
\mathcal{L} = \sum_{i,j} f(X_{ij}) \cdot \big(v_i^\top u_j + b_i + b_j - \log X_{ij}\big)^2,
$$

where $f$ is a weighting that downweights rare and very common pairs. Closed-form intuition: GloVe is matrix factorization of $\log X_{ij}$.

Empirically GloVe and Word2Vec produce comparable embeddings. GloVe is sometimes preferred because the global matrix is reused across iterations.

## Properties of the learned space

- **Linear analogies**: vector arithmetic encodes relations (king - man + woman = queen, walked - walk + run = ran).
- **Cosine similarity** is the standard metric. Magnitudes correlate with frequency, so cosine factors that out.
- **Polysemy**: a word with multiple senses gets one vector that averages them. The cleanest motivation for contextualized embeddings (ELMo, BERT).

<!-- visual:word-analogy-shared-displacement -->
<figure class="learning-figure" aria-labelledby="word-analogy-title">
	<p class="visual-kicker">Relation geometry</p>
	<p class="visual-title" id="word-analogy-title">An analogy asks whether two word pairs have approximately the same displacement.</p>
	<div class="visual-panel plot-panel">
		<svg viewBox="0 0 320 350" role="img" aria-labelledby="word-analogy-svg-title word-analogy-svg-desc">
			<title id="word-analogy-svg-title">Word analogy as a shared vector displacement</title>
			<desc id="word-analogy-svg-desc">In an illustrative two-dimensional projection, man is lower left and woman is above and to its right. King is lower right, and applying the same up-right displacement reaches queen. Solid arrows from man to woman and king to queen are parallel and equal in this teaching construction. The equation king minus man plus woman approximately equals queen is shown below. These are schematic coordinates, not measured word vectors.</desc>
			<defs><marker id="word-analogy-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path class="viz-arrow-forward" d="M0,0 L7,3.5 L0,7 Z"></path></marker></defs>
			<text class="viz-axis-label" x="10" y="17">ILLUSTRATIVE 2D PROJECTION</text>
			<rect class="viz-plot-bg" x="8" y="28" width="304" height="226" rx="5"></rect>
			<path class="viz-gridline" d="M28 218H296M42 238V48"></path>
			<path d="M68 205L126 92" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:4;marker-end:url(#word-analogy-arrow)"></path>
			<path d="M194 205L252 92" style="fill:none;stroke:var(--viz-focus-stroke);stroke-width:4;marker-end:url(#word-analogy-arrow)"></path>
			<path d="M68 205H194M126 92H252" style="fill:none;stroke:var(--viz-edge);stroke-width:1.5;stroke-dasharray:5 4"></path>
			<rect class="viz-node" x="62" y="199" width="12" height="12" rx="1"></rect>
			<circle class="viz-node viz-node--output" cx="126" cy="92" r="7"></circle>
			<rect class="viz-node" x="188" y="199" width="12" height="12" rx="1"></rect>
			<circle class="viz-node viz-node--output" cx="252" cy="92" r="7"></circle>
			<text class="viz-callout" x="68" y="229" text-anchor="middle">man</text>
			<text class="viz-callout" x="126" y="75" text-anchor="middle">woman</text>
			<text class="viz-callout" x="194" y="229" text-anchor="middle">king</text>
			<text class="viz-callout" x="252" y="75" text-anchor="middle">queen</text>
			<text class="viz-axis-label" x="73" y="134" transform="rotate(-63 73 134)">RELATION Δ</text>
			<text class="viz-axis-label" x="199" y="134" transform="rotate(-63 199 134)">SAME Δ</text>
			<rect class="viz-node viz-node--output" x="18" y="272" width="284" height="61" rx="4"></rect>
			<text class="viz-callout" x="160" y="296" text-anchor="middle">king + (woman - man) ≈ queen</text>
			<text class="viz-axis-label" x="160" y="317" text-anchor="middle">TRANSLATE THE RELATION, THEN FIND THE NEAREST WORD</text>
		</svg>
	</div>
	<figcaption><strong>Read it this way:</strong> start with the arrow from <em>man</em> to <em>woman</em>, then move <em>king</em> by that same direction and distance. If the endpoint lands near <em>queen</em>, the two pairs encode a similar relation. This is an original 2D teaching construction, not measured coordinates: real embedding analogies are approximate, corpus-dependent, and do not imply reasoning.</figcaption>
</figure>

## What replaced them

Contextualized embeddings: ELMo, BERT, every modern LLM. The same word gets different vectors in different sentences. Pretrained Word2Vec and GloVe are now mostly used as light-weight features for low-resource scenarios or as a teaching example.

## Common pitfalls

- **Using cosine similarity on context embeddings without L2 normalization.** Most modern stacks normalize before doing the dot product.
- **Treating analogies as deep evidence of "reasoning."** The arithmetic works because of how training data is structured, not because the model "understands" gender or tense.
- **Forgetting subword tokenization.** Modern systems embed BPE pieces, not whole words. "Embeddings" in a 2025 LLM are subword embeddings.

## Related

- [Tokenization](/concepts/tokenization/).
- [Embedding spaces and similarity metrics](/concepts/embedding-spaces-and-similarity/).
- [Approximate nearest neighbors](/concepts/approximate-nearest-neighbors/).
