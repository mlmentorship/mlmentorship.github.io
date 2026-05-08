---
title: "Word embeddings: Word2Vec, GloVe, and the geometry of meaning"
description: "Map words to dense vectors so that similar words land near each other. The breakthrough that proved meaning lives in geometry, not symbols."
date: "2026-05-07"
draft: false
tags: ["concepts"]
category: "concepts"
---

## One-line definition

**Word embeddings** assign each word a dense vector (typically 100 to 300 dimensions) such that distributional similarity in text corresponds to geometric proximity in the embedding space. Trained from co-occurrence patterns, no explicit supervision.

## Why it matters

Pre-2013 NLP represented words as one-hot vectors. The vocabulary was the dimension; "king" and "queen" were as far apart as "king" and "table." Word2Vec ([Mikolov et al., 2013](https://arxiv.org/abs/1301.3781)) showed that learned dense vectors satisfy famous analogies like $\text{vec}(\text{king}) - \text{vec}(\text{man}) + \text{vec}(\text{woman}) \approx \text{vec}(\text{queen})$. The geometry encodes meaning.

Modern transformers learn embeddings end-to-end as part of training. Pretrained Word2Vec / GloVe vectors are mostly historical, but the conceptual frame (meaning as geometry, training from distributional signal) is still the foundation of every embedding-based retrieval system.

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

## What replaced them

Contextualized embeddings: ELMo, BERT, every modern LLM. The same word gets different vectors in different sentences. Pretrained Word2Vec and GloVe are now mostly used as light-weight features for low-resource scenarios or as a teaching example.

## Common pitfalls

- **Using cosine similarity on context embeddings without L2 normalization.** Most modern stacks normalize before doing the dot product.
- **Treating analogies as deep evidence of "reasoning."** The arithmetic works because of how training data is structured, not because the model "understands" gender or tense.
- **Forgetting subword tokenization.** Modern systems embed BPE pieces, not whole words. "Embeddings" in a 2025 LLM are subword embeddings.

## Related

- [Tokenization](/concepts/tokenization-bpe-wordpiece-and-the-llm-era/).
- [Embedding spaces and similarity metrics](/concepts/embedding-spaces-and-similarity-metrics/).
- [Approximate nearest neighbors](/concepts/approximate-nearest-neighbors/).
