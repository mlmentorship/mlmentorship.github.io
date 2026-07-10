---
title: "BERT and masked language modeling"
description: "Train a transformer to fill in randomly masked tokens. The result is a bidirectional encoder that broke a dozen NLP benchmarks at once and defined the pretrain-then-finetune era."
date: "2026-05-07"
draft: false
tags: ["concepts"]
category: "concepts"
---

## One-line definition

**BERT** (Bidirectional Encoder Representations from Transformers, [Devlin et al., 2019](https://arxiv.org/abs/1810.04805)) is a transformer encoder pretrained with masked language modeling: replace 15 percent of input tokens with a special `[MASK]` token, train the model to predict them. The pretrained encoder is then fine-tuned for downstream tasks.

## Why it matters

Pre-BERT, NLP pipelines were task-specific: parse trees for parsing, sequence-to-sequence for translation, hand-crafted features for classification. BERT showed that one bidirectional pretrained encoder, fine-tuned per task, beat the entire task-specific stack on 11 benchmarks at once.

The BERT recipe (pretrain on raw text, fine-tune per task) defined NLP from 2018 to roughly 2022. Decoder-only LLMs (GPT family) eventually dominated for generative work, but BERT-style encoders are still the right answer for classification, retrieval, and embedding tasks. Most production embedding models (Sentence-BERT, modern retrieval encoders) are BERT descendants.

## The pretraining task

### Masked Language Modeling (MLM)

Pick 15 percent of token positions. Of those:

- 80 percent are replaced with `[MASK]`.
- 10 percent are replaced with a random token.
- 10 percent are kept as the original token.

Train the model to predict the original token at each picked position, using cross-entropy. The model only learns from the picked positions; the remaining 85 percent contribute zero gradient.

The 10/10 random/keep split exists because at fine-tuning time there are no `[MASK]` tokens. The model needs to handle every input position consistently.

### Why bidirectional matters

A causal LM (GPT-style) only attends to previous tokens. A masked LM has access to context on both sides. For tasks like classification, NER, or extractive QA where the full input is available, bidirectional context is strictly more informative.

### Next Sentence Prediction (NSP)

The original BERT also predicted whether two sentences appeared consecutively in the corpus. Subsequent work ([RoBERTa](https://arxiv.org/abs/1907.11692)) showed NSP adds little; modern variants drop it.

## Architecture

Standard transformer encoder. Inputs:

- **Token embeddings** (WordPiece subwords).
- **Position embeddings** (learned).
- **Segment embeddings** (which of two sentences the token belongs to).

Special tokens:

- `[CLS]` at position 0. Its final-layer hidden state is used as the sequence representation for classification.
- `[SEP]` between sentences and at the end.

BERT-base: 12 layers, 768 hidden dim, 12 heads, 110M parameters. BERT-large: 24 layers, 1024 hidden, 16 heads, 340M parameters.

## Fine-tuning

Add a small head on top of the pretrained encoder, train end-to-end on the downstream task:

| Task | Head |
|---|---|
| Single-sequence classification | Linear on `[CLS]` |
| Sentence-pair classification (NLI) | Linear on `[CLS]`, both sentences in input |
| Token classification (NER, POS) | Linear on every token's final hidden state |
| Extractive QA | Two linears predicting span start and end positions |

Typical fine-tune: 2 to 5 epochs, learning rate $\sim 2 \cdot 10^{-5}$, small batch.

## Variants

- **RoBERTa** ([Liu et al., 2019](https://arxiv.org/abs/1907.11692)). More data, longer training, no NSP, dynamic masking. The "BERT done right" reference.
- **ALBERT** ([Lan et al., 2019](https://arxiv.org/abs/1909.11942)). Parameter sharing across layers, factorized embeddings.
- **DeBERTa** ([He et al., 2021](https://arxiv.org/abs/2006.03654)). Disentangled position and content attention.
- **Sentence-BERT** ([Reimers & Gurevych, 2019](https://arxiv.org/abs/1908.10084)). BERT fine-tuned with siamese training to produce sentence embeddings useful with cosine similarity.

## When to use BERT in 2026

- Classification, NER, extractive QA: still competitive and much smaller than an LLM.
- Embeddings for retrieval: the modern stack (E5, BGE, GTE) is BERT-family.
- Anywhere bidirectional context helps and you do not need free-form generation.

When to skip: anything generative. Use a decoder-only LLM.

## Common pitfalls

- **Forgetting that fine-tuning is full backprop through the encoder.** Freeze the encoder only if you cannot afford otherwise; full fine-tuning is the strong baseline.
- **Using the `[CLS]` representation directly for sentence similarity.** It was not pretrained for that. Use Sentence-BERT or one of its descendants instead.
- **Treating BERT as a generative model.** It cannot generate left-to-right text; the masking objective is local.

## Related

- [Self-attention vs cross-attention](/concepts/self-attention-vs-cross-attention/).
- [Transformer architecture](/concepts/transformer-architecture/).
- [Two-tower retrieval](/concepts/two-tower-retrieval/).
