---
title: "BERT and masked language modeling"
description: "Train a transformer to fill in randomly masked tokens. The result is a bidirectional encoder that broke a dozen NLP benchmarks at once and defined the pretrain-then-finetune era."
date: "2026-05-07"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

**BERT** (Bidirectional Encoder Representations from Transformers, [Devlin et al., 2019](https://arxiv.org/abs/1810.04805)) is a transformer encoder pretrained with masked language modeling: replace 15 percent of input tokens with a special `[MASK]` token, train the model to predict them. The pretrained encoder is then fine-tuned for downstream tasks.

Pre-BERT, NLP pipelines were task-specific: parse trees for parsing, sequence-to-sequence for translation, hand-crafted features for classification. BERT showed that one bidirectional pretrained encoder, fine-tuned per task, beat the entire task-specific stack on 11 benchmarks at once.

The BERT recipe (pretrain on raw text, fine-tune per task) defined NLP from 2018 to roughly 2022. Decoder-only LLMs (GPT family) eventually dominated for generative work, but BERT-style encoders are still the right answer for classification, retrieval, and embedding tasks. Most production embedding models (Sentence-BERT, modern retrieval encoders) are BERT descendants.

## The pretraining task

### Masked Language Modeling (MLM)

Pick 15 percent of token positions. Of those:

- 80 percent are replaced with `[MASK]`.
- 10 percent are replaced with a random token.
- 10 percent are kept as the original token.

Train the model to predict the original token at each picked position, using cross-entropy. The loss is evaluated only at the picked positions. The remaining 85 percent have no direct prediction loss, but they still supply context and can affect the selected predictions.

The 10/10 random/keep split exists because at fine-tuning time there are no `[MASK]` tokens. The model needs to handle every input position consistently.

<!-- visual:bert-mlm-selection-and-corruption -->
<figure class="learning-figure" aria-labelledby="bert-mlm-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="bert-mlm-title">Which tokens become MLM targets, and what input does BERT see at those positions?</p>
	<div class="visual-grid--two" style="grid-template-columns: 1fr;" role="group" aria-label="Masked language modeling selection and corruption flow">
		<section class="visual-panel plot-panel">
			<svg viewBox="0 0 300 230" role="img" aria-labelledby="bert-selection-title bert-selection-desc">
				<title id="bert-selection-title">Fifteen percent of positions become prediction targets</title>
				<desc id="bert-selection-desc">All input positions provide bidirectional context. A random 15 percent are selected as MLM targets and receive a direct cross-entropy loss. The other 85 percent receive no direct prediction loss, although they can influence predictions as context.</desc>
				<defs><marker id="bert-selection-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto"><path class="viz-arrow-forward" d="M0 0L8 4L0 8Z"></path></marker></defs>
				<text class="viz-axis-label" x="12" y="16">1 · CHOOSE TARGET POSITIONS</text>
				<rect class="viz-plot-bg" x="8" y="25" width="284" height="196" rx="5"></rect>
				<rect class="viz-node viz-node--input" x="50" y="39" width="200" height="46" rx="4"></rect>
				<text class="viz-node-label" x="150" y="58">100% of input positions</text>
				<text class="viz-node-value" x="150" y="74">all supply left + right context</text>
				<path d="M150 85V105" style="fill:none;stroke:var(--viz-edge);stroke-width:2"></path>
				<path d="M150 105L83 126" style="fill:none;stroke:var(--viz-edge);stroke-width:2" marker-end="url(#bert-selection-arrow)"></path>
				<path d="M150 105L217 126" style="fill:none;stroke:var(--viz-edge);stroke-width:2" marker-end="url(#bert-selection-arrow)"></path>
				<text class="viz-edge-label" x="91" y="111">randomly select</text>
				<rect class="viz-node viz-node--focus" x="18" y="127" width="130" height="66" rx="4"></rect>
				<text class="viz-node-label" x="83" y="148">selected 15%</text>
				<text class="viz-node-value" x="83" y="165">predict original token</text>
				<text class="viz-node-value" x="83" y="181">direct cross-entropy loss</text>
				<rect class="viz-node" x="158" y="127" width="124" height="66" rx="4"></rect>
				<text class="viz-node-label" x="220" y="148">other 85%</text>
				<text class="viz-node-value" x="220" y="165">context for targets</text>
				<text class="viz-node-value" x="220" y="181">no direct MLM loss</text>
				<text class="viz-axis-label" x="150" y="211" text-anchor="middle">SELECTION SETS WHERE LOSS IS MEASURED</text>
			</svg>
		</section>
		<section class="visual-panel plot-panel">
			<svg viewBox="0 0 300 230" role="img" aria-labelledby="bert-corruption-title bert-corruption-desc">
				<title id="bert-corruption-title">Every selected position follows one of three input-corruption paths</title>
				<desc id="bert-corruption-desc">Given the selected original token dinner, 80 percent of the time BERT sees MASK, 10 percent it sees a random token such as violin, and 10 percent it sees dinner unchanged. On every path the training target remains dinner and the selected position receives loss.</desc>
				<defs><marker id="bert-corruption-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto"><path class="viz-arrow-forward" d="M0 0L8 4L0 8Z"></path></marker></defs>
				<text class="viz-axis-label" x="12" y="16">2 · CORRUPT EACH SELECTED POSITION</text>
				<rect class="viz-plot-bg" x="8" y="25" width="284" height="196" rx="5"></rect>
				<text class="viz-label" x="18" y="48">original target</text>
				<rect class="viz-node viz-node--input" x="18" y="56" width="72" height="38" rx="4"></rect>
				<text class="viz-callout" x="54" y="79" text-anchor="middle">dinner</text>
				<path d="M90 75H108M108 47V159" style="fill:none;stroke:var(--viz-edge);stroke-width:1.6"></path>
				<path d="M108 47H126M108 103H126M108 159H126" style="fill:none;stroke:var(--viz-edge);stroke-width:1.6"></path>
				<rect class="viz-node viz-node--focus" x="126" y="30" width="84" height="34" rx="4"></rect>
				<rect class="viz-node" x="126" y="86" width="84" height="34" rx="4"></rect>
				<rect class="viz-node viz-node--input" x="126" y="142" width="84" height="34" rx="4"></rect>
				<text class="viz-callout" x="168" y="51" text-anchor="middle">80% · [MASK]</text>
				<text class="viz-callout" x="168" y="107" text-anchor="middle">10% · violin</text>
				<text class="viz-callout" x="168" y="163" text-anchor="middle">10% · dinner</text>
				<path d="M210 47L232 86" style="fill:none;stroke:var(--viz-edge);stroke-width:1.6" marker-end="url(#bert-corruption-arrow)"></path>
				<path d="M210 103H232" style="fill:none;stroke:var(--viz-edge);stroke-width:1.6" marker-end="url(#bert-corruption-arrow)"></path>
				<path d="M210 159L232 120" style="fill:none;stroke:var(--viz-edge);stroke-width:1.6" marker-end="url(#bert-corruption-arrow)"></path>
				<rect class="viz-node viz-node--output" x="232" y="78" width="58" height="52" rx="4"></rect>
				<text class="viz-node-value" x="261" y="96">predict</text>
				<text class="viz-node-label" x="261" y="113">dinner</text>
				<text class="viz-label" x="150" y="194" text-anchor="middle">All 3 paths keep the same target and receive loss.</text>
				<text class="viz-axis-label" x="150" y="211" text-anchor="middle">80 / 10 / 10 IS CONDITIONAL ON SELECTION</text>
			</svg>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> first choose the 15 percent of positions where BERT must recover the original token. Only then choose what input replaces each selected token: `[MASK]`, a random token, or the unchanged token. All three paths predict the same original target; unselected positions provide context without their own MLM loss.</figcaption>
</figure>

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
