---
title: "Tokenization: BPE, WordPiece, and the LLM era"
description: "The critical input layer between text and model. Tokenization mismatch is a frequent source of production LLM bugs."
date: "2026-01-20"
draft: false
tags: ["concepts"]
category: "concepts"
---


## Summary

Tokenization splits text into discrete units (tokens). Modern LLMs use subword schemes (BPE, WordPiece, SentencePiece) that decompose rare words into subword pieces while preserving common words as single tokens.

- Tokenization defines the model's input and output space, it's irreversibly baked into a trained model.
- Token count drives both cost and latency at inference. Different tokenizers give different counts for the same text (sometimes by 30%+).
- Tokenization mismatches between training and serving cause silent quality regressions that are hard to debug.
- Production LLM bugs in multilingual quality, code performance, and unusual-character handling often involve tokenization.

## The main families

### BPE (Byte-Pair Encoding)

Original from text compression (1994), adapted for NLP. Algorithm:
1. Start with a vocabulary of single characters.
2. Find the most frequent adjacent pair in the corpus.
3. Merge that pair into a new vocabulary item.
4. Repeat until target vocabulary size.

**Learning objective:** trace how corpus-wide pair counts create ranked, reusable BPE tokens that compress common words while leaving rarer words as multiple pieces.

<!-- visual:bpe-corpus-merge-trace -->
<figure class="learning-figure" aria-labelledby="bpe-merge-trace-title">
	<p class="visual-kicker">Learning objective · toy byte-level corpus</p>
	<p class="visual-title" id="bpe-merge-trace-title">How does one frequent byte pair grow into reusable subword and whole-word tokens?</p>
	<div class="visual-grid--two" style="grid-template-columns: 1fr;" role="list" aria-label="Three ranked BPE merge rounds followed by tokenization of three words">
		<section class="visual-panel" role="listitem">
			<h4>START · COUNT ADJACENT PAIRS ACROSS ALL OCCURRENCES</h4>
			<p><strong>Corpus:</strong> <code>m a p × 5</code> · <code>m a p s × 3</code> · <code>c a p × 2</code></p>
			<p><strong>Largest count:</strong> <code>a + p = 5 + 3 + 2 = 10</code><br /><strong>Rank 1 merge:</strong> add <code>ap</code> to the vocabulary.</p>
		</section>
		<section class="visual-panel" role="listitem">
			<h4>ROUND 2 · RECOUNT AFTER APPLYING <code>a + p → ap</code></h4>
			<p><strong>Corpus now:</strong> <code>m ap × 5</code> · <code>m ap s × 3</code> · <code>c ap × 2</code></p>
			<p><strong>Largest count:</strong> <code>m + ap = 5 + 3 = 8</code><br /><strong>Rank 2 merge:</strong> add <code>map</code> to the vocabulary.</p>
		</section>
		<section class="visual-panel" role="listitem">
			<h4>ROUND 3 · RECOUNT AFTER APPLYING <code>m + ap → map</code></h4>
			<p><strong>Corpus now:</strong> <code>map × 5</code> · <code>map s × 3</code> · <code>c ap × 2</code></p>
			<p><strong>Largest count:</strong> <code>map + s = 3</code><br /><strong>Rank 3 merge:</strong> add <code>maps</code> to the vocabulary.</p>
		</section>
		<section class="visual-panel" role="listitem">
			<h4>APPLY · REPLAY MERGES IN LEARNED RANK ORDER</h4>
			<p><code>maps</code> → <strong><code>[maps]</code></strong> · common whole word<br /><code>cap</code> → <strong><code>[c] [ap]</code></strong> · shared learned ending<br /><code>traps</code> → <strong><code>[t] [r] [ap] [s]</code></strong> · rarer word, more pieces</p>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> read downward. At each round, count pairs in the current segmentation, merge the largest count everywhere, then recount. The learned order matters at encoding time: <code>ap</code> forms before <code>map</code>, and <code>map</code> before <code>maps</code>. Frequent patterns therefore become larger tokens; an unfamiliar word still falls back to available byte pieces. Original worked example informed by <a href="https://aclanthology.org/P16-1162/">Sennrich et al. (2016)</a> and the <a href="https://github.com/openai/tiktoken">tiktoken documentation</a>.</figcaption>
</figure>

Result: common words become single tokens; rare words decompose into subwords; arbitrary text is always representable (down to characters).

Used in: GPT-2/3/4, LLaMA family, most modern LLMs.

### Byte-level BPE

BPE applied at the *byte* level, not the character level. Vocabulary starts with 256 byte values; merges happen on byte sequences. The advantage: any UTF-8 text is representable without unknown tokens, and the base vocabulary is fixed.

Used in: GPT-2 onward, RoBERTa, most modern LLMs. This is what "BPE" almost always means in 2026.

### WordPiece

Similar to BPE but uses a likelihood-based merging criterion (maximize the probability of the corpus given the vocabulary) instead of frequency. Tokens not at word start are prefixed with `##`.

Used in: BERT, distilBERT.

### SentencePiece (Unigram)

A different paradigm: starts with a large vocabulary, removes pieces that contribute least to corpus likelihood. Trains on raw text with no pre-tokenization (handles whitespace and punctuation as part of the algorithm). Implementation by Google, often paired with their models.

Used in: T5, mT5, ALBERT, multilingual models.

### Tiktoken / cl100k / o200k

Tiktoken is OpenAI's BPE implementation. Their tokenizers (cl100k_base, o200k_base) are byte-level BPE optimized for code and multilingual text. Available open-source.

## What an interviewer expects you to say

If asked about tokenization:

1. Define subword tokenization and why it solves the OOV (out-of-vocabulary) problem.
2. Explain BPE algorithm (frequency-based merging).
3. Distinguish character-level vs byte-level BPE.
4. Mention vocab size trade-offs (small vocab → more tokens per word; large vocab → more parameters in embedding/output layer).
5. Mention that **tokenization choice is fixed at training time**: you can't change it post-training without retraining.

Bonus depth: mention the multilingual tokenization issue (English-trained tokenizers have very different per-character costs for non-Latin scripts, sometimes 5-10x more tokens for the same content).

## Common confusions

- **"Tokens are words."** No. Tokens are arbitrary pieces. "tokenization" itself might be one or two tokens depending on the tokenizer.
- **"Switching tokenizer = small change."** It's a categorical change. The model has to be retrained.
- **"All tokenizers are roughly equivalent in cost."** No. cl100k vs LLaMA tokenizer can give 10-30% different token counts on the same text. For multilingual and code-heavy content, the gap is much larger.
- **"BPE is the same as WordPiece."** Similar idea, different merge criterion (frequency vs likelihood). The differences in practice are small.

## The LLM-era tokenization gotchas

A few tokenization issues that bite in production:

1. **Tokenizer drift between training and serving.** Different versions of the same tokenizer family can subtly differ. Always pin the exact tokenizer version with the model.
2. **Inconsistent BOS/EOS tokens.** Different chat-tuned models expect different special tokens for system prompts, user turns, assistant turns. Send the wrong format → quality regression. Use the model's chat template, not your own.
3. **Multilingual cost asymmetry.** English text is ~4 chars/token; Japanese can be ~1 char/token; some scripts are even worse. A 1000-character Japanese document can cost 10x more than a 1000-character English document.
4. **Code tokenization.** Code-aware tokenizers handle indentation, common keywords, and operators efficiently. Non-code tokenizers (older BERT, etc.) treat code very inefficiently.
5. **Unicode normalization.** Different unicode normalizations of the "same" string (composed vs decomposed) can produce different token sequences. NFC normalization is standard but worth verifying for your domain.
6. **Number tokenization.** Some tokenizers split numbers digit-by-digit (great for arithmetic), others treat numbers as multi-digit tokens (worse for arithmetic). LLaMA and modern models tend to split digits.

## Why interviewers ask

Tokenization questions test:
1. Whether you know the algorithm vs just the vocabulary file.
2. Whether you've debugged a real tokenization bug in production.
3. Whether you know modern LLM tokenization choices (byte-level BPE, digit splitting, code-aware vocab).
4. Whether you're aware of multilingual fairness/cost issues.

A common follow-up asks how to handle an inefficient tokenizer for medical, legal, or code text. Extend the tokenizer and continue pretraining, or train a new tokenizer on domain text and retrain the model.

---

*Related: [Transformer architecture](/concepts/transformer-architecture/), [LLM Evals essay](/guides/llm-evals-the-hardest-part/).*
