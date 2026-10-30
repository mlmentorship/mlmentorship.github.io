---
title: "Tokenization: BPE, WordPiece, and the LLM era"
description: "Trace ranked BPE merges, byte coverage and the versioned text-to-ID contract shared by training and serving."
date: "2026-01-20"
draft: false
tags: ["concepts"]
category: "concepts"
---


## Summary

A tokenizer converts text to model IDs; preprocessing, vocabulary, merge rules and special-token handling belong to the model contract.

Consider a hypothetical corpus containing `map` five times, `maps` three times and `cap` twice.
Use UTF-8 byte base pieces, forbid merges across word boundaries, and resolve ties by a declared fixed ordering.
Which pairs merge, and how should the learned tokenizer encode `maps`, `cap` and unseen `traps`?

## The main families

### BPE (Byte-Pair Encoding)

Count adjacent pairs across the weighted corpus, merge the most frequent eligible pair, apply it, then recount.
Training changes the segmentation; encoding later replays the learned priority without retraining on each input.

<!-- visual:bpe-corpus-merge-trace -->
<figure class="learning-figure" aria-labelledby="bpe-merge-trace-title">
	<p class="visual-kicker">Weighted training counts; ranked encoding</p>
	<p class="visual-title" id="bpe-merge-trace-title">Learn the merge order from the corpus, then reuse it on new words</p><div class="visual-argument" data-visual-argument="bpe-corpus-merge-trace" data-argument-kind="ledger"><p class="visual-argument-premise">Learn merge order once; reuse it during encoding</p><svg viewBox="0 0 360 322" role="img" aria-label="Learn merge order once; reuse it during encoding. The corpus learns a prioritized merge table. A new word applies that fixed order rather than relearning frequencies from itself."><title>Learn merge order once; reuse it during encoding</title><desc>The corpus learns a prioritized merge table. A new word applies that fixed order rather than relearning frequencies from itself.</desc><text x="20" y="28" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Learned merge order</tspan></text><rect data-proof-token="l+o" x="20" y="58" width="102.66666666666667" height="34" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke)"></rect><text x="71.33333333333334" y="79" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="71.33333333333334" dy="0">l+o</tspan></text><rect data-proof-token="lo+w" x="126.66666666666667" y="58" width="102.66666666666667" height="34" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke)"></rect><text x="178" y="79" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="178" dy="0">lo+w</tspan></text><rect data-proof-token="e+r" x="233.33333333333334" y="58" width="102.66666666666667" height="34" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke)"></rect><text x="284.6666666666667" y="79" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="284.6666666666667" dy="0">e+r</tspan></text><text x="20" y="126" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">New word before merges</tspan></text><rect data-proof-token="l" x="20" y="156" width="60" height="34" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke)"></rect><text x="50" y="177" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="50" dy="0">l</tspan></text><rect data-proof-token="o" x="84" y="156" width="60" height="34" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke)"></rect><text x="114" y="177" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="114" dy="0">o</tspan></text><rect data-proof-token="w" x="148" y="156" width="60" height="34" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke)"></rect><text x="178" y="177" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="178" dy="0">w</tspan></text><rect data-proof-token="e" x="212" y="156" width="60" height="34" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke)"></rect><text x="242" y="177" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="242" dy="0">e</tspan></text><rect data-proof-token="r" x="276" y="156" width="60" height="34" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke)"></rect><text x="306" y="177" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="306" dy="0">r</tspan></text><text x="20" y="224" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Apply the fixed merges</tspan></text><rect data-proof-token="low" x="20" y="254" width="156" height="34" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke)"></rect><text x="98" y="275" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="98" dy="0">low</tspan></text><rect data-proof-token="er" x="180" y="254" width="156" height="34" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke)"></rect><text x="258" y="275" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="258" dy="0">er</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
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
	<div class="visual-derivation-caption">recount after each training merge. At encoding time, replay the fixed learned ranks; do not choose pairs using the new word's frequencies.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="bpe-corpus-merge-trace"><strong>Check:</strong> Why does the second winning pair count become eight after the first merge? Why is the new word not used to relearn merge order?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="bpe-corpus-merge-trace">The corpus learns a prioritized merge table. A new word applies that fixed order rather than relearning frequencies from itself.</span></figcaption>
</figure>

Initially, `a+p` occurs $5+3+2=10$ times.
After merging it, `m+ap` occurs eight times; after that merge, `map+s` occurs three times.
All three winners are unique, so the tie rule does not affect this example.

The frozen sequence yields `maps → [maps]`, `cap → [c, ap]` and `traps → [t, r, ap, s]`.
The last word needs available byte pieces `t` and `r`, even though the tiny training corpus never used them.

### Byte-level BPE

Starting with all 256 byte values makes supported UTF-8 text representable before optional merges.
An individual token's bytes need not form a complete Unicode character; decode the whole sequence with the declared policy.
Character-only vocabularies without fallback do not inherit this coverage guarantee.

## Change base units or the deployed vocabulary

### 1. Encode previously unseen Unicode

**New task:** preserve supported text without an unknown-token fallback or silent normalization.

**Picture and change:** U+00E9 has UTF-8 bytes `[195, 169]` before merges.
Its decomposed form, U+0065 followed by U+0301, has different bytes.
Whether to normalize is a task requirement, not a universal cleanup step.

**Code:** use the existing tokenizer implementation, pinned here to `tiktoken==0.12.0` and `cl100k_base`.
This real encoding is separate from the toy corpus above.

```python
import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")
composed = "\u00e9"
decomposed = "e\u0301"
composed_ids = encoding.encode(composed, disallowed_special="all")
decomposed_ids = encoding.encode(decomposed, disallowed_special="all")
assert list(composed.encode("utf-8")) == [195, 169]
assert encoding.decode_bytes(composed_ids) == composed.encode("utf-8")
assert encoding.decode_bytes(decomposed_ids) == decomposed.encode("utf-8")
assert composed_ids != decomposed_ids
print(composed_ids, decomposed_ids)
```

**Check:** both forms round-trip to their original bytes, but their token IDs differ.
Special-token-looking strings need a separate explicit policy; do not grant control-token status to arbitrary user text.

**Use when:** exact code, identifiers or source text must survive preprocessing.

### 2. Add domain-specific pieces

**New task:** reduce fragmentation for domain text without silently changing a deployed model's symbols.

**Picture and change:** version the vocabulary, IDs, embedding/output rows, templates and corresponding continued training together.
Preserving old IDs can avoid one incompatibility, but does not train new rows.

**Check:** reusing ID 17 for another piece without updating coupled weights changes the model's input interpretation.
Revalidate quality and context counts on the paired model/tokenizer artifact.

**Use when:** measured domain fragmentation justifies model adaptation and its costs.

## Failure and transfer

Serving with different BOS/EOS IDs or a different role template can break compatibility despite ordinary text looking correct.
Compare the final serialized IDs, not screenshots of prompts.

Does fewer tokens per document establish a better model?

<details>
<summary>Check your answer</summary>

No. Compression, coverage, quality and embedding/output parameter cost are different quantities; evaluate the complete paired artifact.

</details>

<!-- learning:core-end -->

<div data-learning-extension>

### WordPiece

BERT-style inference commonly uses greedy longest-match pieces within pretokenized words, with `##` marking continuation.
Training details depend on the implementation; a family name does not pin a unique likelihood-based merge algorithm.
Unknown-token behavior differs from complete byte coverage.

### SentencePiece (Unigram)

SentencePiece is a toolkit supporting Unigram and BPE, not a synonym for Unigram.
Unigram uses a probabilistic piece vocabulary and segmentation; training can prune candidates according to its objective.
Normalization and whitespace treatment are configurable parts of the artifact.

### Tiktoken / cl100k / o200k

Tiktoken implements byte-level BPE encodings, including `cl100k_base` and `o200k_base`.
Encoding name, package version, merge assets and special-token policy matter more than a model-family nickname.

## What an interviewer expects you to say

Explain base units, corpus weighting, boundaries, normalization, merge ordering, ties, IDs, special tokens and round-trip policy.
Distinguish training from encoding and model adaptation from silent tokenizer replacement.

## Common confusions

Tokens are not necessarily words, characters or complete UTF-8 sequences.
Smaller vocabularies can increase sequence length; larger ones increase embedding/output rows and can reduce sharing.
Neither tradeoff alone predicts final quality.

## The LLM-era tokenization gotchas

Pin the training/serving contract, including template and BOS/EOS handling.
Measure actual multilingual corpora, indentation/code and number formatting with the exact encoding.
There is no universal characters-per-token ratio or guaranteed arithmetic benefit from digit splitting.
Test normalization against exact-byte requirements rather than silently applying NFC everywhere.

## Why interviewers ask

For inefficient domain tokenization, compare fragmentation, quality, memory and retraining costs before choosing vocabulary adaptation.
Use only truthful production examples; neither vocabulary terminology nor a token-count improvement proves operational experience.

References: [Sennrich et al.](https://aclanthology.org/P16-1162/), [tiktoken](https://github.com/openai/tiktoken), [SentencePiece](https://aclanthology.org/D18-2012/).

*Related: [Transformer architecture](/concepts/transformer-architecture/), [LLM Evals essay](/guides/llm-evals-the-hardest-part/).*

</div>
