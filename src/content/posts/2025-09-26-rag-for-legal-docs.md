---
title: "Design a RAG system for legal documents"
description: "Legal RAG amplifies every standard RAG concern: precise citations, no hallucinations, regulated domain, dense documents with structure. The senior answer addresses each."
date: "2025-09-26"
draft: false
tags: ["questions"]
category: "questions"
---


> *Asked in: legal-tech, fintech, and regulated-domain LLM interviews.*

The L4 answer is generic RAG. The L6 answer is RAG with explicit handling of citation accuracy, structure-aware chunking, refusal behavior, and regulated-domain operational concerns.

## Domain-specific concerns vs generic RAG

What changes for legal:

1. **Citation precision is non-negotiable.** Every claim must trace to a verifiable passage. A hallucinated case citation is a malpractice risk.
2. **Documents are highly structured.** Statutes, contracts, case law have hierarchical structure (titles, sections, paragraphs, sub-clauses). Chunking that strips structure loses retrieval signal.
3. **Vocabulary is specialized.** General-purpose embeddings underperform; BM25 with legal-jargon-aware tokenization often outperforms naive embeddings on exact-match queries (case names, statute numbers, defined terms).
4. **Refusal is a feature.** "I can't answer this from the available sources" is the right answer when retrieval is uncertain. Better than a confident wrong answer.
5. **Audit trails are required.** Every retrieved passage and every generated answer needs to be logged for review.

<!-- visual:legal-rag-claim-lineage -->
<figure class="learning-figure" aria-labelledby="legal-rag-claim-lineage-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="legal-rag-claim-lineage-title">What evidence must survive before a legal RAG claim can be released?</p>
	<div class="visual-grid--two" role="group" aria-label="Four-stage legal RAG claim-lineage record from versioned source to release">
		<section class="visual-panel">
			<h4>1 · PRESERVE THE AUTHORITY</h4>
			<p><strong>Corpus version:</strong> 2026-08-15<br /><strong>Hierarchy:</strong> authority &rsaquo; section &rsaquo; clause<br /><strong>Passage:</strong> P-017</p>
			<p>Keep the source version and parent headings with every chunk; text alone is not enough to locate the authority later.</p>
		</section>
		<section class="visual-panel">
			<h4>2 · REQUIRE SUFFICIENT RETRIEVAL</h4>
			<p><strong>Exact identifiers:</strong> lexical path<br /><strong>Meaning and paraphrase:</strong> dense path<br /><strong>Reranked evidence:</strong> P-017, P-042</p>
			<p>No passage clears the retrieval gate? Refuse before generation rather than invite a plausible legal answer.</p>
		</section>
		<section class="visual-panel">
			<h4>3 · BUILD THE CLAIM LEDGER</h4>
			<p><strong>Claim C-1 &rarr; P-017:</strong> supported<br /><strong>Claim C-2 &rarr; P-042:</strong> unsupported</p>
			<p>A citation that exists is not yet evidence. Check whether the cited passage supports its atomic claim; strip, retrieve again, or refuse when it does not.</p>
		</section>
		<section class="visual-panel">
			<h4>4 · RELEASE WITH LINEAGE</h4>
			<p><strong>Answer:</strong> supported claims + passage IDs<br /><strong>High stakes:</strong> attorney review<br /><strong>Audit:</strong> query, retrieval, draft, verdicts, final output</p>
			<p>The durable record preserves what the system saw, what it rejected, which source version it used, and who approved release.</p>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> follow one claim backward from release. It is defensible only if its citation resolves to an exact passage in the recorded corpus version and that passage supports the claim. Missing retrieval stops generation; failed support triggers removal, another retrieval, refusal, or attorney review, not a confidence-shaped citation. This original ledger was checked against <a href="https://arxiv.org/abs/2408.10343">LegalBench-RAG</a>, <a href="https://aclanthology.org/2023.emnlp-main.398/">Gao et al.'s citation-evaluation work</a>, and the <a href="https://doi.org/10.6028/NIST.AI.600-1">NIST Generative AI Profile</a>.</figcaption>
</figure>

## What an L5 answer sounds like

> "I'd build the standard 7-component RAG architecture (chunking, embedding, lexical index, query understanding, hybrid retrieval, reranking, generation), with these legal-specific changes:
>
> **Chunking**: heading-aware semantic chunking. Each chunk inherits parent headings (Statute name, Section, Subsection) as a prefix. Sliding window with overlap to handle definitions split across boundaries.
>
> **Hybrid retrieval**: BM25 weighted heavier than embedding similarity. Legal queries often look up specific cited authorities; lexical exact-match dominates.
>
> **Reranker**: a strong cross-encoder, possibly fine-tuned on legal relevance judgments.
>
> **Generation**: structured output with required citations. Every claim in the answer must include `[passage_id]`. Post-generation verification: an LLM-judge that takes (claim, cited passage) and verifies entailment. Flag or strip unsupported claims.
>
> **Refusal**: explicit instruction in the prompt: 'If the answer is not supported by the retrieved passages, respond: I cannot answer this from the available sources.' Tune retrieval thresholds: if top-K passages are below a similarity floor, route to refusal directly without generation.
>
> **Eval**: separate retrieval metrics (recall at K on legal Q&A pairs) from generation metrics (citation accuracy, faithfulness). Domain-expert review on a stratified sample."

## What an L6 answer adds

> "...considerations for the regulated context:
>
> **Audit trail**: every query, every retrieval, every generated answer logged with full lineage. Required for regulatory review and for incident investigation.
>
> **Versioned source corpus**: legal texts change (amendments, new case law). The corpus is a tracked artifact with versioning; answers should cite the version of the source they used.
>
> **Domain-fine-tuned embeddings**: out-of-the-box embeddings underperform on legal vocabulary. Fine-tune on (legal-query, relevant-passage) pairs, possibly synthetic via an LLM, possibly mined from real usage logs.
>
> **Hierarchical retrieval for long documents**: a contract or case law document can be hundreds of pages. First retrieve at the document level (which contract is relevant), then at the section level within. Reduces irrelevant chunks polluting the context.
>
> **Adversarial robustness**: users will ask questions designed to elicit confident wrong answers. Test against adversarial query sets; add refusal training data targeted at these patterns.
>
> **Human-in-the-loop for high-stakes outputs**: route legal-research outputs to attorney review; route contract-analysis outputs to a confidence tier (auto-accept, review, reject) based on retrieval and verifier confidence."

## Tells that get you a strong-hire vote

- You bring up **structure-aware chunking** (statutes have hierarchy, contracts have sections).
- You weight **BM25 heavier** than dense embeddings for legal vocabulary.
- You build in **mandatory citations + verifier**.
- You make **refusal a first-class output**.
- You discuss **audit trails** and **source versioning** as regulatory requirements.

## Tells that get you down-leveled

- Generic RAG with no domain-specific changes.
- Embedding-only retrieval (will fail on case names, statute numbers).
- No mention of citation verification.
- Treating refusal as a failure mode rather than a valid output.

## Common follow-up

"How would you measure citation accuracy?"

The L6 answer:

> "Two-step. First, decompose each generated answer into atomic claims. Second, for each claim, verify against its cited passage using a separate LLM judge calibrated on a labeled set. Report 'fraction of claims supported by their citations' as the citation-accuracy metric. Track separately from answer-quality metrics: a model can have high answer quality and bad citations, or vice versa, and they have different remediation paths."

---

*Related: [Designing a RAG system that actually works](/guides/designing-rag-that-works/), [How do you handle hallucinations in production?](/questions/handle-hallucinations-in-production/), [LLM Evals essay](/guides/llm-evals-the-hardest-part/).*
