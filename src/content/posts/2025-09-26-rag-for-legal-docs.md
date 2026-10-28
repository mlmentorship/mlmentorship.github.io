---
title: "Design a RAG system for legal documents"
description: "Legal RAG amplifies every standard RAG concern: precise citations, no hallucinations, regulated domain, dense documents with structure. The senior answer addresses each."
date: "2025-09-26"
draft: false
tags: ["questions"]
category: "questions"
---


> *Asked in: legal-tech, fintech, and regulated-domain LLM interviews.*

Design a document-grounded legal information service whose answers can be checked against the correct authorized source passages.

## Problem

**Context.** A hypothetical organization searches legal documents that vary by jurisdiction, effective date, structure, and access permissions.
Users need answers with source support, including when documents conflict or the evidence is insufficient.

**Task.** Specify ingestion, versioning, retrieval, answer generation, citation verification, abstention, and evaluation.
Explain how a reviewer can trace a claim to the applicable document version and passage.

**Constraints.** Enforce access permissions before evidence reaches generation, preserve provenance, and distinguish superseded material from current authority.
Do not present unsupported synthesis as verified legal advice; define escalation and audit requirements.

## Domain-specific concerns vs generic RAG

What changes for legal:

1. **Citation precision is non-negotiable.** Every claim must trace to a verifiable passage. A hallucinated case citation is a malpractice risk.
2. **Documents are highly structured.** Statutes, contracts, case law have hierarchical structure (titles, sections, paragraphs, sub-clauses). Chunking that strips structure loses retrieval signal.
3. **Vocabulary is specialized.** General-purpose embeddings underperform; BM25 with legal-jargon-aware tokenization often outperforms naive embeddings on exact-match queries (case names, statute numbers, defined terms).
4. **Refusal is a feature.** "I can't answer this from the available sources" is the right answer when retrieval is uncertain. Better than a confident wrong answer.
5. **Audit trails are required.** Every retrieved passage and every generated answer needs to be logged for review.

<!-- visual:legal-rag-claim-lineage -->
<figure class="learning-figure editorial-illustration" aria-labelledby="legal-rag-claim-lineage-edition-title"><span class="visual-fragment-anchor" id="legal-rag-claim-lineage-title" aria-hidden="true"></span><p class="visual-title" id="legal-rag-claim-lineage-edition-title">A resolvable citation can leave the claim unsupported</p><svg viewBox="0 0 360 682" role="img" aria-labelledby="legal-rag-claim-lineage-edition-svg-title legal-rag-claim-lineage-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="legal-rag-claim-lineage-edition-svg-title">A resolvable citation can leave the claim unsupported</title><desc id="legal-rag-claim-lineage-edition-svg-desc">Fictional jurisdiction J authority version 2026-08-15 section four contains P-017 requiring filing within ten days and P-042 saying the clerk records receipt. C1 repeats the ten-day deadline and is supported. C2 infers guaranteed approval from receipt and is held. Release separately requires current jurisdiction/time applicability and required attorney approval.</desc><text x="180" y="22" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Fictional jurisdiction J; not legal advice</tspan></text><g data-editorial-node><rect x="20" y="57" width="320" height="95" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="180" y="81" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Authority V2026-08-15 &gt; section 4 &gt;</tspan><tspan x="180" dy="19">P-017</tspan><tspan x="180" dy="19">&quot;Notice must be filed within 10 days.&quot;</tspan><tspan x="180" dy="19">P-042: &quot;The clerk records receipt.&quot;</tspan></text></g><g data-editorial-node><rect x="20" y="186" width="320" height="72" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.3"></rect><text x="180" y="208" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">C1: notice deadline is 10 days.</tspan><tspan x="180" dy="19">P-017 directly supplies the bound:</tspan><tspan x="180" dy="19">SUPPORT</tspan></text></g><g data-editorial-node><rect x="20" y="290" width="320" height="72" rx="3" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.3"></rect><text x="180" y="312" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">C2: receipt guarantees approval.</tspan><tspan x="180" dy="19">P-042 says nothing about approval:</tspan><tspan x="180" dy="19">HOLD</tspan></text></g><path d="M180 152L180 182" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75"></path><path d="M176.64402122977057,175.8569220667674L180,182L183.35597877022943,175.8569220667674" style="fill:none;stroke:var(--viz-output-stroke);stroke-width:1.75"></path><g data-editorial-node><rect x="20" y="397" width="320" height="138" rx="3" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.3"></rect><text x="180" y="433" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Release only supported C1, IF</tspan><tspan x="180" dy="19">jurisdiction + effective-time checks</tspan><tspan x="180" dy="19">pass</tspan><tspan x="180" dy="19">AND required attorney approval is</tspan><tspan x="180" dy="19">recorded</tspan></text></g><text x="180" y="579" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">C2 is removed or escalated, not rescued by</tspan><tspan x="180" dy="19">its citation.</tspan></text><text x="180" y="630" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Keep query, exact passages, version, verdicts,</tspan><tspan x="180" dy="19">and reviewer.</tspan></text></svg><figcaption><strong>Read it this way:</strong> P-017 supplies C1's exact bound. P-042 does not establish approval, so C2 is removed or escalated. Support, applicability, and release authority remain separate. This is fictional, not legal advice. <span class="visual-reader-check" data-reader-check="legal-rag-claim-lineage"><strong>Check:</strong> Which exact source span supports C1? Why is a resolvable P-042 citation insufficient to release C2?</span></figcaption></figure>

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
