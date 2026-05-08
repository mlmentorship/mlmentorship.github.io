---
title: "Designing a RAG system that actually works"
description: "RAG fails most often at retrieval, not generation. A practitioner's guide to the architecture, the failure modes, and what production teams actually do in 2026."
date: "2026-05-07"
draft: false
tags: ["guides"]
category: "guides"
---


**RAG quality is dominated by retrieval, not by the LLM.** Most production RAG systems that underperform have the same root cause: the team treated it as a generation problem when it is actually a retrieval problem with a generation step at the end.

This essay assumes RAG is the right tool for the problem (see [When would you fine-tune vs prompt vs RAG?](/questions/fine-tune-vs-prompt-vs-rag/) for that decision). What follows is the architecture and operational discipline that separate "we have a RAG system" from "our RAG system works."

## The mental model

A RAG system is a small information-retrieval system bolted to an LLM. The LLM generates answers; everything else is IR.

IR is a 60-year-old field with hard-won lessons that LLM teams routinely re-discover:

- Retrieval quality dominates final quality. If the right document isn't in the top-K, no amount of prompt engineering recovers it.
- Embeddings are not a panacea. Hybrid retrieval (lexical + dense) almost always beats either alone.
- Reranking is the cheapest big quality win available. Most teams skip it.
- Chunking strategy has more impact on quality than embedding model choice. Most teams pick chunk size by intuition and never tune it.

## The reference architecture

A production RAG system in 2026 has roughly seven components:

1. **Document ingestion and chunking.** Parse source documents into retrievable units.
2. **Embedding and indexing.** Compute vectors for each chunk; store them in a vector store.
3. **Lexical index.** A traditional keyword index (BM25, etc.) over the same chunks.
4. **Query understanding.** Reformulate or expand the user's query before retrieval.
5. **Hybrid retrieval.** Run both dense (vector) and lexical retrieval; combine.
6. **Reranking.** Re-order top ~50-200 retrieval candidates down to top ~5-10 for the LLM.
7. **Generation.** The LLM, with the retrieved chunks in context, produces the final answer with citations.

Plus the operational layer: an eval pipeline (see [LLM Evals](/guides/llm-evals-the-hardest-part/)), monitoring, and a feedback loop from production traffic back into the eval set.

Skipping components 4-6 is the single most common failure.

## Component-by-component, with the things that actually matter

### Chunking

The naive approach: split documents into 512-token chunks. This is wrong almost everywhere.

Better approaches, in rough order of sophistication:

- **Semantic chunking**: split on paragraph or section boundaries, then merge to a target token count.
- **Heading-aware chunking**: each chunk inherits the parent document's title and section headers as a prefix, massively improves retrieval for any structured document (technical docs, legal contracts, academic papers).
- **Sliding-window with overlap**: 256-token chunks with 50-token overlap, so context near boundaries isn't lost.
- **Hierarchical chunking**: keep chunks at multiple granularities (paragraph + section + document) and retrieve at the right level for each query.

The best practice depends on your document type. For free-form conversational data, sliding-window is fine. For structured documents, heading-aware semantic chunking is dramatically better. For long technical documents, hierarchical is worth the complexity.

Tune chunk size empirically against your eval set. The right size varies more by domain than people expect; retuning it is often a high-impact, low-effort change.

### Embedding model

The decision space is smaller than people think. Pick a recent strong open model (BGE, E5, GTE, NVIDIA embed family) or a managed option (OpenAI, Cohere, Voyage). The differences between modern embedding models on standard benchmarks are small. The difference between dense embeddings alone and a hybrid setup (dense + BM25) is much larger.

Two non-obvious points:

- **Domain-specific embeddings matter for specialized text.** Out-of-the-box embeddings underperform on legal, medical, or code search. Fine-tuning embeddings on your domain (with synthetic queries) typically helps materially.
- **Embedding model choice locks you in.** Switching means re-embedding your entire corpus. Pick something you'll commit to; don't churn weekly.

### Lexical retrieval (don't skip this)

The most common architecture mistake in RAG is using only dense retrieval. Lexical retrieval (BM25 in particular) wins on:

- Exact-match queries (product codes, error messages, named entities)
- Out-of-distribution terms not well represented in embedding training data
- Acronyms, abbreviations, and domain jargon
- Numeric references ("CVE-2023-1234")

Hybrid retrieval (dense + lexical, fused with reciprocal rank fusion or weighted score combination) consistently beats either alone in production. Set this up from day one. It's a few hundred lines of code; it pays for itself quickly.

### Query understanding

The user's query is rarely the right query for retrieval. A few transformations that help:

- **Query expansion / HyDE**: have an LLM generate a hypothetical answer to the query, then embed *the hypothetical answer* (not the query) and search. Counterintuitively effective; works because the answer is closer in embedding space to actual answer documents than the question is.
- **Query decomposition**: for complex queries ("how does our refund policy interact with the EU consumer protection law"), break into sub-queries, retrieve for each, combine. Especially important for multi-document RAG.
- **Query rewriting**: rephrase ambiguous queries into clearer ones. Often as simple as "given this conversation history, rewrite the user's last message into a self-contained query."

Without query understanding, retrieval depends entirely on user phrasing. With it, retrieval is robust.

### Reranking (the highest-leverage step)

After retrieval returns the top ~50-200 candidates, a reranker scores each (query, candidate) pair more carefully and re-orders.

Why it works: a bi-encoder embedding model has to encode query and document independently, so it can only model coarse similarity. A cross-encoder reranker sees both at once and can model fine-grained relevance, which terms matched, which paraphrases hold, whether the candidate actually answers the question.

Modern rerankers (Cohere Rerank, BGE Reranker, Jina Reranker, etc.) substantially improve top-K precision for modest additional latency, and are typically the single best quality lever in the pipeline.

Skipping reranking is the #1 cause of RAG underperformance.

### Generation

Last and least. By this point most of the work is done. Things that still matter:

- **Cite-by-passage prompting**: instruct the model to cite which retrieved passage each claim comes from, with passage IDs. Improves trustworthiness and makes hallucinations easier to catch.
- **Grounding instructions**: explicitly tell the model "if the answer is not in the provided documents, say so." Without this, models will hallucinate confidently.
- **Context window management**: don't dump 50 chunks into context. Top 5-10 (after reranking) usually beats top 50, both in quality (less distraction) and cost.
- **Structured output when applicable**: for many use cases (Q&A with citations, extraction), structured JSON output is strictly better than free-form prose.

## Common failure modes

In rough frequency order:

1. **Retrieval is bad and team doesn't realize it.** They evaluate the end-to-end RAG output and try to improve it by changing prompts. They never look at whether the right document was even retrieved. **Fix**: separately evaluate retrieval (hit rate at K, MRR) before evaluating generation.

2. **Chunk size is wrong.** Default 512-token chunks for documents where 200 or 1024 would have been right. **Fix**: tune empirically.

3. **No reranker.** Team uses raw bi-encoder retrieval and wonders why precision is low. **Fix**: add a reranker. This is usually a one-day improvement worth 10pp.

4. **No lexical retrieval.** Team relies entirely on embeddings and fails on exact-match queries. **Fix**: add BM25, fuse with reciprocal rank fusion.

5. **Eval set is too small or doesn't reflect real queries.** Team has 30 hand-crafted Q&A pairs, none of which look like real user queries. **Fix**: build the eval set from production traffic samples (see [LLM Evals](/guides/llm-evals-the-hardest-part/)).

6. **Citations don't actually verify.** Model cites passages that don't support its claims. **Fix**: add a citation-verification step where you check each citation against the retrieved passage; flag ungrounded claims.

7. **Stale index.** Team indexes once, then never re-indexes as documents change. **Fix**: incremental indexing pipeline from day one.

8. **Chunking strips structure.** PDF parsing loses headings, tables, lists, and the chunks become decontextualized text. **Fix**: use a structure-aware parser; preserve heading hierarchy.

## What an interviewer expects you to discuss

If asked to "design a RAG system" in a senior loop, the answer should hit:

- The 7-component reference architecture (or a defensible variant).
- Hybrid retrieval (dense + lexical), with explanation of why.
- A reranker as a separate stage.
- Query understanding, especially HyDE or decomposition for hard queries.
- An eval framework with separate retrieval and generation metrics.
- A specific failure mode you've seen and how you fixed it (this is the L6 signal).
- Cost and latency considerations, reranking is fast but indexing is expensive; embedding-update strategies for evolving corpora.

If you list "embedding model + vector store + LLM" and stop there, you've signaled L4 understanding regardless of how good the embedding model is.

## Tools to consider in 2026

- **Vector store**: pgvector for &lt; 10M docs, Qdrant or Weaviate for larger, Vespa if you need both lexical and dense in one system.
- **Embeddings**: BGE-M3 or NVIDIA NV-Embed for open; Cohere v3 or OpenAI text-embedding-3-large for managed.
- **Reranker**: Cohere Rerank, BGE Reranker v2, or a fine-tuned in-house cross-encoder.
- **Lexical**: Elasticsearch / OpenSearch / Tantivy, whatever your team already operates.
- **Orchestration**: don't use LangChain. Use a thin custom layer or LlamaIndex if you want managed primitives.
- **Eval**: Inspect AI, Promptfoo, or Braintrust for the framework; pytest for the lightweight version.

The choice of vendor matters less than the discipline of running them well. Excellent RAG systems exist on the simplest stacks; the most expensive stacks don't guarantee quality.

## The discipline that separates the good teams

Three habits, in order of importance:

1. **They evaluate retrieval and generation separately.** Hit rate at K, MRR, recall at K for retrieval; faithfulness, citation accuracy, answer quality for generation. Both metrics, every release.
2. **They watch the failure tail, not the average.** A 95% top-1 hit rate hides a catastrophic 5% where users get the wrong document and a hallucinated answer. The 5% is what loses trust.
3. **They look at outputs.** Same as the LLM evals advice: read the actual responses every week. The metrics tell you what changed; the outputs tell you what's actually happening.

Build these habits before you build anything fancy. The fancy parts are easy; the discipline is hard.

---

*Related: [LLM Evals: The hardest part of shipping LLMs](/guides/llm-evals-the-hardest-part/), [When would you fine-tune vs prompt vs RAG?](/questions/fine-tune-vs-prompt-vs-rag/).*
