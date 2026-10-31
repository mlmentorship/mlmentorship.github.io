---
title: "RAG: retrieval-augmented generation"
description: "Trace an authorized, versioned passage from corpus to answer, then distinguish citation existence from actual claim support."
date: "2025-09-02"
draft: false
tags: ["concepts"]
category: "concepts"
---


## Summary

Retrieval-augmented generation supplies external evidence as model context without requiring that evidence to be stored in model weights.
Success requires authorized evidence to survive retrieval and selection, then support the actual response claims.
A resolving citation alone proves neither support nor truth.

## The architecture

Offline, parse documents into chunks while retaining source IDs, versions, boundaries, and access metadata.
Build the chosen indexes and update or delete entries when their sources change.
Keep each plan name and retention value together; preserve heading or neighbor references when splitting would separate them.

Online, apply the requester's access policy, retrieve a bounded candidate set, select bounded context, and generate claim-linked citations.
The pictured hybrid retrieval and reranking are possible choices, not mandatory stages for every corpus or latency budget.
Query rewriting can also change intent, so evaluate it rather than assuming improvement.

For a hypothetical fully authorized corpus:

| Passage | Version | Text |
|---------|---------|------|
| P1 | v1 | Plan A retains logs for seven days. |
| P2 | v1 | Plan B retains logs for thirty days. |

Ask about plan B, with candidate limit two and context limit one.
Retrieval returns P1/P2; selection retains P2 with its version and an allowed access decision.
The answer "thirty days" linked to P2 follows the evidence path.
A seven-day answer citing P2 resolves to an existing passage but fails support.
If P2 never enters the candidate set, merely reranking P1 cannot recover it.

<!-- visual:rag-index-request-paths -->
<figure class="learning-figure editorial-illustration" aria-labelledby="rag-index-request-paths-edition-title"><span class="visual-fragment-anchor" id="rag-paths-title" aria-hidden="true"></span><p class="visual-title" id="rag-index-request-paths-edition-title">Trace the answer back through retrieval to one versioned source chunk</p><div class="visual-argument" data-visual-argument="rag-index-request-paths" data-argument-kind="ledger" data-argument-composition="transform"><svg viewBox="0 0 360 376" role="img" aria-label="Keep the answer connected to one versioned source chunk. A stable chunk identity and version connect the retrieved evidence to its source. A citation string alone is not that lineage receipt."><title>Keep the answer connected to one versioned source chunk</title><desc>A stable chunk identity and version connect the retrieved evidence to its source. A citation string alone is not that lineage receipt.</desc><circle cx="23" cy="24" r="10" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><text x="23" y="28" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="23" dy="0">1</tspan></text><text x="44" y="28" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="44" dy="0">Versioned source and indexed chunk</tspan></text><g data-proof-token="doc D v3" data-proof-row="0" data-proof-position="0" data-proof-box="44,60,142,48"><rect x="44" y="60" width="142" height="48" rx="0" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="115" y="87" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="115" dy="0">doc D v3</tspan></text></g><g data-proof-token="chunk C7" data-proof-row="0" data-proof-position="1" data-proof-box="192,60,142,48"><rect x="192" y="60" width="142" height="48" rx="0" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.5"></rect><text x="263" y="87" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="263" dy="0">chunk C7</tspan></text></g><path d="M23 41L23 118" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><path d="M19 112L23 118" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M27 112L23 118" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><circle cx="23" cy="140" r="10" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.75"></circle><text x="23" y="144" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="23" dy="0">2</tspan></text><text x="44" y="144" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="44" dy="0">Request retrieves that same chunk</tspan></text><g data-proof-token="C7" data-proof-row="1" data-proof-position="0" data-proof-box="44,176,142,48"><rect x="44" y="176" width="142" height="48" rx="0" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.5"></rect><text x="115" y="203" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="115" dy="0">C7</tspan></text></g><g data-proof-token="v3" data-proof-row="1" data-proof-position="1" data-proof-box="192,176,142,48"><rect x="192" y="176" width="142" height="48" rx="0" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.5"></rect><text x="263" y="203" style="font-size:14px;font-weight:400;text-anchor:middle;fill:var(--c-text)"><tspan x="263" dy="0">v3</tspan></text></g><path d="M23 157L23 234" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75;stroke-dasharray:4 4"></path><path d="M19 228L23 234" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M27 228L23 234" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><circle cx="23" cy="256" r="10" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="23" y="260" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="23" dy="0">3</tspan></text><text x="44" y="260" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="44" dy="0">Answer evidence record</tspan></text><g data-proof-token="claim" data-proof-row="2" data-proof-position="0" data-proof-box="44,292,142,48"><rect x="44" y="292" width="142" height="48" rx="0" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:2"></rect><text x="115" y="321.5" style="font-size:20px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="115" dy="0">claim</tspan></text></g><g data-proof-token="C7 v3" data-proof-row="2" data-proof-position="1" data-proof-box="192,292,142,48"><rect x="192" y="292" width="142" height="48" rx="0" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:2"></rect><text x="263" y="321.5" style="font-size:20px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="263" dy="0">C7 v3</tspan></text></g></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content"><svg viewBox="0 0 360 624" role="img" aria-labelledby="rag-index-request-paths-edition-svg-title rag-index-request-paths-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="rag-index-request-paths-edition-svg-title">Follow one versioned chunk from the corpus into the answer</title><desc id="rag-index-request-paths-edition-svg-desc">Fictional document D version three has c1 stating thirty-day returns and c2 giving a support email. Lexical retrieval returns c1 and dense retrieval c1/c2. Fusion retains both; the optional reranker removes c2 from answer context. The answer cites D@v3:c1. Corpus refresh need not retrain model weights, and authorization and support still apply.</desc><text x="180" y="22" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Toy authorized request: what is the return</tspan><tspan x="180" dy="19">window?</tspan></text><g data-editorial-node><rect x="25" y="63" width="310" height="72" rx="3" style="fill:var(--viz-input-bg);stroke:var(--viz-input-stroke);stroke-width:1.3"></rect><text x="180" y="85" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">OFFLINE document D@v3</tspan><tspan x="180" dy="19">c1: &quot;Returns within 30 days.&quot;</tspan><tspan x="180" dy="19">c2: &quot;Contact support by email.&quot;</tspan></text></g><path d="M180 135L180 165" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M176.64402122977057,158.8569220667674L180,165L183.35597877022943,158.8569220667674" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><g data-editorial-node><rect x="25" y="169" width="310" height="72" rx="3" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.3"></rect><text x="180" y="191" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Same chunk IDs in both indexes</tspan><tspan x="180" dy="19">lexical -&gt; c1; dense -&gt; c1,c2</tspan><tspan x="180" dy="19">fusion -&gt; {c1,c2}</tspan></text></g><path d="M180 241L180 271" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M176.64402122977057,264.85692206676737L180,271L183.35597877022943,264.85692206676737" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><g data-editorial-node><rect x="25" y="275" width="310" height="72" rx="3" style="fill:var(--viz-focus-bg);stroke:var(--viz-focus-stroke);stroke-width:1.3"></rect><text x="180" y="297" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">ONLINE reranker selects c1</tspan><tspan x="180" dy="19">c2 removed from context</tspan><tspan x="180" dy="19">request authorization retained</tspan></text></g><path d="M180 347L180 377" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M176.64402122977057,370.85692206676737L180,377L183.35597877022943,370.85692206676737" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><g data-editorial-node><rect x="25" y="381" width="310" height="72" rx="3" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.3"></rect><text x="180" y="412.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Answer: &quot;30 days [D@v3:c1]&quot;</tspan><tspan x="180" dy="19">exact source version preserved</tspan></text></g><text x="180" y="499" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">A later corpus v4 changes indexed evidence,</tspan></text><text x="180" y="528" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">not necessarily the language model's weights.</tspan></text><text x="180" y="574" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">A citation still needs an independent</tspan><tspan x="180" dy="19">support check.</tspan></text></svg><div class="visual-derivation-caption">The same c1 identity survives offline indexing, retrieval, selection, and citation. The reranker removes c2 here. Preserve access and source version; a citation alone still does not prove support.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="rag-index-request-paths"><strong>Check:</strong> Which offline chunk becomes the final cited evidence? Which online stage removed its competing candidate?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="rag-index-request-paths">A stable chunk identity and version connect the retrieved evidence to its source. A citation string alone is not that lineage receipt.</span></figcaption></figure>

### Change Matching Needs

**New task:** Retrieve both exact error identifiers and paraphrased descriptions.

**Picture and change:** Compare lexical and dense candidate sets, then fuse deduplicated document identities with their provenance.
Tune candidates, fusion, and limits on development queries, not the final test set.

**Check:** A lexical `ERR-42` match remains eligible rather than being vetoed solely by a low dense score.
Inspect whether subsequent candidate/context cutoffs retain its supporting passage.

**Use when:** Query slices need complementary matching signals, rather than a universally mandated hybrid pipeline.

### Change Evidence Availability

**New task:** The authorized corpus lacks the answer.

**Picture and change:** Return the defined no-evidence or partial-answer outcome instead of inventing a supporting passage.

**Check:** Neither P1 nor P2 supports a claim about plan C.

**Use when:** Coverage or authorization prevents a grounded answer.

Measure retrieval coverage at the actual candidate limit separately from context support, answer correctness, citation resolution, and abstention behavior.
Use judged query slices and report denominators; first-hit utility and relevant-passage coverage are distinct targets.
Time and cost budgets must include retrieval, selection, generation, and any verification actually used.

Revoking access or deleting a document must invalidate old index entries and context caches before serving its text again.
Preserve source/version IDs through citations and recheck request authorization; a cached passage is not a permanent access grant.
Retrieved instructions are untrusted data, not policy.

<!-- learning:core-end -->
<div data-learning-extension>

## What an interviewer expects you to say

Explain one request's evidence path, access decision, limits, and unsupported-answer outcome before discussing optional components.

## Common confusions

Evidence availability does not guarantee evidence use, correct reasoning, or faithful citations.
Embedding changes, reranking, and context selection need workload-specific evaluation, not fixed percentage-point promises.

## Failure modes (in frequency order)

The retained heading is not a measured frequency ordering.
Distinguish missing evidence, lost chunk boundaries, stale versions, access violations, unsupported claims, and distribution-shifted evaluation queries.

## When RAG is and isn't the right tool

Choose retrieval when selecting external, changing evidence fits the task.
Compare direct context, structured queries, prompting, or fine-tuning against the required behavior and evidence lifecycle.

## Tools (mid-2026)

Select storage and orchestration by measured recall, access filtering, update latency, cost, and operational constraints.
No specific framework or document-count threshold follows from the RAG mechanism.

## Why interviewers ask

The useful distinction is between finding evidence and producing claims that the evidence supports.

---

*Related essay: [Designing a RAG system that actually works](/guides/designing-rag-that-works/) for the long form. Related interview: [When would you fine-tune vs prompt vs RAG?](/questions/fine-tune-vs-prompt-vs-rag/).*

</div>
