# Content style audit

**Date:** August 29, 2026

**Scope:** all 283 published Markdown entries

## Result

The library now follows one category-aware editorial contract:

- 185 of 185 concepts start with a Summary that places the definition and practical stakes together.
- 85 of 85 questions state the answer-level framing before the detailed sections.
- 13 of 13 guides state a thesis before the detailed sections.
- 283 of 283 descriptions are 32 words or fewer.
- No ordinary prose sentence exceeds the 48-word automated limit.
- No em dash appears in the repository text.

The canonical rules are in [EDITORIAL_STYLE.md](EDITORIAL_STYLE.md).

## What the audit found

The technical structure was already strong, but the prose templates were uneven.

- 163 concepts used “One-line definition” as the first heading.
- 156 entries had a separate “Why it matters” heading.
- Five entries labeled a claim as “the key insight,” “the key point,” or “the key difference.”
- Eight entries used unsupported “single most important” or “single highest-leverage” claims.
- Three descriptions used “here’s” framing.
- Two entries used the full “not just X, but Y” pattern.
- Several titles or links used “landscape” as a vague name for a set of methods.
- A sentence-length scan identified dense passages that combined many independent claims.
- No use of “seam,” “seamless,” “delve,” or an em dash was present.

## Changes made

### Pyramid order

Every concept now starts with `## Summary`. For 143 concepts, the separate stakes section was merged into that opening Summary. The BatchNorm comparison was folded into its concrete axis comparison. Ten concepts that opened with “Why it matters” were renamed directly. Two question headings were replaced with specific technical outcomes.

Question and guide openings were retained where they already led with the expected answer or thesis.

### Direct language

Canned labels and unsupported superlatives were replaced with concrete claims. Examples:

- “the key insight” became the technical result;
- “single highest-leverage” became the specific role of the method;
- “the distinction matters” became the consequence of the difference;
- “here’s the mental model” became a direct description;
- “not just X, but Y” became two claims.

Valid technical terms remain. The audit did not remove “loss landscape,” “robust estimator,” or other defined domain language.

### Simpler sentences

Dense sentences were split into short claims or ordered lists. Long page descriptions were shortened and moved to answer-first wording.

## Enforcement

The editorial checker now enforces the structure, description limit, sentence limit, banned punctuation, and high-confidence generated-prose patterns. It runs through both `npm run check` and `npm run build`.

The August 29 re-audit expanded the permanent rules to catch standalone “the distinction matters” and “the difference matters” phrasing, contracted note filler, stale coming-soon markers, and inconsistent Related labels. All 283 entries pass the expanded contract.

The checks are intentionally narrow. They reject repeated templates with low ambiguity while leaving technical vocabulary and author judgment intact.
