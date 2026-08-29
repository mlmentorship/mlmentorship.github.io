# Editorial style

**Updated:** August 28, 2026

**Applies to:** all concepts, questions, and guides

## Core rule

Use the pyramid method. State the answer or main claim first. Add evidence, mechanism, limits, and examples after it.

Write for an experienced technical reader who may not be a native English speaker. Keep the technical content precise and the language plain.

## Concepts

Start with `## Summary`.

The Summary should do two jobs:

1. Define the term or method.
2. State the practical consequence, main use, or main limit.

Then explain the mechanism. Put equations, variants, examples, tradeoffs, and interview guidance after the Summary.

Do not add a separate “Why it matters” section. Its useful content belongs in the Summary.

## Questions

Before the first section:

1. Name the interview round when known.
2. State what a basic answer covers.
3. State what a senior answer adds.

Then give the answer in increasing depth. Use level-specific sections when level changes the expected evidence. End with concrete strong and weak signals, followed by one or more changed-condition questions.

## Guides

State the thesis before the first section. Organize the rest around a decision, procedure, or small set of claims. Each section should support the opening thesis.

Long guides may use different section names because their subjects differ. They still need a direct opening and a visible argument.

## Language

- Prefer one claim per sentence.
- Prefer active voice when the actor is known.
- Use “use” instead of “utilize.”
- Define an acronym on first use unless it is universal for the intended reader.
- Keep necessary technical terms. Explain them with ordinary words.
- Use concrete subjects and verbs: “the cache stores,” “the test measures,” “the policy changes.”
- Separate long lists from prose.
- Keep descriptions at 32 words or fewer.
- Keep ordinary prose sentences at 48 words or fewer. Split dense reasoning into several sentences or a list.

## Avoid generated-prose patterns

Do not use:

- em dashes;
- “Why it matters” headings;
- “the key insight,” “the key point,” or “the distinction matters”;
- “the difference matters,” including versions prefixed with “this” or “that”;
- “here’s” framing;
- “not just X, but also Y” and “not merely X, but Y”;
- “seam,” “seamless,” or “seamlessly” as product metaphors;
- “delve,” “at its core,” “serves as,” or “unlock” as filler;
- vague field metaphors such as “the modern method landscape”;
- unsupported superlatives such as “the single most important”;
- inflated words such as “pivotal,” “revolutionary,” “groundbreaking,” or “transformative.”

State the underlying fact instead. For example, replace “the key insight is that communication dominates” with “communication dominates runtime at this batch size.”

Use `Related:` consistently for closing links. Do not leave coming-soon markers after a target exists.

Some words have valid technical meanings. “Loss landscape,” “robust estimator,” and statistical leverage are allowed when they name a defined concept.

## Punctuation

Use periods for separate claims. Use a colon before a short explanation or list. Use semicolons only when two clauses are closely related. Do not use an em dash as a pause or substitute for a sentence boundary.

## Automated checks

`npm run check:style` verifies:

- no em dashes in repository text;
- every concept starts with `## Summary`;
- every Summary contains a substantive answer, not an empty template heading;
- every question and guide has a thesis before its first section;
- descriptions contain at most 32 words;
- ordinary prose sentences contain at most 48 words;
- high-confidence generated-prose patterns are absent.

The checker is a floor, not an editor. Technical accuracy, logical order, evidence quality, and natural wording still require review.
