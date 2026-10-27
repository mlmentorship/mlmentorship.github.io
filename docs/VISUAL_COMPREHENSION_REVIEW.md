# Visual comprehension review

## Current outcome

Expert review is complete for 392 registered visuals across 391 articles and 16 supplemental Mermaid diagrams.
Eleven registered figures and seven supplemental diagrams received substantive revisions.
Twenty-four registered Mermaid figures received missing visible titles; those repairs are not 24 redesigns.
All 16 supplemental diagrams now have stable supplemental identities, visible titles, accessible descriptions and direct captions.
Nine supplemental diagrams retain further refinement recommendations.
The original registered recommendations remain an improvement backlog, not a claim that every figure was redesigned.
The lesson-authoring checkpoint remains 252 of 391; this review did not author the remaining lessons.

Current verification passes 784 registered desktop/phone renders and 80 supplemental five-mode renders.
The eleven revised registered figures also have 55 accepted five-mode cases, with all frames checked for the two changed coding traces.
Native originals were inspected for all eighteen revised visuals, including selected miss/save/hit states, dark-theme views and a static print sequence.
These are source, arithmetic, rendering and expert-composition checks, not observed learner outcomes or PDF-pagination verification.
Changes remain local and uncommitted; no publication was performed.

## Review standard

Judge each illustration by the relationship a reader can explain or predict from its marks.
Technical correctness, accessibility and rendering are necessary but do not establish understanding.
This is an expert illustration review, not a measured learner-comprehension study.
Do not report an estimated comprehension score as an observed learning improvement.

For every figure, identify the question, visible evidence, likely misconception and a concrete improvement.
Keep correct, useful figures; do not redraw a successful explanation merely to impose a uniform style.
Prefer spatial relationships, quantities, transformations and controlled comparisons over boxes repeating the surrounding prose.
Use the smallest number of states that reveals the mechanism, with a specific inference the reader can check.
Labels, examples and captions must support the drawing rather than contain the explanation missing from it.
Interaction must clarify a relationship; the static and print versions must also explain it.

## Coverage

The registered inventory contains 392 visuals across 391 articles, including 106 coding definitions and 797 authored coding states.
The corpus also contains 16 supplemental Mermaid diagrams outside registered markers; they are separately assessed, not added to the registered count.
There are no Markdown/HTML images, canvas or video elements, or unregistered figure wrappers in the source scan.
All 392 registered visuals have individual assessments: originally 62 structural redesign and 330 retain-and-refine recommendations.
Original priorities are 39 high, 217 medium and 136 low; these remain historical judgments, not current unresolved counts or comprehension scores.
Each decision records a retained strength, specific problem, proposed improvement and reader-inference acceptance question.
The structured decisions are in [data/visual-comprehension-review.json](../data/visual-comprehension-review.json).
Its current reconciliation distinguishes implemented repairs, title-only changes, twenty preexisting source changes and the supplemental assessments.
The original problem/improvement fields are retained as history; current reconciliation supersedes the explicitly repaired observations.
For example, current decoding already has common-scale probability bars, delayed labels already include an early-positive branch, and open graphical-model paths say "may depend."

All 797 coding-state notes were reviewed in source, and all 25 historical desktop contact sheets were directly inspected.
Historical native inspection covered 26 captures across 11 figures, plus six later coding-state captures.
That sample included all-reduce, factorization, roofline, causal masking, PCA, ASR and coding traces.
Current native inspection adds the revised figures; neither inspection campaign covers every mobile original or all 797 states.
Technical render coverage, source review, contact-sheet inspection and native inspection are different forms of evidence.
Preserve article routes, question statements, Python fences, visual IDs and authored frame identities during any later redesign.
Do not change the immutable baseline, completed coding guide or frozen planning package.

## Render observations

### Current qualification

All 391 current articles pass desktop-light and mobile-light checks at actual 1280/390 CSS-pixel widths: 784 registered-figure cases.
The source, built HTML and preview response hashes are bound before and after each batch.
The final manifest reuses 127 earlier article results only where source, HTML, preview, renderer, report and PNG hashes agree exactly.
The revised enterprise figure was freshly captured after the earlier version failed its 7.33px text check.
The 24 new visible titles and one already-present delayed-label title resolve the historical title stops.
The loss-selection and LLM-recommender redraws resolve their historical tiny-text stops.

All 16 supplemental diagrams pass desktop light/dark, exact-390 phone light/dark and print-style checks: 80 cases.
Supplemental qualification exposed tiny system maps at 2.23-5.63px, a 6.75px lifecycle and a 7.82px subset-sum sketch.
Those layouts were simplified or refocused; assertions were not relaxed.
The renderer now requires a persistent, labelled Mermaid SVG, stable geometry and accessible title/description markup.
It avoids triggering another theme repaint immediately before capture.
The 8px screen-text floor is only a minimum technical gate, not evidence that every composition is easy to understand.

### Historical observations

The original strict two-mode run accepted 364 articles and stopped on 27: 25 missing visible titles and two tiny-text diagrams.
The old small-text measurements were 7.24px for loss selection and 7.93px for bounded LLM recommendation.
Those failed attempts remain historical failures, not retroactively converted passes.

A separate diagnostic observer captured all 27 stopped articles at both widths without changing their failed status.
Premature Mermaid captures were rejected after the contact sheets exposed raw diagram source.
The repaired observer requires persistent SVG presence and stable geometry before taking each replacement image.
The stable capture index contains 784 native images for all 392 visuals, plus 25 numbered desktop contact sheets.
Generated capture coverage is not evidence of improved human understanding.
The old memory-backed PNG directory is now absent; its manifest is historical, not a claim that those images remain available.

Native inspection exposes a separate composition problem: important comparisons can require horizontal scrolling even on desktop.
The factorization output and part of the PCA explanation are outside their initial desktop views.
Mobile all-reduce and some retained architecture maps still require panning; these retain refinement recommendations.
Loss selection, speculative decoding and LLM recommendation now use vertical drawings instead of their old wide maps.
Reachable scrolling is not the same as a readable, simultaneously visible explanation.

## Implemented repairs

### Make claims and state identities exact

The first revisions target ambiguous inference, not decorative consistency.
The arithmetic and geometry checks establish the stated examples, not whole-article factual qualification.

| Visual | Implemented clarification | Reader must be able to answer |
| --- | --- | --- |
| `fixed-capacity-portfolio-guardrails` | Shared-scale 100% bars show a six-point transfer without moving protected shares. | Where does the proposed capacity come from? |
| `loss-estimation-vs-decision` | The same outcomes produce different MSE/MAE estimates; expected action cost is separate. | Why can the estimate and action change under different losses? |
| `llm-eval-calibration-loop` | Countable 13/20 joint passes remain below the 70% floor and lead to HOLD. | What happens when the candidate fails? |
| `residual-add-gradient-split` | Explicit function and loss determine both gradient branches and their 2.42 total. | Where does each branch's derivative come from? |
| `speculative-decoding-first-rejection-boundary` | Positive-part target-minus-draft correction is drawn after the first rejection. | Why is ordinary target sampling not the whole correction? |
| `fp16-loss-scaling-ledger` | The ledger separates gradient preservation from FP32 update representability. | What does scaling rescue, and what does it not guarantee? |
| `pad-variable-length-sequences-state` | An actual zero token has a true mask; zero padding has a false mask. | Which zero is data, and which zero is padding? |
| `hallucination-claim-release-gates` | Same-passage claims separate citation, support and consequential release clearance. | Can a supported claim still require escalation? |
| `recsys-llm-bounded-augmentation` | A 100ms call misses a 50ms deadline; fallback and catalog gates remain explicit. | Can a late or invented-ID result bypass the completed response? |
| `enterprise-agent-three-plane-boundary` | One credit call crosses authorization, durable dispatch and authoritative reconciliation. | Does a lost reply justify a second logical credit? |
| `two-sum-state` | The complement query connects to a value/original-index row and the returned indices. | Why does value 4 recover index 1, and why can index 0 not pair with itself? |

### Reveal supplemental decisions

Seven supplemental diagrams now focus on a concrete relationship rather than a component inventory:

- Serving stages distinguish retrieved candidates from actual exposure denominators.
- Coding-product paths distinguish shared context from tool authority and user application.
- Coding-task states distinguish verification from approval and cancellation request from confirmed worker stop.
- Data publication joins build artifacts with authorized use and passing checks, with an explicit HOLD branch.
- Reasoning admission separates a desired model route from reserved compute and KV memory.
- LCS uses actual match/mismatch cells for `AB` versus `ACB`, rather than one ambiguous parent sketch.
- Subset sum adds the current item to the old snapshot before union, exposing why 10 is not created by reusing 5.

The other nine supplemental diagrams received titles, accessible descriptions and captions, not structural-redesign claims.
Their individual open refinements and reader questions are in the structured ledger.

## Remaining priorities

### Preserve exact state identity

The histogram trace needs explicit state identity: a pushed zero sentinel disappears before the final pictured stack.
Distinguish the actual stack from unresolved positive-height candidates, or show the intervening removal.
Keep the replacement-window trace's correct distinction between an invalid current window and an earlier valid best.
Its current and historical maxima must remain visibly different during any layout simplification.

### Draw the mechanism, not a checklist

Architecture and evaluation diagrams often name stages without showing the object, evidence or quantity that changes.
For hierarchical all-reduce, show small illustrative rank values, local partials and the identical final result on every rank.
Put the actual crossing on the slower-network boundary instead of asking the caption to explain the traffic reduction.
Distinguish logical reduction from a particular physical collective algorithm.
For evaluation loops, show one candidate, its observed result, the threshold and the failed-candidate route.
For authority diagrams, trace one proposed action, enforced scope and authoritative outcome evidence across the ownership boundaries.
Do not invent empirical outcomes, personal experience or measured performance to make a diagram look concrete.

### Keep the whole inference readable

Recompose factorization and PCA so the input, transformation and consequence can be seen together without panning.
For dense figures, prefer coordinated vertical stages on narrow screens over shrinking a wide diagram into illegible text.
Keep the repaired visible titles, without treating their presence alone as a comprehension improvement.
Keep labels intact: the causal-mask desktop cells split "keep", and the histogram's mobile sentinel label fragments.

### Give coding state the visual hierarchy

Many coding figures have the right data but subordinate it to repeated prompts, metadata and generic teaching panels.
Make the active structure, invariant and decisive transition primary; move repeated recognition and transfer prose out of that path.
Use the representation the pattern needs: indexed lookup, excluded interval, feasible boundary, stack span or subproblem dependency.
Do not impose the same three-panel composition on unrelated mechanisms.
Retain stable controls, color-independent identities, faithful algorithm states and a decisive static or print explanation.

### Preserve successful explanations

The roofline's ceilings, ridge and byte-reduction example already support a quantitative comparison; retain that model.
Keep its ceilings-only warning and do not imply a measured speedup.
The causal-mask example correctly follows one query through masking, normalization and value mixing.
Align its key columns, preserve the exactly zero future-key weight and repair word wrapping.
The ASR example makes a real token revision visible before finalization; retain that temporal contrast.
Label its timings illustrative and distinguish first useful text from endpoint-dependent finality.
The PCA projection geometry and factorization arithmetic are useful; fix composition rather than replacing them with generic boxes.

## Two Sum revision

Original priority: high. The targeted revision is implemented and technically verified, not learner-tested.
Source: [scripts/coding-visuals/problems/two-sum.mjs](../scripts/coding-visuals/problems/two-sum.mjs); visual `two-sum-state`.

The indexed array is enlarged and retained state is explicitly `value | original index`.
The hit frame draws `6 - 4 = 2`, connects it to map key 2, and carries recovered index 1 into `[1, 2]`.
The first miss occurs before saving value 3, making self-pairing impossible in the pictured sequence.
All six frame keys, motion identities and their DOM traversal order remain unchanged.
An opt-in complement-lookup scene leaves other array/map renderers on their original path.
Generator self-tests verify the hit connector, miss-before-save order, recovered indices and generic fallback.
Phone miss/save/hit originals, the desktop-dark result and the complete static print sequence were inspected.
All six print frames remain; reducing print repetition is still a refinement opportunity, and PDF pagination is unverified.

Acceptance question: "Why is the answer [1, 2], and why can index 0 not pair with itself?"
The picture must supply both answers without requiring the surrounding algorithm explanation.

## Acceptance

Each structured `readerCheck` is a concrete acceptance task for the proposed change, not a completed learner test.
Show the redesigned figure without its surrounding solution and ask readers to explain or predict the targeted relationship.
Record correctness, explanation errors and requests for missing information, using the existing figure as a comparison.
Include mobile and static versions and check the likely misconception explicitly.
Do not claim optimized understanding or measured improvement until reader evidence supports that claim.

## Evidence

Current evidence root: `/tmp/mlmentorship-visual-comprehension-20261026/`.

- Current inventory: `inventory-illustration-final-r2-20261026.json`.
- Final preservation, build/preview and capture seal: `illustration-final-qualified-r2.json`.
- Normal site gate exit codes and log hashes: `illustration-final-site-gates-r1.json`.
- Complete registered accounting and source/build/preview bindings: `current-render-library-manifest-r2.json`.
- Accepted registered images include the hash-verified unchanged r1 batches and current r2 batches referenced by that manifest.
- Supplemental five-mode report and images: `supplemental-final-native-r1/`.
- First ten registered revisions and coding frames: `redesign-ten-native-r2/`.
- Final claim-label correction, superseding its five earlier cases: `redesign-claim-labels-r1/`.
- Enterprise action trace: `enterprise-action-native-r1/`.
- Preservation baseline: `redesign-source-baseline-20261026.json`, created after the capacity pilot, not before it.

The old inventory, render manifest, stable capture index and diagnostic receipts remain historical records in that root.
Their `/dev/shm` PNG targets are absent and have not been recreated or relabelled.
Failed and intermediate builds/captures remain separate from accepted current evidence.
Current accepted PNGs are on disk under `/tmp`; they are not permanent repository assets.

The production preview remains available at `http://127.0.0.1:4342/`.
No learner study, whole-article factual qualification, every-state native inspection, PDF pagination or publication is claimed.