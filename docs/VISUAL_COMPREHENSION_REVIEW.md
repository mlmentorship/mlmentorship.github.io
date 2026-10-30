# Visual comprehension review

## Mechanism follow-up

This follow-up builds on the existing individual assessments of 392 registered figures and 16 supplemental drawings.
It adds four targeted improvements, not another 408 redraws or another manual inspection of every retained original.
The structured record is `mechanismFollowUp` in [data/visual-comprehension-review.json](../data/visual-comprehension-review.json).

| Figure | Drawn improvement | Reader inference |
| --- | --- | --- |
| InfoNCE | One denominator enclosure contains all four candidates; its nested positive row has four of ten equal weight units. | The positive belongs to both numerator and denominator; semantic similarity does not override the sampler. |
| Ring all-reduce | A four-rank inset shows all four chunks moving concurrently; one coordinate accumulates 1, 3, 6, 10, then remains 10 during forwarding. | All-gather copies rather than adds; six equal sent chunks account for 3B/2 sent bytes per rank. |
| Sharded matmul | Same-scale grids retain 19 and 50 while showing missing cells before the operand gather and newly computed 22 and 43 afterward. | Uncomputed products are not zero-valued partial sums; gathering inputs enables new multiplication. |
| PCA | The same selected sample has a marked perpendicular residual, a reconstructed vector, and scalar score 85 on a separate shared-scale axis. | The one-number coordinate and reconstructed vector are different objects. |

Normal `npm run check` and `npm run build` pass.
All four revised figures pass five native modes: desktop light/dark, actual-390 phone light/dark, and print-style rendering, totaling 20 cases.
All four phone-light originals and desktop-dark originals were directly inspected.
Final screen PNGs are byte-identical to those inspected before the print-only refinement.
Four actual tagged Chromium PDFs pass physical-page bounds, figure-word completeness, and nonblank-page checks across 30 pages.
Their four complete figure pages were directly viewed; this is not manual inspection of every PDF page.

The first PDF samples exposed titles separated from drawings despite passing text-completeness checks.
Only these four figures now opt into dedicated print pages with height-bounded SVGs.
New assertions require the title, first drawn label, and final drawn label to occur on the same physical page.
The earlier text-only results remain distinct from the final composition-qualified PDFs.

The other 387 article sources remain exact to the preceding qualified tree.
Their built HTML reproduces the previous whole-page hashes after substituting only parsed stylesheet-link filenames.
None matches the new print selectors; all other CSS rules and non-CSS runtime assets are unchanged.
Non-figure lesson content, headings, Python, figure identities, all 797 checkpoints, and 8,739 motion identities remain preserved.
No learner study, measured tenfold improvement, publication-standard certification, push, or publication is claimed.

## Second editorial pass

This pass reviews all 392 registered figures and all 16 supplemental drawings individually.
Each has an evidence-led takeaway in [scripts/illustration-takeaways.mjs](../scripts/illustration-takeaways.mjs), stating the inference visible in that particular figure rather than repeating its topic.
The catalog retains strong existing wording where appropriate; 408 reviewed takeaways does not mean 408 newly rewritten titles or structural redraws.
The review surface now displays those titles instead of reducing them to one-pixel screen-reader boxes.
The native renderer rejects hidden, clipped, transparent and horizontally overflowing titles.

Bayes gains a countable 1,000-cell population grid: one diseased group and 999 healthy groups, each representing 100 people.
Its existing 99-versus-999 positive-result strip retains exact count-proportional geometry.
Actor-critic gains a phone-width composition that separates the forward advantage value from the stopped return gradient while preserving both losses' updates to shared features.
Binary search and rotated target search gain previous/current candidate bands on a shared index scale, without inventing a prior state at initialization.
Human-readable pointer labels replace internal pointer keys in the selected-state comparisons of 79 coding figures.
All 106 coding definitions and all 797 authored checkpoints remain intact.

The exact catalog, applied-title ownership, synchronized drawing models and coding generator pass their focused source checks.
The structured record is `secondPassReconciliation` in [data/visual-comprehension-review.json](../data/visual-comprehension-review.json).
There is no measured tenfold improvement, learner study, publication-standard certification or claim that all figures were structurally redrawn.

### Second-pass qualification

Normal `npm run check` and `npm run build` pass on the isolated second-pass tree.
All 408 figures pass desktop light/dark, actual-390 phone light/dark and print-style checks: 2,040 figure cases.
All 797 authored coding checkpoints are captured in each of the four screen modes: 3,188 required settled state captures.
The accepted render inventory authenticates 7,920 PNG files, including repeated primary-figure captures on supplemental routes and horizontal-scroll endpoints.
This file count is neither a count of unique figures nor a claim of individual manual inspection.
Native interaction checks pass all 106 coding routes and 797 checkpoints at each actual 1280px and 390px width.
They cover forward/backward and direct navigation, keyboard controls, real playback, stable frame/control/page geometry, reduced motion and the no-JavaScript fallback.

All 68 labelled contact sheets were directly reviewed, covering all 408 initial phone views.
All 16 supplemental phone originals and selected registered native originals were inspected separately, including Bayes, actor-critic and the two candidate-band traces.
The contact sheets are scaled composition reviews, not native inspection of every image or every authored checkpoint.
Some retained wide drawings require contained horizontal panning; passing the technical text-size floor does not establish effortless comprehension.
The primary review surface also passes a dedicated visible-title/navigation smoke check; a deliberately one-pixel-clipped control is rejected by the new visibility criteria.

Every non-figure lesson body remains byte-for-byte identical to base commit `be21ddb6c2dc892d8ea593a6277a662f29fbfb25`.
All 391 articles retain their headings, Python fences, ordered registered/supplemental identities and public fragments.
All 797 frame identities and 8,739 ordered motion identities are preserved.
Both pre-existing dirty worktrees are outside this isolated revision and remain preserved.

Evidence is under `/tmp/mlmentorship-illustrations-second-pass-20261028-81b8a701-evidence`.
The original `candidate-source-freeze-r1.json` and `render-qualified-r1.json` remain unchanged.
Final reconciliation allows only the interaction observer and these review records to differ from the rendered source candidate; article/runtime equality is checked separately after the final build.
The observer now bounds CDP waits, reports the current route/stage and observes real playback from the driver instead of awaiting a page-side timer.
Accepted interaction reports are `interactions-1280-r3.json` and `interactions-390-r4.json`.
The interrupted desktop run, the localized observer timeout and the wrong-working-directory phone invocation remain separate, rejected attempts.
Native PNGs are in an owned `/dev/shm` directory and are volatile across host restart; their hashes, reports and contact sheets are retained on disk.
No push, publication, learner experiment or PDF-pagination verification is included in this pass.

## Editorial edition

The new edition covers all 392 registered illustrations across 391 articles and all 16 supplemental illustrations.
It adds 83 purpose-drawn registered SVGs and replaces all 16 supplemental diagrams with original SVG compositions.
All-reduce and Bayes receive two additional manual numerical-geometry revisions.
All 106 coding illustrations gain selected-state comparisons across their 797 existing checkpoints.
Five traces add mechanism-specific drawings: indexed complement lookup, rejected 3Sum pairs, linked tree traversals, palindrome intervals and fixed-width bit destinations.
The histogram distinguishes unresolved positive heights from its actual sentinel-bearing stack.
Every registered figure has an individual inference question, with shared typography, caption hierarchy and responsive refinements.
Successful quantitative plots are retained rather than replaced with generic stage boxes.

The exact changes are recorded in `editionReconciliation` in [data/visual-comprehension-review.json](../data/visual-comprehension-review.json).
The original 392 assessments, 16 supplemental assessments and earlier reconciliation remain historical evidence.
Shared improvements do not mean 408 structural redraws or that every original recommendation is resolved.
There is no measured tenfold improvement, learner study, publication-standard certification or PDF-pagination claim.

### Edition qualification

Normal `npm run check` and `npm run build` pass on the isolated assistant-only edition tree.
All 392 registered and 16 supplemental figures pass desktop light/dark, actual-390 phone light/dark and print-style checks: 2,040 figure cases.
The campaign captures all 797 authored coding checkpoints in each of the four screen modes: 3,188 settled state captures.
Native interaction checks pass all 106 coding routes and 797 checkpoints at both actual 1280px and 390px widths.
They cover forward/backward and direct navigation, keyboard controls, playback, stable frame/control geometry, reduced motion and the no-JavaScript fallback.
All 391 articles retain their headings, Python fences, ordered figure identities and public fragments; all 797 frame identities and 8,739 ordered motion identities are preserved.

All 81 labelled mobile contact sheets were directly reviewed, covering all 408 figures.
Those scaled coding thumbnails show initial states; they do not establish individual native-image inspection of every original or every authored state.
Selected native originals were inspected separately, including numerical geometry, the five mechanism-specific coding additions and the repaired legal-RAG label layout.
The accepted output inventories authenticate 7,574 PNG files; that is a file count, not the number of figure cases or a count of individually inspected images.
Fifteen completed batches covering 300 unchanged articles are reused only after exact source, HTML, runtime-asset, renderer, report and PNG equality; the remaining 91 articles and all supplemental cases were freshly captured on the final rendering candidate.
Failed and superseded attempts remain separate, including the legal-RAG node-fit failure and a Chromium startup failure when shared memory was full.
The latter used an owned `--disable-dev-shm-usage` launcher on retry, without weakening product assertions or deleting unrelated work.

The browser-qualified candidate tree is `2403ceb51ee4f50979f06a5049a799c6de843bab`.
The sealed manifest is `/tmp/mlmentorship-illustration-edition-20261027-81b8a701/production-r5/production-qualified.json`, SHA-256 `65f6eccac269e471bf0f4ab90034813620f4b782f039e57ebaf2b54690e03463`.
Final review-metadata changes are qualified separately against this tree's built article HTML and runtime assets.
The editor worktree's four existing BST statement failures are not part of this edition.
The user's generator, statement and global-style edits remain preserved outside the edition commit.

Generation is reproducible with the existing dependencies:

```sh
node scripts/illustration-redraws.mjs
node scripts/coding-visuals/state-transition.mjs
node scripts/illustration-edition.mjs --self-test
node scripts/illustration-edition.mjs --evidence-dir=/path/to/owned/evidence --capture
node scripts/illustration-edition.mjs --evidence-dir=/path/to/owned/evidence --apply-redraws
node scripts/illustration-edition.mjs --evidence-dir=/path/to/owned/evidence --apply-coding
node scripts/illustration-edition.mjs --evidence-dir=/path/to/owned/evidence --apply-reader-checks
node scripts/illustration-edition.mjs --evidence-dir=/path/to/owned/evidence --apply-takeaways
node scripts/illustration-edition.mjs --evidence-dir=/path/to/owned/evidence --verify
```

Capture once before further edits; the command refuses to overwrite an existing preservation baseline.
Do not regenerate whole lessons merely to update figures or use a dirty generator to apply unrelated changes.

## Responsive follow-up

Every article now promotes its original main illustration before the article body, rather than opting questions and pattern-first lessons out of the opening visual.
Figure identities and authored content remain intact; the no-JavaScript fallback retains the original source placement.
Questions keep closed-book practice ahead of the illustration, with the topic-neutral button label **Practice** and the existing problem-only prompt.

Wide article illustrations use their available content column, including the expanded library sidebar, instead of the full viewport.
Mobile graph and linked-topology canvases retain readable labels inside a contained horizontal scroller.
Search keeps its close control visible while results scroll.
Homepage edition metadata, chapter metadata, coding-frame headings and practice actions wrap at enlarged text sizes without widening the page.
The homepage's primary action context label inherits the button's contrasting foreground.

The responsive review uses actual 320, 390, 768, 1024, 1440 and 1920 CSS-pixel viewports in both themes.
It exercises navigation, populated search, long-title practice, scoring and scratchpad reset, expanded contents, the review explanation and generated prep-map results.
Text-resize cases use 200% root text at phone, tablet and desktop widths.
All 391 main-figure openings are checked at 320, 768 and 1440px for placement, page bounds and the existing 8px SVG-label minimum.
This floor is a technical check, not a claim that every figure is equally readable or a substitute for inspecting native captures.
Failed observations remain separate from the corrected checks, including invalid drawer-state assumptions and tall-panel element captures that did not settle.

## Previous outcome

Expert review is complete for 392 registered visuals across 391 articles and 16 supplemental Mermaid diagrams.
Eleven registered figures and seven supplemental diagrams received substantive revisions.
Twenty-four registered Mermaid figures received missing visible titles; those repairs are not 24 redesigns.
All 16 supplemental diagrams now have stable supplemental identities, visible titles, accessible descriptions and direct captions.
Nine supplemental diagrams retain further refinement recommendations.
The original registered recommendations remain an improvement backlog, not a claim that every figure was redesigned.
The lesson-authoring checkpoint remains 252 of 391; this review did not author the remaining lessons.

At that checkpoint, verification passed 784 registered desktop/phone renders and 80 supplemental five-mode renders.
The eleven revised registered figures also have 55 accepted five-mode cases, with all frames checked for the two changed coding traces.
Native originals were inspected for all eighteen revised visuals, including selected miss/save/hit states, dark-theme views and a static print sequence.
These are source, arithmetic, rendering and expert-composition checks, not observed learner outcomes or PDF-pagination verification.
Those repairs were local at that checkpoint; no publication was performed.

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
The corpus also contains 16 supplemental illustrations outside registered markers; they are separately assessed, not added to the registered count.
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

### Previous qualification

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

## Original priorities

The following paragraphs retain the original recommendations, not the current unresolved backlog.
The new edition implements the all-reduce values, PCA/factorization recomposition, histogram sentinel distinction, causal-mask wrapping and mechanism-specific coding examples described here.
Architecture and authority redraws now follow concrete objects, evidence and consequences.
Further density reduction and comparative reader testing remain opportunities; technical render coverage alone does not establish comprehension.

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

Previous qualification root: `/tmp/mlmentorship-visual-comprehension-20261026/`.
Editorial-edition evidence root: `/tmp/mlmentorship-illustration-edition-20261027-81b8a701/`.
Earlier receipt names below describe the previous qualification, not the new edition.

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

The editor development preview remains available at `http://127.0.0.1:4342/`.
No learner study, whole-article factual qualification, every-state native inspection, PDF pagination or publication is claimed.