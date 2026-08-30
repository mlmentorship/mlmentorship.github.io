# Visual learning system

This document defines how mlmentorship should add visual explanations without turning the field guide into a gallery or copying dense presentation slides.

## Goal

A visual must reduce the effort needed to build a correct mental model. It should answer one question that prose or equations alone make hard to see.

The source deck `Basic_ML_CS_concepts` is useful because it pairs abstract ideas with spatial representations. The website should preserve that learning advantage while redrawing each idea for a narrow reading column, dark mode, mobile, accessibility, and print.

Do not publish screenshots from the deck. Some embedded figures may have separate original sources, and slide layouts are too dense for the web. Redraw the underlying idea and cite a paper when a figure reproduces a paper-specific result.

## Baseline on August 30, 2026

- The source deck has 214 slides and repeatedly uses geometry, plots, tensor slices, computation graphs, and system diagrams.
- The field guide has 283 entries, but only 11 entries contain Mermaid diagrams.
- Most existing diagrams explain large systems. Foundational concepts with the largest visual-learning benefit remain text-only.
- The repository has no instructional SVG, PNG, or WebP assets. Its image files are interface assets only.
- Mermaid currently renders in the browser. Keep it for relationship-heavy diagrams, but prefer prebuilt deterministic visuals for plots and geometry so every visual page does not need Mermaid's runtime.

The first prototype adds a causal-confounding DAG, semantic colors, accessible Mermaid descriptions, responsive compact and wide layouts, and a caption pattern. FlashAttention uses the same visual roles on its existing dataflow.

## Per-article research protocol

Every published entry is part of the visual queue, including questions and guides. A review may conclude that no visual is useful, but it may not skip the review.

Each article-specific agent must:

1. read the complete article and state the mental model that is hardest to form from prose;
2. inspect the source deck at `/mnt/c/Users/saghi/Downloads/Basic_ML_CS_concepts (1).pdf`, including related slides rather than title matches alone;
3. find primary papers, official documentation, or authoritative textbooks for the mechanism;
4. determine whether any candidate source figure can legally be reused;
5. default to an original redraw with citation when licensing is unclear;
6. compare Mermaid, deterministic SVG, semantic HTML, interaction, and no visual;
7. choose one learning objective and the smallest medium that teaches it;
8. implement, test, and update one sidecar in `data/visual-audits/`.

The sidecar records deck review, source review, medium rationale, independent reviewer, implementation IDs, and accessibility. Article markers use `<!-- visual:visual-id -->` so validation can connect research claims to shipped visuals.

Run `npm run visuals:status` to see total coverage and the next unreviewed entries. Normal checks validate every existing sidecar. The final campaign gate is `node scripts/check-visual-coverage.mjs --require-complete`; it fails until every published entry has an independent review outcome.

Parallel agents should own distinct article and sidecar files. Shared styles, components, scripts, and this protocol require a separate integration task to prevent merge conflicts and visual-language drift.

Generate a deterministic batch with `npm run visuals:batch -- --count=6 --output=scratch/visual-batch.json`. The generator selects planned work first and then unreviewed entries from the canonical queue. It creates one dependency-free article task per entry and one dependent cross-review task. Start the batch through the autonomous pipeline, merge only after its integration gate passes, then generate the next batch. Repeat until the strict completeness gate passes.

During cross-review, verify the exact campaign delta with `node scripts/check-visual-coverage.mjs --expect-resolved=slug-one,slug-two --baseline-ref=<pre-batch-revision>`. This rejects unresolved batch entries, duplicate visual IDs, and any resolved article outside the declared batch.

Source and DOM checks are necessary but not sufficient. Cross-review must render every new figure in a real browser and inspect the pixels, not only element bounds or accessibility text. Use an actual 390 CSS pixel viewport and assert `window.innerWidth === 390` before drawing conclusions. Windows headless Chrome enforces a 500 CSS pixel minimum for `--window-size=390` and merely crops the screenshot, so command-line screenshots without device-metric emulation are not valid mobile evidence. Check desktop light, desktop dark, 390-pixel light and dark, and print media. Inspect label glyphs, collisions, clipping, intended internal scrolling, and the drawn geometry behind any stated arithmetic.

Use `scripts/check-visual-rendering.mjs` for this matrix. Build the site, serve `dist` on a local port, then pass the batch slugs, base URL, and an ignored screenshot directory. The script launches Chrome or Edge, applies CDP device metrics, requires an exact 390-pixel mobile layout, checks custom SVG text against each viewBox, captures all five modes, and captures both ends of internally scrollable figures. On WSL with a Windows browser, invoke the script with Windows Node and convert the script and output paths with `wslpath -w`; a Linux process cannot reach that browser's loopback CDP endpoint. The automated pass does not replace opening the saved PNGs and inspecting their pixels.

## Choose the medium from the learning task

| Learning task | Default medium | Examples |
| --- | --- | --- |
| Follow entities, stages, state, or causality | Mermaid | causal DAGs, serving paths, lifecycle states |
| See shape, magnitude, geometry, or a mathematical tradeoff | Deterministic SVG | distributions, ROC/PR curves, PCA, loss surfaces |
| Compare tensor axes, matrix regions, or discrete cases | Semantic HTML or inline SVG | confusion matrices, normalization axes, attention masks |
| Understand how a parameter changes behavior | Small progressive interaction | threshold sweep, regularization, gradient descent |
| Add atmosphere or decoration | Usually omit | generic robots, brains, servers, stock AI art |

Mermaid should not be forced to draw quantitative plots, geometry, heatmaps, or dense tensor layouts. Generated bitmap art should not carry technical meaning.

## Figure anatomy

Every instructional figure needs:

1. **A learning question.** Example: “Which path creates confounding?”
2. **A focal relationship.** Use the warm accent for the idea being taught.
3. **A stable visual grammar.** Inputs, transformations, state, storage, outcomes, warnings, and evidence keep the same styles across articles.
4. **A short caption.** Start with “Read it this way” and state the inference the reader should make.
5. **An accessible name and description.** Mermaid uses `accTitle` and `accDescr`. Custom figures need equivalent text.
6. **A source note when needed.** Cite paper-specific data or figure structure. General mathematical constructions do not need a citation.

A figure should remain useful if its colors are removed. Shape, labels, arrows, line style, and position must carry meaning too.

## Visual grammar

- Neutral surface: ordinary processing or context.
- Blue input: observed data, requests, or supplied values.
- Warm focus: the mechanism currently being explained.
- Violet state: memory, storage, cache, or persistent state.
- Green outcome: output, decision, or verified result.
- Red warning: invalid path, leakage, failure, or unsafe action.
- Solid arrow: ordinary data or causal flow.
- Thick accent arrow: relationship the figure is teaching.
- Dashed arrow: indirect, optional, estimated, or problematic path.

Use system fonts. Prefer rules and restrained fills over shadows, gradients, and decorative icons. Labels should be readable at phone width without zooming whenever the concept allows it.

## Quality gate

Before publishing a visual, verify:

- the figure changes or sharpens the reader's mental model;
- every label is technically correct and agrees with the article;
- the main inference is clear within five seconds;
- text remains readable at 390 CSS pixels;
- the browser actually reports a 390 CSS pixel viewport rather than a cropped wider layout;
- screenshots contain the title, caption, and SVG or Mermaid label glyphs;
- labels do not collide and stated calculations match the geometry that is drawn;
- light and dark themes preserve contrast;
- the figure works in grayscale and print;
- screen readers receive a useful description;
- no external font or runtime image request is added;
- a static fallback preserves the lesson if JavaScript fails;
- the article still makes sense without the figure.

## Initial curriculum

Build a small, mixed-medium set before attempting broad coverage.

| Priority | Existing entry | Medium | Learning objective |
| ---: | --- | --- | --- |
| 1 | Causal inference for ML decisions | Mermaid DAG | See the backdoor path that makes association differ from intervention. |
| 2 | Confusion matrix and classification metrics | HTML/SVG matrix | Derive precision and recall by reading the relevant row and column. |
| 3 | ROC, PR curves, and AUC | Deterministic SVG | See threshold movement and why PR baseline changes with prevalence. |
| 4 | Backpropagation | Inline SVG computation graph | Follow values forward and adjoints backward through local derivatives. |
| 5 | BatchNorm versus LayerNorm | HTML/SVG tensor slices | See exactly which axes contribute to each normalization statistic. |
| 6 | SVD and PCA | SVG geometry | See rotation, projection, retained variance, and reconstruction. |
| 7 | Decision thresholds, asymmetric costs, and abstention | SVG policy strip | See allow, review, and block regions and their cost tradeoff. |
| 8 | Data leakage and point-in-time correctness | Mermaid timeline | See which future information crosses the prediction-time boundary. |
| 9 | Calibration | SVG reliability plot | See confidence, empirical frequency, gap, and temperature scaling. |
| 10 | Activation functions | Generated SVG small multiples | Compare output and derivative shape without a dense formula wall. |
| 11 | Multi-head attention | Mermaid plus tensor labels | See projection, parallel heads, concatenation, and output dimensions. |
| 12 | FlashAttention | Mermaid plus tiled matrix SVG | Separate the ordinary HBM path from the SRAM-tiled exact algorithm. |

After these twelve, measure article completion, figure visibility, search landings, corrections, and qualitative feedback. Do not infer learning from clicks alone.

## Production workflow

1. Write the learning question and caption first.
2. Choose Mermaid, deterministic SVG, semantic HTML, or a small interaction.
3. Generate source with Copilot or a checked-in script.
4. Review the diagram against equations and article claims.
5. Test phone, desktop, dark, light, print, keyboard, and screen-reader output.
6. Commit editable source, not only a raster export.
7. Treat article text and its visual as one unit during future edits.

Copilot CLI can inspect reference images and PDFs, write Mermaid or SVG source, and run render and validation tools. It is not, by itself, a dedicated image-generation model. An external image model could be connected, but core technical figures should remain deterministic, editable, and reviewable.
