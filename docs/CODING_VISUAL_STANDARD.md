# Coding Visual Standard

Coding visuals teach the mechanism of an algorithm, not merely its final answer. Each problem owns one module in `scripts/coding-visuals/problems/<slug>.mjs`; shared factories and scene primitives live in `scripts/coding-visuals/primitives.mjs`. Modules are independent so reviewers can edit different problems without creating registry conflicts.

## Definition contract

Every default export is created with `defineVisual(slug, draft, review)`. A definition contains:

- an objective and at least three authored frames;
- a stable frame key and stable `scene.motion` entity keys for values, pointers, nodes, links, paths, or frontiers that persist between frames;
- a final explicit result;
- review metadata for `pattern`, `recognitionCue`, `invariant`, `stateModel`, `visualRationale`, `rejectedAlternatives`, `transferLesson`, and `reviewStatus`.

Migrated definitions use `reviewStatus: "pending"` until a problem owner replaces migration language with a problem-specific mechanism review. Normal checks permit pending modules so migration and review can proceed independently. `node scripts/check-coding-visuals.mjs --require-reviewed` permits only fully reviewed work and rejects pending, generic, duplicated, unchanged, under-specified, or non-moving traces.

## Mechanism-first rubric

A reviewed visual must let a reader answer:

1. **Recognition:** What wording or input shape suggests this pattern?
2. **Invariant:** What remains true before and after each transition?
3. **State:** Which values, topology, frontier, call, return, or dependency must be retained?
4. **Transition:** What visibly moves or changes, and why is that change safe?
5. **Transfer:** Which related problems can reuse the mechanism?

Use geometry that matches the data structure: indexed array cells and independent pointers/windows; vertical bars and measured areas; graph edges and frontiers; binary-tree and heap parent-child edges; linked-list links and rewires; grid coordinates/frontiers; DP dependencies and fill order; trie prefix edges; interval extents; backtracking paths; bit positions; or tensor dimensions. Do not replace topology with a row of pills and prose describing invisible edges.

## Motion and accessibility

Motion keys identify semantic entities, not DOM positions. Keep a key stable when an entity changes coordinates between adjacent frames. Playback uses those keys to interpolate the entity itself rather than cross-fading a whole panel.

The first frame must be complete HTML and visible without JavaScript. Progressive enhancement may reveal Previous, Next, Play, and timeline controls. Arrow Left/Right and Home/End provide keyboard navigation. Reduced-motion mode advances one authored step per activation and disables interpolation. All state remains readable in light and dark themes, at the exact mobile breakpoint, and in print; print reveals every authored frame and removes controls.

## Parallel workflow

Generate or validate only owned modules with comma-separated slugs:

```sh
node scripts/generate-coding-question-book.mjs --slugs=two-sum,valid-anagram
node scripts/check-coding-visuals.mjs --slugs=two-sum,valid-anagram
node scripts/check-coding-visuals.mjs --slugs=two-sum --require-reviewed
```

Filtered generation writes only the selected article and audit sidecars and does not rewrite the shared chapter registry. Run unfiltered generation only when intentionally synchronizing the complete book.
