import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Count exact odd counts by subtracting two at-most windows.', [
    frame('At most 3 odd values', 'The final valid window begins at index 1; the full array has four odd values.', dualWindow(['1', '1', '2', '1', '1'], { windows: [{ label: 'at most 3', range: [1, 4], count: '14 subarrays' }, { label: 'at most 2', range: [2, 4], count: '12 subarrays' }] })),
    frame('At most 2 odd values', 'The second left boundary moves to index 2, leaving two odd values in the final window.', dualWindow(['1', '1', '2', '1', '1'], { windows: [{ label: 'at most 3', range: [1, 4], count: '14' }, { label: 'at most 2', range: [2, 4], count: '12' }] })),
    frame('Subtract the counts', 'Exactly 3 odds = at_most(3) - at_most(2) = 14 - 12 = 2.', dualWindow(['1', '1', '2', '1', '1'], { windows: [{ label: 'at most 3', range: [1, 4], count: '14' }, { label: 'at most 2', range: [2, 4], count: '12' }], result: '2 nice subarrays' })),
  ]);

export default defineVisual('count-number-of-nice-subarrays', draft, pendingReview(draft.objective));
