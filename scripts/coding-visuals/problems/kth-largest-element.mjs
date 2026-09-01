import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Keep only the largest k values; the smallest of those is the kth largest.', [
    frame('Fill a size-2 heap', 'Read 3 and 2. Both remain candidates for the top two.', heap(['2', '3'], { root: '2', detail: 'size 2' })),
    frame('Replace the weak root', '5 arrives and evicts 2. The heap now protects 3 and 5.', heap(['3', '5'], { root: '3', detail: '2 discarded' })),
    frame('Return the root', 'After all values, heap [5,6] has root 5, the second largest.', heap(['5', '6'], { root: '5', detail: 'kth largest', result: '5' })),
  ]);

export default defineVisual('kth-largest-element', draft, pendingReview(draft.objective));
