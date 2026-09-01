import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Store the minimum so far beside every stack value.', [
    frame('Push 5', 'The first value is also the minimum.', table(['value', 'min so far'], [['5', '5']], [0])),
    frame('Push 2 and 4', '2 becomes the minimum; 4 inherits minimum 2.', table(['value', 'min so far'], [['5', '5'], ['2', '2'], ['4', '2']], [2, 3, 4, 5])),
    frame('Read the minimum', 'The answer is at the top of the min column, without scanning values.', table(['value', 'min so far'], [['5', '5'], ['2', '2'], ['4', '2']], [5], { result: 'get_min() = 2' })),
  ]);

export default defineVisual('min-stack', draft, pendingReview(draft.objective));
