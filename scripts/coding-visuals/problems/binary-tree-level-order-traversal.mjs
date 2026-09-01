import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Read exactly the queue length that existed before adding child nodes.', [
    frame('Queue the root', 'The first layer contains only 3.', queueGrid([['3'], ['9', '20'], ['15', '7']], ['3'], { level: '0' })),
    frame('Read one level', 'Pop 3, then append 9 and 20 for the next layer.', queueGrid([['3'], ['9', '20'], ['15', '7']], ['9', '20'], { level: '1', result: '[[3]]' })),
    frame('Continue by layer', 'The queue boundary gives [[3],[9,20],[15,7]].', queueGrid([['3'], ['9', '20'], ['15', '7']], [], { level: '2', result: '[[3],[9,20],[15,7]]' })),
  ]);

export default defineVisual('binary-tree-level-order-traversal', draft, pendingReview(draft.objective));
