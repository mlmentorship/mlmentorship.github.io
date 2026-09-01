import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Preorder plus explicit null markers preserves both node values and tree shape.', [
    frame('Visit preorder', 'Tree 1 with a right child 2 visits 1, null-left, 2.', tree([['1'], ['-', '2']], [mark(0, 'visit 1', 'focus'), mark(1, 'null', 'state')])),
    frame('Write markers', 'Missing children become # tokens, so the stream is 1,#,2,#,#.', array(['1', '#', '2', '#', '#'], [mark(1, 'shape marker', 'state'), mark(3, 'shape marker', 'state')])),
    frame('Read the same stream', 'The decoder consumes tokens in the same preorder and rebuilds the shape.', tree([['1'], ['-', '2']], [mark(0, 'rebuilt', 'output')], { result: 'same tree' })),
  ]);

export default defineVisual('serialize-and-deserialize-binary-tree', draft, pendingReview(draft.objective));
