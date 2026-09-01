import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Compare both trees at the same position before recursing to children.', [
    frame('Compare roots', 'Both roots are 1, so continue.', table(['position', 'tree A', 'tree B'], [['root', '1', '1'], ['left', '2', '2'], ['right', '3', '3']], [0])),
    frame('Compare children', 'Left and right values match at the same positions.', table(['position', 'tree A', 'tree B'], [['root', '1', '1'], ['left', '2', '2'], ['right', '3', '3']], [3, 6])),
    frame('Accept equal shape', 'The recursive comparisons all return true.', tree([['1'], ['2', '3']], [mark(0, 'same', 'output'), mark(1, 'same', 'output'), mark(2, 'same', 'output')], { result: 'true' })),
  ]);

export default defineVisual('same-tree', draft, pendingReview(draft.objective));
