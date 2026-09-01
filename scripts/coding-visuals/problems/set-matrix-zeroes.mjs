import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Use the first row and column as markers, then apply the marked rows and columns.', [
    frame('Find zeros', 'A zero at row 1, column 1 marks its row and column.', grid([['1', '1', '1'], ['1', '0', '1'], ['1', '1', '1']], [{ row: 1, col: 1, label: '0', tone: 'focus' }], { action: 'mark row 1, col 1' })),
    frame('Read the markers', 'The first row and column now carry the future zero instructions.', grid([['1', '0', '1'], ['0', '0', '1'], ['1', '1', '1']], [{ row: 0, col: 1, label: 'marker', tone: 'state' }, { row: 1, col: 0, label: 'marker', tone: 'state' }])),
    frame('Fill marked cells', 'Zero every cell in the marked row or column.', grid([['1', '0', '1'], ['0', '0', '0'], ['1', '0', '1']], [], { result: 'in place' })),
  ]);

export default defineVisual('set-matrix-zeroes', draft, pendingReview(draft.objective));
