import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Read the four current boundaries, then shrink them after each side.', [
    frame('Read the top and right', 'Consume top row 1,2,3 and right column 6,9.', grid([['1', '2', '3'], ['4', '5', '6'], ['7', '8', '9']], [{ row: 0, col: 0, label: 'top', tone: 'focus' }, { row: 2, col: 2, label: 'right', tone: 'focus' }])),
    frame('Read bottom and left', 'Continue backward across 8,7 and up through 4.', grid([['.', '.', '.'], ['4', '5', '.'], ['7', '8', '.']], [{ row: 2, col: 1, label: 'bottom', tone: 'state' }, { row: 1, col: 0, label: 'left', tone: 'state' }])),
    frame('Finish the inner layer', 'The remaining center is 5.', grid([['.', '.', '.'], ['.', '5', '.'], ['.', '.', '.']], [{ row: 1, col: 1, label: 'last', tone: 'output' }], { result: '[1,2,3,6,9,8,7,4,5]' })),
  ]);

export default defineVisual('spiral-matrix', draft, pendingReview(draft.objective));
