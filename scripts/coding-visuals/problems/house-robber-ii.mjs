import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('A circular solution is the larger of two lines: exclude the first house or exclude the last.', [
    frame('Break the circle', 'Taking both first and last is forbidden, so solve two linear ranges.', array(['2', '3', '2'], [mark(0, 'exclude in case B', 'state'), mark(2, 'exclude in case A', 'state')], { cases: 'houses[0:-1] and houses[1:]' })),
    frame('Solve each line', 'Case A [2,3] gives 3. Case B [3,2] gives 3.', table(['case', 'houses', 'best'], [['A', '[2,3]', '3'], ['B', '[3,2]', '3']], [0, 1])),
    frame('Choose the larger result', 'Both cases tie at 3, which is the circular answer.', array(['2', '3', '2'], [mark(1, 'take', 'output')], { result: '3' })),
  ]);

export default defineVisual('house-robber-ii', draft, pendingReview(draft.objective));
