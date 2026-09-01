import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Keep colder days waiting until a warmer day resolves them.', [
    frame('Hold unresolved days', 'After scanning through 69, days 2, 3, and 4 are still waiting for a warmer temperature.', stack('73 74 75 71 69 72', ['day 2: 75', 'day 3: 71', 'day 4: 69'], { current: '72', action: 'wait' })),
    frame('Resolve from the top', '72 is warmer than 69 and 71, so both waiting days receive distances. Day 2 remains.', stack('73 74 75 71 69 72', ['day 2: 75'], { current: '72', action: 'resolve 69 -> 1, 71 -> 2' })),
    frame('Leave no warmer day as zero', '75 stays in the stack because no later value is warmer.', array(['1', '1', '0', '2', '1', '0'], [mark(2, 'none', 'state'), mark(5, 'none', 'state')], { result: '[1,1,0,2,1,0]' })),
  ]);

export default defineVisual('daily-temperatures', draft, pendingReview(draft.objective));
