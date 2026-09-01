import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Use one frequency signature as the address for each word group.', [
    frame('Build the first bucket', 'eat and tea have the same sorted signature.', buckets([{ count: '[a,e,t]', items: ['eat', 'tea'], tone: 'focus' }, { count: '[a,n,t]', items: [], tone: 'neutral' }])),
    frame('Branch on a new signature', 'tan belongs under [a,n,t], while ate returns to [a,e,t].', buckets([{ count: '[a,e,t]', items: ['eat', 'tea', 'ate'], tone: 'state' }, { count: '[a,n,t]', items: ['tan'], tone: 'focus' }])),
    frame('Read the groups', 'Words sharing a signature are already together.', buckets([{ count: '[a,e,t]', items: ['eat', 'tea', 'ate'] }, { count: '[a,n,t]', items: ['tan', 'nat'] }, { count: '[a,b,t]', items: ['bat'] }], { status: 'three buckets', result: 'three groups' })),
  ]);

export default defineVisual('group-anagrams', draft, pendingReview(draft.objective));
