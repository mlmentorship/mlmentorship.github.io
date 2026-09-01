import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('The first differing character in adjacent words creates a directed ordering edge.', [
    frame('Extract a rule', 'wrt and wrf first differ at t and f, so t -> f.', graph(['w', 'r', 't', 'f'], ['t -> f'], { rule: 't before f' })),
    frame('Collect rules', 'The other adjacent differences add w->e, e->r, and r->t.', graph(['w', 'e', 'r', 't', 'f'], ['w -> e', 'e -> r', 'r -> t', 't -> f'], { indegree: ['w:0', 'e:1', 'r:1', 't:1', 'f:1'] })),
    frame('Topologically order', 'Remove zero-indegree letters and return a valid alien alphabet.', graph(['w', 'e', 'r', 't', 'f'], ['w -> e', 'e -> r', 'r -> t', 't -> f'], { order: ['w', 'e', 'r', 't', 'f'], result: 'wertf' })),
  ]);

export default defineVisual('alien-dictionary', draft, pendingReview(draft.objective));
