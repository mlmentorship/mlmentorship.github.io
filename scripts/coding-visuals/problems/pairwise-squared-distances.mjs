import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Singleton axes create every point-center pair before reducing feature coordinates.', [
    frame('Add singleton axes', 'Points [n,d] become [n,1,d]; centers become [1,k,d].', shapes(['points [n,1,d]', 'centers [1,k,d]'], { action: 'align singleton axes' })),
    frame('Broadcast pairs', 'The difference tensor has one row for every point-center pair.', shapes(['points [n,1,d]', 'centers [1,k,d]', 'difference [n,k,d]'], { action: 'broadcast', focus: 'difference' })),
    frame('Reduce features', 'Summing squared differences over d yields [n,k] distances.', shapes(['difference [n,k,d]', 'sum over d', 'distances [n,k]'], { result: '[n,k]' })),
  ]);

export default defineVisual('pairwise-squared-distances', draft, pendingReview(draft.objective));
