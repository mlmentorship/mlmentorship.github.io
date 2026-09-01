import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Inorder traversal visits BST nodes in ascending order, so stop at the kth visit.', [
    frame('Push the left spine', 'Start by pushing 3, then 1.', tree([['3'], ['1', '4'], ['-', '2', '-', '-']], [mark(0, 'stack', 'state'), mark(1, 'stack', 'focus')])),
    frame('Visit in order', 'Pop 1 first, then 2, then 3. The first visit is the smallest.', tree([['3'], ['1', '4'], ['-', '2', '-', '-']], [mark(1, 'visit 1', 'focus'), mark(4, 'visit 2', 'state')])),
    frame('Stop at k', 'For k=1, return node 1 immediately.', tree([['3'], ['1', '4'], ['-', '2', '-', '-']], [mark(1, 'kth', 'output')], { result: '1' })),
  ]);

export default defineVisual('kth-smallest-element-in-a-bst', draft, pendingReview(draft.objective));
