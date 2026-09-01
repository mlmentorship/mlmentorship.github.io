import { defineVisual, frame, mark, tree, visual } from '../primitives.mjs';

const levels = [['3'], ['9', '20'], ['-', '-', '15', '7']];

const draft = visual('Snapshotting the queue length separates the current tree level from children appended for the next level.', [
  frame(
    'Queue the root',
    'For tree [3,9,20,null,null,15,7], initialize answer = [] and queue = [3]. The drawn edges show 3 -> 9, 3 -> 20, 20 -> 15, and 20 -> 7.',
    tree(levels, [mark(0, 'queued', 'focus')], { queueState: '[3]', answer: '[]' }),
    'queue-root',
  ),
  frame(
    'Freeze level size 1',
    'Read len(queue) = 1 before adding children. Pop 3 into level [3], then enqueue its left child 9 and right child 20.',
    tree(levels, [mark(0, 'level [3]', 'output'), mark(1, 'next queue', 'focus'), mark(2, 'next queue', 'focus')], {
      frozenSize: '1',
      queueState: '[9,20]',
      answer: '[[3]]',
    }),
    'finish-level-0',
  ),
  frame(
    'Freeze level size 2',
    'At the next while iteration, len(queue) = 2. That fixed count means 9 and 20 belong together even though processing them will append children.',
    tree(levels, [mark(1, '1 of 2', 'focus'), mark(2, '2 of 2', 'focus')], {
      frozenSize: '2',
      queueState: '[9,20] before processing',
      level: '[]',
      answer: '[[3]]',
    }),
    'measure-level-1',
  ),
  frame(
    'Process exactly two nodes',
    'Pop 9, which has no children; pop 20, then enqueue 15 and 7. Append level [9,20], leaving queue [15,7] for the next iteration.',
    tree(levels, [
      mark(1, 'level [9,20]', 'output'),
      mark(2, 'level [9,20]', 'output'),
      mark(5, 'next queue', 'focus'),
      mark(6, 'next queue', 'focus'),
    ], { queueState: '[15,7]', answer: '[[3],[9,20]]' }),
    'finish-level-1',
  ),
  frame(
    'Process the leaf level',
    'Freeze len(queue) = 2, pop 15 and 7, and append level [15,7]. Neither leaf adds a child, so the queue becomes empty.',
    tree(levels, [mark(5, 'level [15,7]', 'output'), mark(6, 'level [15,7]', 'output')], {
      frozenSize: '2',
      queueState: '[]',
      answer: '[[3],[9,20],[15,7]]',
      result: '[[3],[9,20],[15,7]]',
    }),
    'finish-level-2',
  ),
]);

const review = {
  pattern: 'Breadth-first tree traversal with a queue-length snapshot for level boundaries.',
  recognitionCue: 'Use this BFS form when a tree answer is grouped by depth, processed left-to-right by level, or must compute one aggregate per level rather than one flat visitation order.',
  invariant: 'At each while-loop start, the queue contains exactly one complete next level in left-to-right order. Iterating the captured length consumes only that level while appending its children for the following level.',
  stateModel: 'The minimal state is a FIFO queue, one frozen queue length, the current level list, and the accumulated answer. Child links determine enqueue order: left before right.',
  visualRationale: 'An edged binary tree exposes parent-child topology while node labels identify the current and next queue levels. The explicit queue and frozen-size annotations make the boundary understandable without color or code.',
  rejectedAlternatives: [
    'A queue-only strip was rejected because it hides the parent-child edges that produce enqueue order.',
    'A depth table was rejected because it presents the answer but not the traversal mechanism.',
    'Recursive DFS was rejected because it does not match the supplied queue implementation or its O(w) frontier state.',
  ],
  transferLesson: 'Capture the frontier size before expanding it whenever output or timing is grouped by BFS depth; this transfers to right-side view, level averages, shortest unweighted paths, and wave simulations.',
  reviewStatus: 'reviewed',
};

export default defineVisual('binary-tree-level-order-traversal', draft, review);
