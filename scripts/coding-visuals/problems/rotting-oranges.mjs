import { defineVisual, frame, grid, visual } from '../primitives.mjs';

const source = (row, col, label, key) => ({ row, col, label, tone: 'focus', key });

const draft = visual('Seeding every rotten cell makes BFS time equal the shortest orthogonal distance from any source; the last timestamp is the required minute.', [
  frame(
    'Seed every time-zero source',
    'In [[2,1,1],[1,1,0],[0,1,1]], enqueue (0,0,0) and count 6 fresh oranges. Here 2 = rotten, 1 = fresh, and 0 = empty.',
    grid([['2', '1', '1'], ['1', '1', '0'], ['0', '1', '1']], [source(0, 0, 'frontier t=0', 'wave-a')], { queueState: '[(0,0,0)]', fresh: '6', minute: '0' }),
    'seed-sources',
  ),
  frame(
    'Spread minute 1',
    'Pop (0,0,0). Its fresh orthogonal neighbors (0,1) and (1,0) rot, fresh falls 6 -> 4, and both enter the queue at time 1.',
    grid([['2', '2', '1'], ['2', '1', '0'], ['0', '1', '1']], [
      source(0, 1, 'frontier t=1', 'wave-a'),
      source(1, 0, 'frontier t=1', 'wave-b'),
    ], { queueState: '[(0,1,1),(1,0,1)]', fresh: '4', arithmetic: '6 - 2 = 4' }),
    'spread-minute-1',
  ),
  frame(
    'Spread minute 2',
    'Pop both time-1 cells. They newly rot (0,2) and (1,1); the second visit to (1,1) sees 2 and does not enqueue it twice. Fresh falls 4 -> 2.',
    grid([['2', '2', '2'], ['2', '2', '0'], ['0', '1', '1']], [
      source(0, 2, 'frontier t=2', 'wave-a'),
      source(1, 1, 'frontier t=2', 'wave-b'),
    ], { queueState: '[(0,2,2),(1,1,2)]', fresh: '2', arithmetic: '4 - 2 = 2' }),
    'spread-minute-2',
  ),
  frame(
    'Spread minute 3',
    'Pop the time-2 cells. Only (2,1) is still fresh and orthogonally adjacent, so it rots and enters at time 3; fresh falls 2 -> 1.',
    grid([['2', '2', '2'], ['2', '2', '0'], ['0', '2', '1']], [
      source(2, 1, 'frontier t=3', 'wave-b'),
    ], { queueState: '[(2,1,3)]', fresh: '1', arithmetic: '2 - 1 = 1' }),
    'spread-minute-3',
  ),
  frame(
    'Spread minute 4',
    'Pop (2,1,3), rot its fresh neighbor (2,2), and enqueue (2,2,4). Fresh falls 1 -> 0, so every orange is now reachable from a source.',
    grid([['2', '2', '2'], ['2', '2', '0'], ['0', '2', '2']], [
      source(2, 2, 'frontier t=4', 'wave-b'),
    ], { queueState: '[(2,2,4)]', fresh: '0', arithmetic: '1 - 1 = 0' }),
    'spread-minute-4',
  ),
  frame(
    'Return the last timestamp',
    'Pop (2,2,4); it adds no fresh neighbor. The queue empties with fresh = 0, so return the last popped timestamp, 4.',
    grid([['2', '2', '2'], ['2', '2', '0'], ['0', '2', '2']], [], { queueState: '[]', fresh: '0', check: 'fresh == 0', result: '4' }),
    'return-minute-4',
  ),
]);

const review = {
  pattern: 'Multi-source breadth-first search on a four-neighbor grid with distance timestamps.',
  recognitionCue: 'Use multi-source BFS for simultaneous unweighted spreading, infection, nearest-source distance, or minimum elapsed steps when every source acts at time zero and each move has equal cost.',
  invariant: 'When (row,col,time) is popped, time is its minimum distance from any initial rotten cell. Marking a fresh neighbor rotten before enqueueing ensures that cell is queued once at time + 1.',
  stateModel: 'The minimal state is the mutated grid as visited state, a queue of (row,col,time), the remaining fresh count, and the last popped time. Each frame shows the exact queue frontier and fresh-count decrement.',
  visualRationale: 'A coordinate grid preserves orthogonal adjacency, obstacles, changed cells, and the advancing wave. Stable wave-a and wave-b markers move through successive frontier cells, while text labels expose time without relying on color.',
  rejectedAlternatives: [
    'A queue-only table was rejected because it hides which cells are adjacent and why diagonal cells cannot spread rot.',
    'A graph node-link diagram was rejected because the regular grid already gives simpler, exact topology.',
    'A single before/after heatmap was rejected because it skips queue order, timestamps, and fresh-count updates.',
  ],
  transferLesson: 'Seed all equal-priority sources before BFS, mark neighbors when enqueued, and carry distance in the queue; the same mechanism solves nearest gate, fire spread, and shortest distance to any facility.',
  reviewStatus: 'reviewed',
};

export default defineVisual('rotting-oranges', draft, review);
