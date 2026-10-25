import { defineVisual, frame, grid, visual } from '../primitives.mjs';

const source = (row, col, label, key) => ({ row, col, label, tone: 'focus', key });

const draft = visual('Seed every rotten cell at time zero. FIFO processes nearest-source arrivals. Mark each fresh cell before enqueueing it.', [
  frame(
    'Seed every time-zero source',
    'Enqueue source (0,0,0) and count six fresh cells. Values are 2 for rotten, 1 for fresh and 0 for empty.',
    grid([['2', '1', '1'], ['1', '1', '0'], ['0', '1', '1']], [source(0, 0, 't=0', 'wave-a')], { queueState: '[(0,0,0)]', fresh: '6', minute: '0' }),
    'seed-sources',
  ),
  frame(
    'Spread minute 1',
    'Pop (0,0,0). Schedule down neighbor (1,0), then right neighbor (0,1), both for time one. Fresh falls from six to four.',
    grid([['2', '2', '1'], ['2', '1', '0'], ['0', '1', '1']], [
      source(1, 0, 'front t=1', 'wave-a'),
      source(0, 1, 'back t=1', 'wave-b'),
    ], { queueState: '[(1,0,1),(0,1,1)]', fresh: '4', arithmetic: '6 - 2 = 4' }),
    'spread-minute-1',
  ),
  frame(
    'Spread minute 2',
    'Pop (1,0,1) and enqueue (1,1,2). Pop (0,1,1), skip discovered (1,1), then enqueue (0,2,2). Fresh falls from four to two.',
    grid([['2', '2', '2'], ['2', '2', '0'], ['0', '1', '1']], [
      source(1, 1, 'front t=2', 'wave-a'),
      source(0, 2, 'back t=2', 'wave-b'),
    ], { queueState: '[(1,1,2),(0,2,2)]', fresh: '2', arithmetic: '4 - 2 = 2' }),
    'spread-minute-2',
  ),
  frame(
    'Spread minute 3',
    'Pop the time-two cells. Only (2,1) is fresh and adjacent, so schedule it for time three. Fresh falls from two to one.',
    grid([['2', '2', '2'], ['2', '2', '0'], ['0', '2', '1']], [
      source(2, 1, 't=3', 'wave-b'),
    ], { queueState: '[(2,1,3)]', fresh: '1', arithmetic: '2 - 1 = 1' }),
    'spread-minute-3',
  ),
  frame(
    'Spread minute 4',
    'Pop (2,1,3) and schedule (2,2) for time four. Fresh reaches zero. Its mark records discovery, not arrival at time three.',
    grid([['2', '2', '2'], ['2', '2', '0'], ['0', '2', '2']], [
      source(2, 2, 't=4', 'wave-b'),
    ], { queueState: '[(2,2,4)]', fresh: '0', arithmetic: '1 - 1 = 0' }),
    'spread-minute-4',
  ),
  frame(
    'Return the last timestamp',
    'Pop (2,2,4). It adds no fresh neighbor. The queue empties with fresh=0, so return the last popped timestamp, four.',
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
