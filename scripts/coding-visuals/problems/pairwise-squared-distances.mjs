import { defineVisual, frame, grid, visual } from '../primitives.mjs';

const motion = (label, x = 0, y = 0) => [{ key: 'pair-cell', kind: 'tensor-cell', x, y, label }];

const draft = visual('Broadcast every point-center pair across a feature axis, square coordinate differences, then reduce that axis.', [
  frame('Lay out points and centers', 'Use points P0=(0,0), P1=(1,2) and centers C0=(1,0), C1=(2,2). The output must contain all four P-C pairs.', grid([
    ['P0 (0,0)', 'P0 (0,0)'],
    ['P1 (1,2)', 'P1 (1,2)'],
  ], [], {
    columns: 'C0 (1,0), C1 (2,2)',
    shape: 'pair grid [n=2,k=2]',
    motion: motion('P0,C0'),
  }), 'inputs'),
  frame('Broadcast coordinate differences', 'P[:,None,:] - C[None,:,:] produces one 2-vector per grid cell: P0-C0=(-1,0), P0-C1=(-2,-2), P1-C0=(0,2), P1-C1=(-1,0).', grid([
    ['(-1,0)', '(-2,-2)'],
    ['(0,2)', '(-1,0)'],
  ], [{ row: 0, col: 0, label: 'pair P0,C0', tone: 'focus', key: 'difference-cell' }], {
    shape: 'differences [2,2,2]',
    motion: motion('P0-C0', 0, 0),
  }), 'differences'),
  frame('Square each feature', 'Elementwise multiplication differences*differences gives (1,0), (4,4), (0,4), and (1,0); signs disappear but the feature axis remains.', grid([
    ['(1,0)', '(4,4)'],
    ['(0,4)', '(1,0)'],
  ], [{ row: 0, col: 1, label: '4,4', tone: 'focus', key: 'squared-cell' }], {
    operation: 'elementwise square',
    motion: motion('(P0-C1)^2', 1, 0),
  }), 'squares'),
  frame('Reduce the feature axis', 'Sum each pair vector over d: 1+0=1, 4+4=8, 0+4=4, and 1+0=1. Rows are points and columns are centers.', grid([
    ['1', '8'],
    ['4', '1'],
  ], [{ row: 1, col: 1, label: 'distance', tone: 'output', key: 'distance-cell' }], {
    arithmetic: 'sum over final axis d',
    motion: motion('distance P1,C1', 1, 1),
    result: '[[1,8],[4,1]]',
  }), 'reduce'),
]);

const review = {
  pattern: 'Tensor broadcasting with complementary singleton axes followed by an elementwise transform and feature-axis reduction.',
  recognitionCue: 'Use it when every item in one batch must pair with every item in another batch and the per-pair calculation is identical across a shared feature dimension.',
  invariant: 'Cell [i,j,:] always belongs to point i and center j; broadcasting changes alignment, not values, and reducing only the final feature axis preserves the pair grid.',
  stateModel: 'The minimal state is points [n,d], centers [k,d], broadcast differences [n,k,d], and output [n,k]. The fixed 2-by-2 grid maintains pair identity across stages.',
  visualRationale: 'A point-by-center grid containing actual feature vectors exposes pair geometry and the reduced axis. Labels and arithmetic remain complete in monochrome static output.',
  rejectedAlternatives: [
    'Shape labels alone were rejected because they cannot verify signs, squares, or resulting distances.',
    'Nested Python loops were rejected because they hide how singleton axes vectorize the same pair grid.',
    'A scatterplot was rejected because it shows Euclidean geometry but not tensor axis alignment or output indexing.',
  ],
  transferLesson: 'For all-pairs tensor operations, insert singleton axes where each operand should repeat, verify the broadcasted axis meaning, and reduce only the intended feature axes.',
  reviewStatus: 'reviewed',
};

export default defineVisual('pairwise-squared-distances', draft, review);
