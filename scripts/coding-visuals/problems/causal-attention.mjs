import { attention, defineVisual, frame, visual } from '../primitives.mjs';

const motion = (label, x = 0, y = 0) => [{ key: 'attention-cell', kind: 'matrix-cell', x, y, label }];

const draft = visual('Form scaled query-key scores, mask the strict upper triangle before softmax, then use prefix-only weights to mix values.', [
  frame('Multiply queries by keys', 'Use Q=K=[[1,0],[0,1]] and V=[[10,0],[0,20]]. QK^T is the 2 by 2 identity score matrix.', attention([
    ['1', '0'],
    ['0', '1'],
  ], {
    axes: 'query rows x key rows',
    motion: motion('score q0,k0'),
  }), 'dot-products'),
  frame('Scale by square root of width', 'Key width is 2, so divide by sqrt(2). Scores become [[0.7071,0],[0,0.7071]].', attention([
    ['0.7071', '0'],
    ['0', '0.7071'],
  ], {
    scale: '1 / sqrt(2)',
    motion: motion('scaled q0,k0'),
  }), 'scale'),
  frame('Mask every future key', 'The strict upper triangle is future context. Set score (query 0,key 1) to -infinity before softmax; row 1 may still read keys 0 and 1.', attention([
    ['0.7071', 'mask'],
    ['0', '0.7071'],
  ], {
    rule: 'key index > query index -> -infinity',
    motion: motion('masked q0,k1', 1, 0),
  }), 'mask'),
  frame('Normalize each allowed prefix', 'Row 0 softmax is [1,0]. Row 1 softmax of [0,0.7071] is approximately [0.3302,0.6698]. Masked future weight is exactly zero.', attention([
    ['1.0000', 'mask'],
    ['0.3302', '0.6698'],
  ], {
    rowSums: '1.0000; 1.0000',
    motion: motion('weight q1,k1', 1, 1),
  }), 'softmax'),
  frame('Mix value rows', 'Query 0 output is 1*V0 = [10,0]. Query 1 output is 0.3302*V0 + 0.6698*V1 = [3.30,13.40].', attention([
    ['10.00', '0.00'],
    ['3.30', '13.40'],
  ], {
    operation: 'weights @ V',
    motion: motion('output token 1 feature 1', 1, 1),
    result: '[[10,0],[3.30,13.40]]',
  }), 'mix-values'),
]);

const review = {
  pattern: 'Scaled dot-product attention with a strict upper-triangular causal mask applied before row-wise softmax.',
  recognitionCue: 'Use it when each sequence position may aggregate only its own and earlier value rows, requiring future key positions to receive exactly zero normalized weight.',
  invariant: 'Attention row i contains scores for query i against every key; after masking and softmax, entries j>i are zero and allowed prefix weights j<=i sum to one.',
  stateModel: 'The minimal state is Q, K, V, square score matrix, upper-triangular Boolean mask, normalized weight matrix, and weight-value product.',
  visualRationale: 'A query-by-key matrix directly exposes causal topology, the masked upper triangle, and row normalization before the output mix. Numeric cells work without color or JavaScript.',
  rejectedAlternatives: [
    'A token-arrow diagram was rejected because it hides scaling, score values, and row normalization.',
    'A generic triangular mask was rejected because it does not verify that masking occurs before softmax or show resulting weights.',
    'Shape-only tensor boxes were rejected because they cannot validate the weighted value arithmetic.',
  ],
  transferLesson: 'For constrained attention, encode allowed information flow as a score-space mask before normalization so forbidden positions receive zero mass, then mix values with the resulting row-stochastic weights.',
  reviewStatus: 'reviewed',
};

export default defineVisual('causal-attention', draft, review);
