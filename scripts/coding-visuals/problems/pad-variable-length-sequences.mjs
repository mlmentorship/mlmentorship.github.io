import { defineVisual, frame, grid, visual } from '../primitives.mjs';

const example = 'sequences = [[3,4], [9], [5,6,7]], pad_value = -1';
const motion = (row, col, label) => [
  { key: 'row-cursor', kind: 'pointer', x: col, y: row, label },
  ...Array.from({ length: 9 }, (_, index) => ({
    key: `cell-${Math.floor(index / 3)}-${index % 3}`,
    kind: 'cell',
    x: index % 3,
    y: Math.floor(index / 3),
    label: `cell ${Math.floor(index / 3)},${index % 3}`,
  })),
];
const state = (rows, row, col, label, extra = {}) => grid(
  rows,
  [{ row, col, label, tone: 'focus', key: 'row-cursor' }],
  {
    example,
    cellFormat: 'token | mask (T valid, F padding)',
    ...extra,
    motion: motion(row, col, label),
  },
);

const allocated = [
  ['-1|F', '-1|F', '-1|F'],
  ['-1|F', '-1|F', '-1|F'],
  ['-1|F', '-1|F', '-1|F'],
];

const draft = visual('Allocate one rectangle, then copy each row and mark the identical slice valid.', [
  frame('Measure the longest sequence', 'Lengths are 2, 1, and 3, so width=max(2,1,3)=3 and batch size is 3.', state(allocated, 0, 0, 'measure lengths', { lengths: '[2,1,3]', width: '3', shape: '(3,3)' }), 'measure-width'),
  frame('Allocate tokens and mask', 'Fill tokens[3,3] with -1 and mask[3,3] with False before copying any real value.', state(allocated, 0, 0, 'allocated', { tokens: 'all -1', mask: 'all False' }), 'allocate-rectangles'),
  frame('Copy sequence 0', 'Assign tokens[0,:2]=[3,4]; the unused third token remains -1.', state([['3|F','4|F','-1|F'], ...allocated.slice(1)], 0, 1, 'copy tokens row 0', { slice: 'tokens[0, :2] = [3,4]' }), 'copy-row-zero'),
  frame('Mark sequence 0 valid', 'Assign mask[0,:2]=True at the same two columns; column 2 stays padding.', state([['3|T','4|T','-1|F'], ...allocated.slice(1)], 0, 1, 'mark mask row 0', { slice: 'mask[0, :2] = True' }), 'mask-row-zero'),
  frame('Copy sequence 1', 'Assign tokens[1,:1]=[9]; columns 1 and 2 remain -1.', state([['3|T','4|T','-1|F'], ['9|F','-1|F','-1|F'], allocated[2]], 1, 0, 'copy tokens row 1', { slice: 'tokens[1, :1] = [9]' }), 'copy-row-one'),
  frame('Mark sequence 1 valid', 'Only mask[1,0] becomes True because the sequence length is 1.', state([['3|T','4|T','-1|F'], ['9|T','-1|F','-1|F'], allocated[2]], 1, 0, 'mark mask row 1', { slice: 'mask[1, :1] = True' }), 'mask-row-one'),
  frame('Copy sequence 2', 'Assign tokens[2,:3]=[5,6,7], filling the full width.', state([['3|T','4|T','-1|F'], ['9|T','-1|F','-1|F'], ['5|F','6|F','7|F']], 2, 2, 'copy tokens row 2', { slice: 'tokens[2, :3] = [5,6,7]' }), 'copy-row-two'),
  frame('Mark sequence 2 valid', 'Assign mask[2,:3]=True, so all three copied cells are valid.', state([['3|T','4|T','-1|F'], ['9|T','-1|F','-1|F'], ['5|T','6|T','7|T']], 2, 2, 'mark mask row 2', { slice: 'mask[2, :3] = True' }), 'mask-row-two'),
  frame('Return aligned rectangles', 'Token -1 appears only where the mask is False; every original integer remains paired with True.', state([['3|T','4|T','-1|F'], ['9|T','-1|F','-1|F'], ['5|T','6|T','7|T']], 2, 2, 'complete', { tokens: '[[3,4,-1],[9,-1,-1],[5,6,7]]', mask: '[[T,T,F],[T,F,F],[T,T,T]]', result: 'tokens shape (3,3) + mask shape (3,3)' }), 'return-tokens-mask'),
]);

const review = {
  pattern: 'Measure maximum length, allocate dense arrays once, then fill matching row slices.',
  recognitionCue: 'A batch contains variable-length sequences but downstream vectorized computation requires one rectangular tensor plus validity information.',
  invariant: 'After row r is processed, its first len(sequence) token cells equal the input and the same mask cells are True; every remaining cell retains pad_value and False.',
  stateModel: 'Retain batch size, maximum width, prefilled token and Boolean arrays, current row, and that row’s length. No per-token append structure is needed.',
  visualRationale: 'A paired token|mask grid keeps shape, slice extent, copied values, and padding validity co-located while a stable row cursor moves through each assignment.',
  rejectedAlternatives: [
    'Ragged list cards show input lengths but not the rectangular output coordinates.',
    'Separate token and mask diagrams make corresponding cells harder to verify.',
    'A tensor-shape arrow omits the actual slice writes and pad values.',
  ],
  transferLesson: 'When rectangular storage introduces synthetic values, carry an aligned validity structure; this transfers to attention masks, packed batches, image padding, and missing-data arrays.',
  reviewStatus: 'reviewed',
};

export default defineVisual('pad-variable-length-sequences', draft, review);
