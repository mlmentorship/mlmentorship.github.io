import { defineVisual, frame, grid, visual } from '../primitives.mjs';

const scene = (rows, row, col, extra = {}) => grid(rows, [
  { row, col, label: 'fill cursor', tone: extra.result ? 'output' : 'focus', key: 'fill-cursor' },
], {
  input: 'sequences=[[3,4],[],[9]], pad_value=0',
  ...extra,
});

const draft = visual('Allocate one batch-width rectangle filled with pad values, then copy each sequence and mark the identical half-open slice as valid.', [
  frame('Find the rectangular width', 'The sequence lengths are 2, 0, and 1, so width=max(2,0,1)=2 and the output shape is (3,2).', scene([
    ['row 0: 3', 'row 0: 4'],
    ['row 1: empty', 'row 1: empty'],
    ['row 2: 9', 'row 2: empty'],
  ], 0, 0, {
    lengths: '[2,0,1]',
    arithmetic: 'max(2,0,1) = 2',
  }), 'measure-width'),
  frame('Allocate tokens and mask once', 'np.full creates three rows of [0,0]; np.zeros creates a matching all-false Boolean mask.', scene([
    ['0 | false', '0 | false'],
    ['0 | false', '0 | false'],
    ['0 | false', '0 | false'],
  ], 0, 0, {
    shapes: 'tokens (3,2); mask (3,2)',
  }), 'allocate'),
  frame('Fill row 0 through slice [0:2]', 'Copy [3,4] into tokens[0,:2] and set mask[0,:2]=True. Token and mask boundaries are identical.', scene([
    ['3 | true', '4 | true'],
    ['0 | false', '0 | false'],
    ['0 | false', '0 | false'],
  ], 0, 1, {
    assignment: 'tokens[0,:2]=[3,4]; mask[0,:2]=True',
  }), 'fill-row-zero'),
  frame('Process the empty row', 'For row 1, len(sequence)=0, so both [1,:0] assignments are empty slices and no cell changes.', scene([
    ['3 | true', '4 | true'],
    ['0 | false', '0 | false'],
    ['0 | false', '0 | false'],
  ], 1, 0, {
    assignment: 'tokens[1,:0]=[]; mask[1,:0]=True',
    decision: 'empty slice changes nothing',
  }), 'keep-empty-row'),
  frame('Fill row 2 through slice [0:1]', 'Copy [9] into the first token cell and mark only that position true; the second cell remains padding.', scene([
    ['3 | true', '4 | true'],
    ['0 | false', '0 | false'],
    ['9 | true', '0 | false'],
  ], 2, 0, {
    assignment: 'tokens[2,:1]=[9]; mask[2,:1]=True',
  }), 'fill-row-two'),
  frame('Return aligned token and mask rectangles', 'The mask distinguishes the real token 0 convention from padding: every true cell came from input and every false cell remained allocated padding.', scene([
    ['3 | true', '4 | true'],
    ['0 | false', '0 | false'],
    ['9 | true', '0 | false'],
  ], 2, 1, {
    tokens: '[[3,4],[0,0],[9,0]]',
    mask: '[[T,T],[F,F],[T,F]]',
    result: 'tokens shape (3,2), mask shape (3,2)',
  }), 'return'),
]);

const review = {
  pattern: 'Allocate a maximum-width batch rectangle once, then fill aligned token and validity slices.',
  recognitionCue: 'Use padding plus a mask when variable-length sequences must enter rectangular tensor operations without treating synthetic pad cells as data.',
  invariant: 'After processing row r, tokens[r,:len(sequence)] equals the original sequence and mask is true on exactly that slice; every remaining cell retains the pad value and false mask.',
  stateModel: 'Keep the batch size, maximum width, prefilled token matrix, all-false mask, current row, and that sequence’s length.',
  visualRationale: 'Each grid cell displays token and Boolean validity together while a stable fill cursor moves through real row slices, including the zero-length assignment and untouched padding.',
  rejectedAlternatives: [
    'A shape-only tensor diagram cannot verify copied values or mask alignment.',
    'Padding without a mask loses the distinction between a real pad-valued token and synthetic padding.',
    'Appending rows dynamically hides the supplied allocate-once slice-assignment mechanism.',
  ],
  transferLesson: 'For ragged-to-dense conversion, derive one shared extent, initialize safe defaults, and update data and validity metadata with the exact same slices.',
  reviewStatus: 'reviewed',
};

export default defineVisual('pad-variable-length-sequences', draft, review);
