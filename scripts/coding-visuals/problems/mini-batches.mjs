import { array, defineVisual, frame, mark, visual } from '../primitives.mjs';

const items = ['A', 'B', 'C', 'D', 'E', 'F', 'G'];
const example = 'items = [A,B,C,D,E,F,G], batch_size = 3';
const state = (start, end, extra = {}) => array(
  items,
  [
    mark(start, `start=${start}`, 'focus', 'slice-start'),
    mark(Math.min(end - 1, items.length - 1), `stop=${end} exclusive`, 'state', 'slice-stop'),
  ],
  { example, coveredRange: `[${start}, ${Math.min(end, items.length)})`, direction: `start advances by +3`, ...extra },
);

const draft = visual('Advance start by batch_size and let slicing clamp the final exclusive stop to the sequence length.', [
  frame('Validate the batch size', 'batch_size=3 is positive, so the generator may enter range(0,7,3).', state(0, 3, { check: '3 <= 0: false', starts: '[0,3,6]' }), 'validate-size'),
  frame('Yield slice 0:3', 'At start 0, items[0:3] contains A, B, C and covers indices [0,3).', state(0, 3, { expression: 'items[0 : 0+3]', yielded: '[A,B,C]' }), 'yield-first'),
  frame('Advance to start 3', 'range adds batch_size: start moves from 0 to 3 without overlap or a gap.', state(3, 6, { movement: '0 + 3 = 3' }), 'advance-three'),
  frame('Yield slice 3:6', 'items[3:6] contains D, E, F and covers the next disjoint range [3,6).', state(3, 6, { expression: 'items[3 : 3+3]', yielded: '[D,E,F]' }), 'yield-second'),
  frame('Advance to start 6', 'range adds 3 again, moving start from 3 to 6 for the remaining item.', state(6, 9, { movement: '3 + 3 = 6' }), 'advance-six'),
  frame('Yield the short final slice', 'The requested stop is 9, but Python clamps items[6:9] at len(items)=7, yielding [G] rather than dropping it.', state(6, 9, { expression: 'items[6:9] -> items[6:7]', yielded: '[G]', finalLength: '1' }), 'yield-remainder'),
  frame('Stop after exhausting range', 'The next start would be 9, which is outside range(0,7,3); all seven items were yielded exactly once.', state(6, 9, { nextStart: '9 >= 7', result: '[[A,B,C],[D,E,F],[G]]' }), 'stop-generator'),
]);

const review = {
  pattern: 'Fixed-stride iteration with half-open sequence slices.',
  recognitionCue: 'Examples must be emitted in bounded groups while preserving order and retaining a remainder smaller than the requested batch size.',
  invariant: 'Before each yield, indices below start were yielded exactly once; the next slice covers [start,min(start+batch_size,n)), and the next start equals the previous requested stop.',
  stateModel: 'Retain the input sequence, positive batch_size, moving start, and exclusive requested stop. Slicing itself handles the final boundary.',
  visualRationale: 'An indexed array with stable start and stop markers shows half-open coverage, +batch_size movement, and why the final slice safely clamps.',
  rejectedAlternatives: [
    'Pre-drawn batch boxes hide the generator’s moving start and slice semantics.',
    'A modulo grouping table complicates a direct stride-and-slice loop.',
    'A full tensor diagram implies equal batch shapes and can conceal the short remainder.',
  ],
  transferLesson: 'Use half-open slices and a fixed stride to partition ordered data exactly once; this transfers to pagination, chunked I/O, windowed inference, and streaming uploads.',
  reviewStatus: 'reviewed',
};

export default defineVisual('mini-batches', draft, review);
