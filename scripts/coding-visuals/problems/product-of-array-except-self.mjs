import { array, defineVisual, frame, mark, visual } from '../primitives.mjs';

const values = (answer) => ['1', '2', '3', '4'].map((num, index) => `num ${num} / out ${answer[index]}`);
const scan = (index, phase) => [mark(index, phase, 'focus', 'scan')];

const draft = visual('At index i, combine the product strictly before i with the product strictly after i.', [
  frame(
    'Initialize the prefix pass',
    'For nums = [1, 2, 3, 4], output starts as [1, 1, 1, 1] and prefix starts at 1.',
    array(values(['1', '1', '1', '1']), scan(0, 'prefix i=0'), {
      phase: 'left to right',
      prefix: '1',
      operation: 'ready to write out[0]',
    }),
    'prefix-initialized',
  ),
  frame(
    'Write index 0',
    'Write out[0] = 1, then update prefix = 1 * nums[0] = 1.',
    array(values(['1', '1', '1', '1']), scan(0, 'prefix i=0'), {
      prefix: '1',
      operation: 'out[0] = 1; prefix = 1 * 1 = 1',
    }),
    'prefix-index-0',
  ),
  frame(
    'Write index 1',
    'Write out[1] = 1, then update prefix = 1 * nums[1] = 2.',
    array(values(['1', '1', '1', '1']), scan(1, 'prefix i=1'), {
      prefix: '2',
      operation: 'out[1] = 1; prefix = 1 * 2 = 2',
    }),
    'prefix-index-1',
  ),
  frame(
    'Write index 2',
    'Write out[2] = 2, then update prefix = 2 * nums[2] = 6.',
    array(values(['1', '1', '2', '1']), scan(2, 'prefix i=2'), {
      prefix: '6',
      operation: 'out[2] = 2; prefix = 2 * 3 = 6',
    }),
    'prefix-index-2',
  ),
  frame(
    'Write index 3',
    'Write out[3] = 6, then update prefix = 6 * nums[3] = 24; every cell now holds its left product.',
    array(values(['1', '1', '2', '6']), scan(3, 'prefix i=3'), {
      prefix: '24',
      operation: 'out[3] = 6; prefix = 6 * 4 = 24',
    }),
    'prefix-index-3',
  ),
  frame(
    'Multiply suffix at index 3',
    'Suffix starts at 1. Set out[3] = 6 * 1 = 6, then suffix = 1 * nums[3] = 4.',
    array(values(['1', '1', '2', '6']), scan(3, 'suffix i=3'), {
      suffix: '4',
      operation: 'out[3] = 6 * 1 = 6; suffix = 1 * 4 = 4',
    }),
    'suffix-index-3',
  ),
  frame(
    'Multiply suffix at index 2',
    'Set out[2] = 2 * 4 = 8, then suffix = 4 * nums[2] = 12.',
    array(values(['1', '1', '8', '6']), scan(2, 'suffix i=2'), {
      suffix: '12',
      operation: 'out[2] = 2 * 4 = 8; suffix = 4 * 3 = 12',
    }),
    'suffix-index-2',
  ),
  frame(
    'Multiply suffix at index 1',
    'Set out[1] = 1 * 12 = 12, then suffix = 12 * nums[1] = 24.',
    array(values(['1', '12', '8', '6']), scan(1, 'suffix i=1'), {
      suffix: '24',
      operation: 'out[1] = 1 * 12 = 12; suffix = 12 * 2 = 24',
    }),
    'suffix-index-1',
  ),
  frame(
    'Multiply suffix at index 0',
    'Set out[0] = 1 * 24 = 24. Every output now contains its left product times its right product.',
    array(values(['24', '12', '8', '6']), scan(0, 'suffix i=0'), {
      suffix: '24',
      operation: 'out[0] = 1 * 24 = 24; suffix = 24 * 1 = 24',
      result: '[24, 12, 8, 6]',
    }),
    'suffix-index-0',
  ),
]);

export default defineVisual('product-of-array-except-self', draft, {
  pattern: 'Prefix and suffix accumulation without division.',
  recognitionCue: 'Each output excludes exactly one array position, division is forbidden, and multiplication can be accumulated from either boundary.',
  invariant: 'Before prefix index i, prefix is the product of nums[0..i-1]. Before suffix index i, suffix is the product of nums[i+1..n-1], while out[i] already stores the left product.',
  stateModel: 'The output array stores completed left products; one scalar prefix scans right, then one scalar suffix scans left. No separate prefix or suffix arrays are required.',
  visualRationale: 'An indexed array whose cells pair each input with its current output makes both passes and every multiplication readable in static HTML; the stable scan key visibly reverses direction between passes.',
  rejectedAlternatives: [
    'A final prefix/suffix table hides the order in which the implementation writes and multiplies each cell.',
    'A division formula does not represent the supplied algorithm and breaks on zeros.',
    'A prose-only trace forces the reader to remember four changing output values.',
  ],
  transferLesson: 'When every answer excludes one position and the operation is associative, save the contribution from one side and sweep the other side with a scalar; the same decomposition works for left/right minima, maxima, and cumulative constraints.',
  reviewStatus: 'reviewed',
});
