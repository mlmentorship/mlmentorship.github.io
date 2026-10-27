import { array, defineVisual, frame, mark, visual } from '../primitives.mjs';

const bitCells = (value) => value.toString(2).padStart(8, '0').split('');
const bitView = { fitArray: true, indexLabels: [7,6,5,4,3,2,1,0] };
const clearFrame = (before, after, count, position, key) => frame(
  `Clear set bit ${count}`,
  `${before} & (${before} - 1) = ${after}; exactly the lowest remaining 1 changes to 0.`,
  array(bitCells(after), [mark(position, `b=${7 - position}`, after === 0 ? 'output' : 'focus', 'lowest-set-bit')], {
    ...bitView,
    input: 'value = 45 (00101101)',
    binaryOperation: `${before.toString(2).padStart(8, '0')} & ${(before - 1).toString(2).padStart(8, '0')} = ${after.toString(2).padStart(8, '0')}`,
    decimalOperation: `${before} & ${before - 1} = ${after}`,
    count: String(count),
    ...(after === 0 ? { result: '4' } : {}),
  }),
  key,
);

const draft = visual('Each value & (value - 1) update removes one lowest set bit, so the number of loop iterations is the Hamming weight.', [
  frame(
    'Initialize value and count',
    'Start at 45 = 00101101 with count zero; the next removed bit is bit 0.',
    array(bitCells(45), [mark(7, 'b=0', 'focus', 'lowest-set-bit')], {
      ...bitView,
      value: '45',
      count: '0',
      loopCondition: '45 != 0',
    }),
    'initialize',
  ),
  clearFrame(45, 44, 1, 7, 'clear-1'),
  clearFrame(44, 40, 2, 5, 'clear-2'),
  clearFrame(40, 32, 3, 4, 'clear-3'),
  clearFrame(32, 0, 4, 2, 'clear-4'),
]);

export default defineVisual('number-of-1-bits', draft, {
  pattern: 'Brian Kernighan bit counting: repeatedly remove the lowest set bit.',
  recognitionCue: 'The task asks for the population count of an integer, and work should scale with the number of 1 bits rather than the fixed bit width.',
  invariant: 'After count iterations, count equals the number of 1 bits removed from the original value, and the current value contains every original set bit not yet removed.',
  stateModel: 'Keep only the changing integer value and an integer count. Subtracting one flips the lowest 1 and all lower zeros; AND with the original preserves higher bits and clears that one 1.',
  visualRationale: 'Eight indexed bit cells show the exact operands and result of every loop update, while the authored lowest-set-bit key moves to each bit that is removed. Decimal equations make the trace readable without color.',
  rejectedAlternatives: [
    'Scanning all bit positions does not expose why runtime depends on the number of set bits.',
    'A decimal-only accumulator hides the bit pattern changed by value & (value - 1).',
    'A single before-and-after diagram skips three executed loop transitions.',
  ],
  transferLesson: 'Use x & (x - 1) whenever progress is defined by deleting one set bit; related uses include power-of-two tests, iterating subsets, and finding the lowest set-bit contribution.',
  independentReview: '3.4 source-to-frame replay',
  reviewStatus: 'reviewed',
});
