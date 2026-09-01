import { array, defineVisual, frame, mark, visual } from '../primitives.mjs';

const original = 43261596;
const originalBits = original.toString(2).padStart(32, '0');
let value = original;
let answer = 0;
const iterations = [];

for (let iteration = 0; iteration < 32; iteration += 1) {
  const bit = value & 1;
  const beforeAnswer = answer;
  answer = (answer << 1) | bit;
  value >>>= 1;
  iterations.push({
    iteration,
    bit,
    beforeAnswer,
    answer,
    remaining: value,
  });
}

const draft = visual('Across exactly 32 iterations, consume input bits from least significant to most significant and append each bit to the growing answer.', [
  frame(
    'Initialize the 32-bit word',
    'Input 43261596 is 00000010100101000001111010011100₂. Set answer = 0 before reading bit position 0.',
    array(originalBits.split(''), [mark(31, 'next: bit 0', 'focus', 'read-cursor')], {
      input: '43261596',
      value: originalBits,
      answer: '00000000000000000000000000000000',
    }),
    'initialize',
  ),
  ...iterations.map(({ iteration, bit, beforeAnswer, answer: nextAnswer, remaining }) => {
    const readIndex = 31 - iteration;
    const outputBits = nextAnswer.toString(2).padStart(32, '0');
    return frame(
      `Read bit ${iteration}: ${bit}`,
      `Iteration ${iteration + 1}: answer = (${beforeAnswer} << 1) | ${bit} = ${nextAnswer}; value shifts right to ${remaining}.`,
      array(originalBits.split(''), [mark(readIndex, `read #${iteration + 1}: ${bit}`, iteration === 31 ? 'output' : 'focus', 'read-cursor')], {
        sourceBitPosition: String(iteration),
        appendOperation: `(${beforeAnswer} << 1) | ${bit} = ${nextAnswer}`,
        answerBits: outputBits,
        remainingInput: remaining.toString(2).padStart(32, '0'),
        ...(iteration === 31 ? { result: '964176192' } : {}),
      }),
      `iteration-${iteration + 1}`,
    );
  }),
]);

export default defineVisual('reverse-bits', draft, {
  pattern: 'Fixed-width bit stream reversal by consume-and-append.',
  recognitionCue: 'The task reverses all 32 positions, including leading zeros, so processing must run a fixed number of times rather than stopping when the numeric input becomes zero.',
  invariant: 'After k iterations, answer contains the original k least-significant bits in reverse read order, while value contains the original bits not yet consumed after k right shifts.',
  stateModel: 'Keep the remaining input integer, the growing answer integer, and a fixed loop counter from 0 through 31. Each step derives one low bit with value & 1.',
  visualRationale: 'A 32-cell fixed-width bit row preserves leading zeros and bit positions. The authored read-cursor key moves from the least-significant end to the most-significant end while every append and shift is printed.',
  rejectedAlternatives: [
    'An 8-bit illustration does not verify the supplied 32-iteration implementation or its leading zeros.',
    'Converting to a string and reversing characters depicts a different algorithm and storage model.',
    'A single mirrored before-and-after row skips all consume, append, and shift transitions.',
  ],
  transferLesson: 'For fixed-width encodings, separate reading order from numeric termination: consume one boundary unit, shift the destination, append, and repeat for the declared width. The pattern transfers to radix conversion and stream packing.',
  independentReview: '3.4 source-to-frame replay',
  reviewStatus: 'reviewed',
});
