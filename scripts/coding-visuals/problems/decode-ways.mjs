import { array, defineVisual, frame, mark, visual } from '../primitives.mjs';

const digits = ['2', '1', '0', '1'];
const current = (index, tone = 'focus') => [mark(index, `index ${index}`, tone, 'current-digit')];

const draft = visual('At each digit, add one-back decodings for a valid single digit and two-back decodings for a valid 10..26 pair.', [
  frame('Initialize after the leading 2', 'For text "2101", the leading digit is nonzero. Set two_back = 1 and one_back = 1: prefix "2" has one decoding.', array(
    digits,
    current(0, 'state'),
    { prefix: '"2"', state: 'two_back = 1; one_back = 1' },
  ), 'initialize'),
  frame('Process index 1', 'Digit 1 is nonzero, so current starts with one_back = 1. Pair 21 is valid, so add two_back = 1: current = 2.', array(
    digits,
    current(1),
    { prefix: '"21"', arithmetic: 'single 1 + pair 1 = 2', update: '(two_back, one_back) = (1,2)' },
  ), 'process-one'),
  frame('Let zero use only pair 10', 'At index 2, digit 0 cannot stand alone, so current starts at 0. Pair 10 is valid and adds two_back = 1: current = 1.', array(
    digits,
    current(2),
    { prefix: '"210"', arithmetic: 'single 0 + pair 1 = 1', update: '(two_back, one_back) = (2,1)' },
  ), 'process-zero'),
  frame('Reject the leading-zero pair', 'At index 3, digit 1 is nonzero and contributes one_back = 1. Pair 01 is invalid, so it adds 0; the only decoding is 2|10|1.', array(
    digits,
    current(3, 'output'),
    { prefix: '"2101"', arithmetic: 'single 1 + invalid pair 0 = 1', result: '1' },
  ), 'process-final-one'),
]);

const review = {
  pattern: 'One-dimensional counting DP with one-digit and two-digit transitions compressed to two rolling values.',
  recognitionCue: 'Use it when each encoded token consumes one or two adjacent symbols, validity depends on the local symbol or pair, and the task asks for the number of complete parses.',
  invariant: 'Before index i, one_back counts decodings through i-1 and two_back through i-2; current sums exactly the valid single-digit and two-digit ways ending at i.',
  stateModel: 'The minimal state is index, two_back, one_back, and current. The indexed digit row exposes the current symbol and adjacent pair, including zero-specific invalid branches.',
  visualRationale: 'A fixed digit row with explicit rolling-state arithmetic directly shows which token widths contribute. It remains understandable without color, JavaScript, or memorized code.',
  rejectedAlternatives: [
    'A letter-mapping chart was rejected because it does not explain how parse counts combine.',
    'A decode tree was rejected because shared suffix states duplicate work and become exponential.',
    'Using only "226" was rejected because it never exercises the decisive zero and invalid-leading-zero branches.',
  ],
  transferLesson: 'For variable-width parsing, sum counts from earlier boundaries only when the token ending at the current boundary is valid; zeros often eliminate the width-one transition.',
  reviewStatus: 'reviewed',
};

export default defineVisual('decode-ways', draft, review);
