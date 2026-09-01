import { array, defineVisual, frame, mark, visual } from '../primitives.mjs';

const places = ['8', '4', '2', '1'];
const example = 'first = 7 (0111), second = 5 (0101), shown in 4 bits';
const bit = (index, label, tone = 'focus', key = 'active-bit') => mark(index, label, tone, key);
const state = (marks, extra = {}) => array(places, marks, { example, columns: '8 4 2 1', ...extra });

const draft = visual('XOR writes sum-without-carry bits while shifted AND carries shared 1 bits into the next pass.', [
  frame(
    'Initialize the two operands',
    'The loop starts with first=0111 and second=0101. Because second is nonzero, another carry-resolution pass is required.',
    state([bit(3, 'inspect operands')], { first: '0111 (7)', second: '0101 (5)', mask: '1111…1111 (32 bits)' }),
    'initialize-operands',
  ),
  frame(
    'Compute pass 1',
    '0111 XOR 0101 = 0010. Their shared 1 bits are 0101, and shifting left gives carry 1010.',
    state([bit(2, 'xor bit', 'state', 'sum-bit'), bit(0, 'shifted carry', 'focus', 'carry-bit')], { xor: '0111 ^ 0101 = 0010', andShift: '(0111 & 0101) << 1 = 1010' }),
    'compute-first-pass',
  ),
  frame(
    'Commit pass 1 simultaneously',
    'The tuple assignment commits first=0010 and second=1010 together; the old operands supplied both expressions.',
    state([bit(2, 'first has 1', 'state', 'sum-bit'), bit(0, 'second carry', 'focus', 'carry-bit')], { first: '0010 (2)', second: '1010 (10)' }),
    'commit-first-pass',
  ),
  frame(
    'Resolve the remaining shared bit',
    '0010 XOR 1010 = 1000. The operands share the 2-bit, so (0010 AND 1010) shifted left is 0100.',
    state([bit(0, 'xor result', 'state', 'sum-bit'), bit(1, 'carry moves here', 'focus', 'carry-bit')], { xor: '0010 ^ 1010 = 1000', andShift: '(0010 & 1010) << 1 = 0100' }),
    'compute-second-pass',
  ),
  frame(
    'Commit pass 2',
    'The next loop state is first=1000 and second=0100. Carry is still nonzero, so continue.',
    state([bit(0, 'first has 1', 'state', 'sum-bit'), bit(1, 'second carry', 'focus', 'carry-bit')], { first: '1000 (8)', second: '0100 (4)' }),
    'commit-second-pass',
  ),
  frame(
    'Clear the final carry',
    '1000 and 0100 have no shared 1 bit: XOR is 1100 and shifted AND is 0000.',
    state([bit(0, 'sum bit', 'output', 'sum-bit'), bit(1, 'sum bit', 'output', 'carry-bit')], { xor: '1000 ^ 0100 = 1100', andShift: '(1000 & 0100) << 1 = 0000', first: '1100 (12)', second: '0000' }),
    'clear-final-carry',
  ),
  frame(
    'Return the 32-bit result',
    'second is zero, so the loop stops. 12 is below the sign bit 0x80000000, therefore return first directly.',
    state([bit(0, '8', 'output', 'sum-bit'), bit(1, '4', 'output', 'carry-bit')], { binary: '1100', decimal: '8 + 4 = 12', signCheck: '12 < 0x80000000', result: '12' }),
    'return-twelve',
  ),
]);

const review = {
  pattern: 'Iterative binary addition: XOR for provisional sum and shifted AND for carry.',
  recognitionCue: 'The prompt forbids arithmetic addition and subtraction but permits bitwise operators on fixed-width signed integers.',
  invariant: 'At every loop boundary, first plus second modulo 2^32 equals the original sum; each pass moves unresolved carries left until second becomes zero.',
  stateModel: 'Retain only two masked 32-bit words: first holds sum bits not yet combined with carry, and second holds the shifted carry. The example displays four low bits because 7+5 does not touch higher bits.',
  visualRationale: 'Named bit-place columns with stable sum-bit and carry-bit keys make the carry move left across passes and keep all binary arithmetic readable without color.',
  rejectedAlternatives: [
    'A decimal number line uses addition geometry and hides XOR/AND behavior.',
    'A truth table explains one bit but does not show carry propagation across repeated word-level passes.',
    'Showing only 7 -> 12 skips the simultaneous tuple update and the termination condition.',
  ],
  transferLesson: 'Separate a binary operation into local output and deferred carry state; the same decomposition underlies half/full adders, bit-mask arithmetic, and fixed-width overflow reasoning.',
  reviewStatus: 'reviewed',
};

export default defineVisual('sum-of-two-integers', draft, review);
