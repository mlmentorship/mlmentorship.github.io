import { array, defineVisual, frame, mark, visual } from '../primitives.mjs';

const rows = [
  { value: 1, shifted: 0, low: 1, answer: [0, 1, 0, 0, 0, 0, 0] },
  { value: 2, shifted: 1, low: 0, answer: [0, 1, 1, 0, 0, 0, 0] },
  { value: 3, shifted: 1, low: 1, answer: [0, 1, 1, 2, 0, 0, 0] },
  { value: 4, shifted: 2, low: 0, answer: [0, 1, 1, 2, 1, 0, 0] },
  { value: 5, shifted: 2, low: 1, answer: [0, 1, 1, 2, 1, 2, 0] },
  { value: 6, shifted: 3, low: 0, answer: [0, 1, 1, 2, 1, 2, 2] },
];

const draft = visual('The set-bit count for value reuses the completed count for value >> 1 and adds value & 1.', [
  frame(
    'Initialize answer[0]',
    'For limit 6, allocate seven cells. Zero has no set bits, so answer[0] = 0 before the loop starts.',
    array(['0', '?', '?', '?', '?', '?', '?'], [mark(0, 'base value 0', 'state', 'current-value')], {
      input: 'limit = 6',
      invariant: 'entries below the next value are complete',
    }),
    'initialize',
  ),
  ...rows.map(({ value, shifted, low, answer }) => frame(
    `Compute answer[${value}]`,
    `${value} is ${value.toString(2)}₂: answer[${value}] = answer[${shifted}] + ${low} = ${answer[shifted]} + ${low} = ${answer[value]}.`,
    array(answer.map((count, index) => index <= value ? String(count) : '?'), [
      mark(shifted, `dependency ${shifted}`, 'state', 'shifted-value'),
      mark(value, `write ${value}`, value === 6 ? 'output' : 'focus', 'current-value'),
    ], {
      shift: `${value} >> 1 = ${shifted}`,
      lowBit: `${value} & 1 = ${low}`,
      recurrence: `answer[${value}] = ${answer[shifted]} + ${low} = ${answer[value]}`,
      ...(value === 6 ? { result: '[0, 1, 1, 2, 1, 2, 2]' } : {}),
    }),
    `value-${value}`,
  )),
]);

export default defineVisual('counting-bits', draft, {
  pattern: 'One-dimensional DP using the number with its least-significant bit removed.',
  recognitionCue: 'The task requests bit counts for every integer in an increasing range, so each answer can reuse a smaller already-computed integer instead of recounting from scratch.',
  invariant: 'Before computing answer[value], all lower indices are correct. Because value >> 1 is smaller, its count is available, and value & 1 contributes exactly the removed low bit.',
  stateModel: 'The output array is the DP state. The current value, dependency index value >> 1, and low bit derive the next entry; no auxiliary table is needed.',
  visualRationale: 'An indexed answer array directly shows fill order and the backward dependency. Stable current-value and shifted-value keys move through all six writes, with binary and decimal recurrence text available statically.',
  rejectedAlternatives: [
    'Independent bit scans for every number hide reuse and cost more total work.',
    'A binary tree of shifts duplicates the same smaller values across branches.',
    'A final count table omits the ordered writes required by the implementation.',
  ],
  transferLesson: 'When a bit operation maps each state to a smaller state, build answers in numeric order and append the removed-bit contribution; similar recurrences use x & (x - 1), highest powers of two, or parity.',
  reviewStatus: 'reviewed',
});
