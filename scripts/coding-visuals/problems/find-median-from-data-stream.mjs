import { defineVisual, frame, heap, table, visual } from '../primitives.mjs';

const valuesFrom = (value) => JSON.parse(value ?? '[]').map(String);
const scene = (nodes, _edges, extra = {}) => {
  const lower = valuesFrom(extra.lowerLogical);
  const upper = valuesFrom(extra.upper);
  const ordered = !lower.length || !upper.length || Number(lower[0]) <= Number(upper[0]);
  const balanced = lower.length === upper.length || lower.length === upper.length + 1;
  return heap([...lower, ...upper], {
    groups: [
      { label: 'lower max-heap', values: lower },
      { label: 'upper min-heap', values: upper },
    ],
    stream: '[5,2,10,4]',
    seen: nodes.join(', '),
    status: extra.complete || extra.result ? 'completed insertion: query allowed' : 'insertion repair: not a query point',
    partitionOrder: ordered ? 'all lower <= all upper' : `violated: lower root ${lower[0]} > upper root ${upper[0]}`,
    sizes: `lower ${lower.length}, upper ${upper.length}: ${balanced ? 'final size rule holds' : 'final size rule violated'}`,
    rootSummary: `lower ${lower[0] ?? 'none'}, upper ${upper[0] ?? 'none'}`,
    ...extra,
  });
};

const draft = visual('After completed insertions, ordered halves and balanced sizes expose the middle ranks at the heap roots.', [
  frame(
    'Initialize two empty heaps',
    'Both heaps start empty, so a median query is undefined. Negated lower values make its stored minimum the logical maximum.',
    table(['heap', 'stored values', 'logical role'], [
      ['lower', '[]', 'max-heap via negatives'],
      ['upper', '[]', 'min-heap'],
    ], [], { stream: '[5,2,10,4]', status: 'initialized: no median yet' }),
    'initialize-heaps',
  ),
  frame(
    'Push 5 into lower',
    'heappush(lower, -5) produces lower stored [-5], whose logical max-root is 5.',
    scene(['5'], [], { current: '5', lowerStored: '[-5]', lowerLogical: '[5]', upper: '[]', phase: 'push lower' }),
    'push-5-lower',
  ),
  frame(
    'Transfer 5 to upper',
    'Pop stored negative five from lower. Push logical five into upper. Sizes zero and one still need repair.',
    scene(['5'], [], { current: '5', lowerStored: '[]', lowerLogical: '[]', upper: '[5]', roots: ['upper=5'], phase: 'lower->upper' }),
    'transfer-5-upper',
  ),
  frame(
    'Rebalance 5 back to lower',
    'Upper is larger, so transfer its root five back to lower. Sizes one and zero complete the insertion.',
    scene(['5'], [], { lowerStored: '[-5]', lowerLogical: '[5]', upper: '[]', roots: ['lower=5'], balance: '1 >= 0 and difference=1', complete: true }),
    'rebalance-5-lower',
  ),
  frame(
    'Read odd median 5',
    'len(lower)=1 is greater than len(upper)=0, so find_median returns float(-lower[0]) = -(-5) = 5.0.',
    scene(['5'], [], { lowerStored: '[-5]', lowerLogical: '[5]', upper: '[]', branch: 'lower larger', result: '5.0' }),
    'median-after-5',
  ),
  frame(
    'Push 2 into lower',
    'heappush(lower, -2) changes stored lower to [-5,-2], representing max-heap root 5 with child 2.',
    scene(['5', '2'], ['5-2'], { current: '2', lowerStored: '[-5,-2]', lowerLogical: '[5,2]', upper: '[]', phase: 'push lower' }),
    'push-2-lower',
  ),
  frame(
    'Transfer lower root 5 to upper',
    'Transfer lower maximum five to upper. Both heaps now have size one. No further repair is needed.',
    scene(['5', '2'], [], { lowerStored: '[-2]', lowerLogical: '[2]', upper: '[5]', roots: ['lower=2', 'upper=5'], rebalance: '1 > 1 is false', complete: true }),
    'transfer-5-after-2',
  ),
  frame(
    'Read even median 3.5',
    'The heap sizes are equal, so return (-lower[0] + upper[0]) / 2 = (2 + 5) / 2 = 3.5.',
    scene(['5', '2'], [], { lowerStored: '[-2]', lowerLogical: '[2]', upper: '[5]', arithmetic: '(2+5)/2', result: '3.5' }),
    'median-after-2',
  ),
  frame(
    'Push 10 into lower',
    'Pushing ten gives logical lower [10,2], above upper root five. Sizes fit, but partition order is temporarily wrong.',
    scene(['5', '2', '10'], ['10-2'], { current: '10', lowerStored: '[-10,-2]', lowerLogical: '[10,2]', upper: '[5]', phase: 'push lower' }),
    'push-10-lower',
  ),
  frame(
    'Transfer lower root 10 to upper',
    'Pop stored -10, leaving lower [-2]. Push 10 into upper [5,10], whose min-root remains 5.',
    scene(['5', '2', '10'], ['5-10'], { lowerStored: '[-2]', lowerLogical: '[2]', upper: '[5,10]', roots: ['lower=2', 'upper=5'], balance: 'upper size 2 > lower size 1' }),
    'transfer-10-upper',
  ),
  frame(
    'Rebalance upper root 5',
    'Upper is larger, so move its minimum five back to lower. Lower [5,2] and upper [10] satisfy both conditions.',
    scene(['5', '2', '10'], ['5-2'], { lowerStored: '[-5,-2]', lowerLogical: '[5,2]', upper: '[10]', roots: ['lower=5', 'upper=10'], balance: 'sizes 2 and 1', complete: true }),
    'rebalance-5-after-10',
  ),
  frame(
    'Read odd median 5',
    'lower has one extra value, so return -lower[0] = -(-5) = 5.0.',
    scene(['5', '2', '10'], ['5-2'], { lowerStored: '[-5,-2]', lowerLogical: '[5,2]', upper: '[10]', branch: 'lower larger', result: '5.0' }),
    'median-after-10',
  ),
  frame(
    'Push 4 into lower',
    'heappush(lower, -4) produces stored [-5,-2,-4], representing logical max-heap [5,2,4].',
    scene(['5', '2', '10', '4'], ['5-2', '5-4'], { current: '4', lowerStored: '[-5,-2,-4]', lowerLogical: '[5,2,4]', upper: '[10]', phase: 'push lower' }),
    'push-4-lower',
  ),
  frame(
    'Transfer lower root 5 to upper',
    'Transfer lower maximum five to upper. Lower [4,2] and upper [5,10] have equal sizes, completing the insertion.',
    scene(['5', '2', '10', '4'], ['4-2', '5-10'], { lowerStored: '[-4,-2]', lowerLogical: '[4,2]', upper: '[5,10]', roots: ['lower=4', 'upper=5'], rebalance: '2 > 2 is false', complete: true }),
    'transfer-5-after-4',
  ),
  frame(
    'Read final even median 4.5',
    'Equal heap sizes select both roots: (-lower[0] + upper[0]) / 2 = (4 + 5) / 2 = 4.5.',
    scene(['5', '2', '10', '4'], ['4-2', '5-10'], { lowerStored: '[-4,-2]', lowerLogical: '[4,2]', upper: '[5,10]', arithmetic: '(4+5)/2', result: '4.5' }),
    'median-after-4',
  ),
]);

const review = {
  pattern: 'Online median maintenance with a negated max-heap for the lower half and a min-heap for the upper half.',
  recognitionCue: 'Use two heaps when numbers arrive online and every median query must avoid sorting the entire history.',
  invariant: 'After each completed insertion, every lower value is at most every upper value, and lower has either equal size or one extra item. These two conditions expose the middle ranks at roots. Repair frames need not satisfy them.',
  stateModel: 'The minimal state is lower and upper. Each insertion pushes a negated value to lower, transfers its maximum to upper, then moves upper minimum back only if upper became larger.',
  visualRationale: 'Both actual heap trees and every original motion identity remain. Stored negative keys and logical roots are separate, and each frame explicitly labels completion or repair, partition order and sizes. The push-ten frame exposes a real order violation rather than claiming every transient root is already a median.',
  rejectedAlternatives: [
    'A fully sorted array was rejected because insertion would be linear and would not expose the supplied heap mechanism.',
    'One heap was rejected because it cannot expose both middle boundaries efficiently.',
    'Only post-insertion snapshots were rejected because they hide the mandatory lower-to-upper transfer and conditional rebalance.',
  ],
  transferLesson: 'Maintain an ordered partition whose size difference is bounded, then answer rank queries from boundary roots. The same balancing idea supports streaming quantiles and running order statistics.',
  reviewStatus: 'reviewed',
};

export default defineVisual('find-median-from-data-stream', draft, review);
