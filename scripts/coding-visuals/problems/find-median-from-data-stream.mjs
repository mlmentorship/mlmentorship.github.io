import { defineVisual, frame, graph, table, visual } from '../primitives.mjs';

const scene = (nodes, edges, extra = {}) => graph(nodes, edges, {
  stream: '[5,2,10,4]',
  ...extra,
});

const draft = visual('The lower max-heap owns the extra item, and its root plus the upper min-heap root are always the middle value or values.', [
  frame(
    'Initialize two empty heaps',
    'lower and upper both start empty. Python stores negated values in lower so its minimum stored key represents the largest lower-half value.',
    table(['heap', 'stored values', 'logical role'], [
      ['lower', '[]', 'max-heap via negatives'],
      ['upper', '[]', 'min-heap'],
    ], [], { stream: '[5,2,10,4]' }),
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
    'heappop(lower) returns -5; negate it and push 5 into upper. lower is empty and upper is [5].',
    scene(['5'], [], { current: '5', lowerStored: '[]', lowerLogical: '[]', upper: '[5]', roots: ['upper=5'], phase: 'lower->upper' }),
    'transfer-5-upper',
  ),
  frame(
    'Rebalance 5 back to lower',
    'upper has size 1 and lower size 0, so pop upper root 5 and push -5 into lower. Sizes become 1 and 0.',
    scene(['5'], [], { lowerStored: '[-5]', lowerLogical: '[5]', upper: '[]', roots: ['lower=5'], balance: '1 >= 0 and difference=1' }),
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
    'Pop stored -5, leaving lower [-2], and push logical 5 into upper. Both heaps now have size 1, so the rebalance condition is false.',
    scene(['5', '2'], [], { lowerStored: '[-2]', lowerLogical: '[2]', upper: '[5]', roots: ['lower=2', 'upper=5'], rebalance: '1 > 1 is false' }),
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
    'heappush(lower, -10) produces stored [-10,-2], temporarily placing logical 10 above 2 in lower.',
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
    'Because upper is larger, pop 5 and push -5 into lower. Stored lower [-5,-2] represents [5,2]; upper becomes [10].',
    scene(['5', '2', '10'], ['5-2'], { lowerStored: '[-5,-2]', lowerLogical: '[5,2]', upper: '[10]', roots: ['lower=5', 'upper=10'], balance: 'sizes 2 and 1' }),
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
    'Pop stored -5, leaving lower [-4,-2], and push 5 into upper [5,10]. Sizes are equal, so no rebalance occurs.',
    scene(['5', '2', '10', '4'], ['4-2', '5-10'], { lowerStored: '[-4,-2]', lowerLogical: '[4,2]', upper: '[5,10]', roots: ['lower=4', 'upper=5'], rebalance: '2 > 2 is false' }),
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
  invariant: 'Every lower value is at most every upper value, and lower has either the same size as upper or exactly one extra item. Therefore the middle values are always heap roots.',
  stateModel: 'The minimal state is lower and upper. Each insertion pushes a negated value to lower, transfers its maximum to upper, then moves upper minimum back only if upper became larger.',
  visualRationale: 'Explicit parent-child edges depict each live heap topology while labels show Python stored negatives, logical lower values, upper values, roots, sizes, and median arithmetic. Stable value nodes persist as ownership changes.',
  rejectedAlternatives: [
    'A fully sorted array was rejected because insertion would be linear and would not expose the supplied heap mechanism.',
    'One heap was rejected because it cannot expose both middle boundaries efficiently.',
    'Only post-insertion snapshots were rejected because they hide the mandatory lower-to-upper transfer and conditional rebalance.',
  ],
  transferLesson: 'Maintain an ordered partition whose size difference is bounded, then answer rank queries from boundary roots. The same balancing idea supports streaming quantiles and running order statistics.',
  reviewStatus: 'reviewed',
};

export default defineVisual('find-median-from-data-stream', draft, review);
