import { defineVisual, frame, stack, visual } from '../primitives.mjs';

const input = '[2,1,5,6,2,3] + sentinel 0';
const stackScene = (index, values, extra) => stack(input, values, {
  ...extra,
  motion: [
    { key: 'scan-index', kind: 'pointer', x: index, y: 0, label: `i=${index}` },
    ...values.map((value, position) => ({
      key: `stack-entry-${value.split(':')[0]}`,
      kind: 'state',
      x: position,
      y: 1,
      label: value,
    })),
  ],
});

const draft = visual('Keep increasing (start,height) candidates; the first shorter bar closes each taller candidate at its exact maximal width.', [
  frame('Initialize the increasing stack', 'Start with an empty stack, best = 0, and scan the real histogram before an appended zero sentinel.', stackScene(0, [], {
    current: 'index 0, height 2',
    best: '0',
  }), 'initialize'),
  frame('Push height 2', 'Nothing taller is waiting, so push (start 0, height 2).', stackScene(0, ['0: height 2'], {
    action: 'push (0,2)',
    best: '0',
  }), 'push-2'),
  frame('Height 1 closes height 2', 'At index 1, 2 > 1, so pop (0,2): area = 2 * (1 - 0) = 2. Carry start = 0 for the shorter bar.', stackScene(1, [], {
    current: 'height 1',
    area: '2 * (1 - 0) = 2',
    carriedStart: 'start: 1 -> 0',
    best: '0 -> 2',
  }), 'pop-2'),
  frame('Push height 1 from carried start', 'Push (0,1), preserving that height 1 can extend back through index 0.', stackScene(1, ['0: height 1'], {
    action: 'push (0,1)',
    best: '2',
  }), 'push-1'),
  frame('Push height 5', 'At index 2, 5 is above stack top 1, so push (2,5).', stackScene(2, ['0: height 1', '2: height 5'], {
    action: 'push (2,5)',
    best: '2',
  }), 'push-5'),
  frame('Push height 6', 'At index 3, 6 is above stack top 5, so push (3,6).', stackScene(3, ['0: height 1', '2: height 5', '3: height 6'], {
    action: 'push (3,6)',
    best: '2',
  }), 'push-6'),
  frame('Height 2 closes height 6', 'At index 4, pop (3,6): 6 cannot cross the shorter height 2, so area = 6 * (4 - 3) = 6 and start becomes 3.', stackScene(4, ['0: height 1', '2: height 5'], {
    current: 'height 2',
    area: '6 * (4 - 3) = 6',
    carriedStart: 'start: 4 -> 3',
    best: '2 -> 6',
  }), 'pop-6'),
  frame('The same bar closes height 5', 'Stack top 5 is still taller than 2, so pop (2,5): area = 5 * (4 - 2) = 10. Carry start back to 2.', stackScene(4, ['0: height 1'], {
    current: 'height 2',
    area: '5 * (4 - 2) = 10',
    carriedStart: 'start: 3 -> 2',
    best: '6 -> 10',
  }), 'pop-5'),
  frame('Push height 2 from index 2', 'Push (2,2), because the popped bars prove height 2 extends left to index 2.', stackScene(4, ['0: height 1', '2: height 2'], {
    action: 'push (2,2)',
    best: '10',
  }), 'push-2-start-2'),
  frame('Push height 3', 'At index 5, 3 is above stack top 2, so push (5,3).', stackScene(5, ['0: height 1', '2: height 2', '5: height 3'], {
    action: 'push (5,3)',
    best: '10',
  }), 'push-3'),
  frame('Sentinel closes height 3', 'The appended 0 at index 6 is shorter than 3, so pop (5,3): area = 3 * (6 - 5) = 3.', stackScene(6, ['0: height 1', '2: height 2'], {
    current: 'sentinel height 0',
    area: '3 * (6 - 5) = 3',
    best: '10',
  }), 'flush-3'),
  frame('Sentinel closes height 2', 'Pop (2,2): height 2 spans indices 2 through 5, so area = 2 * (6 - 2) = 8.', stackScene(6, ['0: height 1'], {
    current: 'sentinel height 0',
    area: '2 * (6 - 2) = 8',
    best: '10',
  }), 'flush-2'),
  frame('Sentinel closes height 1', 'Pop (0,1): height 1 spans the full histogram, so area = 1 * (6 - 0) = 6. No candidate remains unresolved.', stackScene(6, [], {
    current: 'sentinel height 0',
    area: '1 * (6 - 0) = 6',
    best: '10',
  }), 'flush-1'),
  frame('Push the sentinel', 'After the pop loop, the implementation pushes (0,0). It contributes no area but leaves the loop transition fully represented.', stackScene(6, ['0: height 0'], {
    action: 'push (0,0)',
    best: '10',
  }), 'push-sentinel'),
  frame('Return the largest closed area', 'Every bar has now met its first shorter right boundary. The largest recorded rectangle has height 5, width 2, and area 10.', stackScene(6, [], {
    rectangle: 'indices [2..3], minimum height 5',
    arithmetic: '5 * 2 = 10',
    result: '10',
  }), 'result'),
]);

const review = {
  pattern: 'Monotonic increasing stack of each height and its earliest valid start index.',
  recognitionCue: 'Use it when each element needs the widest contiguous span for which it remains the limiting minimum, and the first smaller boundary finalizes that span.',
  invariant: 'Stack heights are nondecreasing; each pair (start,height) can extend from start through index-1, and every popped height receives index as its first smaller right boundary while propagating its start to the incoming shorter bar.',
  stateModel: 'The minimal state is the scan index, best area, current carried start, and a stack of (start,height) pairs. The visible stack preserves unresolved bars and exact width arithmetic at every pop.',
  visualRationale: 'A literal stack beside the fixed histogram input directly shows candidate lifetime, LIFO closures, carried starts, and the sentinel flush; labels and formulas carry the meaning without relying on color or animation.',
  rejectedAlternatives: [
    'A bar chart with only the winning rectangle was rejected because it hides when other candidates become final.',
    'A nearest-smaller table was rejected because it obscures the one-pass stack mechanism and carried start.',
    'A brute-force interval grid was rejected because it adds quadratic clutter instead of exposing monotonic candidate elimination.',
  ],
  transferLesson: 'When the first violating element determines an unresolved candidate’s right boundary, keep candidates monotone and finalize them in reverse order; propagate the oldest valid start across every pop.',
  reviewStatus: 'reviewed',
};

export default defineVisual('largest-rectangle-in-histogram', draft, review);
