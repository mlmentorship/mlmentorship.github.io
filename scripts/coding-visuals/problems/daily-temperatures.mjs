import { defineVisual, frame, grid, visual } from '../primitives.mjs';

const top = (row) => [{ row, col: 0, label: 'top waiting day', tone: 'focus', key: 'stack-top' }];
const waiting = (rows, extra = {}) => grid(rows.map((value) => [value]), top(rows.length - 1), {
  temperatures: '[73, 74, 75, 71, 69, 72, 76, 73]',
  layout: 'oldest to newest',
  ...extra,
});

const draft = visual('A decreasing stack keeps unresolved day indices until the first warmer temperature can pop them.', [
  frame('Initialize unanswered days', 'Start with answer=[0,0,0,0,0,0,0,0] and an empty waiting stack.', waiting(['empty'], {
    current: 'before day 0',
    answer: '[0,0,0,0,0,0,0,0]',
    action: 'initialize',
  }), 'initialize'),
  frame('Push day 0', 'Day 0 is 73. No earlier day waits, so append index 0.', waiting(['day 0: 73'], {
    current: 'day 0: 73',
    answer: '[0,0,0,0,0,0,0,0]',
    action: 'push 0',
  }), 'push-day-0'),
  frame('Resolve day 0 with day 1', '74 > 73, so pop day 0 and write answer[0] = 1 - 0 = 1; then push day 1.', waiting(['day 1: 74'], {
    current: 'day 1: 74',
    answer: '[1,0,0,0,0,0,0,0]',
    action: 'pop 0; 1 - 0 = 1; push 1',
  }), 'resolve-day-0'),
  frame('Resolve day 1 with day 2', '75 > 74, so pop day 1 and write answer[1] = 2 - 1 = 1; then push day 2.', waiting(['day 2: 75'], {
    current: 'day 2: 75',
    answer: '[1,1,0,0,0,0,0,0]',
    action: 'pop 1; 2 - 1 = 1; push 2',
  }), 'resolve-day-1'),
  frame('Keep cooler day 3 waiting', '71 is not warmer than stack-top 75, so no answer is safe yet; push day 3.', waiting(['day 2: 75', 'day 3: 71'], {
    current: 'day 3: 71',
    answer: '[1,1,0,0,0,0,0,0]',
    action: '71 <= 75; push 3',
  }), 'push-day-3'),
  frame('Keep cooler day 4 waiting', '69 is not warmer than stack-top 71, so append day 4 and preserve decreasing temperatures.', waiting(['day 2: 75', 'day 3: 71', 'day 4: 69'], {
    current: 'day 4: 69',
    answer: '[1,1,0,0,0,0,0,0]',
    action: '69 <= 71; push 4',
  }), 'push-day-4'),
  frame('Day 5 resolves day 4', '72 > 69, so the while loop pops day 4 and writes answer[4] = 5 - 4 = 1.', waiting(['day 2: 75', 'day 3: 71'], {
    current: 'day 5: 72',
    answer: '[1,1,0,0,1,0,0,0]',
    action: 'pop 4; 5 - 4 = 1',
  }), 'resolve-day-4'),
  frame('Day 5 also resolves day 3', 'The same 72 > new top 71, so pop day 3, write answer[3] = 5 - 3 = 2, then push day 5 below 75.', waiting(['day 2: 75', 'day 5: 72'], {
    current: 'day 5: 72',
    answer: '[1,1,0,2,1,0,0,0]',
    action: 'pop 3; 5 - 3 = 2; push 5',
  }), 'resolve-day-3'),
  frame('Day 6 resolves day 5', '76 > 72, so pop day 5 and write answer[5] = 6 - 5 = 1.', waiting(['day 2: 75'], {
    current: 'day 6: 76',
    answer: '[1,1,0,2,1,1,0,0]',
    action: 'pop 5; 6 - 5 = 1',
  }), 'resolve-day-5'),
  frame('Day 6 also resolves day 2', '76 > new top 75, so pop day 2, write answer[2] = 6 - 2 = 4, then push day 6.', waiting(['day 6: 76'], {
    current: 'day 6: 76',
    answer: '[1,1,4,2,1,1,0,0]',
    action: 'pop 2; 6 - 2 = 4; push 6',
  }), 'resolve-day-2'),
  frame('Leave final cooler day waiting', 'Day 7 is 73, not warmer than 76, so push it; unresolved days 6 and 7 correctly retain zero.', waiting(['day 6: 76', 'day 7: 73'], {
    current: 'day 7: 73',
    answer: '[1,1,4,2,1,1,0,0]',
    action: '73 <= 76; push 7',
    result: '[1,1,4,2,1,1,0,0]',
  }), 'finish'),
]);

const review = {
  pattern: 'Monotonic decreasing stack of unresolved indices.',
  recognitionCue: 'Look for the first later greater value or the waiting distance to it; those words suggest keeping unresolved indices until a new value is large enough to answer them.',
  invariant: 'After each day is appended, waiting indices increase from bottom to top while their temperatures are non-increasing, and every popped index receives its first strictly warmer later day.',
  stateModel: 'Keep the answer array plus a stack of day indices whose answer is unknown; indices preserve distance arithmetic and temperature lookups, so stored values need not be duplicated.',
  visualRationale: 'A labelled bottom-to-top stack directly shows decreasing temperatures and each while-loop pop; the answer array and subtraction stay visible, and stack-top moves under one stable key.',
  rejectedAlternatives: [
    'A temperature line chart emphasizes shape but obscures the exact unresolved index stack and day-distance subtraction.',
    'A full pairwise comparison matrix visualizes the quadratic brute force rather than the supplied linear mechanism.',
    'An answer-only array hides why one warmer day can resolve several waiting days in LIFO order.',
  ],
  transferLesson: 'For next-greater or next-smaller problems, store unresolved indices in a monotonic stack; the first value that violates the monotonic order resolves every eligible item popped from the top.',
  reviewStatus: 'reviewed',
};

export default defineVisual('daily-temperatures', draft, review);
