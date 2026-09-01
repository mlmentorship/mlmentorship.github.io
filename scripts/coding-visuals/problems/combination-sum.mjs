import { defineVisual, frame, grid, visual } from '../primitives.mjs';

const callStack = (rows, extra = {}) => grid(
  rows,
  [{ row: rows.length - 1, col: 0, label: 'active call', tone: 'focus', key: 'call-top' }],
  { input: 'candidates = [2,3,6,7], target = 7', columns: 'path | start | remaining', ...extra },
);

const draft = visual('Carry the remaining target, reuse the current sorted index, and break when the next choice is too large.', [
  frame('Initialize sorted choices', 'Positive unique choices remain [2,3,6,7]; call choose(0,7) with an empty path.', callStack([['[]', '0', '7']], {
    action: 'choices = [2,3,6,7]',
    answer: '[]',
  }), 'initialize'),
  frame('Choose 2 and keep its index', 'Append 2 and call choose(0,5); passing index 0 permits another 2.', callStack([['[]', '0', '7'], ['[2]', '0', '5']], {
    arithmetic: '7 - 2 = 5',
  }), 'choose-first-two'),
  frame('Reuse 2', 'Append 2 again and call choose(0,3), still allowing choice 2 or any later choice.', callStack([['[]', '0', '7'], ['[2]', '0', '5'], ['[2,2]', '0', '3']], {
    arithmetic: '5 - 2 = 3',
  }), 'choose-second-two'),
  frame('Prune a third 2', 'Trying another 2 reaches remaining 1; there 2 > 1, so break, return, and pop that third 2.', callStack([['[]', '0', '7'], ['[2]', '0', '5'], ['[2,2]', '0', '3'], ['[2,2,2]', '0', '1']], {
    action: '2 > 1; break; pop 2',
  }), 'prune-third-two'),
  frame('Complete 2 + 2 + 3', 'The [2,2] loop advances to choice 3: remaining 3 - 3 = 0, so copy [2,2,3].', callStack([['[]', '0', '7'], ['[2]', '0', '5'], ['[2,2]', '0', '3'], ['[2,2,3]', '1', '0']], {
    arithmetic: '3 - 3 = 0',
    answer: '[[2,2,3]]',
  }), 'record-two-two-three'),
  frame('Prune the 2 + 3 branch', 'Return and pop to [2]; choose 3 leaves 2, but start 1 makes 3 the next option and 3 > 2, so break.', callStack([['[]', '0', '7'], ['[2]', '0', '5'], ['[2,3]', '1', '2']], {
    action: '3 > 2; break; pop 3',
  }), 'prune-two-three'),
  frame('Prune choices above remainder 5', 'Back at [2], the next choice is 6; because choices are sorted and 6 > 5, break the whole loop.', callStack([['[]', '0', '7'], ['[2]', '0', '5']], {
    action: '6 > 5; break; pop outer 2',
  }), 'prune-after-two'),
  frame('Explore the 3 branch', 'At the root choose 3, leaving 4; reusing start 1 allows another 3.', callStack([['[]', '0', '7'], ['[3]', '1', '4']], {
    arithmetic: '7 - 3 = 4',
  }), 'choose-three'),
  frame('Prune 3 + 3', 'A second 3 leaves 1; with start 1 the next choice 3 is too large, so return and pop it.', callStack([['[]', '0', '7'], ['[3]', '1', '4'], ['[3,3]', '1', '1']], {
    action: '3 > 1; break; pop 3; then 6 > 4',
  }), 'prune-three-three'),
  frame('Prune the 6 branch', 'At the root choose 6, leaving 1; start 2 points to 6, and 6 > 1 immediately ends that branch.', callStack([['[]', '0', '7'], ['[6]', '2', '1']], {
    arithmetic: '7 - 6 = 1',
    action: '6 > 1; break; pop 6',
  }), 'prune-six'),
  frame('Complete the 7 branch', 'At the root choose 7; remaining becomes 0, so copy [7] as the second valid combination.', callStack([['[]', '0', '7'], ['[7]', '3', '0']], {
    arithmetic: '7 - 7 = 0',
    result: '[[2,2,3],[7]]',
  }), 'record-seven'),
]);

const review = {
  pattern: 'Backtracking with a sorted start index and remaining target.',
  recognitionCue: 'Use this pattern when combinations must sum to a target, order should not duplicate answers, and the same positive candidate may be selected repeatedly.',
  invariant: 'Every path is nondecreasing by choice index, sum(path) + remaining equals the original target, and choose(start, remaining) may use only start or later indices; remaining zero is exactly a solution.',
  stateModel: 'Keep sorted positive unique choices, mutable path, start index, remaining target, result copies, and recursion stack; recurse with the same index for reuse and pop after return.',
  visualRationale: 'A recursion-stack grid places path, start, and remaining on every active call, making reuse, arithmetic, zero completion, and sorted break pruning readable while call-top moves with depth.',
  rejectedAlternatives: [
    'A dynamic-programming table can count or decide sums but does not enumerate the supplied combination paths.',
    'An unrestricted choice tree generates order duplicates such as [2,2,3] and [3,2,2].',
    'A result-only diagram hides same-index reuse and why one oversized sorted choice prunes all later choices.',
  ],
  transferLesson: 'For reusable combination search, carry a decreasing feasibility measure and a nondecreasing choice boundary; sorted positive choices turn the first oversized option into a safe loop break.',
  reviewStatus: 'reviewed',
};

export default defineVisual('combination-sum', draft, review);
