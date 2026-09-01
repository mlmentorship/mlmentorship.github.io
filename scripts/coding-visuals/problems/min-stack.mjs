import { defineVisual, frame, grid, visual } from '../primitives.mjs';

const top = (row) => [{ row, col: 0, label: 'top', tone: 'focus', key: 'stack-top' }];
const pairs = (rows, extra = {}) => grid(rows, top(rows.length - 1), {
  columns: 'value | minimum so far',
  layout: 'bottom to top',
  operations: 'push(5), push(2), push(4), get_min(), pop(), get_min(), top()',
  ...extra,
});

const draft = visual('Store each value with the minimum for its entire stack prefix, so the top pair always answers get_min().', [
  frame('Initialize an empty MinStack', 'Before any operation, the pair stack is empty.', pairs([['empty', 'empty']], {
    action: 'initialize',
  }), 'initialize'),
  frame('Push the first value', 'push(5) stores (5, 5) because an empty stack has no earlier minimum.', pairs([['5', '5']], {
    action: 'minimum = 5',
  }), 'push-five'),
  frame('Push a new minimum', 'push(2) computes min(2, 5) = 2 and stores (2, 2) above (5, 5).', pairs([['5', '5'], ['2', '2']], {
    action: 'min(2, 5) = 2',
  }), 'push-two'),
  frame('Push above the minimum', 'push(4) computes min(4, 2) = 2, so (4, 2) carries the existing minimum forward.', pairs([['5', '5'], ['2', '2'], ['4', '2']], {
    action: 'min(4, 2) = 2',
  }), 'push-four'),
  frame('Read the current minimum', 'get_min() reads the top pair second field, 2, without inspecting lower entries.', pairs([['5', '5'], ['2', '2'], ['4', '2']], {
    action: 'top pair minimum = 2',
    returned: '2',
  }), 'first-min'),
  frame('Pop the top pair', 'pop() removes (4, 2); the earlier prefix and its saved minimum are exposed unchanged.', pairs([['5', '5'], ['2', '2']], {
    action: 'remove (4, 2)',
  }), 'pop-four'),
  frame('Read the restored minimum', 'get_min() again reads 2 from the exposed top pair (2, 2).', pairs([['5', '5'], ['2', '2']], {
    action: 'top pair minimum = 2',
    returned: '2',
  }), 'second-min'),
  frame('Read the restored top value', 'top() reads the first field of the same top pair, so the final returned value is 2.', pairs([['5', '5'], ['2', '2']], {
    action: 'top pair value = 2',
    result: 'top() = 2; get_min() = 2',
  }), 'read-top'),
]);

const review = {
  pattern: 'Augmented stack that stores a prefix aggregate with every item.',
  recognitionCue: 'Use this pattern when ordinary stack operations must also answer an aggregate such as the current minimum in constant time, including immediately after pops.',
  invariant: 'For every stored pair (value, minimum), minimum equals the smallest value from the bottom through that pair; therefore the top pair summarizes the complete current stack.',
  stateModel: 'One LIFO stack of (value, minimum-so-far) pairs is sufficient: push derives a new prefix minimum, pop removes one pair, and top/get_min read separate fields of the top pair.',
  visualRationale: 'A two-column bottom-to-top stack keeps each value physically attached to its saved prefix minimum; labelled arithmetic proves pushes, while the stable stack-top key moves after pop.',
  rejectedAlternatives: [
    'A single running-min variable cannot restore the previous minimum after the minimum value is popped.',
    'A separate min stack works but requires synchronizing duplicate minima and is less direct than the supplied pair implementation.',
    'A plain operation table records outputs but hides how the earlier minimum is uncovered by a pop.',
  ],
  transferLesson: 'Attach any reversible prefix summary needed later to each stack item; the same technique supports max stacks, depth summaries, and constant-time aggregate reads after rollback.',
  reviewStatus: 'reviewed',
};

export default defineVisual('min-stack', draft, review);
