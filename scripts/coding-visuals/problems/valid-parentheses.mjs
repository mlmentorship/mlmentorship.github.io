import { defineVisual, frame, grid, visual } from '../primitives.mjs';

const top = (row) => [{ row, col: 0, label: 'top', tone: 'focus', key: 'stack-top' }];
const stack = (rows, extra = {}) => grid(rows.map((value) => [value]), top(rows.length - 1), {
  input: '([{}])',
  layout: 'bottom to top',
  ...extra,
});

const draft = visual('A closing bracket is valid only when it removes its matching newest unmatched opening bracket.', [
  frame('Initialize an empty stack', 'Before scanning ([{}]), there is no unmatched opening bracket.', stack(['empty'], {
    current: 'before index 0',
    action: 'initialize',
  }), 'initialize'),
  frame('Push the first opening', 'At index 0, ( is not a closing key, so the implementation appends it.', stack(['('], {
    current: 'index 0: (',
    action: 'push (',
  }), 'push-round'),
  frame('Push the nested opening', 'At index 1, [ is newer unfinished work and sits above (.', stack(['(', '['], {
    current: 'index 1: [',
    action: 'push [',
  }), 'push-square'),
  frame('Push the innermost opening', 'At index 2, { becomes the only opening that the next closer may finish.', stack(['(', '[', '{'], {
    current: 'index 2: {',
    action: 'push {',
  }), 'push-curly'),
  frame('Match the curly pair', 'At index 3, } expects {. pop() returns {, so the scan may continue.', stack(['(', '['], {
    current: 'index 3: }',
    action: 'pop {; expected {',
  }), 'pop-curly'),
  frame('Match the square pair', 'At index 4, ] expects [. pop() returns [, preserving correct nesting.', stack(['('], {
    current: 'index 4: ]',
    action: 'pop [; expected [',
  }), 'pop-square'),
  frame('Match the outer pair', 'At index 5, ) expects (. pop() returns (, leaving no unfinished opening.', stack(['empty'], {
    current: 'index 5: )',
    action: 'pop (; expected (',
    result: 'true',
  }), 'pop-round'),
]);

const review = {
  pattern: 'LIFO stack of unmatched opening brackets.',
  recognitionCue: 'Use this pattern when nested delimiters must close in reverse opening order and each closer must validate the most recent unfinished opener.',
  invariant: 'After each processed character, the stack contains exactly the unmatched opening brackets from the processed prefix, ordered oldest at the bottom and newest at the top.',
  stateModel: 'The only changing state is the stack of opening characters; the fixed closer-to-opener map decides whether a closing character may pop its top.',
  visualRationale: 'A bottom-to-top labelled grid is the simplest semantic-HTML stack: every push and pop changes real stack geometry, and the stable stack-top key moves with the active boundary.',
  rejectedAlternatives: [
    'A character table records events but hides the LIFO geometry that explains nesting.',
    'A parse tree adds parent-child structure that this iterative implementation never stores.',
    'Prose beside the code forces the reader to simulate the unmatched stack mentally.',
  ],
  transferLesson: 'For tags, expression delimiters, and nested scopes, store only unfinished openers and let each closer inspect the newest one; an empty stack at the end proves completion.',
  reviewStatus: 'reviewed',
};

export default defineVisual('valid-parentheses', draft, review);
