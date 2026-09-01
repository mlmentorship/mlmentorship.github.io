import { defineVisual, frame, grid, visual } from '../primitives.mjs';

const top = (row) => [{ row, col: 0, label: 'top saved context', tone: 'focus', key: 'stack-top' }];
const contexts = (rows, extra = {}) => grid(rows.map((value) => [value]), top(rows.length - 1), {
  input: '3[a2[c]]',
  layout: 'outer to inner',
  ...extra,
});

const draft = visual('Each [ saves its outer string and repeat count so ] can restore exactly one nesting level.', [
  frame('Initialize decoder state', 'Before scanning 3[a2[c]], stack=[], current="", and repeat=0.', contexts(['empty'], {
    current: '""',
    repeat: '0',
    action: 'initialize',
  }), 'initialize'),
  frame('Build the outer count', 'Reading digit 3 applies repeat = 0 * 10 + 3 = 3.', contexts(['empty'], {
    current: '""',
    repeat: '3',
    action: '0 * 10 + 3',
  }), 'read-three'),
  frame('Save the outer context', 'At the first [, push ("", 3), then reset current="" and repeat=0.', contexts(['("", 3)'], {
    current: '""',
    repeat: '0',
    action: 'push and reset',
  }), 'open-outer'),
  frame('Build outer text', 'Reading a appends it to the current nesting level: "" + "a" = "a".', contexts(['("", 3)'], {
    current: '"a"',
    repeat: '0',
    action: 'append a',
  }), 'append-a'),
  frame('Build the inner count', 'Reading digit 2 applies repeat = 0 * 10 + 2 = 2 while current stays "a".', contexts(['("", 3)'], {
    current: '"a"',
    repeat: '2',
    action: '0 * 10 + 2',
  }), 'read-two'),
  frame('Save the nested context', 'At the second [, push ("a", 2), then reset current="" and repeat=0.', contexts(['("", 3)', '("a", 2)'], {
    current: '""',
    repeat: '0',
    action: 'push and reset',
  }), 'open-inner'),
  frame('Build the inner text', 'Reading c appends it to the empty inner string, producing current="c".', contexts(['("", 3)', '("a", 2)'], {
    current: '"c"',
    repeat: '0',
    action: 'append c',
  }), 'append-c'),
  frame('Close the inner repeat', 'The first ] pops ("a", 2): "a" + "c" * 2 = "acc".', contexts(['("", 3)'], {
    current: '"acc"',
    repeat: '0',
    action: '"a" + "c" * 2',
  }), 'close-inner'),
  frame('Close the outer repeat', 'The final ] pops ("", 3): "" + "acc" * 3 = "accaccacc".', contexts(['empty'], {
    current: '"accaccacc"',
    repeat: '0',
    action: '"" + "acc" * 3',
    result: 'accaccacc',
  }), 'close-outer'),
]);

const review = {
  pattern: 'Stack of suspended outer decoding contexts.',
  recognitionCue: 'Use a context stack when repeat counts and bracketed substrings can nest, so finishing an inner region must resume text and count saved before its opening bracket.',
  invariant: 'Before each character, current is the decoded text for the active nesting level, repeat is the number parsed immediately before its next [, and every stack entry preserves one suspended outer (text, count) pair.',
  stateModel: 'Maintain current text, the multi-digit repeat accumulator, and a LIFO stack of (previous text, repeat count) pairs; [ pushes and resets, while ] pops and concatenates.',
  visualRationale: 'A bottom-to-top context stack exposes which outer computation is suspended; visible current, repeat, and arithmetic show each branch, while stack-top is a stable moving key.',
  rejectedAlternatives: [
    'A parse tree represents the grammar but introduces nodes and recursion absent from the supplied one-pass stack solution.',
    'A flat character timeline shows scan order but hides the saved outer contexts needed after each ].',
    'Showing only the expanding output skips the push, reset, and restore mechanism.',
  ],
  transferLesson: 'When nested work temporarily replaces an outer accumulator, push exactly the outer state needed to resume it; this transfers to expression evaluation, nested tags, and recursive-descent simulation.',
  reviewStatus: 'reviewed',
};

export default defineVisual('decode-string', draft, review);
