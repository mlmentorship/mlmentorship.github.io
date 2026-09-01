import { array, defineVisual, frame, mark, visual } from '../primitives.mjs';

const prices = ['7', '1', '5', '3', '6', '4'];
const state = (day, lowDay, extra = {}) => array(prices, [
  mark(lowDay, 'lowest earlier buy', 'state', 'lowest-buy'),
  mark(day, 'selling day', 'focus', 'sell-cursor'),
], { scannedPrefix: `[0..${day}]`, ...extra });

const draft = visual('For each possible selling day, subtract the lowest price in its scanned prefix and preserve the largest profit.', [
  frame('Initialize from infinity', 'On day 0 price 7, lowest=min(infinity,7)=7 and best=max(0,7-7)=0.', state(0, 0, {
    arithmetic: 'lowest=7; profit=7-7=0; best=0',
  }), 'day-zero'),
  frame('Move the buy boundary to day 1', 'Price 1 is below 7, so lowest becomes 1 before profit is evaluated; selling the same day yields 0.', state(1, 1, {
    arithmetic: 'lowest=min(7,1)=1; profit=1-1=0; best=0',
    movement: 'lowest-buy moves right because 1 is cheaper',
  }), 'day-one'),
  frame('Evaluate selling at 5', 'On day 2, lowest stays 1 and selling at 5 gives 5-1=4, so best becomes 4.', state(2, 1, {
    arithmetic: 'lowest=1; profit=5-1=4; best=4',
  }), 'day-two'),
  frame('Evaluate selling at 3', 'On day 3, price 3 is not a new low; profit 3-1=2 does not replace best 4.', state(3, 1, {
    arithmetic: 'lowest=1; profit=3-1=2; best=4',
  }), 'day-three'),
  frame('Save the best sale at 6', 'On day 4, lowest remains 1 and profit 6-1=5 raises best from 4 to 5.', state(4, 1, {
    arithmetic: 'lowest=1; profit=6-1=5; best=5',
  }), 'day-four'),
  frame('Finish at price 4', 'On day 5, profit 4-1=3 cannot beat 5; buy day 1 and sell day 4 remain optimal.', state(5, 1, {
    arithmetic: 'lowest=1; profit=4-1=3; best=5',
    result: '5 (buy day 1 at 1; sell day 4 at 6)',
  }), 'day-five'),
]);

const review = {
  pattern: 'Running minimum paired with each later value.',
  recognitionCue: 'Use a running minimum when an ordered one-pass problem asks for the best later-minus-earlier difference and the buy or baseline must occur before the current item.',
  invariant: 'After each day, lowest is the minimum price in the scanned prefix and best is the largest valid sell-minus-earlier-buy profit whose selling day lies in that prefix.',
  stateModel: 'Keep only lowest and best while scanning prices; the visual retains the low day and current selling day so temporal order and the candidate interval are explicit.',
  visualRationale: 'A fixed indexed price array with stable lowest-buy and sell-cursor pointers depicts the actual ordered range, pointer movement, scanned prefix, and every subtraction without relying on color.',
  rejectedAlternatives: [
    'A price line chart is intuitive but less precise for indexed pointer motion and visible arithmetic.',
    'A table of all buy-sell pairs depicts the quadratic brute force rather than the running-minimum mechanism.',
    'Highlighting only prices 1 and 6 hides why earlier and later candidate days cannot improve the answer.',
  ],
  transferLesson: 'For maximum ordered differences, summarize the best earlier baseline while treating each new value as the right endpoint; analogous scans find maximum rise, minimum spread, and best prefix-relative gain.',
  reviewStatus: 'reviewed',
};

export default defineVisual('best-time-to-buy-and-sell-stock', draft, review);
