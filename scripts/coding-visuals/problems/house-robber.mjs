import { array, defineVisual, frame, mark, visual } from '../primitives.mjs';

const houses = ['house 0: $2', 'house 1: $7', 'house 2: $9', 'house 3: $3', 'house 4: $1'];

function robState(index, twoBack, oneBack, skip, take, current, decision, result) {
  return array(houses, [mark(index, `current i=${index}`, result ? 'output' : 'focus', 'current-house')], {
    input: '[2, 7, 9, 3, 1]',
    priorState: `two_back=${twoBack}, one_back=${oneBack}`,
    candidates: `skip=${skip}; take=${twoBack}+${houses[index].match(/\d+$/)[0]}=${take}`,
    decision: `${decision}; current=${current}`,
    shiftedState: `two_back=${oneBack}, one_back=${current}`,
    ...(result ? { result } : {}),
  });
}

const draft = visual('For each prefix of houses, keep the better of skipping the current house or taking it after the best prefix ending two houses earlier.', [
  frame(
    'Initialize before house 0',
    'For money [2, 7, 9, 3, 1], both saved prefix answers start at zero.',
    array(houses, [mark(0, 'next i=0', 'focus', 'current-house')], {
      state: 'two_houses_back=0, one_house_back=0',
      meaning: 'best totals before any house',
    }),
    'initialize',
  ),
  frame(
    'Choose house 0',
    'Skip gives 0; take gives 0 + 2 = 2. Keep current = 2, then shift the saved states to (0, 2).',
    robState(0, 0, 0, 0, 2, 2, 'take is larger'),
    'house-0',
  ),
  frame(
    'Choose house 1 instead',
    'Skip gives 2; take gives 0 + 7 = 7. Keep 7 for the first two houses and shift to (2, 7).',
    robState(1, 0, 2, 2, 7, 7, 'take is larger'),
    'house-1',
  ),
  frame(
    'Combine house 2 with house 0',
    'Skip gives 7; take gives 2 + 9 = 11. Keep 11 and shift to (7, 11).',
    robState(2, 2, 7, 7, 11, 11, 'take is larger'),
    'house-2',
  ),
  frame(
    'Skip house 3',
    'Skip preserves 11; take gives 7 + 3 = 10. Because 11 is larger, current stays 11 and state shifts to (11, 11).',
    robState(3, 7, 11, 11, 10, 11, 'skip is larger'),
    'house-3',
  ),
  frame(
    'Take house 4 and return',
    'Skip gives 11; take gives 11 + 1 = 12. The final saved answer is 12, achieved by houses 0, 2, and 4.',
    robState(4, 11, 11, 11, 12, 12, 'take is larger', '12'),
    'house-4',
  ),
]);

export default defineVisual('house-robber', draft, {
  pattern: 'One-dimensional take-or-skip dynamic programming with two rolling prefix answers.',
  recognitionCue: 'Items lie in a line, adjacent choices conflict, and the objective asks for a maximum total rather than the exact chosen sequence.',
  invariant: 'Before house i, one_house_back is the optimum for houses through i - 1 and two_houses_back is the optimum through i - 2. Therefore max(one_back, two_back + money[i]) is the complete optimum through i.',
  stateModel: 'Retain the current money plus two scalar prefix optima. Skip depends on dp[i - 1], take depends on dp[i - 2], and parallel shifting prepares the same meanings for the next index.',
  visualRationale: 'The actual house line and stable current-house marker expose fill order; every frame prints both dependency values, the chosen branch, and the shifted rolling state, so color and JavaScript are optional.',
  rejectedAlternatives: [
    'A tree of all subsets emphasizes exponential choices and repeats identical prefix states.',
    'Highlighting only the final houses hides the locally optimal skip at house 3.',
    'A full DP table is valid but conceals that the supplied implementation needs only two prior values.',
  ],
  transferLesson: 'When choosing item i only conflicts with a fixed neighborhood, compare the optimum that excludes i with value[i] plus the last compatible optimum. This transfers to weighted independent sets on paths and cooldown scheduling.',
  reviewStatus: 'reviewed',
});
