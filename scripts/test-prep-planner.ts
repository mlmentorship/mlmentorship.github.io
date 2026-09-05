import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readdirSync } from 'node:fs';
import { getPracticeMode, INTERVIEW_ROUNDS } from '../src/data/prepCurriculum';
import { getSubcategoryMap } from '../src/utils/subcategories';
import { buildStudyRoutes, practiceStatus, questionPair, studyBudget } from '../src/utils/prepRoutes';
import { currentPlanWeek, nextPrepTasks, parsePrepBackup, type PrepPlanState } from '../src/utils/prepPlan';
import { markMixedSessionVerified, PREP_PROGRESS_KEY, savePracticeProgress, type PracticeProgressRecord } from '../src/utils/prepProgress';

const today = '2026-09-05';
const plan: PrepPlanState = {
  version: 1, createdOn: today, updatedOn: today, role: 'applied-scientist', roleLabel: 'Applied Scientist',
  level: 'l5', domain: 'general', weeklyHours: 5, availableWeeks: 4, horizon: 4,
  estimatedHours: { low: 10, high: 20 }, selectedRounds: ['ml-breadth', 'ml-system-design'],
  areaRatings: { fundamentals: 3, 'system-design': 1 }, topGaps: [], externalRounds: [],
};
const record = (slug: string, changes: Partial<PracticeProgressRecord> = {}): PracticeProgressRecord => ({
  slug, title: slug, mode: 'breadth', score: 'Review', weakDimensions: [], dimensionMisses: {},
  attempts: 1, successfulAttempts: 0, lastSuccessfulOn: null, mixedVerifiedOn: null,
  lastAttemptOn: '2026-09-01', dueOn: '2026-09-08', ...changes,
});

test('confirmed rounds are the only routes, with weakest evidence first', () => {
  const routes = buildStudyRoutes(plan, [], today);
  assert.deepEqual(routes.map(route => route.id), ['ml-system-design', 'ml-breadth']);
  assert.equal(nextPrepTasks(plan, [], today)[0].href, routes[0].steps[0].href);
  assert.ok(nextPrepTasks({ ...plan, level: 'l7' }, [], today).every(task => !task.href.includes('level-paths')));
});

test('role, domain, and level change representative questions', () => {
  assert.equal(questionPair(plan, 'ml-system-design').slugs[0], 'design-fraud-detection');
  assert.equal(questionPair({ ...plan, role: 'ml-engineer' }, 'ml-system-design').slugs[0], 'design-ml-monitoring');
  assert.equal(questionPair({ ...plan, level: 'l7' }, 'ml-system-design').slugs[0], 'design-multi-team-ml-platform');
  assert.equal(questionPair({ ...plan, domain: 'recsys' }, 'ml-system-design').slugs[0], 'design-youtube-recommender');
  assert.equal(questionPair({ ...plan, role: 'research-engineer' }, 'coding').slugs[0], 'implement-attention-from-scratch');
  assert.equal(questionPair({ ...plan, domain: 'llm' }, 'product-experimentation').slugs[0], 'how-would-you-evaluate-an-llm-application');
  assert.equal(questionPair({ ...plan, domain: 'platform' }, 'systems-infrastructure').slugs[0], 'design-fault-tolerant-distributed-training');
});

test('priority puts the weakest round first without losing its route', () => {
  const configured = { ...plan, selectedRounds: ['ml-breadth', 'ml-system-design'], areaRatings: { fundamentals: 4, 'system-design': 0 } };
  const routes = buildStudyRoutes(configured, [record('design-fraud-detection', { score: 'Weak', weakDimensions: ['framing'], dueOn: null })], today);
  assert.equal(routes[0].id, 'ml-system-design');
  assert.equal(routes[0].steps[0].slug, 'design-fraud-detection');
  const researchPlan = { ...plan, role: 'research-scientist', level: 'l7', domain: 'research', selectedRounds: ['ml-system-design', 'technical-strategy'], areaRatings: { 'system-design': 1, behavioral: 1 } };
  const researchRoutes = buildStudyRoutes(researchPlan, [], today);
  assert.deepEqual(researchRoutes.find(route => route.id === 'ml-system-design')?.steps.map(step => step.slug), ['design-reasoning-model-fixed-budget', 'design-ml-system-fixed-budget']);
  assert.deepEqual(researchRoutes.find(route => route.id === 'technical-strategy')?.steps.map(step => step.slug), ['design-post-training-data-and-rl-environment', 'design-enterprise-agent-platform']);
});

test('future retries are withheld and become first priority when due', () => {
  const pending = record('design-fraud-detection');
  assert.ok(nextPrepTasks(plan, [pending], today).every(task => !task.href.includes(pending.slug)));
  assert.equal(nextPrepTasks(plan, [pending], '2026-09-08')[0].kind, 'retry');
  assert.equal(nextPrepTasks(plan, [pending], '2026-09-08')[0].href, '/questions/design-fraud-detection/?practice=1');
});

test('out-of-map practice is retained but does not displace selected work', () => {
  const outside = record('implement-attention-from-scratch', { dueOn: '2026-08-01' });
  assert.ok(nextPrepTasks(plan, [outside], today).every(task => !task.href.includes(outside.slug)));
});

test('transfer follows a clean diagnostic, not merely a failed attempt', () => {
  const oneRound = { ...plan, selectedRounds: ['ml-breadth'] };
  assert.equal(nextPrepTasks(oneRound, [record('bias-variance-tradeoff')], today).length, 0);
  const clean = record('bias-variance-tradeoff', { score: 'Confident', successfulAttempts: 1 });
  assert.equal(nextPrepTasks(oneRound, [clean], today)[0].href, '/questions/how-to-choose-loss-function/?practice=1');
});

test('mixed checks need clean spaced evidence and a later day', () => {
  const clean = record('bias-variance-tradeoff', { score: 'Confident', successfulAttempts: 2, dueOn: null, lastSuccessfulOn: today, lastAttemptOn: today });
  assert.equal(practiceStatus({ ...clean, mixedVerifiedOn: today }, today), 'mixed');
  assert.equal(practiceStatus({ ...clean, mixedVerifiedOn: '2026-09-06' }, '2026-09-06'), 'mastered');
  assert.equal(practiceStatus({ ...clean, weakDimensions: ['framing'] }, today), 'due');
  assert.ok(nextPrepTasks(plan, [clean], today).every(task => task.id !== `mixed:${clean.slug}`));
});

test('shared prompts appear only once in the queue', () => {
  const tasks = nextPrepTasks({ ...plan, selectedRounds: ['research-depth', 'research-work-sample'] }, [], today);
  assert.equal(new Set(tasks.map(task => task.href)).size, tasks.length);
});

test('weekly recommendations leave half the budget for review and do not overspend', () => {
  for (const weeklyHours of [1, 5, 8, 12]) {
    const configured = { ...plan, weeklyHours, selectedRounds: INTERVIEW_ROUNDS.map(round => round.id) };
    const budget = studyBudget(configured, buildStudyRoutes(configured, [], today));
    assert.equal(budget.sessions.reduce((sum, session) => sum + session.minutes, 0) + budget.repairMinutes + budget.unassignedMinutes, weeklyHours * 60);
  }
});

test('all mapped questions exist and method matches content metadata', () => {
  const files = readdirSync('src/content/posts');
  for (const domain of ['general', 'llm', 'recsys', 'platform', 'research', 'post-training', 'alignment', 'multimodal', 'product']) {
    for (const role of ['applied-scientist', 'ml-engineer', 'research-engineer', 'research-scientist']) {
      const routes = buildStudyRoutes({ ...plan, domain, role, selectedRounds: INTERVIEW_ROUNDS.map(round => round.id) }, [], today);
      assert.equal(routes.length, INTERVIEW_ROUNDS.length, `${role}/${domain}: missing round route`);
      const mappedSlugs: string[] = [];
      for (const route of routes) for (const step of route.steps) {
        assert.equal(route.steps.length, 2, `${role}/${domain}/${route.id}: route needs diagnostic and transfer`);
        assert.notEqual(route.steps[0].slug, route.steps[1].slug, `${role}/${domain}/${route.id}: duplicate pair`);
        mappedSlugs.push(step.slug);
        const file = files.find(name => name.endsWith(`-${step.slug}.md`));
        assert.ok(file, step.slug);
        const category = getSubcategoryMap('questions')?.map[step.slug];
        assert.ok(category, `No subcategory: ${file}`);
        assert.equal(step.practice.id, getPracticeMode(step.slug, category).id, file);
        const repairSlug = step.visualHref.split('#')[1];
        assert.ok(files.some(name => name.endsWith(`-${repairSlug}.md`)), repairSlug);
        assert.notEqual(step.repairLabel, 'Worked visual for this prompt', step.slug);
      }
      assert.equal(new Set(mappedSlugs).size, mappedSlugs.length, `${role}/${domain}: duplicate prompt across rounds`);
    }
  }
});

test('existing version-one plans and backups remain compatible', () => {
  assert.deepEqual(parsePrepBackup({ version: 1, plan, records: [] })?.plan, plan);
  assert.equal(currentPlanWeek({ ...plan, availableWeeks: 2, horizon: 8 }, new Date('2026-09-26')), 2);
});

test('recorder uses two/seven day retries and rejects same-day mixed verification', context => {
  const storage = new Map<string, string>();
  context.mock.timers.enable({ apis: ['Date'], now: new Date('2026-09-05T12:00:00').getTime() });
  Object.defineProperty(globalThis, 'localStorage', { configurable: true, value: { getItem: (key: string) => storage.get(key) ?? null, setItem: (key: string, value: string) => storage.set(key, value) } });
  context.after(() => { Reflect.deleteProperty(globalThis, 'localStorage'); });
  const input = { slug: 'bias-variance-tradeoff', title: 'Bias variance', mode: 'breadth', weakDimensions: [] };
  assert.equal(savePracticeProgress({ ...input, score: 'Weak' }).dueOn, '2026-09-07');
  assert.equal(savePracticeProgress({ ...input, score: 'Review' }).dueOn, '2026-09-12');
  storage.set(PREP_PROGRESS_KEY, JSON.stringify([record(input.slug, { score: 'Confident', successfulAttempts: 2, lastSuccessfulOn: today, lastAttemptOn: today, dueOn: null })]));
  assert.equal(markMixedSessionVerified(input.slug), null);
  context.mock.timers.setTime(new Date('2026-09-06T12:00:00').getTime());
  assert.equal(markMixedSessionVerified(input.slug)?.mixedVerifiedOn, '2026-09-06');
});