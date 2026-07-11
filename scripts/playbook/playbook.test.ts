import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { buildCatalog } from './catalog';
import { buildPersonalizedPlaybook } from './engine';
import { generatePlaybook } from './generate';
import { findChrome } from './pdf';
import { renderPlaybookHtml } from './render';
import { parseIntake } from './schema';
import { LEVELS, ROLES, type PlaybookIntake } from './types';

const repoRoot = process.cwd();
const sample = parseIntake(JSON.parse(fs.readFileSync(path.join(repoRoot, 'examples/playbook/sample-intake.json'), 'utf8')));
const catalog = buildCatalog(repoRoot);
const generatedOn = '2026-07-10';

function build(intake: PlaybookIntake) {
  return buildPersonalizedPlaybook(intake, catalog, generatedOn);
}

function withChanges(changes: Partial<PlaybookIntake>): PlaybookIntake {
  return parseIntake({ ...sample, ...changes });
}

test('catalog covers every published content item with unique canonical routes', () => {
  assert.equal(catalog.length, 194);
  assert.equal(new Set(catalog.map((resource) => resource.slug)).size, catalog.length);
  assert.equal(new Set(catalog.map((resource) => resource.route)).size, catalog.length);
  for (const resource of catalog) {
    assert.match(resource.route, /^\/(questions|guides|concepts)\/[a-z0-9-]+\/$/);
    assert.ok(resource.title.length > 2);
    assert.ok(resource.estimatedMinutes >= 5);
  }
});

test('sample plan is deterministic, bounded, and includes mandatory calibration', () => {
  const first = build(sample);
  const second = build(sample);
  assert.deepEqual(first, second);
  assert.equal(first.planId, second.planId);
  assert.equal(first.weeks.length, sample.weeks);
  assert.ok(first.resourceAppendix.some((resource) => resource.slug === 'as-vs-mle-vs-re'));
  assert.ok(first.resourceAppendix.some((resource) => resource.slug === 'most-ambitious-project'));
  assert.ok(first.resourceAppendix.some((resource) => resource.slug === 'l5-vs-l6-faang-ml'));
  assert.equal(first.totals.taskCounts.simulation, 1);
  assert.ok(first.totals.taskCounts.review >= 3);
  for (const week of first.weeks) {
    assert.ok(week.plannedMinutes <= week.budgetMinutes);
    assert.ok(week.tasks.length > 0);
  }
});

test('weak areas become critical priorities and influence selected resources', () => {
  const intake = withChanges({
    selfRatings: { ...sample.selfRatings, 'llm-systems': 1, behavioral: 5 },
    rounds: ['llm-systems', 'ml-system-design'],
  });
  const plan = build(intake);
  assert.equal(plan.readiness[0].area, 'llm-systems');
  assert.equal(plan.readiness[0].priority, 'critical');
  const llmTasks = plan.weeks.flatMap((week) => week.tasks).filter((task) => task.area === 'llm-systems');
  assert.ok(llmTasks.length >= 3);
});

test('role changes alter readiness weighting and plan identity', () => {
  const mle = build(sample);
  const appliedScientist = build(withChanges({ role: 'applied-scientist' }));
  const researchEngineer = build(withChanges({ role: 'research-engineer' }));
  assert.notEqual(mle.planId, appliedScientist.planId);
  assert.notEqual(mle.planId, researchEngineer.planId);

  const mleProduction = mle.readiness.find((item) => item.area === 'production')!;
  const asProduction = appliedScientist.readiness.find((item) => item.area === 'production')!;
  const reMath = researchEngineer.readiness.find((item) => item.area === 'math-research')!;
  const mleMath = mle.readiness.find((item) => item.area === 'math-research')!;
  assert.ok(mleProduction.weight > asProduction.weight);
  assert.ok(reMath.weight > mleMath.weight);
});

test('every scheduled resource resolves and first attempts are unique', () => {
  const plan = build(sample);
  const canonical = new Set(catalog.map((resource) => resource.slug));
  const firstAttemptSlugs = plan.weeks
    .flatMap((week) => week.tasks)
    .filter((task) => task.type !== 'review')
    .map((task) => task.resourceSlug)
    .filter((slug): slug is string => Boolean(slug));
  assert.equal(new Set(firstAttemptSlugs).size, firstAttemptSlugs.length);
  for (const slug of firstAttemptSlugs) assert.ok(canonical.has(slug), `missing ${slug}`);

  const firstAttempts = new Map(plan.weeks.flatMap((week) => week.tasks).map((task) => [task.id, task]));
  for (const review of plan.weeks.flatMap((week) => week.tasks).filter((task) => task.type === 'review')) {
    const original = firstAttempts.get(review.reviewOf!);
    assert.ok(original, `review target missing: ${review.reviewOf}`);
    assert.ok(review.week > original!.week, `review ${review.id} must follow ${original!.id}`);
  }
});

test('domain-specific content does not leak into an unrelated plan', () => {
  const plan = build(sample);
  const slugs = new Set(plan.resourceAppendix.map((resource) => resource.slug));
  assert.equal(slugs.has('design-youtube-recommender'), false);
  assert.equal(slugs.has('evaluate-search-ranker'), false);
  assert.ok(slugs.has('how-would-you-evaluate-an-llm-application'));
});

test('HTML is self-contained, personalized, and links to selected resources', () => {
  const plan = build(sample);
  const html = renderPlaybookHtml(plan);
  assert.match(html, /<!doctype html>/i);
  assert.match(html, new RegExp(plan.planId));
  assert.match(html, /Sample Candidate/);
  assert.match(html, /https:\/\/mlmentorship\.com\/questions\//);
  assert.match(html, /@page \{ size: A4/);
  assert.doesNotMatch(html, /<script/i);
});

test('invalid intake fails with actionable field paths', () => {
  assert.throws(
    () => parseIntake({ ...sample, weeks: 1, selfRatings: { ...sample.selfRatings, coding: 8 } }),
    /weeks:|selfRatings\.coding:/,
  );
  assert.throws(
    () => parseIntake({ ...sample, interviewDate: sample.startDate }),
    /interviewDate: interviewDate must be after startDate/,
  );
  assert.throws(
    () => parseIntake({ ...sample, startDate: '2026-02-31' }),
    /startDate: startDate is not a valid calendar date/,
  );
  assert.throws(
    () => parseIntake({ ...sample, interviewDate: '2026-07-20' }),
    /interviewDate must fall within the final week/,
  );
});

test('circular prerequisites fail with a clear planner error', () => {
  const cyclicCatalog = catalog.map((resource) => {
    if (resource.slug === 'as-vs-mle-vs-re') return { ...resource, prerequisites: ['most-ambitious-project'] };
    if (resource.slug === 'most-ambitious-project') return { ...resource, prerequisites: ['as-vs-mle-vs-re'] };
    return resource;
  });
  assert.throws(
    () => buildPersonalizedPlaybook(sample, cyclicCatalog, generatedOn),
    /Circular playbook prerequisite detected/,
  );
});

test('all role/level combinations produce bounded plans at short and long horizons', () => {
  for (const role of ROLES) {
    for (const targetLevel of LEVELS) {
      for (const weeks of [2, 8]) {
        const intake = withChanges({
          role,
          targetLevel,
          weeks,
          hoursPerWeek: weeks === 2 ? 5 : 10,
          interviewDate: undefined,
        });
        const plan = build(intake);
        assert.equal(plan.weeks.length, weeks);
        assert.ok(plan.totals.uniqueResources >= 8);
        assert.ok(plan.totals.scheduledMinutes <= weeks * intake.hoursPerWeek * 60);
        assert.ok(plan.weeks.every((week) => week.tasks.length > 0));
        assert.ok(plan.weeks.every((week) => week.plannedMinutes <= week.budgetMinutes));
      }
    }
  }
});

test('generation writes plan, HTML, manifest, and a valid PDF', { skip: !findChrome() }, async () => {
  const outputDirectory = fs.mkdtempSync(path.join(os.tmpdir(), 'mlmentorship-playbook-test-'));
  try {
    const result = await generatePlaybook({
      intakePath: path.join(repoRoot, 'examples/playbook/sample-intake.json'),
      outputDirectory,
      repoRoot,
    });
    assert.ok(fs.existsSync(result.planPath));
    assert.ok(fs.existsSync(result.htmlPath));
    assert.ok(result.pdfPath && fs.existsSync(result.pdfPath));
    assert.ok(fs.existsSync(result.manifestPath));
    assert.equal(fs.readFileSync(result.pdfPath!).subarray(0, 5).toString('ascii'), '%PDF-');
    assert.ok(fs.statSync(result.pdfPath!).size > 20_000);
    const manifest = JSON.parse(fs.readFileSync(result.manifestPath, 'utf8'));
    assert.equal(manifest.planId, result.playbook.planId);
    assert.deepEqual(manifest.files.map((file: { name: string }) => file.name), ['plan.json', 'playbook.html', 'playbook.pdf']);
  } finally {
    fs.rmSync(outputDirectory, { recursive: true, force: true });
  }
});

test('anonymized generation removes direct identity and marks the manifest', async () => {
  const outputDirectory = fs.mkdtempSync(path.join(os.tmpdir(), 'mlmentorship-playbook-anon-'));
  try {
    const result = await generatePlaybook({
      intakePath: path.join(repoRoot, 'examples/playbook/sample-intake.json'),
      outputDirectory,
      repoRoot,
      htmlOnly: true,
      anonymize: true,
    });
    const plan = JSON.parse(fs.readFileSync(result.planPath, 'utf8'));
    const html = fs.readFileSync(result.htmlPath, 'utf8');
    const manifest = JSON.parse(fs.readFileSync(result.manifestPath, 'utf8'));
    assert.equal(plan.generatedFor, 'Candidate');
    assert.equal(plan.intake.experienceSummary, undefined);
    assert.deepEqual(plan.intake.constraints, []);
    assert.deepEqual(plan.intake.priorities, []);
    assert.doesNotMatch(html, /Sample Candidate/);
    assert.doesNotMatch(html, /One weekend day must remain completely free/);
    assert.equal(manifest.containsPersonalData, false);
  } finally {
    fs.rmSync(outputDirectory, { recursive: true, force: true });
  }
});
