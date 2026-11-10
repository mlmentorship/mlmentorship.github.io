import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { test } from 'node:test';
import { fromMarkdown } from 'mdast-util-from-markdown';
import { parseFragment, type DefaultTreeAdapterMap } from 'parse5';
import { placeArticleVisualLesson, renderVisualLessonSteps } from '../src/utils/visualLessonHtml';
import { printPanelRanges } from '../src/utils/visualLessonControls';
import type { VisualLesson } from '../src/utils/visualAudits';
import catalog from '../data/visual-lessons.json';

type HtmlNode = DefaultTreeAdapterMap['node'];
function elements(node: HtmlNode): DefaultTreeAdapterMap['element'][] {
  return [...('tagName' in node ? [node] : []), ...('childNodes' in node ? node.childNodes.flatMap(elements) : [])];
}
const attribute = (node: DefaultTreeAdapterMap['element'], name: string) => node.attrs.find(item => item.name === name)?.value;
const fixture = '<figure id="source" class="learning-figure" data-coding-visual="fixture" aria-labelledby="heading"><h3 id="heading" class="visual-title">Old title</h3><div data-visual-argument="fixture"><svg viewBox="0 0 360 120" aria-labelledby="picture"><title id="picture">Primary</title><text font-size="14" x="20" y="50">Input</text></svg></div><figcaption>Primary caption</figcaption><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content"><svg viewBox="0 0 720 120"><title>Detailed mechanism</title><text font-size="10">State</text></svg><p class="visual-derivation-caption">Detail caption</p><div><div data-frame-key="start"></div><div data-frame-key="finish"></div></div></div></details></figure>';

test('server-rendered selection exposes the detail and namespaces its references without altering its source', () => {
  const lesson = { steps: [{ visualId: 'fixture', view: 'detail' }, { visualId: 'fixture', view: 'mechanism' }] } as VisualLesson;
  const rendered = renderVisualLessonSteps(fixture, lesson, 'article');
  const detail = elements(parseFragment(rendered[0]));
  assert.equal(attribute(detail[0], 'data-visual-lesson-step'), 'fixture:detail');
  assert.equal(attribute(detail[0], 'aria-labelledby'), 'article-1-heading');
  assert.equal(attribute(detail[0], 'aria-describedby'), 'article-1-fixture-lesson-caption');
  assert.ok(!detail.some(node => node.tagName === 'details' || attribute(node, 'data-visual-argument')));
  assert.ok(rendered[0].includes('Detailed mechanism'));
  assert.ok(rendered[0].includes('min-width:900px'));
  assert.ok(rendered[0].includes('height:auto') && rendered[0].includes('max-height:none'));
  assert.ok(rendered[1].includes('Primary caption') && !rendered[1].includes('Detail caption'));
  assert.ok(!rendered[1].includes('data-coding-visual'));
  assert.ok(rendered[0].includes('data-coding-visual="fixture"'));
  const ids = rendered.flatMap(html => elements(parseFragment(html)).map(node => attribute(node, 'id')).filter(Boolean));
  assert.equal(new Set(ids).size, ids.length);
  assert.ok(fixture.includes('<details class="visual-derivation">'));
});

test('authored print panels retain the screen asset and keep every SVG reference unique', () => {
  const lesson = { steps: [{ visualId: 'fixture', view: 'mechanism', printPanels: ['adam-two-memory-route', 'adam-decay-bypass'] }] } as VisualLesson;
  const [rendered] = renderVisualLessonSteps(fixture, lesson, 'print');
  const nodes = elements(parseFragment(rendered));
  assert.equal(nodes.filter(node => attribute(node, 'data-visual-lesson-step')).length, 1);
  assert.equal(nodes.filter(node => attribute(node, 'data-visual-lesson-screen') !== undefined).length, 1);
  assert.deepEqual(nodes.map(node => attribute(node, 'data-visual-lesson-print-panel')).filter(Boolean), lesson.steps[0].printPanels);
  assert.ok(rendered.includes('Primary caption') && rendered.includes('Primary</title>'));
  const ids = nodes.map(node => attribute(node, 'id')).filter(Boolean);
  assert.equal(new Set(ids).size, ids.length);
  nodes.forEach(node => ['aria-labelledby', 'aria-describedby'].forEach(name => {
    const references = attribute(node, name);
    if (references) references.split(/\s+/).forEach(identity => assert.ok(ids.includes(identity), identity));
  }));
});

test('print windows cover an oversized drawing without cutting measured labels', () => {
  const bands = [{ top: 580, bottom: 625 }, { top: 1150, bottom: 1190 }];
  const ranges = printPanelRanges(1500, bands, 600);
  assert.equal(ranges[0].top, 0);
  assert.equal(ranges.at(-1)!.bottom, 1500);
  ranges.forEach((range, index) => {
    assert.ok(range.bottom > range.top && range.bottom - range.top <= 600);
    if (index) assert.equal(range.top, ranges[index - 1].bottom);
    bands.forEach(band => assert.ok(!(band.top < range.bottom && band.bottom > range.bottom)));
  });
  assert.deepEqual(printPanelRanges(300, [], 600), [{ top: 0, bottom: 300 }]);
});

test('print windows keep a complete output node with its label', () => {
  const output = { top: 564, bottom: 615 };
  const ranges = printPanelRanges(640, [{ top: 575, bottom: 591 }, output], 600);
  assert.ok(ranges[0].bottom < output.top);
  assert.ok(ranges.some(range => range.top <= output.top && range.bottom >= output.bottom));
  assert.equal(ranges.at(-1)!.bottom, 640);
});

test('raw source and Astro-normalized class tokens produce identical selected markup', () => {
  const lesson = { steps: [{ visualId: 'fixture', view: 'mechanism' }] } as VisualLesson;
  const raw = fixture.replace('class="learning-figure"', 'class=" learning-figure  "');
  assert.deepEqual(renderVisualLessonSteps(raw, lesson, 'article'), renderVisualLessonSteps(fixture, lesson, 'article'));
});

test('all authored lessons materialize their exact order with all selected execution checkpoints', () => {
  const lessons = catalog.lessons as Record<string, VisualLesson>;
  const postSlugs = readdirSync(new URL('../src/content/posts/', import.meta.url)).filter(filename => filename.endsWith('.md')).map(filename => filename.replace(/^\d{4}-\d{2}-\d{2}-/, '').replace(/\.md$/, '')).sort();
  assert.deepEqual(Object.keys(lessons).sort(), postSlugs);
  for (const [slug, lesson] of Object.entries(lessons)) {
    const audit = JSON.parse(readFileSync(new URL(`../data/visual-audits/${slug}.json`, import.meta.url), 'utf8'));
    const source = readFileSync(new URL(`../${audit.article}`, import.meta.url), 'utf8');
    const html: string[] = [];
    function collect(node: ReturnType<typeof fromMarkdown> | ReturnType<typeof fromMarkdown>['children'][number]): void {
      if (node.type === 'html') html.push(node.value);
      if ('children' in node) node.children.forEach(collect);
    }
    collect(fromMarkdown(source));
    const article = placeArticleVisualLesson(html.join('\n'), lesson);
    const rendered = elements(parseFragment(article.lead + article.body)).filter(node => attribute(node, 'data-visual-lesson-step')).map(node => {
      const parent = node.parentNode as DefaultTreeAdapterMap['element'];
      assert.notEqual(parent.tagName, 'details', slug);
      return node;
    });
    assert.deepEqual(rendered.map(node => attribute(node, 'data-visual-lesson-step')), lesson.steps.map(step => `${step.visualId}:${step.view}`), slug);
    const ids = elements(parseFragment(article.lead + article.body)).map(node => attribute(node, 'id')).filter(Boolean);
    assert.equal(new Set(ids).size, ids.length, slug);
    rendered.forEach((value, index) => {
      const nodes = elements(value);
      assert.ok(!nodes.some(node => attribute(node, 'class')?.split(/\s+/).includes('visual-derivation')), slug);
      if (lesson.steps[index].view === 'trace') {
        const original = elements(parseFragment(html.join('\n'))).find(node => node.tagName === 'figure' && elements(node).some(child => attribute(child, 'data-visual-argument') === lesson.steps[index].visualId))!;
        assert.deepEqual(nodes.map(node => attribute(node, 'data-frame-key')).filter(Boolean), elements(original).map(node => attribute(node, 'data-frame-key')).filter(Boolean), slug);
      }
    });
  }
});

test('article placement exposes every selected step after the problem without changing prose or original identities', () => {
  const lesson = { steps: [{ visualId: 'fixture', view: 'detail' }, { visualId: 'fixture', view: 'mechanism' }] } as VisualLesson;
  const source = `<h2 id="problem">Problem</h2><p>Exact constraints.</p><h2 id="picture">Picture</h2><!-- visual:fixture -->${fixture}<h2 id="code">Code</h2><p>Unchanged explanation.</p>`;
  const article = placeArticleVisualLesson(source, lesson);
  const nodes = elements(parseFragment(article.lead + article.body));
  assert.deepEqual(nodes.map(node => attribute(node, 'data-visual-lesson-step')).filter(Boolean), ['fixture:detail', 'fixture:mechanism']);
  assert.ok(article.body.indexOf('Exact constraints.') < article.body.indexOf('data-article-lesson-support'));
  assert.ok(article.body.includes('Unchanged explanation.') && article.body.includes('id="source"'));
  assert.match(article.body, /<!-- visual:fixture -->\s*<figure/);
  assert.equal(nodes.find(node => attribute(node, 'data-visual-lesson-step') === 'fixture:mechanism')?.parentNode && attribute(nodes.find(node => attribute(node, 'data-visual-lesson-step') === 'fixture:mechanism')!.parentNode as DefaultTreeAdapterMap['element'], 'class'), 'article-lesson-support');
  assert.equal(nodes.filter(node => attribute(node, 'class') === 'article-visual-reference').length, 1);
});
