import { parseFragment, serialize, serializeOuter, type DefaultTreeAdapterMap } from 'parse5';
import artwork from '../../data/visual-lesson-artwork.json';
import type { VisualLesson, VisualLessonStep } from './visualAudits';

type HtmlNode = DefaultTreeAdapterMap['node'];
type HtmlElement = DefaultTreeAdapterMap['element'];
const scenes = artwork.scenes as Record<string, { html: string; sha256: string }>;

function elements(node: HtmlNode): HtmlElement[] {
  const found: HtmlElement[] = [];
  if ('tagName' in node) found.push(node);
  if ('childNodes' in node) node.childNodes.forEach(child => found.push(...elements(child)));
  return found;
}

function attribute(element: HtmlElement, name: string): string | undefined {
  return element.attrs.find(item => item.name === name)?.value;
}

function setAttribute(element: HtmlElement, name: string, value: string): void {
  const existing = element.attrs.find(item => item.name === name);
  if (existing) existing.value = value;
  else element.attrs.push({ name, value });
}

function hasClass(element: HtmlElement, name: string): boolean {
  return (attribute(element, 'class') ?? '').split(/\s+/).includes(name);
}

function remove(element: HtmlElement): void {
  const parent = element.parentNode!;
  parent.childNodes.splice(parent.childNodes.indexOf(element), 1);
}

function textContent(node: HtmlNode): string {
  if ('value' in node) return node.value;
  return 'childNodes' in node ? node.childNodes.map(textContent).join('') : '';
}

function inheritedFontSize(element: HtmlElement): number {
  let current: HtmlNode | null = element;
  while (current && 'tagName' in current) {
    const declared = attribute(current, 'font-size') ?? attribute(current, 'style')?.match(/(?:^|;)\s*font-size:\s*([\d.]+)(?:px)?(?:;|$)/)?.[1];
    if (declared) return Number.parseFloat(declared);
    current = current.parentNode;
  }
  return 16;
}

function namespaceReferences(children: HtmlElement[], prefix: string): void {
  const identities = new Map(children.filter(element => attribute(element, 'id')).map(element => [attribute(element, 'id')!, `${prefix}-${attribute(element, 'id')}`]));
  children.forEach(element => element.attrs.forEach(item => {
    if (item.name === 'id') item.value = identities.get(item.value)!;
    else if (item.name === 'class') item.value = item.value.trim().split(/\s+/).join(' ');
    else if (['aria-labelledby', 'aria-describedby', 'aria-controls', 'aria-owns', 'for'].includes(item.name)) item.value = item.value.split(/\s+/).map(value => identities.get(value) ?? value).join(' ');
    else if (item.name === 'href' && item.value.startsWith('#') && identities.has(item.value.slice(1))) item.value = `#${identities.get(item.value.slice(1))}`;
    else if (item.value.includes('url(#')) item.value = item.value.replace(/url\(#([^)]+)\)/g, (match, identity: string) => identities.has(identity) ? `url(#${identities.get(identity)})` : match);
  }));
}

function prepareFigure(figure: HtmlElement, step: VisualLessonStep, prefix: string): string {
  const descendants = elements(figure);
  const derivation = descendants.find(element => hasClass(element, 'visual-derivation'));
  if (step.view === 'mechanism') {
    if (derivation) remove(derivation);
    figure.attrs = figure.attrs.filter(item => !['data-coding-visual', 'data-coding-enhanced', 'data-coding-playing', 'data-active-frame', 'tabindex'].includes(item.name));
  } else {
    const content = derivation && elements(derivation).find(element => hasClass(element, 'visual-derivation-content'));
    if (!derivation || !content) throw new Error(`Missing ${step.view} content for ${step.visualId}`);
    const primary = descendants.find(element => attribute(element, 'data-visual-argument') !== undefined);
    if (primary) remove(primary);
    figure.childNodes.filter(node => 'tagName' in node && node.tagName === 'figcaption').forEach(node => remove(node as HtmlElement));
    const parent = derivation.parentNode!;
    parent.childNodes.splice(parent.childNodes.indexOf(derivation), 1, ...content.childNodes);
    content.childNodes.forEach(node => { node.parentNode = parent; });
    const caption = elements(figure).find(element => hasClass(element, 'visual-derivation-caption'));
    if (caption) {
      caption.tagName = 'figcaption';
      caption.nodeName = 'figcaption';
      setAttribute(caption, 'class', 'diagram-caption');
    }
  }
  if (step.printPanels?.length) {
    const screen = elements(figure).find(element => element.tagName === 'svg')!;
    setAttribute(screen, 'data-visual-lesson-screen', '');
    const panels = elements(parseFragment('<div class="visual-lesson-print-panels"></div>'))[0];
    step.printPanels.forEach((visualId, index) => {
      const source = scenes[visualId];
      if (!source) throw new Error(`Missing print panel ${visualId}`);
      const svg = elements(parseFragment(source.html)).find(element => element.tagName === 'svg')!;
      namespaceReferences(elements(svg), `print-${index + 1}`);
      setAttribute(svg, 'data-visual-lesson-print-panel', visualId);
      panels.childNodes.push(svg);
      svg.parentNode = panels;
    });
    const parent = screen.parentNode!;
    parent.childNodes.splice(parent.childNodes.indexOf(screen) + 1, 0, panels);
    panels.parentNode = parent;
  }
  const children = elements(figure);
  const heading = children.find(element => hasClass(element, 'visual-title'));
  const detailTitle = step.view === 'detail' ? children.find(element => element.tagName === 'title') : undefined;
  const title = step.title ?? (detailTitle ? textContent(detailTitle) : undefined);
  if (heading && title) heading.childNodes = [{ nodeName: '#text', value: title, parentNode: heading }];
  const selectedCaption = children.find(element => element.tagName === 'figcaption');
  if (selectedCaption) {
    const captionId = attribute(selectedCaption, 'id') ?? `${step.visualId}-lesson-caption`;
    setAttribute(selectedCaption, 'id', captionId);
    setAttribute(figure, 'aria-describedby', captionId);
  } else figure.attrs = figure.attrs.filter(item => item.name !== 'aria-describedby');
  setAttribute(figure, 'data-visual-lesson-step', `${step.visualId}:${step.view}`);
  setAttribute(figure, 'class', `${attribute(figure, 'class') ?? ''} visual-lesson-step`.trim());
  namespaceReferences(children, prefix);
  children.filter(element => element.tagName === 'svg').forEach(svg => {
    const viewBox = attribute(svg, 'viewBox') ?? attribute(svg, 'viewbox');
    const labels = elements(svg).filter(element => element.tagName === 'text');
    if (!viewBox || labels.length === 0) return;
    const width = Number(viewBox.trim().split(/[\s,]+/)[2]);
    const minimumSize = Math.min(...labels.map(inheritedFontSize));
    const readableWidth = Math.ceil(width * 12.5 / minimumSize);
    const canvas = elements(parseFragment('<div class="visual-lesson-canvas"></div>'))[0];
    const parent = svg.parentNode!;
    parent.childNodes.splice(parent.childNodes.indexOf(svg), 1, canvas);
    canvas.parentNode = parent;
    canvas.childNodes.push(svg);
    svg.parentNode = canvas;
    setAttribute(svg, 'style', `${attribute(svg, 'style') ?? ''};width:100%;height:auto;min-width:${readableWidth}px;max-width:none;max-height:none`);
  });
  return serializeOuter(figure);
}

export function renderVisualLessonSteps(articleHtml: string, lesson: VisualLesson, prefix: string): string[] {
  const articleFigures = elements(parseFragment(articleHtml)).filter(element => element.tagName === 'figure' && hasClass(element, 'learning-figure'));
  return lesson.steps.map((step, index) => {
    const source = scenes[step.visualId]?.html ?? articleFigures.find(figure => elements(figure).some(element => attribute(element, 'data-visual-argument') === step.visualId));
    if (!source) throw new Error(`Missing visual lesson asset ${step.visualId}`);
    const figure = elements(parseFragment(typeof source === 'string' ? source : serializeOuter(source))).find(element => element.tagName === 'figure')!;
    return prepareFigure(figure, step, `${prefix}-${index + 1}`);
  });
}

export function placeArticleVisualLesson(articleHtml: string, lesson: VisualLesson): { lead: string; body: string } {
  const rendered = renderVisualLessonSteps(articleHtml, lesson, 'article-lesson');
  const body = parseFragment(articleHtml);
  const originals = elements(body).filter(element => element.tagName === 'figure' && hasClass(element, 'learning-figure'));
  const insertions = new Map<HtmlNode, string[]>();
  let previousPosition = 0;
  lesson.steps.slice(1).forEach((step, index) => {
    let anchor = originals.find(figure => elements(figure).some(element => attribute(element, 'data-visual-argument') === step.visualId)) ?? originals[0];
    while (anchor.parentNode !== body) anchor = anchor.parentNode as HtmlElement;
    const position = Math.max(previousPosition, body.childNodes.indexOf(anchor));
    previousPosition = position;
    const anchorNode = body.childNodes[position];
    const group = insertions.get(anchorNode) ?? [];
    group.push(`<div class="article-lesson-support" data-article-lesson-support>${rendered[index + 1]}</div>`);
    insertions.set(anchorNode, group);
  });
  originals.forEach(figure => {
    const reference = elements(parseFragment('<details class="article-visual-reference"><summary>Reference drawing</summary></details>'))[0];
    const parent = figure.parentNode!;
    let markerIndex = parent.childNodes.indexOf(figure) - 1;
    while (markerIndex >= 0 && parent.childNodes[markerIndex].nodeName === '#text' && !textContent(parent.childNodes[markerIndex]).trim()) markerIndex -= 1;
    const marker = parent.childNodes[markerIndex];
    if (marker?.nodeName === '#comment' && 'data' in marker && marker.data.trim().startsWith('visual:')) {
      parent.childNodes.splice(markerIndex, 1);
      reference.childNodes.push(marker);
      marker.parentNode = reference;
    }
    parent.childNodes.splice(parent.childNodes.indexOf(figure), 1, reference);
    reference.parentNode = parent;
    reference.childNodes.push(figure);
    figure.parentNode = reference;
    const additions = insertions.get(figure);
    if (additions) {
      insertions.delete(figure);
      insertions.set(reference, additions);
    }
  });
  body.childNodes = body.childNodes.flatMap(node => {
    const additions = parseFragment((insertions.get(node) ?? []).join('')).childNodes;
    additions.forEach(child => { child.parentNode = body; });
    return [...additions, node];
  });
  return { lead: rendered[0], body: serialize(body) };
}