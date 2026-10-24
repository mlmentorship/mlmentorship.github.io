import { enhanceCodingVisuals } from './codingVisuals';

export interface ExtractedLearningVisual {
  marker?: Comment;
  figure: HTMLElement;
  source?: HTMLElement;
}

function siblingElement(node: Node, direction: 'next' | 'previous'): HTMLElement | undefined {
  let sibling = direction === 'next' ? node.nextSibling : node.previousSibling;
  while (sibling) {
    if (sibling.nodeType === Node.ELEMENT_NODE) return sibling as HTMLElement;
    sibling = direction === 'next' ? sibling.nextSibling : sibling.previousSibling;
  }
  return undefined;
}

function visualMarker(root: ParentNode, visualId?: string): Comment | undefined {
  if (!visualId) return undefined;
  const documentRoot = root instanceof Document ? root : root.ownerDocument;
  if (!documentRoot || !(root instanceof Node)) return undefined;
  const walker = documentRoot.createTreeWalker(root, NodeFilter.SHOW_COMMENT);
  let node = walker.nextNode();
  while (node) {
    if ((node.nodeValue ?? '').trim() === `visual:${visualId}`) return node as Comment;
    node = walker.nextNode();
  }
  return undefined;
}

function sourceAfter(element: Element | undefined): HTMLElement | undefined {
  const sibling = element?.nextElementSibling;
  return sibling instanceof HTMLElement && sibling.classList.contains('diagram-source') ? sibling : undefined;
}

export function extractLearningVisual(root: ParentNode, visualId?: string): ExtractedLearningVisual | undefined {
  const marker = visualMarker(root, visualId);
  if (visualId && !marker) return undefined;
  const markedElement = marker ? siblingElement(marker, 'next') : undefined;
  const existingFigure = markedElement?.classList.contains('learning-figure')
    ? markedElement
    : visualId ? undefined : root.querySelector<HTMLElement>('.learning-figure');

  if (existingFigure) return { marker, figure: existingFigure, source: sourceAfter(existingFigure) };

  const mermaid = markedElement?.matches('pre.mermaid, .mermaid')
    ? markedElement
    : visualId ? undefined : root.querySelector<HTMLElement>('pre.mermaid');
  if (!mermaid) return undefined;

  const caption = mermaid.nextElementSibling instanceof HTMLElement && mermaid.nextElementSibling.classList.contains('diagram-caption')
    ? mermaid.nextElementSibling
    : undefined;
  const source = sourceAfter(caption ?? mermaid);
  const visualTitle = marker ? siblingElement(marker, 'previous') : undefined;
  const visualKicker = visualTitle?.previousElementSibling instanceof HTMLElement
    ? visualTitle.previousElementSibling
    : undefined;
  const documentRoot = mermaid.ownerDocument;
  const figure = documentRoot.createElement('figure');
  const figcaption = caption ? documentRoot.createElement('figcaption') : undefined;
  figure.className = 'learning-figure review-generated-figure';
  mermaid.classList.add('visual-scroll');
  mermaid.parentNode?.insertBefore(figure, mermaid);

  if (visualKicker?.classList.contains('visual-kicker')) figure.append(visualKicker);
  if (visualTitle?.classList.contains('visual-title')) {
    if (!visualTitle.id) visualTitle.id = `visual-${visualId ?? 'mermaid'}-title`;
    figure.setAttribute('aria-labelledby', visualTitle.id);
    figure.append(visualTitle);
  }
  figure.append(mermaid);
  if (caption && figcaption) {
    figcaption.className = caption.className;
    while (caption.firstChild) figcaption.append(caption.firstChild);
    caption.remove();
    figure.append(figcaption);
  }
  return { marker, figure, source };
}

export function absolutizeVisualLinks(element: Element, articleHref: string) {
  const base = new URL(articleHref, window.location.origin);
  element.querySelectorAll<HTMLElement>('[href], [src]').forEach((child) => {
    for (const attribute of ['href', 'src']) {
      const value = child.getAttribute(attribute);
      if (!value || value.startsWith('#') || value.startsWith('data:')) continue;
      try { child.setAttribute(attribute, new URL(value, base).toString()); } catch { /* Preserve non-URL values. */ }
    }
  });
}

const previewArticleCache = new Map<string, Promise<Document>>();
const initializedPreviews = new WeakSet<HTMLElement>();

async function loadVisualPreview(target: HTMLElement): Promise<void> {
  const href = target.dataset.landingVisualHref;
  const visualId = target.dataset.landingVisualId;
  if (!href || !visualId || target.dataset.loaded === 'true') return;
  target.setAttribute('aria-busy', 'true');
  try {
    let request = previewArticleCache.get(href);
    if (!request) {
      request = fetch(href, { headers: { Accept: 'text/html' } })
        .then((response) => {
          if (!response.ok) throw new Error(`Article returned ${response.status}`);
          return response.text();
        })
        .then((html) => new DOMParser().parseFromString(html, 'text/html'))
        .catch((error) => {
          previewArticleCache.delete(href);
          throw error;
        });
      previewArticleCache.set(href, request);
    }
    const extracted = extractLearningVisual(await request, visualId);
    if (!extracted) throw new Error('No completed visual was found.');
    const figure = document.importNode(extracted.figure, true);
    absolutizeVisualLinks(figure, href);
    target.replaceChildren(figure);
    const diagrams = [...target.querySelectorAll<HTMLElement>('pre.mermaid')];
    if (diagrams.length > 0) {
      const { default: mermaid } = await import('mermaid');
      mermaid.initialize({ startOnLoad: false, theme: document.documentElement.dataset.theme === 'dark' ? 'dark' : 'default' });
      await mermaid.run({ nodes: diagrams });
    }
    enhanceCodingVisuals(target);
    target.dataset.loaded = 'true';
  } catch (error) {
    const message = document.createElement('p');
    message.className = 'landing-visual-error';
    message.textContent = 'Visual unavailable here.';
    const link = document.createElement('a');
    link.href = href;
    link.textContent = 'Read the lesson';
    target.replaceChildren(message, link);
    console.error('Visual preview could not be loaded.', error);
  } finally {
    target.setAttribute('aria-busy', 'false');
  }
}

export function initializeVisualPreviews(root: ParentNode = document): void {
  const targets = [...root.querySelectorAll<HTMLElement>('[data-landing-visual]')].filter((target) => !initializedPreviews.has(target));
  if (targets.length === 0) return;
  const observer = 'IntersectionObserver' in window ? new IntersectionObserver((observations) => {
    observations.filter((observation) => observation.isIntersecting).forEach((observation) => {
      observer?.unobserve(observation.target);
      void loadVisualPreview(observation.target as HTMLElement);
    });
  }, { rootMargin: '600px 0px' }) : undefined;
  targets.forEach((target) => {
    initializedPreviews.add(target);
    if (target.dataset.landingVisualLoad === 'eager' || !observer) void loadVisualPreview(target);
    else observer.observe(target);
  });
}