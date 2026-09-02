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
  const tracedFigure = visualId
    ? [...root.querySelectorAll<HTMLElement>('[data-article-trace]')]
      .find((element) => element.dataset.articleTrace === visualId)
    : undefined;
  if (tracedFigure) {
    return { marker: visualMarker(root, visualId), figure: tracedFigure, source: sourceAfter(tracedFigure) };
  }

  const marker = visualMarker(root, visualId);
  const markedElement = marker ? siblingElement(marker, 'next') : undefined;
  const existingFigure = markedElement?.classList.contains('learning-figure')
    ? markedElement
    : root.querySelector<HTMLElement>('.learning-figure');

  if (existingFigure) return { marker, figure: existingFigure, source: sourceAfter(existingFigure) };

  const mermaid = markedElement?.matches('pre.mermaid, .mermaid')
    ? markedElement
    : root.querySelector<HTMLElement>('pre.mermaid');
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