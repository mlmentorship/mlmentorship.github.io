import { stopCodingVisuals } from './codingVisuals';

export function printPanelRanges(height: number, bands: { top: number; bottom: number }[], maximumHeight: number): { top: number; bottom: number }[] {
  const ranges: { top: number; bottom: number }[] = [];
  let top = 0;
  while (top < height) {
    let bottom = Math.min(height, top + maximumHeight);
    if (bottom < height) {
      let crossing = bands.find(band => band.top < bottom && band.bottom > bottom);
      while (crossing) {
        bottom = crossing.top - 2;
        crossing = bands.find(band => band.top < bottom && band.bottom > bottom);
      }
    }
    if (bottom <= top) throw new Error('A visual label is taller than a print panel');
    ranges.push({ top, bottom });
    top = bottom;
  }
  return ranges;
}

function addPrintPanels(svg: SVGSVGElement, readableWidth: number): void {
  if (svg.matches('[data-visual-lesson-print-panel], [data-visual-lesson-screen]')) return;
  if (!svg.getBoundingClientRect().width) return;
  const view = svg.viewBox.baseVal;
  const scale = Math.max(1, readableWidth / view.width);
  const maximumHeight = 600 / scale;
  if (view.height <= maximumHeight) return;
  const matrix = svg.getCTM()!.inverse();
  const bounds = (element: SVGGraphicsElement) => {
    const box = element.getBBox();
    const transform = matrix.multiply(element.getCTM()!);
    const points = [new DOMPoint(box.x, box.y), new DOMPoint(box.x + box.width, box.y), new DOMPoint(box.x, box.y + box.height), new DOMPoint(box.x + box.width, box.y + box.height)].map(point => point.matrixTransform(transform));
    return { top: Math.min(...points.map(point => point.y)) - view.y, bottom: Math.max(...points.map(point => point.y)) - view.y };
  };
  const bands = [...svg.querySelectorAll<SVGTextElement>('text')].map(bounds);
  const nodeBands = [...svg.querySelectorAll<SVGGraphicsElement>('rect, circle, ellipse, polygon')].map(bounds).filter(band => band.bottom - band.top <= maximumHeight);
  const panels = document.createElement('div');
  panels.className = 'visual-lesson-print-panels';
  printPanelRanges(view.height, [...bands, ...nodeBands], maximumHeight).forEach((range, index) => {
    const panel = svg.cloneNode(true) as SVGSVGElement;
    panel.setAttribute('viewBox', `${view.x} ${view.y + range.top} ${view.width} ${range.bottom - range.top}`);
    [...panel.querySelectorAll('text')].forEach((label, labelIndex) => {
      const band = bands[labelIndex];
      if (band.bottom <= range.top || band.top >= range.bottom) label.remove();
    });
    panel.dataset.visualLessonPrintPanel = String(index + 1);
    panel.style.overflow = 'hidden';
    const descendants = [panel, ...panel.querySelectorAll<SVGElement>('*')];
    const identities = new Map(descendants.filter(node => node.id).map(node => [node.id, `${node.id}-print-${index + 1}`]));
    descendants.forEach(node => [...node.attributes].forEach(attribute => {
      if (attribute.name === 'id') node.id = identities.get(attribute.value)!;
      else if (['aria-labelledby', 'aria-describedby', 'aria-controls'].includes(attribute.name)) node.setAttribute(attribute.name, attribute.value.split(/\s+/).map(value => identities.get(value) ?? value).join(' '));
      else if (attribute.value.startsWith('#') && identities.has(attribute.value.slice(1))) node.setAttribute(attribute.name, `#${identities.get(attribute.value.slice(1))}`);
      else if (attribute.value.includes('url(#')) node.setAttribute(attribute.name, attribute.value.replace(/url\(#([^)]+)\)/g, (match, identity: string) => identities.has(identity) ? `url(#${identities.get(identity)})` : match));
    }));
    const canvas = document.createElement('div');
    canvas.className = 'visual-lesson-canvas';
    canvas.append(panel);
    panels.append(canvas);
  });
  svg.dataset.visualLessonScreen = '';
  svg.parentElement!.after(panels);
}

export function enhanceVisualLesson(host: HTMLElement): void {
  const figures = [...host.querySelectorAll<HTMLElement>(':scope > [data-visual-lesson-step]')];
  if (figures.length === 0 || host.querySelector(':scope > [data-visual-lesson-controls]')) return;
  figures.forEach(figure => figure.classList.add('visual-lesson-step'));
  figures.forEach(figure => figure.querySelectorAll<SVGSVGElement>('svg').forEach(svg => {
    const sizes = [...svg.querySelectorAll('text')].map(text => parseFloat(getComputedStyle(text).fontSize)).filter(size => size > 0);
    if (!sizes.length || !svg.viewBox.baseVal.width) return;
    if (!svg.parentElement?.classList.contains('visual-lesson-canvas')) {
      const canvas = document.createElement('div');
      canvas.className = 'visual-lesson-canvas';
      svg.before(canvas);
      canvas.append(svg);
    }
    const readableWidth = Math.ceil(svg.viewBox.baseVal.width * 12.5 / Math.min(...sizes));
    svg.style.minWidth = `${readableWidth}px`;
    svg.style.setProperty('--visual-lesson-print-width', `${Math.max(svg.viewBox.baseVal.width, readableWidth)}px`);
    svg.style.maxWidth = 'none';
    svg.style.width = '100%';
    addPrintPanels(svg, readableWidth);
  }));
  if (figures.length === 1) return;

  const controls = document.createElement('nav');
  controls.className = 'visual-lesson-controls';
  controls.dataset.visualLessonControls = '';
  controls.setAttribute('aria-label', 'Visual lesson steps');
  const previous = document.createElement('button');
  previous.type = 'button';
  previous.dataset.visualLessonPrevious = '';
  previous.title = 'Previous visual step';
  previous.setAttribute('aria-label', previous.title);
  previous.innerHTML = '<span aria-hidden="true">&larr;</span>';
  const next = document.createElement('button');
  next.type = 'button';
  next.dataset.visualLessonNext = '';
  next.title = 'Next visual step';
  next.setAttribute('aria-label', next.title);
  next.innerHTML = '<span aria-hidden="true">&rarr;</span>';
  const position = document.createElement('output');
  position.setAttribute('aria-live', 'polite');
  let current = 0;

  function show(index: number): void {
    current = Math.max(0, Math.min(figures.length - 1, index));
    figures.forEach((figure, figureIndex) => {
      const selected = figureIndex === current;
      if (!selected) stopCodingVisuals(figure);
      figure.hidden = !selected;
      if (selected) figure.setAttribute('aria-current', 'step');
      else figure.removeAttribute('aria-current');
    });
    host.dataset.visualLessonIndex = String(current);
    position.value = `${current + 1} / ${figures.length}`;
    position.textContent = position.value;
    previous.disabled = current === 0;
    next.disabled = current === figures.length - 1;
  }

  previous.addEventListener('click', () => show(current - 1));
  next.addEventListener('click', () => show(current + 1));
  controls.addEventListener('keydown', event => {
    if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
    event.preventDefault();
    show(current + (event.key === 'ArrowRight' ? 1 : -1));
  });
  controls.append(previous, position, next);
  host.prepend(controls);
  show(0);
}