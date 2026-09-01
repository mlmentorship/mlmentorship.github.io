const activeTimers = new Set<number>();
let lifecycleBound = false;

function stopTimer(timer: number | undefined): void {
  if (timer === undefined) return;
  window.clearInterval(timer);
  activeTimers.delete(timer);
}

function bindLifecycle(): void {
  if (lifecycleBound) return;
  lifecycleBound = true;
  document.addEventListener('astro:before-preparation', () => {
    for (const timer of activeTimers) window.clearInterval(timer);
    activeTimers.clear();
  });
}

function setFrame(
  visual: HTMLElement,
  frames: HTMLElement[],
  frameButtons: HTMLButtonElement[],
  progress: HTMLOutputElement,
  status: HTMLElement | null,
  previous: HTMLButtonElement,
  next: HTMLButtonElement,
  index: number,
  announce: boolean,
): void {
  const boundedIndex = Math.max(0, Math.min(frames.length - 1, index));
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const previousPositions = new Map<string, DOMRect>();
  if (!reducedMotion) {
    visual.querySelectorAll<HTMLElement | SVGElement>('[data-motion-key]').forEach((element) => {
      if (element.getClientRects().length > 0) previousPositions.set(element.dataset.motionKey ?? '', element.getBoundingClientRect());
    });
  }
  visual.dataset.activeFrame = String(boundedIndex);
  frames.forEach((frame, frameIndex) => {
    frame.hidden = frameIndex !== boundedIndex;
    if (frameIndex === boundedIndex) frame.setAttribute('aria-current', 'step');
    else frame.removeAttribute('aria-current');
  });
  frameButtons.forEach((button, buttonIndex) => {
    if (buttonIndex === boundedIndex) button.setAttribute('aria-current', 'step');
    else button.removeAttribute('aria-current');
  });
  const activeFrame = frames[boundedIndex];
  if (!reducedMotion && activeFrame) {
    activeFrame.querySelectorAll<HTMLElement | SVGElement>('[data-motion-key]').forEach((element) => {
      const previousPosition = previousPositions.get(element.dataset.motionKey ?? '');
      if (!previousPosition) return;
      const position = element.getBoundingClientRect();
      const deltaX = previousPosition.left - position.left;
      const deltaY = previousPosition.top - position.top;
      if (Math.abs(deltaX) < 1 && Math.abs(deltaY) < 1) return;
      element.animate(
        [
          { transform: `translate(${deltaX}px, ${deltaY}px)` },
          { transform: 'translate(0, 0)' },
        ],
        { duration: 420, easing: 'cubic-bezier(.2,.8,.2,1)' },
      );
    });
  }
  const activeLabel = activeFrame?.getAttribute('aria-label') ?? '';
  const activeNote = activeFrame?.querySelector<HTMLElement>('.coding-trace-frame-heading strong')?.textContent?.trim() ?? '';
  progress.value = `Step ${boundedIndex + 1} of ${frames.length}`;
  progress.textContent = progress.value;
  previous.disabled = boundedIndex === 0;
  next.disabled = boundedIndex === frames.length - 1;
  if (announce && status) status.textContent = `Step ${boundedIndex + 1} of ${frames.length}: ${activeLabel}. ${activeNote}`;
}

export function enhanceCodingVisuals(root: ParentNode = document): void {
  bindLifecycle();
  const candidates: HTMLElement[] = [];
  if (root instanceof HTMLElement && root.matches('[data-coding-visual]')) candidates.push(root);
  candidates.push(...root.querySelectorAll<HTMLElement>('[data-coding-visual]'));

  for (const visual of candidates) {
    if (visual.dataset.codingEnhanced === 'true') continue;
    const controls = visual.querySelector<HTMLElement>('[data-coding-controls]');
    const progress = visual.querySelector<HTMLOutputElement>('[data-coding-progress]');
    const previous = visual.querySelector<HTMLButtonElement>('[data-coding-previous]');
    const play = visual.querySelector<HTMLButtonElement>('[data-coding-play]');
    const playLabel = visual.querySelector<HTMLElement>('[data-coding-play-label]');
    const next = visual.querySelector<HTMLButtonElement>('[data-coding-next]');
    const status = visual.querySelector<HTMLElement>('[data-coding-status]');
    const timeline = visual.querySelector<HTMLElement>('[data-coding-timeline]');
    const frames = [...visual.querySelectorAll<HTMLElement>('[data-coding-frame]')];
    const frameButtons = [...visual.querySelectorAll<HTMLButtonElement>('[data-coding-frame-button]')];
    if (!controls || !progress || !previous || !play || !playLabel || !next || !timeline || frames.length < 2 || frameButtons.length !== frames.length) continue;

    let currentStep = 0;
    let timer: number | undefined;
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const stop = () => {
      stopTimer(timer);
      timer = undefined;
      visual.dataset.codingPlaying = 'false';
      playLabel.textContent = currentStep === frames.length - 1 ? 'Replay trace' : 'Play trace';
      play.setAttribute('aria-label', playLabel.textContent);
      play.querySelector('[aria-hidden="true"]')?.replaceChildren(document.createTextNode('>'));
    };
    const update = (index: number, announce = true) => {
      currentStep = Math.max(0, Math.min(frames.length - 1, index));
      setFrame(visual, frames, frameButtons, progress, status, previous, next, currentStep, announce);
    };
    const tick = () => {
      if (!document.contains(visual)) {
        stop();
        return;
      }
      if (currentStep >= frames.length - 1) {
        stop();
        return;
      }
      update(currentStep + 1);
      if (currentStep >= frames.length - 1) stop();
    };

    previous.addEventListener('click', () => {
      stop();
      update(currentStep - 1);
    });
    next.addEventListener('click', () => {
      stop();
      update(currentStep + 1);
    });
    play.addEventListener('click', () => {
      if (reducedMotion) {
        update(currentStep >= frames.length - 1 ? 0 : currentStep + 1);
        return;
      }
      if (timer !== undefined) {
        stop();
        return;
      }
      if (currentStep >= frames.length - 1) update(0, false);
      visual.dataset.codingPlaying = 'true';
      playLabel.textContent = 'Pause trace';
      play.setAttribute('aria-label', 'Pause trace');
      play.querySelector('[aria-hidden="true"]')?.replaceChildren(document.createTextNode('||'));
      timer = window.setInterval(tick, 900);
      activeTimers.add(timer);
    });
    frameButtons.forEach((button, frameIndex) => {
      button.addEventListener('click', () => {
        stop();
        update(frameIndex);
      });
    });
    visual.addEventListener('keydown', (event) => {
      if (event.altKey || event.ctrlKey || event.metaKey) return;
      if (event.key === 'ArrowLeft') {
        event.preventDefault();
        stop();
        update(currentStep - 1);
      } else if (event.key === 'ArrowRight') {
        event.preventDefault();
        stop();
        update(currentStep + 1);
      } else if (event.key === 'Home') {
        event.preventDefault();
        stop();
        update(0);
      } else if (event.key === 'End') {
        event.preventDefault();
        stop();
        update(frames.length - 1);
      }
    });

    controls.hidden = false;
    timeline.hidden = false;
    visual.classList.add('is-enhanced');
    visual.dataset.codingEnhanced = 'true';
    visual.dataset.codingPlaying = 'false';
    if (reducedMotion) {
      playLabel.textContent = 'Next step';
      play.setAttribute('aria-label', 'Next step');
    }
    update(0, false);
  }
}
