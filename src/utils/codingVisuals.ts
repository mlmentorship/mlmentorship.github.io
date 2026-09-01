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

function setStep(
  visual: HTMLElement,
  steps: HTMLElement[],
  progress: HTMLOutputElement,
  status: HTMLElement | null,
  previous: HTMLButtonElement,
  next: HTMLButtonElement,
  index: number,
  announce: boolean,
): void {
  const boundedIndex = Math.max(0, Math.min(steps.length - 1, index));
  visual.dataset.activeStep = String(boundedIndex);
  steps.forEach((step, stepIndex) => {
    step.classList.toggle('is-active', stepIndex === boundedIndex);
    if (stepIndex === boundedIndex) step.setAttribute('aria-current', 'step');
    else step.removeAttribute('aria-current');
  });
  const activeStep = steps[boundedIndex];
  const activeLabel = activeStep?.querySelector<HTMLElement>('.coding-visual-step-label')?.textContent?.trim() ?? '';
  const activeValue = activeStep?.querySelector('strong')?.textContent?.trim() ?? '';
  const activeDetail = activeStep?.querySelector('small')?.textContent?.trim() ?? '';
  progress.value = `Step ${boundedIndex + 1} of ${steps.length}`;
  previous.disabled = boundedIndex === 0;
  next.disabled = boundedIndex === steps.length - 1;
  if (announce && status) status.textContent = `Step ${boundedIndex + 1} of ${steps.length}: ${activeLabel}, ${activeValue}. ${activeDetail}`;
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
    const steps = [...visual.querySelectorAll<HTMLElement>('[data-coding-step]')];
    if (!controls || !progress || !previous || !play || !playLabel || !next || steps.length < 2) continue;

    let currentStep = 0;
    let timer: number | undefined;
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const stop = () => {
      stopTimer(timer);
      timer = undefined;
      visual.dataset.codingPlaying = 'false';
      playLabel.textContent = currentStep === steps.length - 1 ? 'Replay trace' : 'Play trace';
      play.setAttribute('aria-label', playLabel.textContent);
      play.querySelector('[aria-hidden="true"]')?.replaceChildren(document.createTextNode('>'));
    };
    const update = (index: number, announce = true) => {
      currentStep = Math.max(0, Math.min(steps.length - 1, index));
      setStep(visual, steps, progress, status, previous, next, currentStep, announce);
    };
    const tick = () => {
      if (!document.contains(visual)) {
        stop();
        return;
      }
      if (currentStep >= steps.length - 1) {
        stop();
        return;
      }
      update(currentStep + 1);
      if (currentStep >= steps.length - 1) stop();
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
        update(currentStep >= steps.length - 1 ? 0 : currentStep + 1);
        return;
      }
      if (timer !== undefined) {
        stop();
        return;
      }
      if (currentStep >= steps.length - 1) update(0, false);
      visual.dataset.codingPlaying = 'true';
      playLabel.textContent = 'Pause trace';
      play.setAttribute('aria-label', 'Pause trace');
      play.querySelector('[aria-hidden="true"]')?.replaceChildren(document.createTextNode('||'));
      timer = window.setInterval(tick, 900);
      activeTimers.add(timer);
    });
    steps.forEach((step) => {
      step.setAttribute('tabindex', '0');
      step.setAttribute('role', 'button');
      step.addEventListener('click', () => {
        stop();
        update(Number(step.dataset.codingStep));
      });
      step.addEventListener('keydown', (event) => {
        if (event.key !== 'Enter' && event.key !== ' ') return;
        event.preventDefault();
        stop();
        update(Number(step.dataset.codingStep));
      });
    });

    controls.hidden = false;
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
