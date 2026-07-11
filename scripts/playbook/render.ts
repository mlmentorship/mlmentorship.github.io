import type { PersonalizedPlaybook, PlanTask, ReadinessArea, TaskType } from './types';
import { LEVEL_LABELS, ROLE_LABELS, ROUND_LABELS } from './rules';

function escapeHtml(value: unknown): string {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function formatDate(value: string): string {
  return new Intl.DateTimeFormat('en-US', { month: 'short', day: 'numeric', year: 'numeric', timeZone: 'UTC' })
    .format(new Date(`${value}T00:00:00Z`));
}

function addDays(value: string, days: number): Date {
  const date = new Date(`${value}T00:00:00Z`);
  date.setUTCDate(date.getUTCDate() + days);
  return date;
}

function weekRange(startDate: string, week: number): string {
  const start = addDays(startDate, (week - 1) * 7);
  const end = addDays(startDate, week * 7 - 1);
  const formatter = new Intl.DateTimeFormat('en-US', { month: 'short', day: 'numeric', timeZone: 'UTC' });
  return `${formatter.format(start)}–${formatter.format(end)}`;
}

function taskBadge(type: TaskType): string {
  return {
    read: 'Learn',
    practice: 'Practice',
    design: 'Design',
    story: 'Story',
    derive: 'Derive',
    review: 'Retry',
    simulation: 'Simulate',
  }[type];
}

function readinessRow(item: ReadinessArea): string {
  const scoreWidth = Math.max(8, item.rating * 20);
  return `<tr>
    <td><strong>${escapeHtml(item.label)}</strong></td>
    <td><span class="priority priority--${item.priority}">${escapeHtml(item.priority)}</span></td>
    <td><div class="rating"><span style="width:${scoreWidth}%"></span></div><small>${item.rating}/5</small></td>
    <td>${escapeHtml(item.rationale)}</td>
  </tr>`;
}

function taskCard(task: PlanTask): string {
  const title = task.absoluteUrl
    ? `<a href="${escapeHtml(task.absoluteUrl)}">${escapeHtml(task.title)}</a>`
    : escapeHtml(task.title);
  return `<article class="task task--${task.type}">
    <div class="task-head">
      <span class="task-badge">${taskBadge(task.type)}</span>
      <span class="task-time">${task.minutes} min</span>
    </div>
    <h4>${title}</h4>
    <p class="task-why">${escapeHtml(task.why)}</p>
    <ol>${task.instructions.map((item) => `<li>${escapeHtml(item)}</li>`).join('')}</ol>
  </article>`;
}

function weekSection(playbook: PersonalizedPlaybook, weekIndex: number): string {
  const week = playbook.weeks[weekIndex];
  const days = new Map<number, PlanTask[]>();
  for (const task of week.tasks) {
    const list = days.get(task.day) ?? [];
    list.push(task);
    days.set(task.day, list);
  }

  return `<section class="week page-break-before">
    <header class="week-head">
      <div>
        <p class="eyebrow">Week ${week.week} · ${weekRange(playbook.intake.startDate, week.week)}</p>
        <h2>${escapeHtml(week.theme)}</h2>
        <p>${escapeHtml(week.objective)}</p>
      </div>
      <div class="week-hours"><strong>${(week.plannedMinutes / 60).toFixed(1)}h</strong><span>scheduled</span><small>of ${playbook.intake.hoursPerWeek}h available</small></div>
    </header>
    <aside class="exit-criteria">
      <h3>Exit criteria</h3>
      <ul>${week.exitCriteria.map((item) => `<li>${escapeHtml(item)}</li>`).join('')}</ul>
    </aside>
    ${[...days.entries()].sort(([a], [b]) => a - b).map(([day, tasks]) => `
      <div class="day">
        <h3>Session ${day}</h3>
        ${tasks.map(taskCard).join('')}
      </div>`).join('')}
  </section>`;
}

function checkboxList(items: string[]): string {
  return `<ul class="checklist">${items.map((item) => `<li><span class="checkbox"></span>${escapeHtml(item)}</li>`).join('')}</ul>`;
}

export function renderPlaybookHtml(playbook: PersonalizedPlaybook): string {
  const intake = playbook.intake;
  const selectedRounds = intake.rounds.map((round) => ROUND_LABELS[round]).join(' · ');
  const selectedDomains = intake.domainTracks.length > 0
    ? intake.domainTracks.map((domain) => domain.replace(/-/g, ' ')).join(' · ')
    : 'General ML';

  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>${escapeHtml(playbook.executiveSummary.headline)} · mlmentorship</title>
  <style>
    :root {
      --ink: #111827;
      --soft: #475569;
      --muted: #64748b;
      --rule: #dbe2ea;
      --paper: #ffffff;
      --tint: #f7f8fa;
      --orange: #c2410c;
      --orange-soft: #ffedd5;
      --green: #166534;
      --green-soft: #dcfce7;
      --amber: #92400e;
      --amber-soft: #fef3c7;
      --red: #991b1b;
      --red-soft: #fee2e2;
    }
    * { box-sizing: border-box; }
    html { color: var(--ink); background: #e5e7eb; font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; line-height: 1.5; }
    body { width: 210mm; margin: 0 auto; background: var(--paper); font-size: 10.2pt; }
    a { color: var(--orange); text-decoration-thickness: 0.8px; text-underline-offset: 2px; }
    h1, h2, h3, h4, p { margin-top: 0; }
    h1 { font-size: 32pt; line-height: 1.02; letter-spacing: -1.2px; margin-bottom: 7mm; }
    h2 { font-size: 21pt; line-height: 1.1; letter-spacing: -0.5px; margin-bottom: 4mm; }
    h3 { font-size: 12pt; margin-bottom: 3mm; }
    h4 { font-size: 11pt; line-height: 1.25; margin: 0 0 2mm; }
    p { margin-bottom: 4mm; }
    small { color: var(--muted); }
    .page { min-height: 277mm; padding: 18mm 17mm 16mm; }
    .page-break-before { break-before: page; page-break-before: always; }
    .cover { position: relative; display: flex; min-height: 297mm; padding: 20mm; flex-direction: column; overflow: hidden; background: linear-gradient(145deg, #111827 0 70%, #1f2937 70%); color: white; }
    .cover::after { content: ''; position: absolute; width: 105mm; height: 105mm; right: -38mm; bottom: -30mm; border: 16mm solid #ea580c; border-radius: 50%; opacity: .95; }
    .brand { display: flex; align-items: center; gap: 3mm; font-weight: 700; letter-spacing: -.2px; }
    .brand-mark { display: inline-flex; width: 11mm; height: 9mm; border-radius: 2mm; align-items: center; justify-content: center; background: #fdba74; color: #1c1917; font-size: 9pt; }
    .cover-main { margin: auto 0; max-width: 155mm; }
    .cover .eyebrow { color: #fdba74; }
    .cover h1 { font-size: 40pt; }
    .cover-subtitle { font-size: 16pt; line-height: 1.35; color: #d1d5db; max-width: 140mm; }
    .cover-meta { display: grid; grid-template-columns: repeat(3, 1fr); gap: 5mm; position: relative; z-index: 1; }
    .cover-meta div { padding-top: 4mm; border-top: .4mm solid #4b5563; }
    .cover-meta strong, .cover-meta span { display: block; }
    .cover-meta span { margin-top: 1mm; color: #9ca3af; font-size: 8.5pt; }
    .eyebrow { margin-bottom: 3mm; font-size: 8pt; font-weight: 700; letter-spacing: 1.4px; text-transform: uppercase; color: var(--orange); }
    .lede { font-size: 14pt; line-height: 1.45; color: var(--soft); max-width: 160mm; }
    .summary-grid { display: grid; grid-template-columns: 1.25fr .75fr; gap: 7mm; margin: 7mm 0; }
    .card { padding: 6mm; border: .35mm solid var(--rule); border-radius: 3mm; background: var(--paper); break-inside: avoid; }
    .card--tint { background: var(--tint); }
    .card ul { margin: 0; padding-left: 5mm; }
    .meta-table { width: 100%; border-collapse: collapse; margin: 5mm 0 8mm; }
    .meta-table td { padding: 3mm 2mm; border-bottom: .3mm solid var(--rule); vertical-align: top; }
    .meta-table td:first-child { width: 36mm; color: var(--muted); font-size: 8.5pt; text-transform: uppercase; letter-spacing: .5px; }
    .readiness { width: 100%; border-collapse: collapse; font-size: 8.7pt; }
    .readiness th { padding: 2.5mm; text-align: left; color: var(--muted); font-size: 7.5pt; text-transform: uppercase; letter-spacing: .5px; border-bottom: .5mm solid var(--ink); }
    .readiness td { padding: 3mm 2.5mm; border-bottom: .3mm solid var(--rule); vertical-align: top; }
    .readiness td:nth-child(1) { width: 34mm; }
    .readiness td:nth-child(2) { width: 22mm; }
    .readiness td:nth-child(3) { width: 38mm; }
    .rating { display: inline-block; width: 25mm; height: 2.2mm; margin-right: 2mm; border-radius: 2mm; background: #e5e7eb; overflow: hidden; vertical-align: middle; }
    .rating span { display: block; height: 100%; background: var(--orange); }
    .priority { display: inline-block; padding: .8mm 2mm; border-radius: 8mm; font-size: 7pt; font-weight: 700; text-transform: uppercase; }
    .priority--critical { color: var(--red); background: var(--red-soft); }
    .priority--high { color: var(--amber); background: var(--amber-soft); }
    .priority--maintain { color: var(--green); background: var(--green-soft); }
    .week { padding: 16mm 17mm; }
    .week-head { display: grid; grid-template-columns: 1fr 35mm; gap: 8mm; padding-bottom: 6mm; margin-bottom: 7mm; border-bottom: .6mm solid var(--ink); }
    .week-head p { color: var(--soft); }
    .week-hours { text-align: right; padding: 3mm; background: var(--tint); border-radius: 3mm; }
    .week-hours strong { display: block; font-size: 18pt; }
    .week-hours span, .week-hours small { display: block; }
    .day { margin: 0 0 7mm; break-inside: avoid; }
    .day > h3 { display: flex; align-items: center; gap: 3mm; color: var(--muted); font-size: 8pt; letter-spacing: 1px; text-transform: uppercase; }
    .day > h3::after { content: ''; flex: 1; height: .3mm; background: var(--rule); }
    .task { margin: 0 0 4mm; padding: 4.5mm 5mm; border-left: 1.4mm solid var(--orange); background: var(--tint); break-inside: avoid; }
    .task--review { border-left-color: #2563eb; }
    .task--simulation { border-left-color: var(--green); }
    .task-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 2mm; }
    .task-badge { padding: .8mm 2mm; border-radius: 5mm; background: var(--orange-soft); color: var(--orange); font-size: 7pt; font-weight: 700; text-transform: uppercase; letter-spacing: .5px; }
    .task-time { color: var(--muted); font-size: 8pt; }
    .task-why { color: var(--soft); font-size: 8.7pt; }
    .task ol { margin: 0; padding-left: 5mm; font-size: 8.5pt; }
    .exit-criteria { margin: 0 0 6mm; padding: 4mm 5mm; border: .4mm solid var(--rule); border-radius: 3mm; break-inside: avoid; }
    .exit-criteria h3 { margin-bottom: 2mm; }
    .exit-criteria ul { display: grid; grid-template-columns: repeat(3, 1fr); gap: 4mm; margin: 0; padding: 0; list-style-position: inside; }
    .protocol-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 5mm; margin: 7mm 0; }
    .rubric { display: grid; grid-template-columns: 36mm 1fr; gap: 0; border: .3mm solid var(--rule); }
    .rubric div { padding: 3mm; border-bottom: .3mm solid var(--rule); }
    .rubric div:nth-child(odd) { background: var(--tint); font-weight: 700; }
    .story { margin-bottom: 4mm; padding-bottom: 4mm; border-bottom: .3mm solid var(--rule); break-inside: avoid; }
    .story ul { margin-bottom: 0; }
    .checklist { list-style: none; padding: 0; }
    .checklist li { display: flex; gap: 3mm; margin: 0 0 3mm; break-inside: avoid; }
    .checkbox { flex: 0 0 auto; width: 4mm; height: 4mm; margin-top: 1mm; border: .4mm solid var(--muted); border-radius: .8mm; }
    .compact-rules .checklist { columns: 2; column-gap: 8mm; }
    .resource-list { padding: 0; list-style: none; columns: 2; column-gap: 8mm; font-size: 8pt; line-height: 1.25; }
    .resource-list li { break-inside: avoid; margin-bottom: 2mm; padding-bottom: 1.6mm; border-bottom: .3mm solid var(--rule); }
    .resource-list strong, .resource-list span { display: block; }
    .resource-list span { color: var(--muted); font-size: 8pt; }
    .footer-note { margin-top: 10mm; padding-top: 5mm; border-top: .3mm solid var(--rule); color: var(--muted); font-size: 8pt; }
    .executive h1 { font-size: 28pt; margin-bottom: 5mm; }
    .executive .lede { font-size: 12pt; line-height: 1.35; margin-bottom: 3mm; }
    .executive .meta-table { margin: 3mm 0 4mm; font-size: 9pt; }
    .executive .meta-table td { padding: 2.2mm 2mm; }
    .executive .summary-grid { gap: 5mm; margin: 4mm 0 0; }
    .executive .summary-grid .card { padding: 4mm; font-size: 8.4pt; line-height: 1.35; }
    .executive .summary-grid ol,
    .executive .summary-grid ul { padding-left: 4.5mm; }
    .executive .summary-grid li { margin-bottom: 1mm; }
    @page { size: A4; margin: 0; }
    @media print {
      html, body { background: white; }
      body { margin: 0; }
      a { color: var(--orange); }
    }
  </style>
</head>
<body>
  <section class="cover">
    <div class="brand"><span class="brand-mark">ml</span><span>mentorship</span></div>
    <div class="cover-main">
      <p class="eyebrow">Personalized interview prep playbook</p>
      <h1>${escapeHtml(playbook.generatedFor)}</h1>
      <p class="cover-subtitle">${escapeHtml(ROLE_LABELS[intake.role])} · ${escapeHtml(LEVEL_LABELS[intake.targetLevel])} · ${escapeHtml(intake.weeks)}-week preparation sprint</p>
    </div>
    <div class="cover-meta">
      <div><strong>${escapeHtml(playbook.profile.horizonLabel)}</strong><span>Preparation horizon</span></div>
      <div><strong>${escapeHtml(playbook.totals.uniqueResources)} resources</strong><span>Selected from the senior ML library</span></div>
      <div><strong>${escapeHtml(playbook.planId)}</strong><span>Plan ID · Engine ${escapeHtml(playbook.engineVersion)}</span></div>
    </div>
  </section>

  <section class="page executive page-break-before">
    <p class="eyebrow">Executive brief</p>
    <h1>${escapeHtml(playbook.executiveSummary.headline)}</h1>
    <p class="lede">${escapeHtml(playbook.executiveSummary.strategy)}</p>
    <table class="meta-table">
      <tr><td>Target</td><td>${escapeHtml(playbook.profile.roleLabel)} · ${escapeHtml(playbook.profile.levelLabel)}</td></tr>
      <tr><td>Schedule</td><td>${escapeHtml(playbook.profile.horizonLabel)} · ${escapeHtml(playbook.totals.scheduledHours)} structured hours</td></tr>
      <tr><td>Rounds</td><td>${escapeHtml(selectedRounds)}</td></tr>
      <tr><td>Domain</td><td>${escapeHtml(selectedDomains)}</td></tr>
      ${playbook.profile.interviewDate ? `<tr><td>Interview date</td><td>${formatDate(playbook.profile.interviewDate)}</td></tr>` : ''}
      ${intake.experienceSummary ? `<tr><td>Context</td><td>${escapeHtml(intake.experienceSummary)}</td></tr>` : ''}
    </table>
    <div class="summary-grid">
      <div class="card">
        <h3>Top priorities</h3>
        <ol>${playbook.executiveSummary.topPriorities.map((item) => `<li>${escapeHtml(item)}</li>`).join('')}</ol>
      </div>
      <div class="card card--tint">
        <h3>Risks to manage</h3>
        <ul>${playbook.executiveSummary.risks.map((item) => `<li>${escapeHtml(item)}</li>`).join('')}</ul>
      </div>
    </div>
  </section>

  <section class="page page-break-before">
    <p class="eyebrow">Readiness map</p>
    <h2>Where to spend the next ${escapeHtml(intake.weeks)} weeks</h2>
    <p class="lede">The ranking combines your self-rating, stated interview rounds, and the expected emphasis for ${escapeHtml(ROLE_LABELS[intake.role])} at ${escapeHtml(LEVEL_LABELS[intake.targetLevel])}.</p>
    <table class="readiness">
      <thead><tr><th>Area</th><th>Priority</th><th>Baseline</th><th>Why it is placed here</th></tr></thead>
      <tbody>${playbook.readiness.map(readinessRow).join('')}</tbody>
    </table>
    <div class="card card--tint compact-rules" style="margin-top:8mm">
      <h3>Operating rules</h3>
      ${checkboxList(playbook.executiveSummary.operatingRules)}
    </div>
  </section>

  ${playbook.weeks.map((_, index) => weekSection(playbook, index)).join('')}

  <section class="page page-break-before">
    <p class="eyebrow">Practice protocol</p>
    <h2>Turn reading into interview performance</h2>
    <div class="protocol-grid">
      <div class="card"><h3>Before</h3>${checkboxList(playbook.practiceProtocol.before)}</div>
      <div class="card"><h3>During</h3>${checkboxList(playbook.practiceProtocol.during)}</div>
      <div class="card"><h3>After</h3>${checkboxList(playbook.practiceProtocol.after)}</div>
    </div>
    <h3>Five-dimension scoring rubric</h3>
    <div class="rubric">${playbook.practiceProtocol.scoringRubric.map((item) => `<div>${escapeHtml(item.dimension)}</div><div>${escapeHtml(item.strongSignal)}</div>`).join('')}</div>
    <div class="card card--tint" style="margin-top:8mm">
      <h3>Self-score after every attempt</h3>
      <p><strong>1–2:</strong> weak; add a prerequisite and retry in 2 days. <strong>3:</strong> review; retry in 7 days. <strong>4–5:</strong> confident; retain only a final-week check.</p>
    </div>
  </section>

  <section class="page page-break-before">
    <p class="eyebrow">Story bank</p>
    <h2>Prepare evidence, not scripts</h2>
    <p class="lede">Each story needs a concise version, a deep version, and evidence strong enough to survive follow-up.</p>
    ${playbook.storyBank.map((story) => `<article class="story"><h3>${escapeHtml(story.prompt)}</h3><ul>${story.evidenceToPrepare.map((item) => `<li>${escapeHtml(item)}</li>`).join('')}</ul></article>`).join('')}
  </section>

  <section class="page page-break-before">
    <p class="eyebrow">Final week</p>
    <h2>Protect performance</h2>
    ${checkboxList(playbook.finalWeekChecklist)}
    <div class="card card--tint" style="margin-top:9mm">
      <h3>Your one-page miss list</h3>
      <p>Keep this page blank until the final week. Add only errors that appeared in closed-book attempts.</p>
      <p style="height:85mm;border-bottom:.3mm solid var(--rule)"></p>
    </div>
  </section>

  <section class="page page-break-before">
    <p class="eyebrow">Selected library</p>
    <h2>${escapeHtml(playbook.totals.uniqueResources)} resources chosen for this plan</h2>
    <ul class="resource-list">${playbook.resourceAppendix.map((resource) => `<li><strong><a href="${escapeHtml(resource.absoluteUrl)}">${escapeHtml(resource.title)}</a></strong><span>${escapeHtml(resource.category)} · ${escapeHtml(resource.subcategory)} · ~${resource.estimatedMinutes} min</span></li>`).join('')}</ul>
    <p class="footer-note">Generated ${formatDate(playbook.generatedOn)} · Plan ${escapeHtml(playbook.planId)} · Engine ${escapeHtml(playbook.engineVersion)}<br>${escapeHtml(playbook.disclaimer)}</p>
  </section>
</body>
</html>`;
}
