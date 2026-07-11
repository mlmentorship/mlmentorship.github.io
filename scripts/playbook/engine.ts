import { createHash } from 'node:crypto';
import type {
  CatalogResource,
  PersonalizedPlaybook,
  PlanArea,
  PlanTask,
  PlanWeek,
  PlaybookIntake,
  TaskType,
} from './types';
import {
  AREA_LABELS,
  ENGINE_VERSION,
  LEVEL_LABELS,
  ROLE_LABELS,
  ROLE_WEIGHTS,
  ROUND_AREA,
  ROUND_LABELS,
} from './rules';

const LEVEL_RANK = { l4: 4, l5: 5, l6: 6 } as const;

function stableJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(stableJson).join(',')}]`;
  if (value && typeof value === 'object') {
    const object = value as Record<string, unknown>;
    return `{${Object.keys(object).sort().map((key) => `${JSON.stringify(key)}:${stableJson(object[key])}`).join(',')}}`;
  }
  return JSON.stringify(value);
}

function planId(intake: PlaybookIntake): string {
  return createHash('sha256')
    .update(`${ENGINE_VERSION}:${stableJson(intake)}`)
    .digest('hex')
    .slice(0, 12);
}

function titleCase(value: string): string {
  return value.replace(/(^|[-\s])\w/g, (match) => match.toUpperCase()).replace(/-/g, ' ');
}

function primaryArea(resource: CatalogResource): PlanArea {
  return resource.areas[0] ?? 'fundamentals';
}

function targetAreas(intake: PlaybookIntake): Set<PlanArea> {
  return new Set(intake.rounds.map((round) => ROUND_AREA[round]));
}

function areaUrgency(intake: PlaybookIntake, area: PlanArea): number {
  const rating = intake.selfRatings[area];
  const roleWeight = ROLE_WEIGHTS[intake.role][area];
  const roundBoost = targetAreas(intake).has(area) ? 0.45 : 0;
  return (6 - rating) * (roleWeight + roundBoost);
}

function scoreResource(resource: CatalogResource, intake: PlaybookIntake): number {
  const areas = targetAreas(intake);
  let score = resource.priority;
  score += resource.roles.includes(intake.role) ? 34 : -16;
  score += resource.levels.includes(intake.targetLevel) ? 22 : -30;
  score += resource.areas.some((area) => areas.has(area)) ? 30 : 0;
  score += Math.max(...resource.areas.map((area) => areaUrgency(intake, area))) * 6;
  score += resource.domainTracks.some((domain) => intake.domainTracks.includes(domain)) ? 24 : 0;
  score += resource.category === 'questions' ? 8 : 0;

  const targetRank = LEVEL_RANK[intake.targetLevel];
  const highestResourceRank = Math.max(...resource.levels.map((level) => LEVEL_RANK[level]));
  if (highestResourceRank < targetRank) score += 5;
  if (intake.weeks <= 3 && resource.category === 'concepts') score -= 8;
  if (intake.selfRatings[primaryArea(resource)] === 5) score -= 10;
  return score;
}

function readiness(intake: PlaybookIntake): PersonalizedPlaybook['readiness'] {
  return (Object.keys(intake.selfRatings) as PlanArea[])
    .map((area) => {
      const rating = intake.selfRatings[area];
      const weight = ROLE_WEIGHTS[intake.role][area] + (targetAreas(intake).has(area) ? 0.45 : 0);
      const urgency = Number(((6 - rating) * weight).toFixed(2));
      const priority: 'critical' | 'high' | 'maintain' = rating <= 2 || urgency >= 5
        ? 'critical'
        : rating <= 3 || urgency >= 3
          ? 'high'
          : 'maintain';
      const rationale = targetAreas(intake).has(area)
        ? `${AREA_LABELS[area]} is in the stated interview loop and is self-rated ${rating}/5.`
        : `${AREA_LABELS[area]} is ${rating}/5 and carries ${weight.toFixed(1)}× relevance for the ${ROLE_LABELS[intake.role]} track.`;
      return { area, label: AREA_LABELS[area], rating, weight: Number(weight.toFixed(2)), urgency, priority, rationale };
    })
    .sort((a, b) => b.urgency - a.urgency || a.label.localeCompare(b.label));
}

function mandatorySlugs(intake: PlaybookIntake): string[] {
  const slugs = ['as-vs-mle-vs-re', 'most-ambitious-project'];
  if (intake.targetLevel !== 'l4') slugs.push('l5-vs-l6-faang-ml');
  if (intake.rounds.includes('behavioral')) slugs.push('scope-ambiguous-problem');
  if (intake.rounds.includes('llm-systems')) slugs.push('llm-evals-the-hardest-part', 'how-would-you-evaluate-an-llm-application');
  if (intake.rounds.includes('recsys-search')) slugs.push('two-tower-retrieval', 'design-youtube-recommender');
  if (intake.rounds.includes('coding')) slugs.push(intake.role === 'applied-scientist' ? 'implement-knn' : 'debug-training-loop');
  if (intake.rounds.includes('ml-system-design')) {
    slugs.push(intake.role === 'ml-engineer' ? 'design-ml-monitoring' : 'design-fraud-detection');
  }
  if (intake.rounds.includes('production-debugging')) slugs.push('debug-model-not-learning');
  if (intake.rounds.includes('math-research')) slugs.push('explain-backprop', 'derive-logistic-regression');
  if (intake.role === 'research-engineer' && intake.domainTracks.includes('llm')) slugs.push('implement-attention-from-scratch');
  return [...new Set(slugs)];
}

function taskInstructions(type: TaskType): string[] {
  switch (type) {
    case 'read':
      return ['Read for structure, not memorization.', 'Write three takeaways and one unresolved question.', 'Explain when this idea matters in a real system.'];
    case 'practice':
      return ['Answer aloud before reading the rubric.', 'Time-box the first answer to 8 minutes.', 'Compare your answer with the level-calibrated signals and record one miss.'];
    case 'design':
      return ['Spend 5 minutes scoping users, objective, constraints, and failure cost.', 'Draw the system and defend the top three trade-offs.', 'Finish with evaluation, rollout, monitoring, and one concrete failure story.'];
    case 'story':
      return ['Write a two-minute version and a six-minute version.', 'Name your decision, personal contribution, measurable result, and what changed afterward.', 'Prepare two follow-ups that expose conflict, uncertainty, or a failed assumption.'];
    case 'derive':
      return ['Derive from a blank page without notes.', 'State dimensions and assumptions at every step.', 'Connect the result to one modeling or production decision.'];
    case 'review':
      return ['Retry from a blank page.', 'Focus only on the misses from the first attempt.', 'Promote to confident only if the answer is structured and specific.'];
    case 'simulation':
      return ['Use no notes and record the session.', 'Run one technical question, one design question, and one behavioral story.', 'Score immediately, then write the three highest-value corrections.'];
  }
}

function whyResource(resource: CatalogResource, intake: PlaybookIntake): string {
  const area = primaryArea(resource);
  const rating = intake.selfRatings[area];
  const roundMatch = resource.rounds.find((round) => intake.rounds.includes(round));
  if (roundMatch) return `${ROUND_LABELS[roundMatch]} is in the target loop; this resource develops ${AREA_LABELS[area].toLowerCase()} from a ${rating}/5 baseline.`;
  return `This is a high-signal ${ROLE_LABELS[intake.role]} resource for ${AREA_LABELS[area].toLowerCase()} at the ${LEVEL_LABELS[intake.targetLevel]} bar.`;
}

function weekTheme(week: number, totalWeeks: number): { theme: string; objective: string } {
  if (week === 1) return { theme: 'Calibrate the bar', objective: 'Understand the role and level, establish answer structure, and close prerequisite gaps.' };
  if (week === totalWeeks) return { theme: 'Simulate and consolidate', objective: 'Perform under interview conditions, repair final weak areas, and protect recovery time.' };
  if (week === totalWeeks - 1) return { theme: 'Integrate the rounds', objective: 'Combine technical depth, system judgment, and concise communication in realistic sessions.' };
  return { theme: `Build round-specific depth`, objective: 'Practice the highest-urgency rounds and convert passive knowledge into repeatable answers.' };
}

function exitCriteria(week: number, totalWeeks: number): string[] {
  if (week === 1) return ['Can state the target role and level bar in one minute.', 'Has a first-pass project story with measurable impact.', 'Has a written list of the three biggest gaps.'];
  if (week === totalWeeks) return ['Completes a timed simulation without notes.', 'Has no unresolved critical gap.', 'Can name the final-week sleep, logistics, and review plan.'];
  return ['Completes every scheduled attempt before revealing answers.', 'Records at least one correction per practice task.', 'Retries any answer scored below 3/5.'];
}

function assignDays(tasks: PlanTask[]): PlanTask[] {
  const dayLoads = [0, 0, 0, 0, 0];
  return tasks.map((task, index) => {
    const minLoad = Math.min(...dayLoads);
    const dayIndex = dayLoads.indexOf(minLoad);
    dayLoads[dayIndex] += task.minutes;
    return { ...task, day: dayIndex + 1, sequence: index + 1 };
  });
}

function selectResources(intake: PlaybookIntake, catalog: CatalogResource[], targetMinutes: number): CatalogResource[] {
  const bySlug = new Map(catalog.map((resource) => [resource.slug, resource]));
  const selected: CatalogResource[] = [];
  const selectedSlugs = new Set<string>();
  const visiting = new Set<string>();
  const requestedAreas = targetAreas(intake);

  const add = (slug: string) => {
    if (selectedSlugs.has(slug)) return;
    if (visiting.has(slug)) {
      throw new Error(`Circular playbook prerequisite detected at: ${slug}`);
    }
    const resource = bySlug.get(slug);
    if (!resource) throw new Error(`Planner rule references missing resource: ${slug}`);
    visiting.add(slug);
    for (const prerequisite of resource.prerequisites) add(prerequisite);
    visiting.delete(slug);
    selected.push(resource);
    selectedSlugs.add(slug);
  };

  for (const slug of mandatorySlugs(intake)) add(slug);

  const ranked = catalog
    .filter((resource) => !selectedSlugs.has(resource.slug))
    .filter((resource) => resource.areas.some((area) => requestedAreas.has(area)))
    .filter((resource) =>
      resource.domainTracks.length === 0 ||
      resource.domainTracks.some((domain) => intake.domainTracks.includes(domain)),
    )
    .map((resource) => ({ resource, score: scoreResource(resource, intake) }))
    .sort((a, b) => b.score - a.score || a.resource.route.localeCompare(b.resource.route));

  let minutes = selected.reduce((sum, resource) => sum + resource.estimatedMinutes, 0);
  for (const { resource } of ranked) {
    if (minutes >= targetMinutes) break;
    const prerequisiteMinutes = resource.prerequisites
      .filter((slug) => !selectedSlugs.has(slug))
      .map((slug) => bySlug.get(slug)?.estimatedMinutes ?? 0)
      .reduce((sum, value) => sum + value, 0);
    if (minutes + prerequisiteMinutes + resource.estimatedMinutes > targetMinutes * 1.08) continue;
    add(resource.slug);
    minutes = selected.reduce((sum, item) => sum + item.estimatedMinutes, 0);
  }

  return selected;
}

function idealWeek(resource: CatalogResource, totalWeeks: number): number {
  if (resource.category === 'guides') return 1;
  if (resource.prerequisites.length > 0 || resource.category === 'concepts') return Math.min(2, totalWeeks);
  if (resource.taskType === 'story') return Math.max(1, Math.floor(totalWeeks / 2) - 1);
  if (resource.taskType === 'design') return Math.max(2, Math.ceil(totalWeeks / 2));
  return Math.max(1, Math.floor(totalWeeks / 2));
}

function schedule(intake: PlaybookIntake, selected: CatalogResource[]): PlanWeek[] {
  const budgetMinutes = Math.round(intake.hoursPerWeek * 60);
  const structuredBudget = Math.min(Math.round(budgetMinutes * 0.82), 600);
  const simulationMinutes = budgetMinutes >= 240
    ? Math.min(75, Math.max(45, Math.round(budgetMinutes * 0.18)))
    : 0;
  const weeks: PlanWeek[] = Array.from({ length: intake.weeks }, (_, index) => {
    const week = index + 1;
    const copy = weekTheme(week, intake.weeks);
    return {
      week,
      ...copy,
      plannedMinutes: 0,
      budgetMinutes,
      tasks: [],
      exitCriteria: exitCriteria(week, intake.weeks),
    };
  });

  const placeTask = (
    task: Omit<PlanTask, 'week' | 'day' | 'sequence'>,
    preferredWeek: number,
    latestWeek = intake.weeks,
    allowEarlier = true,
  ) => {
    const candidates = Array.from({ length: latestWeek - preferredWeek + 1 }, (_, index) => preferredWeek + index);
    const fallback = allowEarlier
      ? Array.from({ length: preferredWeek - 1 }, (_, index) => index + 1).reverse()
      : [];
    const weekNumber = [...candidates, ...fallback].find((candidate) => {
      const week = weeks[candidate - 1];
      const capacity = candidate === intake.weeks
        ? structuredBudget - simulationMinutes
        : structuredBudget;
      return week.plannedMinutes + task.minutes <= capacity;
    });
    if (!weekNumber) return false;
    const week = weeks[weekNumber - 1];
    week.tasks.push({ ...task, week: weekNumber, day: 1, sequence: 1 });
    week.plannedMinutes += task.minutes;
    return true;
  };

  for (const resource of selected) {
    const area = primaryArea(resource);
    placeTask({
      id: `resource:${resource.slug}`,
      type: resource.taskType,
      title: resource.title,
      area,
      minutes: resource.estimatedMinutes,
      route: resource.route,
      absoluteUrl: resource.absoluteUrl,
      resourceSlug: resource.slug,
      why: whyResource(resource, intake),
      instructions: taskInstructions(resource.taskType),
    }, idealWeek(resource, intake.weeks), resource.taskType === 'read' ? Math.max(1, intake.weeks - 1) : intake.weeks);
  }

  const firstAttempts = weeks.flatMap((week) => week.tasks)
    .filter((task) => ['practice', 'design', 'story', 'derive'].includes(task.type));
  for (const task of firstAttempts) {
    if (task.week >= intake.weeks) continue;
    placeTask({
      id: `review:${task.id}`,
      type: 'review',
      title: `Retry: ${task.title}`,
      area: task.area,
      minutes: 15,
      route: task.route,
      absoluteUrl: task.absoluteUrl,
      resourceSlug: task.resourceSlug,
      why: 'A delayed retry turns recognition into retrieval and exposes whether the correction held.',
      instructions: taskInstructions('review'),
      reviewOf: task.id,
    }, task.week + 1, Math.min(intake.weeks, task.week + 2), false);
  }

  if (simulationMinutes > 0) {
    const lastWeek = weeks[weeks.length - 1];
    lastWeek.tasks.push({
      id: 'simulation:final',
      week: intake.weeks,
      day: 1,
      sequence: 1,
      type: 'simulation',
      title: 'Full-loop simulation and correction pass',
      area: 'system-design',
      minutes: simulationMinutes,
      why: 'The final test is performance across rounds under time pressure, not isolated recognition.',
      instructions: taskInstructions('simulation'),
    });
    lastWeek.plannedMinutes += simulationMinutes;
  }

  for (const week of weeks) {
    week.tasks = assignDays(week.tasks);
    week.plannedMinutes = week.tasks.reduce((sum, task) => sum + task.minutes, 0);
  }
  return weeks;
}

function storyBank(intake: PlaybookIntake): PersonalizedPlaybook['storyBank'] {
  const roleSpecific = intake.role === 'applied-scientist'
    ? 'A modeling or experiment decision that changed the product direction'
    : intake.role === 'ml-engineer'
      ? 'A reliability or scale problem where the model was not the hardest part'
      : 'A research-to-production decision that required both algorithmic and implementation depth';
  return [
    { prompt: 'Your most ambitious project', evidenceToPrepare: ['Scope and why it mattered', 'Your decisions and personal contribution', 'Metric movement and durable impact', 'What you would change now'] },
    { prompt: 'A disagreement with someone senior', evidenceToPrepare: ['The actual technical or product disagreement', 'How you made the disagreement legible', 'What evidence changed the decision', 'How the relationship evolved'] },
    { prompt: 'A failure or invalidated assumption', evidenceToPrepare: ['What you believed', 'The signal that contradicted it', 'How quickly you changed course', 'What process changed afterward'] },
    { prompt: 'A decision under ambiguity', evidenceToPrepare: ['What was unknown', 'How you reduced the decision space', 'The reversible and irreversible choices', 'The checkpoint that prevented drift'] },
    { prompt: roleSpecific, evidenceToPrepare: ['Initial constraint', 'Trade-offs considered', 'Decision and execution', 'Measured outcome and second-order effects'] },
  ];
}

export function buildPersonalizedPlaybook(
  intake: PlaybookIntake,
  catalog: CatalogResource[],
  generatedOn = new Date().toISOString().slice(0, 10),
): PersonalizedPlaybook {
  if (catalog.length < 150) throw new Error(`Catalog unexpectedly small: ${catalog.length}`);
  const ready = readiness(intake);
  const totalBudgetMinutes = intake.weeks * intake.hoursPerWeek * 60;
  const targetResourceMinutes = Math.min(Math.round(totalBudgetMinutes * 0.55), 3600);
  const selected = selectResources(intake, catalog, targetResourceMinutes);
  const weeks = schedule(intake, selected);
  const tasks = weeks.flatMap((week) => week.tasks);
  const selectedSlugs = new Set(tasks.map((task) => task.resourceSlug).filter(Boolean));
  const appendix = selected.filter((resource) => selectedSlugs.has(resource.slug));
  const taskCounts = Object.fromEntries(
    ['read', 'practice', 'design', 'story', 'derive', 'review', 'simulation'].map((type) => [type, tasks.filter((task) => task.type === type).length]),
  ) as Record<TaskType, number>;
  const scheduledMinutes = tasks.reduce((sum, task) => sum + task.minutes, 0);
  const top = ready.slice(0, 3);
  const riskItems = ready.filter((item) => item.priority === 'critical').map((item) => `${item.label} starts at ${item.rating}/5 while carrying high relevance.`);
  const risks = [...riskItems, ...(intake.constraints ?? [])].slice(0, 5);

  return {
    schemaVersion: 1,
    engineVersion: ENGINE_VERSION,
    planId: planId(intake),
    generatedFor: intake.candidateName,
    generatedOn,
    intake,
    profile: {
      roleLabel: ROLE_LABELS[intake.role],
      levelLabel: LEVEL_LABELS[intake.targetLevel],
      horizonLabel: `${intake.weeks} weeks · ${intake.hoursPerWeek} hours/week`,
      totalBudgetHours: intake.weeks * intake.hoursPerWeek,
      interviewDate: intake.interviewDate,
    },
    executiveSummary: {
      headline: `${intake.candidateName}'s ${intake.weeks}-week ${ROLE_LABELS[intake.role]} preparation plan`,
      strategy: `Prioritize ${top.map((item) => item.label.toLowerCase()).join(', ')}, move from calibrated examples to closed-book attempts, and reserve the final week for integrated simulation rather than new material.`,
      topPriorities: top.map((item) => `${item.label}: ${item.rationale}`),
      risks: risks.length > 0 ? risks : ['No critical self-rated gap was reported; the main risk is passive reading without timed retrieval practice.'],
      operatingRules: [
        'Answer before revealing the rubric.',
        'Record one concrete miss after every attempt.',
        'Retry weak answers after a delay; do not reread immediately.',
        'Use real project evidence, not hypothetical accomplishments.',
        'Protect at least 18% of weekly availability as flex, recovery, and overflow time.',
      ],
    },
    readiness: ready,
    weeks,
    practiceProtocol: {
      before: ['Choose a quiet setting and close reference material.', 'State the question and clarify assumptions aloud.', 'Set the prescribed timer before starting.'],
      during: ['Lead with structure and scope before details.', 'Make trade-offs explicit and tie them to user or system constraints.', 'Check for interviewer alignment rather than performing a monologue.'],
      after: ['Score the attempt before reading the reference answer.', 'Write the highest-value missing idea in one sentence.', 'Schedule a blank-page retry instead of an immediate reread.'],
      scoringRubric: [
        { dimension: 'Problem framing', strongSignal: 'Clarifies user, objective, constraints, and failure cost before proposing techniques.' },
        { dimension: 'Technical depth', strongSignal: 'Explains mechanism, not just terminology, and handles a follow-up one layer deeper.' },
        { dimension: 'Trade-offs', strongSignal: 'Names alternatives, decision criteria, and what would reverse the choice.' },
        { dimension: 'Evidence', strongSignal: 'Uses specific experience, metrics, and failure modes rather than generic claims.' },
        { dimension: 'Communication', strongSignal: 'Structures the answer, checks alignment, and stays concise under follow-up.' },
      ],
    },
    storyBank: storyBank(intake),
    finalWeekChecklist: [
      'Complete one full-loop simulation at the same time of day as the interview.',
      'Freeze new topics 48 hours before the interview.',
      'Prepare environment, links, IDE, paper, and backup connectivity.',
      'Review only the one-page miss list and story-bank headlines.',
      'Protect sleep and a short warm-up; do not run a full mock the night before.',
      'Write three questions for the interview team that demonstrate role-level judgment.',
    ],
    resourceAppendix: appendix,
    totals: {
      scheduledMinutes,
      scheduledHours: Number((scheduledMinutes / 60).toFixed(1)),
      uniqueResources: appendix.length,
      taskCounts,
    },
    disclaimer: 'This playbook is educational preparation based on self-reported information. It does not guarantee interview performance, leveling, or employment outcomes.',
  };
}

export function describePlan(playbook: PersonalizedPlaybook): string {
  return `${playbook.profile.roleLabel} ${playbook.profile.levelLabel}; ${playbook.profile.horizonLabel}; ${playbook.totals.uniqueResources} resources; ${playbook.totals.scheduledHours} scheduled hours.`;
}

export function displayArea(area: PlanArea): string {
  return AREA_LABELS[area] ?? titleCase(area);
}
