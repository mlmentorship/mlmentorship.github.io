import { z } from 'zod';
import { DOMAIN_TRACKS, LEVELS, PLAN_AREAS, ROLES, ROUNDS, type PlaybookIntake } from './types';

const isoDate = /^\d{4}-\d{2}-\d{2}$/;

function validIsoDate(value: string): boolean {
  if (!isoDate.test(value)) return false;
  const [year, month, day] = value.split('-').map(Number);
  const date = new Date(Date.UTC(year, month - 1, day));
  return date.getUTCFullYear() === year && date.getUTCMonth() === month - 1 && date.getUTCDate() === day;
}

const ratingsShape = Object.fromEntries(
  PLAN_AREAS.map((area) => [area, z.number().int().min(1).max(5)]),
) as Record<(typeof PLAN_AREAS)[number], z.ZodNumber>;

export const intakeSchema = z.object({
  version: z.literal(1),
  candidateName: z.string().trim().min(1).max(80),
  role: z.enum(ROLES),
  targetLevel: z.enum(LEVELS),
  startDate: z.string().regex(isoDate, 'startDate must use YYYY-MM-DD'),
  weeks: z.number().int().min(2).max(8),
  hoursPerWeek: z.number().min(3).max(20),
  rounds: z.array(z.enum(ROUNDS)).min(2).max(8).transform((items) => [...new Set(items)]),
  domainTracks: z.array(z.enum(DOMAIN_TRACKS)).max(6).default([]).transform((items) => [...new Set(items)]),
  selfRatings: z.object(ratingsShape).strict(),
  interviewDate: z.string().regex(isoDate, 'interviewDate must use YYYY-MM-DD').optional(),
  experienceSummary: z.string().trim().max(1200).optional(),
  constraints: z.array(z.string().trim().min(1).max(240)).max(10).default([]),
  priorities: z.array(z.string().trim().min(1).max(240)).max(10).default([]),
}).strict().superRefine((value, context) => {
  const startValid = validIsoDate(value.startDate);
  const start = startValid ? Date.parse(`${value.startDate}T00:00:00Z`) : Number.NaN;
  if (!startValid) {
    context.addIssue({ code: 'custom', path: ['startDate'], message: 'startDate is not a valid calendar date' });
  }

  if (value.interviewDate) {
    const interviewValid = validIsoDate(value.interviewDate);
    const interview = interviewValid ? Date.parse(`${value.interviewDate}T00:00:00Z`) : Number.NaN;
    if (!interviewValid) {
      context.addIssue({ code: 'custom', path: ['interviewDate'], message: 'interviewDate is not a valid calendar date' });
    } else if (!Number.isNaN(start) && interview <= start) {
      context.addIssue({ code: 'custom', path: ['interviewDate'], message: 'interviewDate must be after startDate' });
    } else if (!Number.isNaN(start)) {
      const finalWeekStart = start + (value.weeks - 1) * 7 * 86_400_000;
      const planEnd = start + value.weeks * 7 * 86_400_000;
      if (interview < finalWeekStart || interview > planEnd) {
        context.addIssue({
          code: 'custom',
          path: ['interviewDate'],
          message: 'interviewDate must fall within the final week of the selected plan horizon',
        });
      }
    }
  }
});

export function parseIntake(input: unknown): PlaybookIntake {
  const result = intakeSchema.safeParse(input);
  if (result.success) return result.data as PlaybookIntake;

  const details = result.error.issues
    .map((issue) => `${issue.path.join('.') || 'intake'}: ${issue.message}`)
    .join('\n');
  throw new Error(`Invalid playbook intake:\n${details}`);
}
