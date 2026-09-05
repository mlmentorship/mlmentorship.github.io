export const PUBLIC_EVENT_NAMES = [
  'homepage_plan_clicked',
  'homepage_workbook_clicked',
  'homepage_practice_clicked',
  'homepage_curriculum_clicked',
  'homepage_start_clicked',
  'homepage_signup_clicked',
  'email_signup_viewed_home',
  'email_signup_attempted_home',
  'email_signup_confirmation_sent_home',
  'email_signup_viewed_entry',
  'email_signup_attempted_entry',
  'email_signup_confirmation_sent_entry',
  'email_signup_dismissed_entry',
  'email_signup_viewed_review',
  'email_signup_attempted_review',
  'email_signup_confirmation_sent_review',
  'email_signup_viewed_workbook',
  'email_signup_attempted_workbook',
  'email_signup_confirmation_sent_workbook',
  'readiness_started',
  'readiness_completed',
  'practice_started',
  'rubric_revealed',
  'self_score_selected',
  'generic_plan_opened',
  'simulation_opened',
  'workbook_opened',
] as const;

export type PublicEventName = (typeof PUBLIC_EVENT_NAMES)[number];

declare global {
  interface Window {
    goatcounter?: {
      count?: (options: { path: string; title: string; event: true }) => void;
    };
  }
}

export function trackPublicEvent(name: PublicEventName): void {
  if (!PUBLIC_EVENT_NAMES.includes(name)) return;
  const payload = {
    path: `${window.location.host}/events/${name}`,
    title: `mlmentorship: ${name.replaceAll('_', ' ')}`,
    event: true as const,
  };
  const send = (attempt = 0) => {
    if (typeof window.goatcounter?.count === 'function') {
      window.goatcounter.count(payload);
    } else if (attempt < 20) {
      window.setTimeout(() => send(attempt + 1), 100);
    }
  };
  send();
}
