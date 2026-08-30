// Site-wide constants.
export const HSAGHIR_URL = 'https://hsaghir.com';
export const LOOPLET_URL = 'https://github.com/hsaghir/looplet';

export const SITE = {
  title: 'mlmentorship',
  description: 'Build a private senior ML interview plan from your role, level, rounds, available time, and recent closed-book evidence. Free and browser-local.',
  author: 'Hamidreza Saghir',
  authorBio: 'Notes on senior ML interviews, system design, and applied ML practice.',
  email: 'ml.mentorship@gmail.com',
  url: 'https://mlmentorship.com',
  locale: 'en',
  avatar: '/images/avatar.jpg',
  social: {
    github: '',
    scholar: '',
    linkedin: 'https://www.linkedin.com/in/hamidrezasaghir',
    twitter: '',
    mastodon: '',
  },
  newsletter: {
    // Dedicated form for the mlmentorship audience, under the shared MailerLite
    // account. This keeps source attribution and audience segmentation explicit.
    provider: 'mailerlite' as 'mailerlite' | 'none',
    accountId: '2284644',
    formId: 'rgOeEV',
    blurb: 'Occasional senior-level answer patterns, interview-process changes, and new deep cases. No fixed schedule.',
  },
  analytics: {
    // Shared with hsaghir.com; BaseHead prefixes paths with the hostname so the
    // dashboard can distinguish the two sites and reveal the cross-site funnel.
    provider: 'goatcounter' as 'goatcounter' | 'none',
    code: 'hsaghir',
  },
  giscus: {
    repo: '',
    repoId: '',
    category: '',
    categoryId: '',
    mapping: 'pathname',
    reactionsEnabled: '0',
    theme: 'preferred_color_scheme',
  },
  // Feature flags
  features: {
    // Build and expose the browser-only preparation subsystem. When false,
    // navigation and Practice Mode disappear and /prep/* routes redirect to
    // the core Questions library.
    prepTools: import.meta.env.PUBLIC_PREP_TOOLS === undefined || import.meta.env.PUBLIC_PREP_TOOLS === 'true',
  },
  // Keep the header short. The wordmark links to the table of contents.
  // URLs unchanged to preserve search-engine indexing and any inbound links.
  nav: [
    { label: 'Contents', href: '/' },
    { label: 'Questions', href: '/questions/' },
    { label: 'Workbook', href: '/prep/' },
    { label: 'About', href: '/about/' },
  ],
  postsPerPage: 30,
} as const;
