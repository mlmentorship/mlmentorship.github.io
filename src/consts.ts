// Site-wide constants.
export const HSAGHIR_URL = 'https://hsaghir.com';
export const LOOPLET_URL = 'https://github.com/hsaghir/looplet';

export const SITE = {
  title: 'mlmentorship',
  description: 'A free visual field guide for senior ML, AI systems, and frontier-lab interviews, with ordered lessons, coding traces, and a private workbook.',
  author: 'Hamidreza Saghir',
  authorBio: 'Notes on senior ML interviews, AI systems, and ML from primitives.',
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
    // Dedicated Beehiiv form for the mlmentorship audience.
    provider: 'beehiiv' as 'beehiiv' | 'none',
    formId: 'ec384504-44fb-4f84-84ce-df038d14e1c1',
    blurb: 'ML from Primitives: occasional notes on senior ML interviews, AI systems, kernels, scaling, and level calibration.',
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
  // Keep the header task-oriented. The wordmark remains the direct home link.
  nav: [
    { label: 'Library', href: '/#curriculum' },
    { label: 'Map', href: '/map/' },
    { label: 'Questions', href: '/questions/' },
    { label: 'Coding', href: '/library/coding-interview/' },
    { label: 'Workbook', href: '/prep/' },
    { label: 'About', href: '/about/' },
  ],
  postsPerPage: 30,
} as const;
