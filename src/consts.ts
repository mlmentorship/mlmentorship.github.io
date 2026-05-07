// Site-wide constants.
export const HSAGHIR_URL = 'https://hsaghir.com';
export const LOOPLET_URL = 'https://github.com/hsaghir/looplet';

export const SITE = {
  title: 'mlmentorship',
  description: 'Senior ML interview prep. Essays, interview questions with leveled answers (L4/L5/L6), reference notes, system design case studies.',
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
    provider: 'none' as 'mailerlite' | 'none',
    accountId: '',
    formId: '',
    blurb: 'Occasional notes on senior ML interviews. No spam, unsubscribe any time.',
  },
  analytics: {
    provider: 'none' as 'goatcounter' | 'none',
    code: '',
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
    // Show mock-interview booking CTAs (header nav, footer button, in-content CTA, home band).
    // Default off until the site has traction; the /interview/ page itself stays live and
    // is reachable from /about/ for anyone who lands directly.
    bookingCTA: false,
  },
  // Navigation (top of every page). Kept to 5 items by convention; About
  // lives in the footer. Mock-interview link is appended dynamically when
  // features.bookingCTA is enabled.
  nav: [
    { label: 'Start here', href: '/start-here/' },
    { label: 'Interviews', href: '/interviews/' },
    { label: 'Essays', href: '/essays/' },
    { label: 'Reference', href: '/reference/' },
    { label: 'Hamidreza Saghir', href: HSAGHIR_URL, external: true },
  ],
  postsPerPage: 30,
} as const;
