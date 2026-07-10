// Site-wide constants.
export const HSAGHIR_URL = 'https://hsaghir.com';
export const LOOPLET_URL = 'https://github.com/hsaghir/looplet';

export const SITE = {
  title: 'mlmentorship',
  description: 'Senior ML & AI interview prep. Essays, interview questions with leveled answers (L4/L5/L6), reference notes, system design case studies. Free.',
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
    blurb: 'Occasional notes on senior ML interviews. No spam, unsubscribe any time.',
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
    // Show mock-interview booking CTAs (header nav, footer button, in-content CTA, home band).
    // Default off until the site has traction; the /interview/ page itself stays live and
    // is reachable from /about/ for anyone who lands directly.
    bookingCTA: false,
  },
  // Navigation (top of every page). 'Start here' lives in the docs sidebar
  // since it's a navigation aid, not a content section. About lives in
  // both nav and footer. Mock-interview link is appended dynamically when
  // features.bookingCTA is enabled.
  // Order: most-clicked first (Questions > Guides > Concepts).
  // URLs unchanged to preserve search-engine indexing and any inbound links.
  nav: [
    { label: 'Questions', href: '/questions/' },
    { label: 'Guides', href: '/guides/' },
    { label: 'Concepts', href: '/concepts/' },
    { label: 'About', href: '/about/' },
  ],
  postsPerPage: 30,
} as const;
