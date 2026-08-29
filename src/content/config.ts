import { defineCollection, z } from 'astro:content';

const posts = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    description: z.string().optional(),
    date: z.coerce.date(),
    updated: z.coerce.date().optional(),
    reviewed: z.coerce.date().optional(),
    draft: z.boolean().default(false),
    tags: z.array(z.string()).default([]),
    // One of: 'guides' | 'questions' | 'concepts'
    category: z.enum(['questions', 'guides', 'concepts']),
    cover: z.string().optional(),
    coverAlt: z.string().optional(),
    featured: z.boolean().default(false),
    archived: z.boolean().default(false),
    aliases: z.array(z.string()).default([]),
    roles: z.array(z.string()).default([]),
    rounds: z.array(z.string()).default([]),
    difficulty: z.enum(['Foundation', 'Intermediate', 'Advanced', 'Mixed']).optional(),
    priority: z.enum(['Core', 'Role-specific', 'Specialist']).optional(),
    prerequisites: z.array(z.string()).default([]),
  }),
});

export const collections = { posts };
