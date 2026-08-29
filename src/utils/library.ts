import { getCollection } from 'astro:content';
import type { LibraryChapter, LibraryContentCategory } from '../data/library';

export interface LibraryEntry {
  slug: string;
  title: string;
  description?: string;
  category: LibraryContentCategory;
  href: string;
  date: Date;
}

export async function getLibraryEntries(): Promise<LibraryEntry[]> {
  const posts = await getCollection('posts', ({ data }) => !data.draft);
  return posts.map((post) => {
    const slug = post.slug.replace(/^\d{4}-\d{2}-\d{2}-/, '');
    return {
      slug,
      title: post.data.title,
      description: post.data.description,
      category: post.data.category,
      href: `/${post.data.category}/${slug}/`,
      date: post.data.date,
    };
  });
}

export function entriesForChapter(entries: LibraryEntry[], chapter: LibraryChapter): LibraryEntry[] {
  const bySlug = new Map(entries.map((entry) => [entry.slug, entry]));
  return chapter.slugs.map((slug) => bySlug.get(slug)).filter((entry): entry is LibraryEntry => entry !== undefined);
}