import fs from 'node:fs';
import path from 'node:path';
import matter from 'gray-matter';
import {
  INTERVIEW_SUBCATEGORY,
  REFERENCE_SUBCATEGORY,
} from '../../src/utils/subcategories';
import type {
  CatalogResource,
  DomainTrack,
  Level,
  PlanArea,
  ResourceCategory,
  TaskType,
} from './types';
import {
  AREA_ROLES,
  AREA_ROUNDS,
  SUBCATEGORY_AREA,
  applyResourceOverride,
} from './rules';

function stripDatePrefix(value: string): string {
  return value.replace(/^\d{4}-\d{2}-\d{2}-/, '');
}

function stripMarkdown(markdown: string): string {
  return markdown
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/`[^`]*`/g, ' ')
    .replace(/!\[[^\]]*\]\([^)]*\)/g, ' ')
    .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1')
    .replace(/<[^>]+>/g, ' ')
    .replace(/[#>*_~|$\\{}[\]]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function guideAreas(slug: string, title: string, tags: string[]): PlanArea[] {
  const text = `${slug} ${title} ${tags.join(' ')}`.toLowerCase();
  if (/role|interview|l5|l6/.test(text)) return ['behavioral'];
  if (/rag|llm|eval/.test(text)) return ['llm-systems', 'system-design'];
  if (/inference|training|pretraining/.test(text)) return ['production', 'llm-systems'];
  if (/search|recsys|ranking/.test(text)) return ['recsys-search', 'system-design'];
  return ['system-design'];
}

function inferAreas(category: ResourceCategory, slug: string, title: string, tags: string[]): PlanArea[] {
  if (category === 'questions') {
    const subcategory = INTERVIEW_SUBCATEGORY[slug] ?? 'ML Fundamentals';
    return [SUBCATEGORY_AREA[subcategory] ?? 'fundamentals'];
  }
  if (category === 'concepts') {
    const subcategory = REFERENCE_SUBCATEGORY[slug] ?? 'Classical ML';
    return [SUBCATEGORY_AREA[subcategory] ?? 'fundamentals'];
  }
  return guideAreas(slug, title, tags);
}

function inferSubcategory(category: ResourceCategory, slug: string): string {
  if (category === 'questions') return INTERVIEW_SUBCATEGORY[slug] ?? 'Other';
  if (category === 'concepts') return REFERENCE_SUBCATEGORY[slug] ?? 'Other';
  return 'Guides';
}

function inferTaskType(category: ResourceCategory, primaryArea: PlanArea): Exclude<TaskType, 'review' | 'simulation'> {
  if (category === 'guides' || category === 'concepts') return 'read';
  if (primaryArea === 'behavioral') return 'story';
  if (primaryArea === 'math-research') return 'derive';
  if (primaryArea === 'system-design' || primaryArea === 'llm-systems' || primaryArea === 'recsys-search') return 'design';
  return 'practice';
}

function inferLevels(category: ResourceCategory): Level[] {
  if (category === 'guides') return ['l5', 'l6'];
  return ['l4', 'l5', 'l6'];
}

function inferDomains(subcategory: string, title: string, tags: string[]): DomainTrack[] {
  const text = `${subcategory} ${title} ${tags.join(' ')}`.toLowerCase();
  const domains: DomainTrack[] = [];
  if (/llm|transformer|attention|rag|language model/.test(text)) domains.push('llm');
  if (/recsys|recommend|retrieval|search|ranking|two-tower/.test(text)) domains.push('recsys-search');
  if (/classical|tree|regression|svm|cluster/.test(text)) domains.push('classical-ml');
  if (/deep learning|training|neural|distributed|gpu/.test(text)) domains.push('deep-learning');
  if (/nlp|speech|language|token|bert|rnn/.test(text)) domains.push('nlp-speech');
  if (/vision|image|object detection|cnn/.test(text)) domains.push('computer-vision');
  return [...new Set(domains)];
}

function estimatedMinutes(category: ResourceCategory, taskType: Exclude<TaskType, 'review' | 'simulation'>, wordCount: number): number {
  const readTime = Math.max(5, Math.ceil(wordCount / 220));
  if (category === 'concepts') return Math.min(30, readTime + 10);
  if (category === 'guides') return Math.min(45, readTime + 15);
  if (taskType === 'story' || taskType === 'derive') return Math.min(50, readTime + 30);
  if (taskType === 'design') return Math.min(60, readTime + 35);
  return Math.min(50, readTime + 25);
}

export function buildCatalog(repoRoot = process.cwd()): CatalogResource[] {
  const postsDirectory = path.join(repoRoot, 'src/content/posts');
  if (!fs.existsSync(postsDirectory)) throw new Error(`Content directory not found: ${postsDirectory}`);

  const seen = new Set<string>();
  const resources = fs.readdirSync(postsDirectory)
    .filter((file) => /\.mdx?$/.test(file))
    .sort()
    .map((file) => {
      const filePath = path.join(postsDirectory, file);
      const parsed = matter(fs.readFileSync(filePath, 'utf8'));
      const category = parsed.data.category as ResourceCategory;
      if (!['questions', 'guides', 'concepts'].includes(category)) {
        throw new Error(`Unsupported category in ${file}: ${String(parsed.data.category)}`);
      }

      const slug = stripDatePrefix(file.replace(/\.mdx?$/, ''));
      if (seen.has(slug)) throw new Error(`Duplicate content slug: ${slug}`);
      seen.add(slug);

      const title = String(parsed.data.title ?? slug);
      const description = String(parsed.data.description ?? '');
      const tags = Array.isArray(parsed.data.tags) ? parsed.data.tags.map(String) : [];
      const subcategory = inferSubcategory(category, slug);
      const areas = inferAreas(category, slug, title, tags);
      const primaryArea = areas[0];
      const taskType = inferTaskType(category, primaryArea);
      const plainText = stripMarkdown(parsed.content);
      const wordCount = plainText ? plainText.split(/\s+/).length : 0;
      const route = `/${category}/${slug}/`;

      const resource: CatalogResource = {
        slug,
        title,
        description,
        category,
        subcategory,
        route,
        absoluteUrl: `https://mlmentorship.com${route}`,
        tags,
        wordCount,
        readingMinutes: Math.max(1, Math.ceil(wordCount / 220)),
        taskType,
        estimatedMinutes: estimatedMinutes(category, taskType, wordCount),
        areas,
        roles: AREA_ROLES[primaryArea],
        levels: inferLevels(category),
        rounds: AREA_ROUNDS[primaryArea],
        domainTracks: inferDomains(subcategory, title, tags),
        priority: category === 'questions' ? 35 : category === 'guides' ? 30 : 20,
        prerequisites: [],
      };
      return applyResourceOverride(resource);
    });

  return resources.sort((a, b) => a.route.localeCompare(b.route));
}

export function catalogBySlug(catalog: CatalogResource[]): Map<string, CatalogResource> {
  return new Map(catalog.map((resource) => [resource.slug, resource]));
}
